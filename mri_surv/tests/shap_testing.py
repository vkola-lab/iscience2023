import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from simple_mlps.datas import ParcellationData
from simple_mlps.simple_models import _MLP_Surv

class NNSurvLoss(nn.Module):
    def __init__(self, bins: torch.Tensor):
        self.bins = bins
        self.bin_centers = (bins[0, 1:] + bins[0, :-1])/2
        super(NNSurvLoss, self).__init__()

    def forward(self,
                preds: torch.Tensor,
                obss: torch.Tensor,
                hits: torch.Tensor):
        censored = obss.view(-1,1) * (1-hits.view(-1,1))
        survived_bins_censored = censored >= self.bin_centers
        bins_hits = obss.view(-1,1) * hits.view(-1,1)
        hit_bins = bins_hits >= self.bins[0,1:]
        survived_bins = (survived_bins_censored+hit_bins)^survived_bins_censored*hit_bins
        survived_event_bins_ = obss.view(-1,1) >= self.bins[0,:-1]
        not_survived_event_bins_ = obss.view(-1,1) < self.bins[0,1:]
        # compute bins where the event occured
        event_bins = survived_event_bins_*not_survived_event_bins_
        hit_bins = event_bins*hits.view(-1, 1)
        l_h_x = 1+survived_bins*(preds-1)
        n_l_h_x = 1-hit_bins*preds
        cat_tensor = torch.cat((l_h_x, n_l_h_x), axis=0)
        total = -torch.log(torch.clamp(cat_tensor, min=1e-12))
        pos_sum = torch.sum(total)
        neg_sum = torch.sum(pos_sum)
        return neg_sum

class SurvivalWrapper:
    def __init__(self, fold_number):
        self.model = _MLP_Surv(
            in_size=66,
            drop_rate=0.1,
            fil_num=100,
            output_shape=3).cuda()
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss = NNSurvLoss(torch.Tensor([[0, 24, 48, 108]]).cuda())
        self.dataset = ParcellationData
        self.train_data = self.dataset(fold_number, stage='train')
        self.train_dataloader = DataLoader(self.train_data, batch_size=60, shuffle=True, drop_last=True)
        self.valid_data = self.dataset(fold_number, stage='valid')
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=len(self.valid_data), shuffle=False)
        self.test_data = self.dataset(fold_number, stage='test')
        self.test_dataloader = DataLoader(self.test_data, batch_size=len(self.test_data), shuffle=False)

    def train(self, n_epochs: int):
        min_loss = float('inf')
        for epoch in range(n_epochs):
            self.train_epoch()
            loss = self.validate()
            if loss < min_loss:
                self.save_epoch(epoch)

    def save_epoch(self):
        raise NotImplementedError
        pass

    def train_epoch(self):
        self.model.train()
        for data, obss, hits, _ in self.train_dataloader:
            self.model.zero_grad()
            pred = self.model(data.cuda())
            loss = self.loss(pred, obss.cuda(), hits.cuda())
            loss.backward()
            self.optimizer.step()

    def validate(self):
        with self.model.eval():
            data, obss, hits, _ = self.valid_data[:]
            print(data)
            pred = self.model(data.cuda())
            loss = self.loss(pred, obss.cuda(), hits.cuda())
        return loss

    def test(self):
        pass

def main():
    wrapper = SurvivalWrapper(1)
    wrapper.train(n_epochs=10)

if __name__ == '__main__':
    main()