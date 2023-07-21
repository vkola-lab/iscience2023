import torch
from torch import nn, Tensor

def sur_loss(preds, obss, hits, bins=Tensor([[0, 24, 48, 108]])):
    bin_centers = (bins[0, 1:] + bins[0, :-1])/2
    survived_bins_censored = torch.ge(torch.mul(obss.view(-1, 1),1-hits.view(
            -1,1)), bin_centers)
    survived_bins_hits = torch.ge(torch.mul(obss.view(-1,1), hits.view(-1,1)),
                                  bins[0,1:])
    survived_bins = torch.logical_or(survived_bins_censored, survived_bins_hits)
    survived_bins = torch.where(survived_bins, 1, 0)
    event_bins = torch.logical_and(torch.ge(obss.view(-1, 1), bins[0, :-1]),
                 torch.lt(obss.view(-1, 1), bins[0, 1:]))
    event_bins = torch.where(event_bins, 1, 0)
    hit_bins = torch.mul(event_bins, hits.view(-1, 1))
    l_h_x = 1+survived_bins*(preds-1)
    n_l_h_x = 1-hit_bins*preds
    cat_tensor = torch.cat((l_h_x, n_l_h_x), axis=0)
    total = -torch.log(torch.clamp(cat_tensor, min=1e-12))
    pos_sum = torch.sum(total)
    neg_sum = torch.sum(pos_sum)
    return neg_sum

class NNSurvLoss(nn.Module):
    def __init__(self, bins):
        self.bins = bins
        self.bin_centers = (bins[0, 1:] + bins[0, :-1])/2
        super(NNSurvLoss, self).__init__()

    def forward(self,
                preds: torch.Tensor,
                obss: torch.Tensor,
                hits: torch.Tensor):
        survived_bins_censored = torch.ge(torch.mul(obss.view(-1, 1),1-hits.view(
                -1,1)), self.bin_centers)
        survived_bins_hits = torch.ge(torch.mul(obss.view(-1,1), hits.view(-1,1)),
                                      self.bins[0,1:])
        survived_bins = torch.logical_or(survived_bins_censored, survived_bins_hits)
        survived_bins = torch.where(survived_bins, 1, 0)
        event_bins = torch.logical_and(torch.ge(obss.view(-1, 1), self.bins[0,:-1]),
                     torch.lt(obss.view(-1, 1), self.bins[0, 1:]))
        event_bins = torch.where(event_bins, 1, 0)
        hit_bins = torch.mul(event_bins, hits.view(-1, 1))
        l_h_x = 1+survived_bins*(preds-1)
        n_l_h_x = 1-hit_bins*preds
        cat_tensor = torch.cat((l_h_x, n_l_h_x), axis=0)
        total = -torch.log(torch.clamp(cat_tensor, min=1e-12))
        pos_sum = torch.sum(total)
        neg_sum = torch.sum(pos_sum)
        return neg_sum

class NNSurvLossSimple(nn.Module):
    def __init__(self, bins: torch.Tensor):
        self.bins = bins
        self.bin_centers = (bins[0, 1:] + bins[0, :-1])/2
        super(NNSurvLossSimple, self).__init__()

    def forward(self,
                preds: torch.Tensor,
                obss: torch.Tensor,
                hits: torch.Tensor):
        censored = obss.view(-1,1) * (1-hits.view(-1,1))
        survived_bins_censored = censored >= self.bin_centers
        bins_hits = obss.view(-1,1) * hits.view(-1,1)
        hit_bins = bins_hits >= self.bins[0,1:]
        survived_bins = survived_bins_censored+hit_bins-survived_bins_censored*hit_bins
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