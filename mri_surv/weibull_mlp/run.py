import torch
import os, sys
import glob
from scipy import interpolate
import pandas as pd
import numpy as np
from torchviz import make_dot
from tqdm import tqdm
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from torch.optim import Adam
from weibull_mlp.datas import ParcellationData
from weibull_mlp.losses import WeibullLoss
from weibull_mlp.weibull import WeibullMLP, weibull_cdf, weibull_cdf_debug

DEVICE = torch.device("cpu")

NB_EPOCH = 1000
BATCH_SIZE = 128


def make_struc_array(hits: np.ndarray, obss: np.ndarray) -> np.ndarray:
    """
    Creates a structured array from boolean hits (progresses == True) and
    obss (time of censoring or hit)

    Args:
        hits (np.ndarray): 1 or 0 progresses or does not progress
        obss (np.ndarray): time of event or censoring

    Returns:
        np.ndarray: structured array with fields "hit" and "time"
    """
    return np.array(
        [(x, y) for x, y in zip(hits == 1, obss)],
        dtype=[("hit", bool), ("time", float)],
    )



class WeibullWrapper:
    """
    This has the functionality to run the Weibull MLP
    """

    def __init__(self, exp_idx=1) -> None:
        torch.manual_seed(exp_idx)  # set seed
        self.seed = exp_idx
        self.model_name = "weibull_mlp"
        self.exp_idx = exp_idx
        self.device = DEVICE  # cpu for now
        self.c_index = []
        self.checkpoint_dir = "./checkpoint_dir/{}_exp{}/".format("weibull", exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.dataset = ParcellationData
        self.prepare_dataloader(exp_idx)
        self.criterion = WeibullLoss()
        self.model = WeibullMLP(in_size=self.in_size).float()
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.optimal_path = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch = -1

    def save_checkpoint(self, loss: torch.Tensor) -> None:
        """
        loss tensor that is 0d with a single value. saves the current model
        if the loss is less than the previous optimal valid metric loss

        Args:
            loss (torch.Tensor): 0d tensor with loss on current valid data
        """

        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for _, _, files in os.walk(self.checkpoint_dir):
                for file in files:
                    if file.endswith(".pth"):
                        os.remove(self.checkpoint_dir + file)
            torch.save(
                self.model.state_dict(),
                "{}{}_{}.pth".format(
                    self.checkpoint_dir, self.model_name, self.optimal_epoch
                ),
            )

    def load_checkpoint(self) -> None:
        """
        retrieves checkpoint from directory given model name
        """
        fi = glob.glob(f"{self.checkpoint_dir}{self.model_name}_*.pth")
        assert len(fi) == 1
        self.model.load_state_dict(torch.load(fi[0]))
        self.optimal_path = fi[0]

    def prepare_dataloader(self, exp_idx: int) -> None:
        """
        Given an exp_idx, prepares the dataloaders to be used
        in the experiment, including training, validation, testing, and external data.


        Args:
            exp_idx (int): integer, only tested for values between 0 and 4
        """
        assert exp_idx >= 0 and exp_idx <= 4
        train_data = self.dataset(exp_idx=exp_idx, stage="train", dataset="ADNI")
        assert len(train_data) == 108 * 3
        self.features = train_data.get_features()
        valid_data = self.dataset(exp_idx=exp_idx, stage="valid", dataset="ADNI")
        assert len(valid_data) == 108
        test_data = self.dataset(exp_idx=exp_idx, stage="test", dataset="ADNI")
        assert len(test_data) == 108
        nacc_data = self.dataset(exp_idx=exp_idx, stage="all", dataset="NACC")
        adni_data_all = self.dataset(exp_idx=exp_idx, stage="all", dataset="ADNI")
        self.train_dataloader = DataLoader(
            train_data, batch_size=BATCH_SIZE, drop_last=False
        )
        self.valid_dataloader = DataLoader(valid_data, batch_size=len(valid_data))
        self.test_dataloader = DataLoader(
            test_data, batch_size=len(test_data), shuffle=False
        )
        self.nacc_dataloader = DataLoader(
            nacc_data, batch_size=len(nacc_data), shuffle=False
        )
        self.all_dataloader = DataLoader(
            adni_data_all, batch_size=len(adni_data_all), shuffle=False
        )

        self.in_size = train_data.data.shape[1]
        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data
        self.nacc_data = nacc_data

    def train(self) -> float:
        """
        Trains using number of epochs global constant NB_EPOCHs,
        saving the minimum loss on validation data. No early stopping

        Returns:
            float: validation metric on optimal parameters
        """
        # no L2 norm in adam, we will add it in ourselves
        for self.epoch in tqdm(range(NB_EPOCH)):
            self.train_model_epoch(self.optimizer)
            val_loss = self.valid_model_epoch()
            self.save_checkpoint(val_loss)
        self.optimal_path = f'{self.checkpoint_dir}{self.model_name}_{self.optimal_epoch}.pth'
        return self.optimal_valid_metric

    def train_model_epoch(self, optimizer) -> None:
        """
        This trains a single epoch

        Args:
            optimizer (_type_): _description_
        """
        self.model.train(True)
        for inputs, _, _, _, truth in self.train_dataloader:
            self.model.zero_grad()
            preds = self.model(inputs.to(self.device).float())
            preds = weibull_cdf(preds)
            loss = self.criterion(preds, truth)
            # adding in l1, l2 norm for just the hidden layer parameters

            for layer, val in self.model.named_parameters():
                if "fc" in layer:
                    loss += 0.001 * val.abs().sum()
                    loss += 0.001 * (val**2).sum()
            loss.backward()
            optimizer.step()
        # make_dot(
        #     loss,
        #     params=dict(self.model.named_parameters()),
        #     show_attrs=True,
        #     show_saved=True,

        # ).render("torchviz_orig", format="png")

    def valid_model_epoch(self) -> torch.Tensor:
        """
        Tests the current model on the validation dataset

        Returns:
            torch.Tensor: 0-d tensor of loss for current epoch
        """

        with torch.no_grad():
            self.model.train(False)
            for data, _, _, _, truth in self.valid_dataloader:
                preds = self.model(data.to(self.device).float())
                preds = weibull_cdf(preds)
                loss = self.criterion(preds, truth)
                for layer, val in self.model.named_parameters():
                    if "fc" in layer:
                        loss += 0.001 * val.abs().sum()
                        loss += 0.001 * (val**2).sum()
        return loss

    def eval_data_optimal_epoch(self, external_data=False):
    
        if external_data:
            dataloader = self.nacc_dataloader
        else:
            dataloader = self.test_dataloader
        with torch.no_grad():
            self.load_checkpoint()
            self.model.train(False)
            for data, _, _, rids, truth in dataloader:
                preds = self.model(data.to(self.device)).to("cpu")
                preds = weibull_cdf(preds)
                return preds, rids, truth

    def retrieve_testing_data(self, external_data, train_on_all=False):
        if external_data:
            dataloader = self.nacc_dataloader
        else:
            dataloader = self.test_dataloader
        with torch.no_grad():
            self.load_checkpoint()
            self.model.train(False)
            for data, obss, hits, rids, _ in dataloader:
                preds = self.model(data.float().to(self.device)).to("cpu")
                # preds = weibull_cdf(preds)
                preds = weibull_cdf(preds)
                rids = rids
                test_struc = make_struc_array(hits, obss)
            if train_on_all and external_data:
                train_dataloader = self.nacc_dataloader
            elif train_on_all:
                train_dataloader = self.all_dataloader
            else:
                train_dataloader = self.train_dataloader
            hits_train = []
            obss_train = []
            for _, obss, hits, _, _ in train_dataloader:
                hits_train.append(hits)
                obss_train.append(obss)
            hits_train = torch.cat(hits_train, dim=-1)
            obss_train = torch.cat(obss_train, dim=-1)
            train_struc = make_struc_array(hits_train, obss_train)
        return preds, train_struc, test_struc, rids

    def get_preds(self, external_data: bool):
        preds, _, test_struc, rids = self.retrieve_testing_data(external_data)
        samples, bins = preds.shape
        labels = np.arange(0, 12*bins, 12).reshape(1, -1)
        labels = np.repeat(labels, samples, axis=0)
        rids = np.asarray(rids).reshape(-1, 1)
        rids = np.repeat(rids, bins, axis=1)
        hit = test_struc['hit'].reshape(-1, 1)
        hit = np.where(np.repeat(hit, bins, axis=1), 1, 0)
        time = test_struc['time'].reshape(-1, 1)
        time = np.repeat(time, bins, axis=1)
        dataset = 'NACC' if external_data else 'ADNI'
        dataset = [dataset]*samples*bins
        experiment = [self.exp_idx]*samples*bins
        df = pd.DataFrame({
            'RID': rids.reshape(-1, 1).squeeze(),
            'Bins': labels.reshape(-1, 1).squeeze(),
            'Predictions': 1-preds.reshape(-1, 1).squeeze(),
            'Dataset': dataset,
            'Experiment': experiment,
            'Progresses': hit.reshape(-1, 1).squeeze(),
            'Time': time.reshape(-1, 1).squeeze()
        })

        return df

    def ci_surv_data_optimal_epoch(
        self, bin_=3, external_data=False, return_preds=False, train_on_all=False
    ) -> float:
        preds, _, test_struc, _ = self.retrieve_testing_data(
            external_data, train_on_all=train_on_all
        )
        c_index = concordance_index_censored(
            test_struc["hit"], test_struc["time"], np.squeeze(preds[:, bin_])
        )
        return c_index[0]

    def brier_surv_data_optimal_epoch(self, external_data=False, train_on_all=False):
        preds, train_struc, test_struc, _ = self.retrieve_testing_data(
            external_data, train_on_all=train_on_all
        )
        brier_scores = retrieve_brier_scores(train_struc, test_struc, 1 - preds)
        return brier_scores[0]


def retrieve_brier_scores(train_struc, test_struc, preds_raw):
    new_max = min(float(max(test_struc['time'])), 108)
    bins = list(range(0, 120, 12))
    truncated_bins = [0, 24, 48, new_max-1]
    interp = interpolate.PchipInterpolator(bins, preds_raw, axis=1)
    preds_brier = interp(truncated_bins)
    brier_scores = integrated_brier_score(
        train_struc, test_struc, preds_brier, truncated_bins)
    return brier_scores, interp


def train_weibull() -> None:
    mlps = []
    ci_adni = []
    bs_adni = []
    bs_adni_all = []

    ci_nacc = []
    bs_nacc = []
    bs_nacc_all = []
    preds = []

    for idx in range(5):
        mlps.append(WeibullWrapper(idx))
        mlps[idx].train()

        ci_adni.append(mlps[idx].ci_surv_data_optimal_epoch(external_data=False))
        bs_adni.append(mlps[idx].brier_surv_data_optimal_epoch(external_data=False))
        bs_adni_all.append(
            mlps[idx].brier_surv_data_optimal_epoch(
                external_data=False,
                train_on_all=True
            )
        )
        preds.append(mlps[idx].get_preds(False))

        ci_nacc.append(mlps[idx].ci_surv_data_optimal_epoch(external_data=True))
        bs_nacc.append(mlps[idx].brier_surv_data_optimal_epoch(external_data=True))
        bs_nacc_all.append(
            mlps[idx].brier_surv_data_optimal_epoch(
                external_data=True, train_on_all=True
            )
        )
        preds.append(mlps[idx].get_preds(True))

        print("Cycle {}/5 complete".format(idx + 1))

    preds = pd.concat(preds, axis=0, ignore_index=True)

    preds.to_csv(
        './metadata/data_processed/weibull_predictions.csv', index=False)

    print("ADNI_____")
    print(ci_adni, np.mean(ci_adni), np.std(ci_adni))
    print(bs_adni, np.mean(bs_adni), np.std(bs_adni))
    print(bs_adni_all, np.mean(bs_adni_all), np.std(bs_adni_all))

    print("NACC_____")
    print(ci_nacc, np.mean(ci_nacc), np.std(ci_nacc))
    print(bs_nacc, np.mean(bs_nacc), np.std(bs_nacc))
    print(bs_nacc_all, np.mean(bs_nacc_all), np.std(bs_nacc_all))
