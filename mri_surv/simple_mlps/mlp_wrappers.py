from copy import deepcopy
import numpy as np
import torch
import os, sys
import json
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score, cumulative_dynamic_auc
from scipy import interpolate
from simple_mlps.loss_functions import sur_loss, NNSurvLoss
from simple_mlps.datas import ParcellationDataVentricles, \
    ParcellationDataVentriclesNacc, ParcellationDataGMVCSF, ParcellationDataCensored, ParcellationDataCSF
from simple_mlps.simple_models import _MLP, _MLP_Surv
import logging
import shap
from tabulate import tabulate
from tests.tests import mutable_argument_tester
import pandas as pd
import seaborn
import glob
from icecream import ic
import tqdm
import torchviz

logging.basicConfig(filename='logs/mlp_training_scores.log',
                    level=logging.INFO)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

@mutable_argument_tester
def write_raw_score(f, preds, labels):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')

class MLP_Wrapper_Meta:
    def __init__(self, exp_idx,
                 model_name, lr, weight_decay, model, model_kwargs,
                 dataset_kwargs,
                 criterion, dataset=ParcellationDataVentricles,
                 dataset_external=ParcellationDataVentriclesNacc):
        self.seed = exp_idx
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.device = DEVICE
        self.dataset = dataset
        self.dataset_kwargs = dataset_kwargs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dataset_external = dataset_external
        self.c_index = []
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.prepare_dataloader(exp_idx)
        self.criterion = criterion
        torch.manual_seed(exp_idx)
        self.model = model(in_size=self.in_size, **model_kwargs).float()
        self.model.to(self.device)

    def save_checkpoint(self, loss):
        # if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:

        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for _, _, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(),
                       '{}{}_{}.pth'.format(
                    self.checkpoint_dir, self.model_name, self.optimal_epoch)
                       )

    def load_checkpoint(self):
        fi = glob.glob(f'{self.checkpoint_dir}{self.model_name}_*.pth')
        assert(len(fi) == 1)
        self.model.load_state_dict(torch.load(fi[0]))
        self.optimal_path = fi[0]

    def prepare_dataloader(self, seed):
        train_data = self.dataset(seed=seed, stage = 'train',
                                  **self.dataset_kwargs)
        self.features = train_data.get_features()
        valid_data = self.dataset(seed=seed, stage = 'valid',
                                  **self.dataset_kwargs)
        test_data = self.dataset(seed=seed, stage = 'test',
                                 **self.dataset_kwargs)
        all_data = self.dataset(seed=seed, stage='all', **self.dataset_kwargs)
        nacc_data = self.dataset_external(self.seed, stage='all')
        self.train_dataloader = DataLoader(train_data, batch_size=len(train_data))
        self.valid_dataloader = DataLoader(valid_data, batch_size=len(valid_data))
        self.test_dataloader = DataLoader(test_data, batch_size=len(test_data),
                                          shuffle=False)
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data),
                                         shuffle=False)
        self.nacc_dataloader = DataLoader(nacc_data, batch_size=len(nacc_data),
                                shuffle=False)
        self.in_size = train_data.data.shape[1]
        self.all_data = all_data
        self.train_data = train_data
        self.test_data = test_data
        self.nacc_data = nacc_data

    def train(self, epochs):
        self.val_loss = []
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(
                0.5, 0.999), weight_decay=self.weight_decay)
        for self.epoch in range(epochs):
                self.train_model_epoch(self.optimizer)
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                self.val_loss.append(val_loss)
                if self.epoch % 300 == 0:
                #     print('{}: {}th epoch validation score: {}'.format(
                #             self.model_name, self.epoch, val_loss))
                    logging.debug('{}: {}th epoch validation score:{:.4f}'.format(
                            self.model_name, self.epoch, val_loss))
                    logging.debug('{}:')

        # print('Best model saved at the {}th epoch; cox-based loss: {}'.format(
                # self.optimal_epoch, self.optimal_valid_metric))
        self.optimal_path = '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)
        # print('Location: '.format(self.optimal_path))
        return self.optimal_valid_metric

    def train_model_epoch(self, optimizer):
        self.model.train(True)
        for inputs, obss, hits, _ in self.train_dataloader:
            if torch.sum(hits) == 0:
                continue
            self.model.zero_grad()
            preds = self.model(inputs.to(self.device))
            loss = self.criterion(preds.to('cpu'), obss, hits)
            if self.epoch == 0:
                torchviz.make_dot(
                    loss,
                    params=dict(self.model.named_parameters()),
                    ).render("figures/mlp_loss_function", format="png")
            loss.backward()
            optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            for data, obss, hits, _ in self.valid_dataloader:
                preds = self.model(data.to(self.device))
                loss = self.criterion(preds.to('cpu'), obss,
                                      hits)
        return loss

    def eval_data_optimal_epoch(self, external_data=False):
        if external_data:
            dataloader = self.nacc_dataloader
        else:
            dataloader = self.test_dataloader
        with torch.no_grad():
            self.load_checkpoint()
            self.model.train(False)
            for data, _, _, rids in dataloader:
                preds = self.model(data.to(self.device)).to('cpu')
                return preds, rids

    def retrieve_testing_data(self, external_data, train_on_all=False):
        if external_data:
            dataloader = self.nacc_dataloader
        else:
            dataloader = self.test_dataloader
        with torch.no_grad():
            self.load_checkpoint()
            self.model.train(False)
            for data, obss, hits, rids in dataloader:
                preds = self.model(data.to(self.device)).to('cpu')
                rids = rids
                test_struc = make_struc_array(hits, obss)
            if train_on_all and external_data:
                train_dataloader = self.nacc_dataloader
            elif train_on_all:
                train_dataloader = self.all_dataloader 
            else:
                train_dataloader = self.train_dataloader
            for _, obss_train, hits_train, _ in train_dataloader:
                train_struc = make_struc_array(hits_train, obss_train)
        return preds, train_struc, test_struc, rids

    def test_surv_data_optimal_epoch(self, bins, concordance_time=24,
                                     external_data=False, return_preds = False, train_on_all=False):
        preds, train_struc, test_struc, rids = self.retrieve_testing_data(external_data, train_on_all=train_on_all)                             
        preds_raw = np.concatenate((np.ones((preds.shape[0],1)),
                          np.cumprod(preds.numpy(), axis=1)), axis=1)
        c_index = concordance_index_censored(test_struc['hit'], test_struc['time'],
                                             1-np.squeeze(preds_raw[:,bins==concordance_time]))
        brier_scores, interp = retrieve_brier_scores(bins, preds_raw, train_struc, test_struc)
        if return_preds:
            return c_index[0], brier_scores, preds_raw, rids, interp
        return c_index[0], brier_scores

    def test_surv_data_optimal_epoch_fixed(self,
                                     external_data=False, 
                                     train_on_all=False):
        preds, train_struc, test_struc, rids = self.retrieve_testing_data(external_data, train_on_all=train_on_all)                             
        preds_raw = np.concatenate((np.ones((preds.shape[0],1)),
                          np.cumprod(preds.numpy(), axis=1)), axis=1)
        brier_scores = _retrieve_brier_scores_fixed_time(train_struc, test_struc, preds_raw)
        return brier_scores

def make_struc_array(hits, obss):
    return np.array([(x,y) for x,y in zip(hits == 1, obss)], dtype=[('hit',bool),('time',float)])

def retrieve_brier_scores(bins, preds_raw, train_struc, test_struc):
    bins = bins.copy()
    new_max = min(float(max(test_struc['time'])),108)
    truncated_bins = np.concatenate([bins[:-1],[new_max-1]], axis=-1)
    interp = interpolate.PchipInterpolator(bins, preds_raw, axis=1)
    preds_brier = interp(truncated_bins)
    brier_scores = integrated_brier_score(train_struc, test_struc, preds_brier, truncated_bins)
    return brier_scores, interp

def _retrieve_brier_scores_fixed_time(train_struc, test_struc, preds, times=(0,24,48,84,)):
    interp = interpolate.PchipInterpolator((0,24,48,108,), preds, axis=1)
    preds_brier = interp(times)
    brier_scores = brier_score(train_struc, test_struc, preds_brier, times)
    return brier_scores

class SurLossUtility:
    def __init__(self, model_name="mlp_csf_sur_loss", mlp_repeat=5,
                 fname='./results/test_model.txt', train_on_all=False):
        self._json_props = read_json(
                './simple_mlps/mlp_config.json')
        self.mlp_setting = self._json_props[model_name]
        self.model_name = model_name
        self.bins = self.mlp_setting["bins"]
        if len(self.bins) == 0:
            self._compute_bins()
        self.c_index = {}
        self.fixed_brier = {}
        self.fname = fname
        self.mlp_repeat = mlp_repeat
        self.mlps = self._create_mlps()
        self.features = self.mlps[0].features
        self.train_on_all = train_on_all

    def _compute_bins(self):
        self.bins = np.array([0, 24, 48, 108])

    def train(self, force=False):
        for mlp in self.mlps:
            if not force:
                try:
                    mlp.load_checkpoint()
                except:
                    print('Could not load file! Trying to generate checkpoints rn')
                    mlp.train(epochs=self.mlp_setting['train_epochs'])
                    mlp.load_checkpoint()
            else:
                mlp.train(epochs=self.mlp_setting['train_epochs'])
                mlp.load_checkpoint()

    def get_c_index(self, concordance_time=24):
        self.c_index['ADNI'] = []
        self.fixed_brier['ADNI'] = []
        for mlp in self.mlps:
            self.c_index['ADNI'].append(list(mlp.test_surv_data_optimal_epoch(self.bins,
                                                                 concordance_time, train_on_all=self.train_on_all)))
            brier_scores_fixed = mlp.test_surv_data_optimal_epoch_fixed(
                                     external_data=False, 
                                     train_on_all=self.train_on_all)
            self.fixed_brier['ADNI'].append(brier_scores_fixed)
            
    def get_c_index_external(self, concordance_time=24):
        self.c_index['NACC'] = []
        self.fixed_brier['NACC'] = []
        for mlp in self.mlps:
            c_idx, brier_scores, = mlp.test_surv_data_optimal_epoch(
                    self.bins, concordance_time,external_data=True, train_on_all=self.train_on_all)
            self.c_index['NACC'].append([c_idx, brier_scores])
            brier_scores_fixed = mlp.test_surv_data_optimal_epoch_fixed(
                                     external_data=True, 
                                     train_on_all=self.train_on_all)
            self.fixed_brier['NACC'].append(brier_scores_fixed)

    def write_c_index(self):
        with open(self.fname, 'a') as fi:
            fi.write('\n' + self.model_name + ', censoring using all data: ' + str(self.train_on_all) + '\n')
            tbl = [['Dataset','Statistic', 'Mean', 'Std']]
            for ds in self.c_index.keys():
                tbl.append([ds, 'C-index:', str(np.mean(self.c_index[ds],0)[0]), str(np.std(self.c_index[ds],0)[0])])
                tbl.append([ds, 'Brier Score:', str(np.mean(self.c_index[ds],0)[1]), str(np.std(self.c_index[ds],0)[1])])
            fi.write(
                tabulate(
                    tbl, tablefmt="tsv"
                )
            )

    def write_fixed_brier(self):
        with open(self.fname, 'a') as fi:
            fi.write('\n' + self.model_name + ' Fixed Brier Scores' + ', censoring using all data: ' + str(self.train_on_all) + '\n')
            tbl  = []
            headers = ('Dataset','Statistic','Mean','Std',)
            for dataset in self.fixed_brier.keys():
                months = self.fixed_brier[dataset][0][0]
                values = np.concatenate([y.reshape(1,-1) for _, y in self.fixed_brier[dataset]], axis=0)
                mn = np.mean(values, axis=0)
                sd = np.std(values, axis=0)
                tbl += [[dataset,f'Brier score {int(x)} mos',y,z] for x,y,z in zip(months,mn,sd)]
            fi.write(tabulate(tbl, headers=headers,tablefmt="tsv"))
            
    def _create_mlps(self):
        mlps = []
        for repe_idx in range(self.mlp_repeat):
            mlps.append(
                MLP_Wrapper_Meta(
                    repe_idx,
                    model_name=self.model_name,
                    lr=self.mlp_setting['learning_rate'],
                    weight_decay=self.mlp_setting['weight_decay'],
                    model=eval(self.mlp_setting['model']),
                    model_kwargs={
                            'fil_num'  : self.mlp_setting["fil_num"],
                            'drop_rate': self.mlp_setting["drop_rate"],
                            'output_shape': len(self.bins)-1
                    },
                    dataset_kwargs={

                    },
                    criterion=lambda x,y,z: eval(self.mlp_setting[
                        'criterion'])(x,y,z,torch.Tensor([self.bins])),
                    dataset=eval(self.mlp_setting['dataset']),
                    dataset_external=eval(self.mlp_setting[
                                            'dataset_external'])
                )
            )
        return mlps

    def write_model_predictions(self):
        preds = []
        labels = []
        exp_no = []
        dataset = []
        for data in ['ADNI','NACC']:
            for idx, mlp in enumerate(self.mlps):
                pred, label = mlp.eval_data_optimal_epoch(
                        external_data=(data == 'NACC'))
                pred = _format_pred(pred)
                preds.append(pred)
                labels.append(label)
                exp_no.append(np.ones(pred.shape)*idx)
                dataset.append(np.asarray([data]*pred.shape[0]))

        labels = np.concatenate(labels, axis=0).reshape((-1,1))
        preds = np.concatenate(preds, axis=0)
        exp_no = np.concatenate(exp_no, axis=0)
        dataset = np.concatenate(dataset,axis=0).reshape((-1,1))
        preds, labels, exp_no, bins, dataset = _format_mlp_labels(
                preds, labels, exp_no, self.bins.reshape((1,-1)), dataset)
        write_preds_labels(preds, labels, exp_no, self.model_name, bins=bins, dataset=dataset)

    def shapify(self, internal=True):
        shap_values = []
        test_values = []
        rid_values = []
        exp_no = []
        for idx, mlp in enumerate(self.mlps):
            if internal:
                test_dataloader = mlp.test_dataloader
            else:
                test_dataloader = mlp.nacc_dataloader
            for train_inputs, _, _, _ in mlp.train_dataloader:
                background = train_inputs.to('cpu')
                mlp.load_checkpoint()
                e = shap.DeepExplainer(mlp.model.to('cpu'), background)
                for test_inputs, _, _, rid in test_dataloader:
                    shap_values_current = e.shap_values(test_inputs.to('cpu'))
                    shap_values.append(np.asarray(shap_values_current))
                    test_inputs = np.expand_dims(np.asarray(test_inputs),0)
                    test_values.append(np.concatenate([test_inputs]*len(shap_values_current), axis=0))
                    rid = np.expand_dims(np.asarray(rid),0)
                    rid_values.append(np.concatenate([rid]*len(shap_values_current), axis=0))
                    ids = idx*np.ones((len(shap_values_current),
                                            test_inputs.shape[1]))
                    exp_no.append(ids)
        shap_values = np.concatenate(shap_values, axis=1)
        test_values = np.concatenate(test_values, axis=1)
        rid_values = np.concatenate(rid_values, axis=1)
        exp_no = np.concatenate(exp_no, axis=1)
        return _reshape_shap_values(shap_values, test_values, self.features,
                                    rid_values, exp_no)

    def shapify_all_and_write(self, use_external=True):
        if use_external:
            shap_dfs = []
            for val in [True, False]:
                shap_dfs.append(self.shapify(val))
            labels = np.array(['ADNI']).repeat(shap_dfs[0].shape[0])
            labels_nacc = np.array(['NACC']).repeat(shap_dfs[1].shape[0])
            labels = np.concatenate([labels, labels_nacc])
            shap_dfs = pd.concat(shap_dfs, ignore_index=False, axis=0)
            shap_dfs['Dataset'] = labels
            self.shap_df = shap_dfs.set_index(
                    ['Dataset', 'RID', 'Bin', 'ExpNo'])
            self.shap_df.to_csv(self._json_props['datadir'] +
                                self.model_name + '_shap_values.csv')
        else:
            shap_dfs = self.shapify(True)
            labels = np.array(['ADNI']).repeat(shap_dfs.shape[0])
            shap_dfs['Dataset'] = labels
            self.shap_df = shap_dfs.set_index(
                    ['Dataset', 'RID', 'Bin', 'ExpNo'])
            self.shap_df.to_csv(self._json_props['datadir'] +
                                self.model_name + '_shap_values.csv')

def _format_pred(pred):
    pred = np.concatenate([np.ones((pred.shape[0],1)),pred], axis=1)
    pred = np.cumprod(pred, axis=1)
    pred = np.clip(pred, a_min=0, a_max=1)
    return pred

def _get_bin_rid_labels(nbins, rid_values, exp_no):
    bin_labels = np.asarray(list(range(nbins))).reshape((-1,1))
    bin_labels = np.repeat(bin_labels, repeats=rid_values.shape[1], axis=0)
    bin_labels = bin_labels.reshape((-1,1), order='C').squeeze()
    rid_values = rid_values.reshape((-1,1), order='C').squeeze()
    exp_no = exp_no.reshape((-1,1), order='C').squeeze()
    return bin_labels, rid_values, exp_no

def _get_ndarrays(test_ndarray, shap_ndarray):
    nfeatures = shap_ndarray.shape[-1]
    test_ndarray = test_ndarray.reshape((-1,nfeatures), order='C')
    shap_ndarray = shap_ndarray.reshape((-1,nfeatures), order='C')
    return test_ndarray, shap_ndarray

@mutable_argument_tester
def _reshape_shap_values(shap_ndarray, test_ndarray, features, rid_labels,
                         exp_no):
    assert(shap_ndarray.ndim == 3)
    nbins, nsamples, nfeatures = shap_ndarray.shape
    bin_labels, rid_labels, exp_no = \
        _get_bin_rid_labels(nbins, rid_labels,exp_no)
    test_ndarray, shap_ndarray = _get_ndarrays(test_ndarray, shap_ndarray)
    index_df = pd.DataFrame(data={'RID': rid_labels, 'Bin': bin_labels,
                                  'ExpNo': exp_no})
    index = pd.MultiIndex.from_frame(index_df)
    shap_df = pd.DataFrame(data=shap_ndarray, columns=features,
                        index=index).reset_index()
    shap_df = pd.melt(shap_df, id_vars=['RID', 'Bin', 'ExpNo'])
    test_df = pd.DataFrame(data=test_ndarray, columns=features,
                           index=index).reset_index()
    test_df = pd.melt(test_df, id_vars=['RID', 'Bin','ExpNo'])
    return shap_df.merge(test_df, how="inner", left_on=['RID','Bin','ExpNo',
                                                        'variable'],
                         right_on=['RID', 'Bin', 'ExpNo','variable'],
                        suffixes=("", "_gmvol"), validate='one_to_one')

@mutable_argument_tester
def _format_mlp_labels(predictions, labels, exp_no, bins, dataset):
    labels = np.tile(labels, (1,predictions.shape[1]))
    bins = np.tile(bins, (predictions.shape[0],1))
    dataset = np.tile(dataset, (1,predictions.shape[1]))
    predictions = np.reshape(predictions, (-1, 1), order='C').squeeze()
    labels = np.reshape(labels, (-1,1), order='C').squeeze()
    exp_no = np.reshape(exp_no, (-1,1), order='C').squeeze()
    bins = np.reshape(bins, (-1, 1), order='C').squeeze()
    dataset = np.reshape(dataset, (-1,1), order='C').squeeze()
    return predictions, labels, exp_no, bins, dataset

@mutable_argument_tester
def write_preds_labels(preds, labels, exp_no, fname, dataset, bins):
    _json_props = read_json('./simple_mlps/mlp_config.json')
    df = pd.DataFrame(data={'Predictions': preds, 'Experiment':
        exp_no, 'Bins': bins, 'Dataset': dataset},
                      index=labels)
    df.to_csv(_json_props['datadir'] + fname + '.csv', index_label='RID')

def gmv_csf_sur_loss(force=True):
    gmv_csf_sur = SurLossUtility("mlp_parcellation_gmv_csf_sur_loss")
    gmv_csf_sur.train(force=force)
    gmv_csf_sur.get_c_index()
    gmv_csf_sur.get_c_index_external()
    gmv_csf_sur.write_c_index()
    gmv_csf_sur.write_fixed_brier()
    gmv_csf_sur.write_model_predictions()
    
def gmv_sensitivity():
    bins = np.array([0, 24, 48, 108])
    reps = list(range(10))
    risk_prs = list(np.arange(0, 20, 2))
    df = pd.DataFrame(index=pd.MultiIndex.from_product([risk_prs, reps], names=["Risk", "Rep"]), columns=["Brier", "CI"])
    max_and_min = {
            "train": {"max": np.inf, "min": 0},
            "test": {"max": np.inf, "min": 0}
        }
    for risk_pr in tqdm.tqdm(risk_prs):
        for j in reps:
            mlp = MLP_Wrapper_Meta(
                    exp_idx = 1,
                    model_name=f'mlp_parcellation_sensitivity_{j}',
                    lr=0.01,
                    weight_decay=0,
                    model=_MLP_Surv,
                    model_kwargs={
                            'fil_num'  : 100,
                            'drop_rate': 0.5,
                            'output_shape': len(bins)-1
                    },
                    dataset_kwargs={
                    },
                    criterion=lambda x,y,z: sur_loss(x,y,z,torch.Tensor([bins])),
                    dataset=lambda seed, stage: ParcellationDataCensored(
                        seed=seed, stage=stage, rand_exp=j, risk_pr=risk_pr/100),
                    dataset_external=ParcellationDataVentriclesNacc
                )
            mlp.train(epochs=1000)
            mlp.load_checkpoint()
            c_idx = []
            for bin_ in (24, 48, 108):
                preds, train_struc, test_struc, _ = mlp.retrieve_testing_data(external_data=True, train_on_all=False)                             
                preds_raw = np.concatenate((np.ones((preds.shape[0],1)),
                                np.cumprod(preds.numpy(), axis=1)), axis=1)
                c_index = concordance_index_censored(test_struc['hit'], test_struc['time'],
                                                    1-np.squeeze(preds_raw[:,bins==bin_]))
                c_idx.append(c_index[0])
            print(c_idx)
            print(max([x for x,y in zip(train_struc['time'], train_struc['hit']) if y == 1 ]))
            
            interp = interpolate.PchipInterpolator(bins, preds_raw, axis=1)
            preds_brier = interp([0, 24, 48])
            brier_score = integrated_brier_score(train_struc, test_struc, preds_brier, [0, 24, 48])
    
            print(brier_score)
            
            df.loc[(risk_pr, j),"Brier"] = brier_score
            df.loc[(risk_pr, j), "CI"] = np.mean(c_idx)
    df.to_csv("./metadata/data_processed/gmv_corr_with_ad_stats.csv")
    return df

def main(force=False):
    for censor_type in (True, False,):
        sur = SurLossUtility('mlp_parcellation_ventricles_sur_loss', train_on_all=censor_type)
        sur.train(force=force)
        sur.get_c_index()
        sur.get_c_index_external()
        sur.write_c_index()
        sur.write_fixed_brier()
        # sur.write_model_predictions()
        # sur.shapify_all_and_write()

        gmv_csf_sur = SurLossUtility("mlp_parcellation_gmv_csf_sur_loss", train_on_all=censor_type)
        gmv_csf_sur.train(force=force)
        gmv_csf_sur.get_c_index()
        gmv_csf_sur.get_c_index_external()
        gmv_csf_sur.write_c_index()
        gmv_csf_sur.write_fixed_brier()

        csfsur = SurLossUtility('mlp_csf_sur_loss', train_on_all=censor_type)
        csfsur.train(force=force)
        csfsur.get_c_index()
        csfsur.write_c_index()
        csfsur.write_fixed_brier()

        agesur = SurLossUtility('mlp_parcellation_ventricles_sur_loss_age', train_on_all=censor_type)
        agesur.train(force=force)
        agesur.get_c_index()
        agesur.write_c_index()
        agesur.write_fixed_brier()

        mmsesur = SurLossUtility('mlp_parcellation_ventricles_sur_loss_mmse', train_on_all=censor_type)
        mmsesur.train(force=force)
        mmsesur.get_c_index()
        mmsesur.write_c_index()
        mmsesur.write_fixed_brier()

        mmsesur = SurLossUtility('mlp_parcellation_ventricles_sur_loss_age_mmse', train_on_all=censor_type)
        mmsesur.train(force=force)
        mmsesur.get_c_index()
        mmsesur.write_c_index()
        mmsesur.write_fixed_brier()