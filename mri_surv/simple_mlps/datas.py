from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import random
import glob
import pandas as pd
import numpy as np
import csv
import re
import torch
import nibabel as nib
import json
import logging
import datetime
from icecream import ic
from lifelines import KaplanMeierFitter
from sksurv.nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator, check_y_survival, _compute_counts
from simple_mlps.model_ibs.ibs import make_struc_array
import matplotlib.pyplot as plt

fname = 'logs/datas.log'
with open(fname, 'w') as fi:
    fi.write(str(datetime.datetime.today()))

logging.basicConfig(filename=fname, level=logging.DEBUG)

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def _read_csv_cox(filename, skip_ids: list=None):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fileIDs, time_obs, hit, age, mmse = [], [], [], [], []
        for r in reader:
            if skip_ids is not None:
                if r['RID'] in skip_ids:
                    continue
            fileIDs += [str(r['RID'])]
            time_obs += [float(r['TIMES'])]  # changed to TIMES_ROUNDED. consider switching so observations for progressors are all < 1 year
            hit += [int(float(r['PROGRESSES']))]
            age += [float(r['AGE'])]
            if 'MMSCORE_mmse' in r.keys():
                mmse += [float(r['MMSCORE_mmse'])]
            else:
                mmse += [np.nan if r['MMSE'] == '' else float(r['MMSE'])]
    return fileIDs, np.asarray(time_obs), np.asarray(hit), age, mmse

def _read_csv_csf(filename):
    parcellation_tbl = pd.read_csv(filename)
    valid_columns = ["abeta", "tau","ptau"]
    parcellation_tbl = parcellation_tbl[valid_columns + ['RID']].copy()
    return parcellation_tbl

def _retrieve_kfold_partition(idxs, stage, folds=5, exp_idx=1, shuffle=True,
                              random_state=120):
    idxs = np.asarray(idxs).copy()
    if shuffle:
        np.random.seed(random_state)
        idxs = np.random.permutation(idxs)
    if 'all' in stage:
        return idxs
    if len(idxs.shape) > 1: raise ValueError
    fold_len = len(idxs) // folds
    folds_stitched = []
    for f in range(folds):
        folds_stitched.append(idxs[f*fold_len:(f+1)*fold_len])
    test_idx = exp_idx
    valid_idx = (exp_idx+1) % folds
    train_idx = np.setdiff1d(np.arange(0,folds,1),[test_idx, valid_idx])
    if 'test' in stage:
        return folds_stitched[test_idx]
    elif 'valid' in stage:
        return folds_stitched[valid_idx]
    elif 'train' in stage:
        return np.concatenate([folds_stitched[x] for x in train_idx], axis=0)
    else:
        raise ValueError

def deabbreviate_parcellation_columns(df):
    df_dict = pd.read_csv(
            './metadata/data_raw/neuromorphometrics/neuromorphometrics.csv',
                          usecols=['ROIabbr','ROIname'],sep=';')
    df_dict = df_dict.loc[[x[0] == 'l' for x in df_dict['ROIabbr']],:]
    df_dict['ROIabbr'] = df_dict['ROIabbr'].apply(
            lambda x: x[1:]
    )
    df_dict['ROIname'] = df_dict['ROIname'].apply(
            lambda x: x.replace('Left ', '')
    )
    df_dict = df_dict.set_index('ROIabbr').to_dict()['ROIname']
    df.rename(columns=df_dict, inplace=True)

def drop_ventricles(df, ventricle_list):
    df.drop(columns=ventricle_list, inplace=True)

def add_ventricle_info(parcellation_df, ventricle_df, ventricles):
    return parcellation_df.merge(ventricle_df[['RID'] + ventricles], on='RID',
                          validate="one_to_one")

def _average_hemispheric_gmv(parcellation_df) -> pd.DataFrame:
    reg = re.compile(r'corr_vol_(?P<region>.*)')
    cols = list(filter(lambda x: reg.match(x), parcellation_df.columns))
    parcellation_df = parcellation_df[cols].copy()
    parcellation_df = parcellation_df.rename(cols = lambda x: reg.findall(x)['region'])

class ParcellationDataMeta(Dataset):
    def __init__(self, seed, **kwargs):
        random.seed(1000)
        self.exp_idx = seed
        json_props = read_json('./simple_mlps/mlp_config'
                             '.json')
        self._json_props = json_props
        self.csv_directory = json_props['datadir']
        self.ventricles = json_props['ventricles']  
        self.csvname = self.csv_directory + json_props['metadata_fi']
        self.parcellation_file = self.csv_directory + json_props['parcellation_fi']
        self.parcellation_file_csf = self.csv_directory + json_props[
            'parcellation_csf_fi']
        self.rids, self.time_obs, self.hit, self.age, self.mmse = \
            _read_csv_cox(self.csvname)

    def _prep_data(self, feature_df, stage):
        idxs = list(range(len(self.rids)))
        self.index_list = _retrieve_kfold_partition(idxs, stage, 5, self.exp_idx)
        self.rid = np.array(self.rids)
        logging.warning(f'selecting indices\n{self.rid[self.index_list]}\n\t '
                        f'for stage'
                        f'{stage} and random seed 1000')
        self.labels = feature_df.columns
        self.data_l = feature_df.to_numpy()
        self.data_l = torch.FloatTensor(self.data_l)
        self.data = self.data_l

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx_transformed = self.index_list[idx]
        x = self.data[idx_transformed]
        obs = self.time_obs[idx_transformed]
        hit = self.hit[idx_transformed]
        rid = self.rid[idx_transformed]
        return x, obs, hit, rid

    def get_features(self):
        return self.labels

    def get_data(self):
        return self.data

class ParcellationDataCSF(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2), add_age=False,
                 add_mmse=False):
        super().__init__(seed, stage=stage, ratio=ratio)
        parcellation_df = _read_csv_csf(self.csvname)
        parcellation_df['RID'] = parcellation_df['RID'].apply(
                lambda x: str(int(x)).zfill(4)
        )
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        self._prep_data(parcellation_df, stage)


class ParcellationDataVentricles(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2),
                 ventricle_info=False, add_age=False,
                 add_mmse=False):
        super().__init__(seed, stage=stage)
        parcellation_df = pd.read_csv(self.parcellation_file, dtype={'RID':
                                                                         str})
        ventricle_df = pd.read_csv(self.parcellation_file_csf, dtype={'RID':
                                                                          str})
        drop_ventricles(parcellation_df, self.ventricles)
        if ventricle_info:
           parcellation_df = add_ventricle_info(parcellation_df, ventricle_df,
                                  self.ventricles)
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        deabbreviate_parcellation_columns(parcellation_df)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        self.parcellation_df = parcellation_df
        self._prep_data(parcellation_df, stage)


class ParcellationDataVentriclesNacc(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2),
                 ventricle_info=False, add_age=False,
                 add_mmse=False):
        self.seed = seed
        random.seed(1000)
        json_props = read_json('./simple_mlps/mlp_config'
                             '.json')
        self._json_props = json_props
        self.ventricles = json_props['ventricles']
        self.csv_directory = json_props['datadir']
        self.csvname = self.csv_directory + json_props['metadata_fi_nacc']
        self.rids, self.time_obs, self.hit, self.age, self.mmse = \
            _read_csv_cox(self.csvname)
        csvname2 = self.csv_directory + json_props['parcellation_fi_nacc']
        csvname3 = self.csv_directory + json_props['parcellation_csf_fi_nacc']
        parcellation_df = pd.read_csv(csvname2, dtype={'RID': str})
        ventricle_df = pd.read_csv(csvname3, dtype={'RID': str})
        drop_ventricles(parcellation_df, self.ventricles)
        if ventricle_info:
            parcellation_df = add_ventricle_info(parcellation_df, ventricle_df,
                                  self.ventricles)
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        deabbreviate_parcellation_columns(parcellation_df)
        self.exp_idx = 1
        self._prep_data(parcellation_df, stage)

class ParcellationDataGMVCSF(Dataset):
    def __init__(self, seed, stage, dataset='ADNI', ratio=(0.6, 0.2, 0.2), partitioner=_retrieve_kfold_partition):
        random.seed(1000)
        self.exp_idx = seed
        self.ratio = ratio
        self.stage = stage
        self.partitioner = partitioner
        json_props = read_json('./simple_mlps/mlp_config.json')
        self.csv_directory = json_props['datadir']
        if dataset == "ADNI":
            self.csvname = self.csv_directory + json_props['metadata_fi']
        elif dataset == "NACC":
            self.csvname = self.csv_directory + json_props['metadata_fi_nacc']
        self.parcellation_file = pd.read_csv(
            self.csv_directory + "mri3_cat12_combined_csf_gmv.csv", dtype={'RID': str})
        self.parcellation_file = self.parcellation_file.query(
            'Dataset == @dataset').drop(columns=['Dataset', 'PROGRESSION_CATEGORY', 'TIV']).copy()
        self.rids, self.time_obs, self.hit, self.age, self.mmse = \
            _read_csv_cox(self.csvname)
        
        if dataset == "ADNI":
            self.parcellation_file['RID'] = self.parcellation_file['RID'].apply(
                lambda x: x.zfill(4)
            )
        self.parcellation_file.set_index('RID', inplace=True)
        self.parcellation_file = self.parcellation_file.loc[self.rids,:].reset_index(
                drop=True)
        self._prep_data(self.parcellation_file)

    def _prep_data(self, feature_df):
        idxs = list(range(len(self.rids)))
        self.index_list = self.partitioner(idxs, stage=self.stage, exp_idx=self.exp_idx)
        self.rid = np.array(self.rids)
        self.labels = feature_df.columns
        self.data_l = feature_df.to_numpy()
        self.data_l = torch.FloatTensor(self.data_l)
        self.data = self.data_l

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx_transformed = self.index_list[idx]
        x = self.data[idx_transformed]
        obs = self.time_obs[idx_transformed]
        hit = self.hit[idx_transformed]
        rid = self.rid[idx_transformed]
        return x, obs, hit, rid

    def get_features(self):
        return self.labels

    def get_data(self):
        return self.data
    

class ParcellationDataCensored(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2),
                 rand_exp=1, risk_pr=0):
        super().__init__(seed, stage=stage)
        parcellation_df = pd.read_csv(self.parcellation_file, dtype={'RID':
                                                                         str})
        drop_ventricles(parcellation_df, self.ventricles)
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        deabbreviate_parcellation_columns(parcellation_df)
        self._prep_data(parcellation_df, stage)
        self.time_obs, self.hit = censor_df(np.array(self.time_obs), np.array(self.hit), seed=rand_exp, risk_pr=risk_pr)
    


def censor_df(time_obs: np.array, hit: np.array, seed: float, risk_pr: float):


    def retrieve_nacc_data():
        json_props = read_json('./simple_mlps/mlp_config'
                             '.json')
        csv_directory = json_props['datadir']
        json_props = read_json('./simple_mlps/mlp_config'
                             '.json')
        csvname = csv_directory + json_props['metadata_fi_nacc']
        _, time_obs_nacc, hit_nacc, _, _ = \
            _read_csv_cox(csvname)
        return np.array(time_obs_nacc), np.array(hit_nacc)

    time_obs_nacc, hit_nacc = retrieve_nacc_data()

    unique_times = pd.unique(time_obs)
    unique_times.sort()

    unique_times = np.concatenate([unique_times, [np.inf]])

    genr = np.random.default_rng(seed=seed)

    time_obs_new = time_obs.copy()

    hit_new = hit.copy()

    for t_idx in range(len(unique_times)-1):

        # compute number of persons at risk and not due to be censored in the current time interval

        at_risk_idx_hit = np.where(
                (time_obs_new > unique_times[t_idx]) &
                (hit_new == 1)  # survived at least to this point in time
            )[0]

        n_at_risk = len(at_risk_idx_hit)

        if len(at_risk_idx_hit) == 0:
            continue

        n_to_censor = int(np.clip(
                np.round(n_at_risk*risk_pr),
                a_min=0,
                a_max=n_at_risk - 1
            ))

        print(n_at_risk)
        print(risk_pr)
        print(n_to_censor)

        idx_cens = genr.choice(at_risk_idx_hit, n_to_censor, replace=False)

        time_obs_new[idx_cens] = unique_times[t_idx]
        print(sum(hit_new))
        hit_new[idx_cens] = 0
        print(sum(hit_new))

    r0 = pd.DataFrame({"Time": time_obs, "Hit": hit})  # compute risk overall
    r1 = pd.DataFrame({"Time": time_obs_new, "Hit": hit_new})
    r1["Dataset"] = "ADNICensored"
    r2 = pd.DataFrame({"Time": time_obs_nacc, "Hit": hit_nacc})
    r2["Dataset"] = "NACC"
    r0["Dataset"] = "ADNI"
    df = pd.concat([r0, r1, r2], axis=0, ignore_index=True)

    print("done")

    df.to_csv(f"metadata/data_processed/censoring_probabilities_seed{seed}_riskpr{risk_pr}.csv")

    return time_obs_new, hit_new

def test_hemisphere_averaging():
    for stage in ("train", "test", "validate"):
        for seed in range(5):
            ds = ParcellationDataGMVCSF(seed=seed, stage=stage, dataset = "ADNI")
            reg = re.compile(r'^corr_vol_[lr]{1}(?P<region>.*)')
            cols = list(filter(lambda x: reg.match(x), ds.labels))
            ds2 = ds.parcellation_file[cols]
            ds2 = ds2.rename(columns = lambda x: reg.match(x)['region'])
            ds2 = ds2.groupby(by=ds2.columns, axis=1).agg(np.mean)
            deabbreviate_parcellation_columns(ds2)
            ds_orig = ParcellationDataVentricles(seed, stage=stage)
            cols = ds_orig.parcellation_df.columns
            ds2 = ds2[cols]
            assert np.allclose(ds2.to_numpy(), ds_orig.parcellation_df.to_numpy())

def test_rid_selection_for_gmv_csf() -> None:
    for stage in ("train", "test", "validate"):
        for seed in range(5):
            ds = ParcellationDataGMVCSF(seed=seed, stage=stage, dataset = "ADNI")
            ds2 = ParcellationDataVentricles(seed=seed, stage=stage)
            for j in range(len(ds)):
                _, obs, hit, rid = ds[j]
                _, obs2, hit2, rid2 = ds2[j]
                assert(np.allclose(obs, obs2))
                assert(np.allclose(hit, hit2))
                assert(np.array_equal(rid, rid2))

def test_censored_dataset():
    for stage in ("train", "test", "validate"):
        for seed in range(5):
            ds = ParcellationDataCensored(seed=seed, stage=stage)

if __name__ == "__main__":
    # test_hemisphere_averaging()
    # test_rid_selection_for_gmv_csf()
    # print("PASS")
    test_censored_dataset()