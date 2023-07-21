# dataloader.py: Prepare the dataloader needed for the neural networks
# Created: 5/21/2021
# Status: OK

import numpy as np
from sklearn.model_selection import StratifiedKFold
import glob
import torch
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from .utils import read_csv_cox2, read_csv_pre, rescale, read_csv_cox_ext
from simple_mlps.datas import _retrieve_kfold_partition
import random
import sys
import os
import csv
import nibabel as nib
import pandas as pd

def _read_csv_cox(filename, skip_ids: list=None):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fileIDs, time_obs, hit, age, mmse, sex, educ = [], [], [], [], [], [], []
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
            if 'PTGENDER_demo' in r.keys():
                sex += [str(r['PTGENDER_demo'])]
            else:
                sex += [str(r['SEX'])]
            if 'PTEDUCAT_demo' in r.keys():
                educ += [float(r['PTEDUCAT_demo'])]
            else:
                educ += [float(r['EDUC'])]
    df = pd.DataFrame({'RID': fileIDs,
        'TIMES': np.asarray(time_obs),
        'HITS': np.asarray(hit),
        'AGE': np.asarray(age),
        'MMSE': np.asarray(mmse),
        'SEX': np.asarray(sex),
        'EDUC': np.asarray(educ)
    })
    return df

class CNN_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, ratio=(0.6, 0.2, 0.2), seed=1000, name='', fold=[], external=False):
        random.seed(seed)

        self.stage = stage
        self.cache = []
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir

        self.Data_list = glob.glob(Data_dir + '*nii*')

        csvname = 'metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv'

        if external:
            csvname = 'metadata/data_processed/merged_dataframe_cox_test_pruned_final.csv'

        metadata = _read_csv_cox(csvname)
        metadata['MRI_fi'] = ''

        self.name = 'mri'
        self.Data_list.sort() #ensure the order is correct

        tmp_f = []
        tmp_d = []
        metadata.set_index('RID', inplace=True)
        for rid in metadata.index:
            for d in self.Data_list:
                fname = os.path.basename(d)    
                if rid in fname:
                    metadata.loc[rid, 'MRI_fi'] = d
                    break
        
        empty_idx = metadata[metadata['MRI_fi'] == ''].index

        idxs = list(range(len(metadata)))
        self.index_list = _retrieve_kfold_partition(idxs, self.stage, 5, self.exp_idx)

        self.fileIDs = np.array(metadata.index)[self.index_list] #Note: this only for csv generation not used for data retrival
        self.metadata = metadata.copy()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        rid = self.fileIDs[idx]

        obs = self.metadata.loc[rid, 'TIMES']
        hit = self.metadata.loc[rid, 'HITS']
        age = self.metadata.loc[rid, 'AGE']
        mmse = self.metadata.loc[rid, 'MMSE']
        sex = self.metadata.loc[rid, 'SEX']
        educ = self.metadata.loc[rid, 'EDUC']
        data = nib.load(self.metadata.loc[rid, 'MRI_fi']).get_fdata().astype(np.float32)
        
        data[data != data] = 0
        SCALE = True

        if SCALE:
            data = rescale(data, (0, 2.5))
        data = np.expand_dims(data, axis=0)
        return data, obs, hit, age, mmse, sex, educ

if __name__ == "__main__":
    train_data = CNN_Data("/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/", 0, stage='train', seed=1, name='a', fold=5)
    valid_data = CNN_Data("/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/", 0, stage='valid', seed=1, name='a', fold=5)
    test_data = CNN_Data("/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/", 0, stage='test', seed=1, name='a', fold=5)
    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))
