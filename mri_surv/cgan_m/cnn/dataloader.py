# dataloader.py: Prepare the dataloader needed for the neural networks
# Created: 5/21/2021
# Status: OK

import numpy as np
from sklearn.model_selection import StratifiedKFold
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from utils import read_csv_cox2, read_csv_cox, read_csv_pre, rescale, read_csv2, read_csv_ord, read_csv_cox_ext
import random
import sys
import os
import nibabel as nib
import csv
import pandas as pd

#SKULL: True indicate SKULL removed, False indicate SKULL remained
SKULL = True
#True indicate scaling the data in dataloader
SCALE = False
#True to get cross set where all models correctly identified
CROSS_VALID=False

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

class CNN_Data_Append(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, ratio=(0.6, 0.2, 0.2), seed=1000, name='', fold=[], external=False):
        random.seed(seed)

        self.stage = stage
        self.cache = []
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir

        self.Data_list = glob.glob(Data_dir + '*nii*')
        
        # csvname = '~/mri-pet/mri_surv/metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv'
        # if external:
        #     csvname = '~/mri-pet/mri_surv/metadata/data_processed/merged_dataframe_cox_test_pruned_final.csv'

        csvname = '../../metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv'
        if external:
            csvname = '../../metadata/data_processed/merged_dataframe_cox_test_pruned_final.csv'

        metadata = _read_csv_cox(csvname)
        # print(metadata)
        # print(metadata['TIMES'])
        # sys.exit()
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
        sex = self.metadata.loc[rid, 'SEX'] == 'M'
        educ = self.metadata.loc[rid, 'EDUC']
        data = nib.load(self.metadata.loc[rid, 'MRI_fi']).get_fdata().astype(np.float32)
        
        data[data != data] = 0
        SCALE = True

        if SCALE:
            data = rescale(data, (0, 2.5))
        data = np.expand_dims(data, axis=0)
        return data, obs, hit, age, mmse, sex, educ
        # return data, obs, hit

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

class San_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, ratio=(0.6, 0.2, 0.2), seed=1000, name='', fold=[]):
        random.seed(seed)

        self.stage = stage

        self.cache = []
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir

        # self.Data_list = glob.glob(Data_dir + 'coregistered*nii*')
        self.Data_list = glob.glob(Data_dir + '*nii*')

        # csvname = '~/mri-pet/metadata/data_processed/merged_dataframe_cox_pruned_final.csv'
        csvname = '~/mri-pet/mri_surv/metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv'
        csvname = os.path.expanduser(csvname)
        fileIDs, time_obs, time_hit = read_csv_cox2(csvname) #training file
        # print(len(fileIDs))
        # print(len(time_obs))
        # print(len(time_hit))
        # sys.exit()

        # print(np.array([time_hit[:18],time_obs[:18]]))

        if 'mri' in name:
            name = 'mri'
        elif 'amyloid' in name:
            name = 'amyloid'
        else:
            name = 'fdg'
        self.name = name
        self.Data_list.sort() #ensure the order is correct

        tmp_f = []
        tmp_d = []
        for d in self.Data_list:
            for f in fileIDs:
                fname = os.path.basename(d)
                if f in fname:
                    tmp_f.append(f)
                    tmp_d.append(d)
                    break
        self.Data_list = tmp_d

        skip = list(set(tmp_f) ^ set(fileIDs))

        # print(skip, 'not found, skipped')
        for f in skip:
            idx = fileIDs.index(f)
            fileIDs.pop(idx)
            time_obs.pop(idx)
            time_hit.pop(idx)
        #sanity check
        for f, d in zip(fileIDs, self.Data_list):
            if f not in d:
                print('inconsistent pair!')
                print(f, d)
                sys.exit()
        self.time_obs = time_obs
        self.time_hit = time_hit
        # print(len(self.time_obs))
        # print(len(self.Data_list))
        # print(len(self.time_hit))
        # print((self.Data_list[:10]))
        # print((fileIDs[:10]))
        # sys.exit()

        FIXED = False
        if FIXED:
            # 1 fold for testing, 1 fold for validation, the rest for training
            # Note: in this case, MLP supports only 5 fold cross validation!
            # Not in use, need to test before use!
            num_fold = 5
            idxs = list(range(len(fileIDs)))

            skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=120)

            i = 0
            for train_index, test_index in skf.split(idxs, labels):
                if i == exp_idx:
                    if 'train' in stage:
                        self.index_list = train_index[:-len(test_index)]
                    elif 'valid' in stage:
                        self.index_list = train_index[-len(test_index):]
                    elif 'test' in stage:
                        self.index_list = test_index
                    elif 'all' in stage:
                        self.index_list = idxs
                    else:
                        self.index_list = []
                    break
                i += 1

        else:
            l = len(self.Data_list)
            split1 = int(l*ratio[0])
            split2 = int(l*(ratio[0]+ratio[1]))
            idxs = list(range(len(fileIDs)))
            random.shuffle(idxs)
            if 'train' in stage:
                self.index_list = idxs[:split1]
            elif 'valid' in stage:
                self.index_list = idxs[split1:split2]
            elif 'test' in stage:
                self.index_list = idxs[split2:]
            elif 'all' in stage:
                self.index_list = idxs
            else:
                raise Exception('Unexpected Stage for FCN_Cox_Data!')
        self.fileIDs = np.array(fileIDs)[self.index_list] #Note: this only for csv generation not used for data retrival

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        obs = self.time_obs[idx]
        hit = self.time_hit[idx]

        if type(self.Data_dir) == type([]):
            self.Data_list2 = [dl.replace(self.Data_dir[0], self.Data_dir[1]) for dl in self.Data_list]
            if ('mri' in self.name) and ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('mri' in self.name) and ('amyloid' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
            elif ('mri' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('amyloid', 'fdg') for d in self.Data_list2]
            else:
                print('error: case not supported. make sure the model name contains input scan types (mri, amyloid, or fdg)')
                print(self.name)
                sys.exit()
            if len(self.Data_dir) == 2:
                try:
                    data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                    data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                except:
                    data1 = np.load(self.Data_list[idx]).astype(np.float32)
                    data2 = np.load(self.Data_list2[idx]).astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data = np.array([data1, data2])
            else:
                data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                data3 = nib.load(self.Data_list3[idx]).get_fdata().astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data3[data3 != data3] = 0
                data = np.array([data1, data2, data3])
        else:
            try:
                data = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
            except:
                data = np.load(self.Data_list[idx]).astype(np.float32)
            data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
        data = np.expand_dims(data, axis=0)
        return data, obs, hit

    def get_sample_weights(self):
        num_classes = len(set(self.time_hit))
        counts = [self.time_hit.count(i) for i in range(num_classes)]
        count = len(self.time_hit)
        weights = [count / counts[i] for i in self.time_hit]
        class_weights = [count/c for c in counts]
        return weights, class_weights

class AE_Cox_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, ratio=(0.6, 0.2, 0.2), seed=1000, name='', fold=[], external=False):
        random.seed(seed)

        self.stage = stage
        self.cache = []
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir

        # self.Data_list = glob.glob(Data_dir + 'coregistered*nii*')
        self.Data_list = glob.glob(Data_dir + '*nii*')
        # print(len(self.Data_list))

        # csvname = '~/mri-pet/metadata/data_processed/merged_dataframe_cox_pruned_final.csv'
        csvname = '~/mri-pet/mri_surv/metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv'
        if external:
            csvname = '~/mri-pet/mri_surv/metadata/data_processed/merged_dataframe_cox_test_pruned_final.csv'

        csvname = os.path.expanduser(csvname)

        if external:
            fileIDs, time_obs, time_hit = read_csv_cox_ext(csvname) #training file
        else:
            fileIDs, time_obs, time_hit = read_csv_cox2(csvname) #training file
        # fileIDs, time_obs, time_hit = read_csv_cox(csvname) #training file

        # print(np.array([time_hit[:18],time_obs[:18]]))

        if 'mri' in name:
            name = 'mri'
        elif 'amyloid' in name:
            name = 'amyloid'
        else:
            name = 'fdg'
        self.name = name
        self.Data_list.sort() #ensure the order is correct

        tmp_f = []
        tmp_d = []
        for d in self.Data_list:
            for f in fileIDs:
                fname = os.path.basename(d)
                if f in fname:
                    tmp_f.append(f)
                    tmp_d.append(d)
                    break
        self.Data_list = tmp_d

        skip = list(set(tmp_f) ^ set(fileIDs))

        # print(skip, 'not found, skipped')
        for f in skip:
            idx = fileIDs.index(f)
            fileIDs.pop(idx)
            time_obs.pop(idx)
            time_hit.pop(idx)
        #sanity check
        for f, d in zip(fileIDs, self.Data_list):
            if f not in d:
                print('inconsistent pair!')
                print(f, d)
                sys.exit()
        self.time_obs = time_obs
        self.time_hit = time_hit
        # print(len(time_hit))
        # sys.exit()

        FOLDS = True
        if FOLDS:
            # 1 fold for testing, 1 fold for validation, the rest for training
            # Using custom for unity experiments
            # print('using k-fold')
            # sys.exit()
            idxs = list(range(len(fileIDs)))
            self.index_list = _retrieve_kfold_partition(idxs, self.stage, 5, self.exp_idx)

            # ver 2
            # num_fold = 5
            # idxs = list(range(len(fileIDs)))
            #
            # skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=120)
            #
            # i = 0
            # for train_index, test_index in skf.split(idxs, labels):
            #     if i == exp_idx:
            #         if 'train' in stage:
            #             self.index_list = train_index[:-len(test_index)]
            #         elif 'valid' in stage:
            #             self.index_list = train_index[-len(test_index):]
            #         elif 'test' in stage:
            #             self.index_list = test_index
            #         elif 'all' in stage:
            #             self.index_list = idxs
            #         else:
            #             self.index_list = []
            #         break
            #     i += 1

        else:
            l = len(self.Data_list)
            split1 = int(l*ratio[0])
            split2 = int(l*(ratio[0]+ratio[1]))
            idxs = list(range(len(fileIDs)))
            random.shuffle(idxs)
            if 'train' in stage:
                self.index_list = idxs[:split1]
            elif 'valid' in stage:
                self.index_list = idxs[split1:split2]
            elif 'test' in stage:
                self.index_list = idxs[split2:]
            elif 'all' in stage:
                self.index_list = idxs
            else:
                raise Exception('Unexpected Stage for FCN_Cox_Data!')
        self.fileIDs = np.array(fileIDs)[self.index_list] #Note: this only for csv generation not used for data retrival
        # print(len(self.time_obs))
        # print(len(self.Data_list))
        # print(len(self.time_hit))
        # print((self.Data_list[:10]))
        # print((fileIDs[:10]))
        # sys.exit()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        obs = self.time_obs[idx]
        hit = self.time_hit[idx]

        if type(self.Data_dir) == type([]):
            self.Data_list2 = [dl.replace(self.Data_dir[0], self.Data_dir[1]) for dl in self.Data_list]
            if ('mri' in self.name) and ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('mri' in self.name) and ('amyloid' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
            elif ('mri' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('amyloid', 'fdg') for d in self.Data_list2]
            else:
                print('error: case not supported. make sure the model name contains input scan types (mri, amyloid, or fdg)')
                print(self.name)
                sys.exit()
            if len(self.Data_dir) == 2:
                try:
                    data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                    data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                except:
                    data1 = np.load(self.Data_list[idx]).astype(np.float32)
                    data2 = np.load(self.Data_list2[idx]).astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data = np.array([data1, data2])
            else:
                data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                data3 = nib.load(self.Data_list3[idx]).get_fdata().astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data3[data3 != data3] = 0
                data = np.array([data1, data2, data3])
        else:
            try:
                data = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
            except:
                data = np.load(self.Data_list[idx]).astype(np.float32)
            data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
        data = np.expand_dims(data, axis=0)
        return data, obs, hit

    def get_sample_weights(self):
        num_classes = len(set(self.time_hit))
        counts = [self.time_hit.count(i) for i in range(num_classes)]
        count = len(self.time_hit)
        weights = [count / counts[i] for i in self.time_hit]
        class_weights = [count/c for c in counts]
        return weights, class_weights

class CNN_Data_Pre(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, ratio=(0.6, 0.2, 0.2), seed=1000, name='', fold=[]):
        random.seed(seed)

        self.stage = stage
        self.cache = []
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir

        self.Data_list = glob.glob(Data_dir + '*nii*')

        csvname = '~/mri-pet/mri_surv/metadata/data_processed/merged_dataframe_unused_cox_pruned.csv'
        csvname = os.path.expanduser(csvname)
        fileIDs, labels = read_csv_pre(csvname) #training file
        # self.Data_list.sort()
        # args = np.argsort(fileIDs)
        # fileIDs = np.array(fileIDs)[args]
        # labels = np.array(labels)[args]

        # print(len(fileIDs))
        # print(len(self.Data_list))
        # print(set(labels))
        # print((labels))
        # sys.exit()


        # print(np.array([time_hit[:18],time_obs[:18]]))

        if 'mri' in name:
            name = 'mri'
        elif 'amyloid' in name:
            name = 'amyloid'
        else:
            name = 'fdg'
        self.name = name

        tmp_l = []
        tmp_d = []
        for d in self.Data_list:
            for f in fileIDs:
                dname = os.path.basename(d)
                if f in dname:
                    tmp_l.append(labels[fileIDs.index(f)])
                    tmp_d.append(d)
                    break
        self.Data_list = tmp_d
        self.labels = tmp_l

        # print(len(self.Data_list))
        # print(len(self.labels))
        # sys.exit()

        FOLDS = True
        if FOLDS:
            # 1 fold for testing, 1 fold for validation, the rest for training
            # Using custom for unity experiments
            # print('using k-fold')
            # sys.exit()
            idxs = list(range(len(self.Data_list)))
            self.index_list = _retrieve_kfold_partition(idxs, self.stage, 5, self.exp_idx)

        else:
            l = len(self.Data_list)
            split1 = int(l*ratio[0])
            split2 = int(l*(ratio[0]+ratio[1]))
            idxs = list(range(l))
            random.shuffle(idxs)
            if 'train' in stage:
                self.index_list = idxs[:split1]
            elif 'valid' in stage:
                self.index_list = idxs[split1:split2]
            elif 'test' in stage:
                self.index_list = idxs[split2:]
            elif 'all' in stage:
                self.index_list = idxs
            else:
                raise Exception('Unexpected Stage for FCN_Cox_Data!')
        self.fileIDs = np.array(fileIDs)[self.index_list] #Note: this only for csv generation not used for data retrival

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        label = self.labels[idx]

        if type(self.Data_dir) == type([]):
            self.Data_list2 = [dl.replace(self.Data_dir[0], self.Data_dir[1]) for dl in self.Data_list]
            if ('mri' in self.name) and ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('mri' in self.name) and ('amyloid' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
            elif ('mri' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('amyloid', 'fdg') for d in self.Data_list2]
            else:
                print('error: case not supported. make sure the model name contains input scan types (mri, amyloid, or fdg)')
                print(self.name)
                sys.exit()
            if len(self.Data_dir) == 2:
                try:
                    data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                    data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                except:
                    data1 = np.load(self.Data_list[idx]).astype(np.float32)
                    data2 = np.load(self.Data_list2[idx]).astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data = np.array([data1, data2])
            else:
                data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                data3 = nib.load(self.Data_list3[idx]).get_fdata().astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data3[data3 != data3] = 0
                data = np.array([data1, data2, data3])
        else:
            try:
                data = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
            except:
                data = np.load(self.Data_list[idx]).astype(np.float32)
            data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
        data = np.expand_dims(data, axis=0)
        return data, label


    def get_sample_weights(self):
        num_classes = len(set(self.labels))
        counts = [self.labels.count(i) for i in range(num_classes)]
        count = len(self.labels)
        weights = [count / counts[i] for i in self.labels]
        class_weights = [count/c for c in counts]
        return np.array(weights)[self.index_list], class_weights

class CNN_Data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, ratio=(0.5, 0.25, 0.25), seed=1000, name='', fold=[]):
        random.seed(seed)

        self.stage = stage
        self.cache = []
        self.exp_idx = exp_idx
        self.Data_dir = Data_dir

        self.Data_list = glob.glob(Data_dir + '*nii*')
        self.Data_list.sort() #ensure the order is correct

        csvname = '~/mri-pet/mri_surv/metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv'
        csvname = os.path.expanduser(csvname)
        # fileIDs, labels = read_csv2(csvname) #training file

        fileIDs, labels = read_csv_ord(csvname) #training file
        labels = [4 if l == -1 else l for l in labels]
        labels = [l-1 for l in labels]
        #ordinal classes
        num_classes = len(set(labels))

        # print(len(fileIDs))
        # print(len(self.Data_list))
        # print(set(labels))
        # print((labels[:10]))
        # sys.exit()


        # print(np.array([time_hit[:18],time_obs[:18]]))

        if 'mri' in name:
            name = 'mri'
        elif 'amyloid' in name:
            name = 'amyloid'
        else:
            name = 'fdg'
        self.name = name

        tmp_l = []
        tmp_d = []
        for d in self.Data_list:
            for f in fileIDs:
                dname = os.path.basename(d)
                if f in dname:
                    tmp_l.append(labels[fileIDs.index(f)])
                    tmp_d.append(d)
                    break
        self.Data_list = tmp_d
        self.labels = tmp_l
        #sanity check
        for f, d in zip(fileIDs, self.Data_list):
            if f not in d:
                print('inconsistent pair!')
                print(f, d)
                sys.exit()
        # print(len(self.Data_list))
        # print(len(self.labels))
        # sys.exit()

        l = len(self.Data_list)
        split1 = int(l*ratio[0])
        split2 = int(l*(ratio[0]+ratio[1]))
        idxs = list(range(l))
        random.shuffle(idxs)
        # print([labels.count(i) for i in range(4)])
        if 'train' in stage:
            self.index_list = idxs[:split1]
        elif 'valid' in stage:
            self.index_list = idxs[split1:split2]
        elif 'test' in stage:
            self.index_list = idxs[split2:]
        elif 'all' in stage:
            self.index_list = idxs
        else:
            raise Exception('Unexpected Stage for FCN_Cox_Data!')
        self.fileIDs = np.array(fileIDs)[self.index_list] #Note: this only for csv generation not used for data retrival

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        label = self.labels[idx]

        if type(self.Data_dir) == type([]):
            self.Data_list2 = [dl.replace(self.Data_dir[0], self.Data_dir[1]) for dl in self.Data_list]
            if ('mri' in self.name) and ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('mri' in self.name) and ('amyloid' in self.name):
                self.Data_list2 = [d.replace('mri', 'amyloid') for d in self.Data_list2]
            elif ('mri' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('mri', 'fdg') for d in self.Data_list2]
            elif ('amyloid' in self.name) and ('fdg' in self.name):
                self.Data_list3 = [d.replace('amyloid', 'fdg') for d in self.Data_list2]
            else:
                print('error: case not supported. make sure the model name contains input scan types (mri, amyloid, or fdg)')
                print(self.name)
                sys.exit()
            if len(self.Data_dir) == 2:
                try:
                    data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                    data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                except:
                    data1 = np.load(self.Data_list[idx]).astype(np.float32)
                    data2 = np.load(self.Data_list2[idx]).astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data = np.array([data1, data2])
            else:
                data1 = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
                data2 = nib.load(self.Data_list2[idx]).get_fdata().astype(np.float32)
                data3 = nib.load(self.Data_list3[idx]).get_fdata().astype(np.float32)
                data1[data1 != data1] = 0
                data2[data2 != data2] = 0
                data3[data3 != data3] = 0
                data = np.array([data1, data2, data3])
        else:
            try:
                data = nib.load(self.Data_list[idx]).get_fdata().astype(np.float32)
            except:
                data = np.load(self.Data_list[idx]).astype(np.float32)
            data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
        data = np.expand_dims(data, axis=0)
        return data, label


    def get_sample_weights(self):
        num_classes = len(set(self.labels))
        counts = [self.labels.count(i) for i in range(num_classes)]
        count = len(self.labels)
        weights = [count / counts[i] for i in self.labels]
        class_weights = [count/c for c in counts]
        return np.array(weights)[self.index_list], class_weights

if __name__ == "__main__":

    train_data = CNN_Data_Append("/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/", 0, stage='train', seed=1, name='a', fold=5)
    valid_data = CNN_Data_Append("/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/", 0, stage='valid', seed=1, name='a', fold=5)
    test_data = CNN_Data_Append("/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/", 0, stage='test', seed=1, name='a', fold=5)
    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))
    for item in test_data:
        data, obs, hit, age, mmse, sex, educ = item
        print('obs, hit')
        print(obs, hit)
        sys.exit()
    # dataset = CNN_Data_Pre(Data_dir="/data2/MRI_PET_DATA/processed_images_final_unused_cox/brain_stripped_unused_cox/", exp_idx=0, stage='all')
    # print(len(dataset))
    # dataset = FCN_Data(Data_dir='/data2/ADNIP/', exp_idx=0, stage='valid_patch')
    # dataset = FCN_Data(Data_dir='/home/mfromano/data/adni/processed_images_test/', exp_idx=0, stage='train', whole_volume=True)
    # train_dataloader = DataLoader(dataset, batch_size=1)
    # sample_weight, _ = dataset.get_sample_weights()
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
    # train_w_dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    # for scan1, label in train_w_dataloader:
    # t = []
    # for scan1, label in dataset:
    #     if scan1.shape in t:
    #         pass
    #     else:
    #         t.append(scan1.shape)
    # print(t)

    train_data = AE_Cox_Data("/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/", 0, stage='train', seed=1, name='a', fold=5)
    valid_data = AE_Cox_Data("/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/", 0, stage='valid', seed=1, name='a', fold=5)
    test_data = AE_Cox_Data("/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/", 0, stage='test', seed=1, name='a', fold=5)
    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))
    # train_dataloader = DataLoader(dataset, batch_size=1)
    # t = []
    # for scan1, scan2, label in dataset:
    #     if scan1.shape in t:
    #         pass
    #     else:
    #         t.append(scan1.shape)
    # print(t)
