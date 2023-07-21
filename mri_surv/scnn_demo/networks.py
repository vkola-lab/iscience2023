# Network Wrappers
# Created: 5/21/2021
# Status: OK

import os
import sys
import collections
import shutil

from matplotlib.font_manager import json_load
import nilearn
import time
import glob
import csv
import torch
import shap

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import pandas as pd

from scnn_demo.models import CNN_Surv
from scnn_demo.dataloader import CNN_Data
# from mlp_cox_csf import MLP_Data as CSF_Data
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored, integrated_brier_score
sys.path.insert(1, './plot/')
from sklearn.metrics import confusion_matrix, classification_report
# import cv2
from scipy import interpolate

def make_struc_array(hits, obss):
    return np.array([(x,y) for x,y in zip(np.asarray(hits) == 1, obss)], dtype=[('hit',bool),('time',float)])

def sur_loss(preds, obss, hits, bins=torch.Tensor([[0, 24, 48, 108]])):
    if torch.cuda.is_available():
        bins = bins.cuda()
    bin_centers = (bins[0, 1:] + bins[0, :-1])/2
    survived_bins_censored = torch.ge(torch.mul(obss.view(-1, 1),1-hits.view(-1,1)), bin_centers)
    survived_bins_hits = torch.ge(torch.mul(obss.view(-1,1), hits.view(-1,1)), bins[0,1:])
    survived_bins = torch.logical_or(survived_bins_censored, survived_bins_hits)
    survived_bins = torch.where(survived_bins, 1, 0)
    event_bins = torch.logical_and(torch.ge(obss.view(-1, 1), bins[0, :-1]), torch.lt(obss.view(-1, 1), bins[0, 1:]))
    event_bins = torch.where(event_bins, 1, 0)
    hit_bins = torch.mul(event_bins, hits.view(-1, 1))
    l_h_x = 1+survived_bins*(preds-1)
    n_l_h_x = 1-hit_bins*preds
    cat_tensor = torch.cat((l_h_x, n_l_h_x), axis=0)
    total = -torch.log(torch.clamp(cat_tensor, min=1e-12))
    pos_sum = torch.sum(total)
    neg_sum = torch.sum(pos_sum)
    return neg_sum

class SCNN_Wrapper:
    def __init__(self, fil_num, drop_rate, seed, batch_size, Data_dir, exp_idx, num_fold, model_name, metric='concord', lr=0.01):
        self.gpu = 1
        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.model_name = model_name
        torch.manual_seed(seed)
        self.num_classes = 3
        self.targets = list(range(self.num_classes))
        self.model = CNN_Surv(drop_rate, fil_num=fil_num, out_channels=self.num_classes).cuda()
        if self.gpu != 1:
            self.model = self.model.cpu()
        self.prepare_dataloader(batch_size, Data_dir)
        self.criterion = sur_loss
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def prepare_dataloader(self, b_s, Data_dir):
        train_data = CNN_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.train_dataloader = DataLoader(train_data, batch_size=b_s, drop_last=True)
        valid_data = CNN_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

        test_data  = CNN_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = CNN_Data(Data_dir, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=b_s)

    def _train_model_epoch(self):
        self.model.train(True)
        s_loss = 0
        num = 0
        for data, obs, hit, age, mmse, sex, educ in self.train_dataloader:
            num += 1
            inputs, obs, hit = data.cuda(), obs.cuda(), hit.cuda()

            self.model.zero_grad()

            preds = self.model(inputs)

            loss = self.criterion(preds, obs, hit)

            s_loss += loss

            # torch.use_deterministic_algorithms(False)
            loss.backward()
            # torch.use_deterministic_algorithms(True)
            self.optimizer.step()

    def train(self, n_epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        for self.epoch in range(n_epochs):
            self._train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self._valid_model_epoch()
                self._save_checkpoint(val_loss)
                print('{}th epoch validation loss [CE]:'.format(self.epoch), '%.3f' % (val_loss))
                # if self.epoch % (epochs//10) == 0:
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def test(self, trained=True):
        print('testing...')
        if trained:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
            # print(self.model.state_dict())
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            obs_all = []
            hit_all = []
            preds_all = []
            for data, obs, hit, age, mmse, sex, educ in self.test_dataloader:
                preds_all += [self.model(data.cuda())]
                obs_all += [obs]
                hit_all += [hit]
        loss = self.criterion(preds_all, obs_all, hit_all)
        print(loss)
        return loss

    def _valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            obs_all = []
            hit_all = []
            for data, obs, hit, age, mmse, sex, educ in self.valid_dataloader:
                obs_all += [obs]
                hit_all += [hit]
                preds_all += [self.model(data.cuda()).cpu().numpy().squeeze()]
            obs_all = torch.tensor(obs_all).cuda()
            hit_all = torch.tensor(hit_all).cuda()
            preds_all = torch.tensor(preds_all).cuda()
            print(preds_all)
            print(obs_all)
            print(hit_all)

            loss = self.criterion(preds_all, obs_all, hit_all)
        return loss

    def _save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def concord(self, load=True, all=False):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        cis = []
        bss = []
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [self.test_dataloader, DataLoader(self.valid_dataloader, batch_size=1, shuffle=False)]
            if all:
                train_dl = DataLoader(self.train_data, batch_size=1, shuffle=False)
                ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)
                dls += [train_dl, ext_dl]
            obss_all, hits_all = [], []
            for _, obss, hits in self.train_dataloader:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc = make_struc_array(hits_all, obss_all)
            for dl in dls:
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    preds_all.append(preds)
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0]]
                preds_all = np.concatenate(preds_all,axis=0)
                c_index = concordance_index_censored(np.asarray(hits_all) == 1, obss_all, -preds_all[:,1])
                interp = interpolate.PchipInterpolator(bins, preds_all, axis=1)
                new_bins = bins[:-1].copy() + [min(max(obss_all),108)-1]
                preds_all_brier = interp(new_bins)
                test_struc = make_struc_array(hits_all, obss_all)
                bs = integrated_brier_score(train_struc, test_struc, preds_all_brier, new_bins)
                cis += [c_index]
                bss += [bs]
        # print(cis, bss)
        return cis, bss

def main():

    json ={"fil_num":              10,
        "drop_rate":            0.3,
        "batch_size":           10,
        "balanced":             0,
        "metric":               "sur_loss",
        "Data_dir":             "/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/",
        "learning_rate":        0.01,
        "train_epochs":         2000}
    exp_idx = 1
    scnn = SCNN_Wrapper(
        fil_num=json['fil_num'],
        drop_rate=json['drop_rate'],
        seed=exp_idx,
        batch_size=json['batch_size'],
        Data_dir=json['Data_dir'],
        exp_idx=exp_idx,
        num_fold=5,
        model_name=exp_idx,
        metric='concord',
        lr=0.01)
    scnn.train(50)
    scnn.test()