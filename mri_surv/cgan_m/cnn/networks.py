# Network Wrappers
# Created: 5/21/2021
# Status: OK

import os
import sys
import collections
import shutil
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

from models import _CNN_Pre, _CNN, _CNN_Surv, _CNN_Surv_Res, _CNN_Surv_Append
from dataloader import AE_Cox_Data, CNN_Data_Pre, CNN_Data, San_Data, CNN_Data_Append
# from mlp_cox_csf import MLP_Data as CSF_Data
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored, integrated_brier_score, brier_score
sys.path.insert(1, './plot/')
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
# import cv2
from scipy import interpolate
from utils import w_estimator

CROSS_VALID = False
SWEEP = 1

def retrieve_brier_scores(train_struc, test_struc, preds_raw, bins):
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

def make_struc_array(hits, obss):
    return np.array([(x,y) for x,y in zip(np.asarray(hits) == 1, obss)], dtype=[('hit',bool),('time',float)])

# import matlab.engine

def cus_loss_cus(preds, obss, hits, all_logs=None, all_obss=None, debug=False, ver=0):
    '''
    Don't use entire set for now.
    Problems: inf value may happen
    Potential reason:
        1. patch-based, random picked -> unstable result
        2. lr too high
    '''

    # if ver == 0: # custom loss
    #     # requires full dataset
    #     preds, y, e = preds, obss, hits
    #     mask = torch.ones(y.shape[0], y.shape[0])
    #     # mask[(y.T - y) > 0] = 0
    #     mask[(y.view(-1, 1) - y) < 0] = 0 #chaned from > to <!
    #     # mask = mask.cuda() # whole trainig set does not fit
    #     log_loss = torch.mm(mask, torch.exp(preds))
    #     log_loss = torch.log(log_loss)
    #     neg_log_loss = -torch.sum((preds-log_loss) * e) / torch.sum(e)
    #
    #     if torch.isnan(neg_log_loss):
    #         print('nan')
    #         print(log_loss)
    #         print(preds)
    #         sys.exit()
    #         return None
    #     return neg_log_loss

    if ver == 0: # custom loss
        # requires full dataset
        preds, y, e = preds, obss, hits
        mask = torch.ones(y.shape[0], y.shape[0])
        mask2 = torch.ones(y.shape[0], y.shape[0])
        # mask[(y.T - y) > 0] = 0
        mask[(y.view(-1, 1) - y) < 0] = 0 #chaned from > to <!
        mask2[(y.view(-1, 1) - y) > 0] = 0 #chaned from > to <! (patients who still 'at risk' at time of patient y's progression time, i.e. observed before T_y)
        # mask = mask.cuda() # whole trainig set does not fit
        # print(preds[:10])
        preds = torch.exp(torch.abs(preds))
        # print(preds[:10])
        # sys.exit()
        sum_loss = torch.mm(mask, preds) / len(preds)
        sum_loss2 = torch.mm(mask2, preds) / len(preds)
        neg_sum_loss = -torch.sum((preds-sum_loss) * e) - torch.sum((sum_loss2-preds) * e)

        if torch.isnan(neg_sum_loss):
            print('nan')
            print(sum_loss)
            print(preds)
            sys.exit()
            return None
        return neg_sum_loss

    elif ver == 1: # local cox loss
        #\frac{1}{N_D} \sum_{i \in D}[F(x_i,\theta) - log(\sum_{j \in R_i} e^F(x_j,\theta))] - \lambda P(\theta)
        '''
        where:
            D is the set of observed events
            N_D is the number of observed events
            R_i is the set of examples that are still alive at time of death t_j
            F(x,\theta) = log hazard rate
        '''

        # #consider l1
        idxs = torch.argsort(obss, dim=0, descending=True)
        h_x = preds[idxs]
        obss = obss[idxs]
        hits = hits[idxs]

        num_hits = torch.sum(hits)

        e_h_x = torch.exp(h_x)
        log_e = torch.log(torch.cumsum(e_h_x, dim=0))
        diff = h_x - log_e
        hits = torch.reshape(hits, diff.shape) #convert into same shape, prevent errors
        diff = torch.sum(diff*hits)
        loss = -diff / num_hits
        if debug:
            print(preds.shape)
            print(h_x.shape)
            sys.exit()

        if torch.isnan(loss):
            print('nan')
            print(h_x)
            print(e_h_x)
            sys.exit()
            return None
        return loss

    return 0

def cox_loss(risk_pred, y, e, ver=0):
    risk_pred = risk_pred.view(-1,1)
    y = y.view(-1,1)
    e = e.reshape(-1,1)
    mask = torch.ones(y.shape[0], y.shape[0])
    mask[(y.T - y) > 0] = 0
    log_loss = torch.exp(risk_pred) * mask
    log_loss = torch.sum(log_loss, dim=0) #/ torch.sum(mask, dim=0)
    log_loss = torch.log(log_loss).reshape(-1, 1)
    neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
    return neg_log_loss

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

class CNN_Surv_Abstract:
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

class CNN_Surv_Wrapper:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, lr):
        self.cox_local = 0 #0: global, 1: local
        self.loss_type = 1 # 1 for categorical
        # self.categorical = 0
        self.metric = metric

        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.model_name = model_name
        #'macro avg' or 'weighted avg'
        torch.manual_seed(seed)
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        # self.time_intervals = list(set(self.all_data.time_obs))
        # self.time_intervals.sort()
        # self.time_class = {}
        # self.class_time = {}
        # for i,t in enumerate(self.time_intervals):
        #     self.time_class[t] = i
        #     self.class_time[i] = t

        # in_size = 121*145*121
        vector_len = 4
        # vector_len = len(self.time_intervals)
        self.model = _CNN_Surv(drop_rate=drop_rate, fil_num=fil_num, out_channels=vector_len).cuda()
        # for n, p in self.model.named_parameters():
            # print(n,p.data)
        # print(sum(p.numel() for p in self.model.parameters()))
        # sys.exit()
        if self.cox_local != 1:
            self.model = self.model.cpu()

        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        # self.criterion = cox_loss
        self.criterion = sur_loss
        # self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            if self.cox_local != 1:
                self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            if self.cox_local != 1:
                self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, drop_last=True)
        self.valid_dataloader = AE_Cox_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)

        test_data  = AE_Cox_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=len(train_data))

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, obss, hits in self.train_dataloader:
            # if self.categorical:
            #     obss = torch.tensor([self.time_class[o.item()] for o in obss])

            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.cox_local != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()



            self.model.zero_grad()
            preds = self.model(inputs)

            if self.loss_type:
                loss = self.criterion(preds, obss, hits)
            else:
                loss = self.criterion(preds, obss, hits, ver=self.cox_local)
            loss.backward()
            clip = 1
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            # print('obss', obss[:10])
            # print('hits', hits[:10])
            # print(preds[:10])
            # print(loss)
            # preds = self.model(inputs)
            # loss = self.criterion(preds, obss, hits)
            # print(preds[:10])
            # print(loss)
            # sys.exit()
        return loss

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            train_loss = self.train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                if self.epoch % (epochs//10) == 0:
                    print('{}th epoch validation loss [surv]:'.format(self.epoch), '%.3f' % (val_loss), 'train_loss: %.3f' % (train_loss))
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            patches_all = []
            obss_all = []
            hits_all = []
            # for patches, labels in self.valid_dataloader:
            for data, obss, hits in self.valid_dataloader:
                # if torch.sum(hits) == 0:
                    # continue # because 0 indicates nothing to learn in this batch, we skip it
                patches, obs, hit = data, obss, hits

                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                patch = patches
                patches_all += [patch]
                # patches_all += [patch.numpy()]
                # obss_all += [obss.numpy()[0]]
                # hits_all += [hits.numpy()[0]]
                obss_all += [obss]
                hits_all += [hits]

            # idxs = np.argsort(obss_all, axis=0)[::-1]
            patches_all = np.array(patches_all)
            obss_all = np.array(obss_all)
            hits_all = np.array(hits_all)
            # if self.categorical:
            #     obss_all = torch.tensor([self.time_class[o] for o in obss_all])

            patches_all = torch.tensor(patches_all)

            if self.cox_local != 1:
                preds_all = self.model(patches_all)

                if self.loss_type:
                    # print(preds_all.shape, obss_all.shape, hits_all.shape)
                    loss = self.criterion(torch.tensor(preds_all), torch.tensor(obss_all), torch.tensor(hits_all))
                else:
                    preds_all, obss_all, hits_all = preds_all.view(-1, 1), torch.tensor(obss_all).view(-1), torch.tensor(hits_all).view(-1)
                    loss = self.criterion(preds_all, obss_all, hits_all, ver=self.cox_local)

            else:
                preds_all = self.model(patches_all.cuda()).cpu()
                # preds_all, obss_all, hits_all = preds_all.view(-1, 1).cuda(), torch.tensor(obss_all).view(-1).cuda(), torch.tensor(hits_all).view(-1).cuda()
                preds_all, obss_all, hits_all = preds_all.view(-1, 1).cuda(), torch.tensor(obss_all).view(-1).cuda(), torch.tensor(hits_all).view(-1).cuda()

            # loss = self.criterion(preds_all, torch.tensor(obss_all.reshape(preds_all.shape)).float())
            # loss = self.criterion(preds_all, torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)
            # loss = self.criterion(preds_all, torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)

            # loss = self.criterion(preds_all, obss_all, hits_all, ver=self.cox_local)
        return loss

    def save_checkpoint(self, loss):
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

    def predict_plot(self, id=[10,30], average=False):
        # id: element id in the dataset to plot

        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = self.all_dataloader
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]

        times = train_obss

        data = self.test_data

        fig, ax = plt.subplots()
        self.model = self.model.cpu()
        with torch.no_grad():
            self.model.zero_grad()
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            input, obs, hit = data[id[0]]
            if average:
                inputs = []
                for d in data:
                    item = d[0]
                    if self.csf:
                        item = item.numpy()
                    inputs += [item]
                input = np.mean(inputs,axis=0)
            pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            if average:
                title = 'Average plot'
                ax.plot(times, surv_chances, label=title)
                ax.set(xlabel='time (m)', ylabel='Surv', title='')
                ax.grid()
                ax.legend()
                fig.savefig("likelihood.png")
                return
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred1 = pred

            input, obs, hit = data[id[1]]
            pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred2 = pred
        # print('hit', hit, obs)
        # print(pred)
        # print(sums)
        title = 'ratio (h(x_1)/h(x_2)): %.3f' % (pred1/pred2)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("likelihood.png")
        # plt.show()
        # print(len(surv_chances))
        # print(len(times))
        # print(pred)
        # print(sums[:10])
        # print(surv_chances[:10])
        # print(times[:10])
        # sys.exit()

        return

    def predict_plot_general(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = self.all_dataloader
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]

        times = train_obss

        data = self.test_data

        inputs_0, inputs_1 = [], []
        for d in data:
            item = d[0]
            if d[-1] == 0: #hit
                inputs_0 += [item]
            else:
                inputs_1 += [item]
        input_0 = np.mean(inputs_0,axis=0)
        input_1 = np.mean(inputs_1,axis=0)

        with torch.no_grad():
            self.model.zero_grad()
            input_0 = torch.tensor(input_0).unsqueeze(dim=0)
            input_1 = torch.tensor(input_1).unsqueeze(dim=0)
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            pred_0 = torch.exp(self.model(input_0))
            pred_1 = torch.exp(self.model(input_1))

        fig, ax = plt.subplots()
        event_chances = pred_0 / (sums+pred_0) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 0')

        event_chances = pred_1 / (sums+pred_1) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 1')

        title = 'ratio (AD_0/AD_1): %.3f' % (pred_0/pred_1)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("likelihood_general.png")

        return

    def predict_plot_scatter(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        preds = []
        times = []
        for inputs, obss, hits in self.test_dataloader:

            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.cox_local != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()

            preds += [self.model(inputs).item()]
            times += [obss.item()]
        fig, ax = plt.subplots()

        ax.scatter(times, preds)
        title = 'Scatter plot, h_x vs time'
        ax.set(xlabel='time (m)', ylabel='h_x', title=title)
        ax.grid()

        fig.savefig("scatter_plot.png")

        return

    def concord(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        preds_all = []
        obss_all = []
        hits_all = []
        with torch.no_grad():
            for data, obss, hits in self.test_dataloader:
                preds = self.model(data)
                preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                if self.loss_type: # surv loss
                    interp = interpolate.interp1d([0, 12, 24, 48, 108], preds, axis=-1, kind='quadratic')
                    concordance_time = 24
                    preds = interp(concordance_time) # approx @ 24
                    preds_all += [preds[0]]
                else:
                    preds_all += list(np.array(preds)[0])

                # print(preds_all)
                # print(hits)
                # print(obss)
                # sys.exit()
                obss_all += [np.array(obss)[0]]
                hits_all += [np.array(hits)[0] == 1]
            # print(preds_all[:10])
            c_index = concordance_index_censored(hits_all, obss_all, preds_all)
        # self.c_index.append(c_index)
        print(c_index)
        return c_index

    def shap(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        with torch.no_grad():
            exer = shap.Explainer(self.model)
            for data, obss, hits in self.test_dataloader:
                print(data.shape)
                sys.exit()
                shap_vals = exer(data)
                plt.clf()
                shap.plots.waterfall(shap_vals[0],matplotlib=True,show=False)
                plt.savefig('shap.png')
                break

class CNN_Wrapper_Pre:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, lr):
        self.gpu = 1

        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.model_name = model_name
        torch.manual_seed(seed)
        # in_size = 167*191*167
        vector_len = 2
        self.targets = list(range(vector_len))
        self.model = _CNN_Pre(drop_rate, fil_num=fil_num, out_channels=vector_len).cuda()
        if self.gpu != 1:
            self.model = self.model.cpu()

        self.prepare_dataloader(batch_size, balanced, Data_dir)
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        # self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.0)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def load(self, dir):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        self.model.load_state_dict(st, strict=False)
        print('loaded.')

    def prepare_dataloader(self, b_s, balanced, Data_dir):
        train_data = CNN_Data_Pre(Data_dir, self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            # sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=b_s, sampler=sampler)
            # if self.cox_local != 1:
                # self.train_dataloader = DataLoader(train_data, batch_size=b_s, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=b_s, drop_last=True)
            # if self.cox_local != 1:
                # self.train_dataloader = DataLoader(train_data, batch_size=b_s, shuffle=True, drop_last=True)
        self.valid_dataloader = CNN_Data_Pre(Data_dir, self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)

        test_data  = CNN_Data_Pre(Data_dir, self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = CNN_Data_Pre(Data_dir, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=b_s)

    def train_model_epoch(self):
        self.model.train(True)
        s_loss = 0
        num = 0
        for inputs, labels in self.train_dataloader:
            num += 1
            # if self.categorical:
            #     obss = torch.tensor([self.time_class[o.item()] for o in obss])

            inputs, labels = inputs.cuda(), labels.cuda()
            if self.gpu != 1:
                inputs, labels = inputs.cpu(), labels.cpu()

            self.model.zero_grad()
            preds = self.model(inputs)

            loss = self.criterion(preds, labels)
            s_loss += loss
            # loss = self.criterion(preds, labels, labels == 2)
            torch.use_deterministic_algorithms(False)
            loss.backward()
            torch.use_deterministic_algorithms(True)
            self.optimizer.step()
            # print('old loss', loss)
            # print('new loss', self.criterion(self.model(inputs), labels))
            # sys.exit()
        print(self.epoch,'training loss:', (s_loss/num).item())

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        # val_loss = self.valid_model_epoch() # sanity check
        for self.epoch in range(epochs):
            self.train_model_epoch()
            if self.epoch % 1 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                print('{}th epoch validation loss [CE]:'.format(self.epoch), '%.3f' % (val_loss))
                # if self.epoch % (epochs//10) == 0:
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def test(self, trained=False):
        print('testing...')
        if trained:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
            # print(self.model.state_dict())
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            labels_all = []
            for data, label in self.test_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                preds_all += [self.model(data.cuda())]
                labels_all += [label]
            target_names = ['class ' + str(i) for i in self.targets]
            preds_all = [torch.argmax(p).item() for p in preds_all]
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=self.targets, target_names=target_names, zero_division=0, output_dict=False)
            print(report)
            # loss = self.criterion(torch.tensor(preds_all), torch.tensor(labels_all))

        # return loss

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            labels_all = []
            for data, label in self.valid_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                data = torch.tensor(np.expand_dims(data, axis=0))
                data = data.cuda()
                if self.gpu != 1:
                    data = data.cpu()
                labels_all += [label]
                preds_all += [self.model(data).cpu().numpy().squeeze()]

            labels_all = torch.tensor(labels_all).cuda()
            preds_all = torch.tensor(preds_all).cuda()
            # print(preds_all)
            # print(preds_all.shape)
            # print(labels_all.shape)
            # sys.exit()
            if self.gpu != 1:
                preds_all = torch.tensor(preds_all).cpu()
                labels_all = torch.tensor(labels_all).cpu()

            loss = self.criterion(preds_all, labels_all)
            # loss = self.criterion(preds_all, labels_all, labels_all == 2)
            target_names = ['class ' + str(i) for i in self.targets]
            preds_all = [torch.argmax(p).item() for p in preds_all]
            labels_all = labels_all.tolist()
            print([preds_all.count(i) for i in self.targets])
            print([labels_all.count(i) for i in self.targets])
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=self.targets, target_names=target_names, zero_division=0, output_dict=False)
            print(report)
            # print(report['accuracy'])
            # loss = -report['accuracy']

        return loss

    def save_checkpoint(self, loss):
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

    def concord(self, load=True):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        preds_all = []
        obss_all = []
        hits_all = []
        with torch.no_grad():
            for data, hits in self.test_dataloader:
                preds = self.model(data.cuda()).cpu()
                if self.criterion == sur_loss:
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                if 1:
                    if self.criterion != sur_loss:# CE loss
                        m = nn.Softmax(dim=1)
                        preds_all += [m(preds).tolist()[0][0]]
                    else: #Surv loss
                        interp = interpolate.interp1d([0, 1], preds, axis=-1, kind='linear')
                        concordance_time = 0.5
                        preds = interp(concordance_time) # approx @ 24
                        preds_all += [preds[0]]

                hits_all += [np.array(hits)[0] == 1]
            preds_all = [-p for p in preds_all]
            c_index = concordance_index_censored(hits_all, [1]*len(hits_all), preds_all)
            # print(c_index)
        # self.c_index.append(c_index)
        return c_index

    def concord_valid(self, load=True):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        preds_all = []
        obss_all = []
        hits_all = []
        with torch.no_grad():
            dl = DataLoader(self.valid_dataloader, batch_size=1, shuffle=False)
            for data, hits in dl:
                preds = self.model(data.cuda()).cpu()
                if self.criterion == sur_loss:
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                if 1:
                    if self.criterion != sur_loss:# CE loss
                        m = nn.Softmax(dim=1)
                        preds_all += [m(preds).tolist()[0][0]]
                    else: #Surv loss
                        interp = interpolate.interp1d([0, 1], preds, axis=-1, kind='linear')
                        concordance_time = 0.5
                        preds = interp(concordance_time) # approx @ 24
                        preds_all += [preds[0]]

                hits_all += [np.array(hits)[0] == 1]
            preds_all = [-p for p in preds_all]
            c_index = concordance_index_censored(hits_all, [1]*len(hits_all), preds_all)
            # print(c_index)
        # self.c_index.append(c_index)
        return c_index

class CNN_Wrapper:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, lr):
        self.gpu = 1
        self.ordinal = 1

        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.model_name = model_name
        torch.manual_seed(seed)
        # in_size = 167*191*167
        self.num_classes = 4
        self.targets = list(range(self.num_classes))
        self.model = _CNN(drop_rate, fil_num=fil_num, out_channels=self.num_classes).cuda()
        if self.gpu != 1:
            self.model = self.model.cpu()

        self.prepare_dataloader(batch_size, balanced, Data_dir)
        # print(self.imbalanced_ratio)
        # sys.exit()
        # self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        self.criterion = nn.BCEWithLogitsLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        # self.criterion = nn.BCEWithLogitsLoss().cuda()
        # self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.0)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def load(self, dir):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        del st['l2.weight']
        del st['l2.bias']
        self.model.load_state_dict(st, strict=False)
        print('loaded.')

    def prepare_dataloader(self, b_s, balanced, Data_dir):
        train_data = CNN_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            # sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=b_s, sampler=sampler)
            # if self.cox_local != 1:
                # self.train_dataloader = DataLoader(train_data, batch_size=b_s, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=b_s, drop_last=True)
            # if self.cox_local != 1:
                # self.train_dataloader = DataLoader(train_data, batch_size=b_s, shuffle=True, drop_last=True)
        self.valid_dataloader = CNN_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)

        test_data  = CNN_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = CNN_Data(Data_dir, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=b_s)

    def train_model_epoch(self):
        self.model.train(True)
        s_loss = 0
        num = 0
        for inputs, labels in self.train_dataloader:
            if self.ordinal:
                labels = torch.tensor([[1.0]*l+[0.0]*(self.num_classes-l) for l in labels])
            num += 1
            # if self.categorical:
            #     obss = torch.tensor([self.time_class[o.item()] for o in obss])

            inputs, labels = inputs.cuda(), labels.cuda()
            if self.gpu != 1:
                inputs, labels = inputs.cpu(), labels.cpu()

            self.model.zero_grad()
            preds = self.model(inputs)

            loss = self.criterion(preds, labels)
            s_loss += loss
            # loss = self.criterion(preds, labels, labels == 2)
            torch.use_deterministic_algorithms(False)
            loss.backward()
            torch.use_deterministic_algorithms(True)
            self.optimizer.step()
            # print('old loss', loss)
            # print('new loss', self.criterion(self.model(inputs), labels))
            # sys.exit()
        # print(self.epoch,'training loss:', (s_loss/num).item())

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        # val_loss = self.valid_model_epoch() # sanity check
        for self.epoch in range(epochs):
            self.train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                print('{}th epoch validation loss [CE]:'.format(self.epoch), '%.3f' % (val_loss))
                # if self.epoch % (epochs//10) == 0:
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def test(self, trained=False):
        print('testing...')
        if trained:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
            # print(self.model.state_dict())
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            labels_all = []
            for data, label in self.test_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                preds_all += [self.model(data.cuda())]
                labels_all += [label]
            target_names = ['class ' + str(i) for i in self.targets]
            preds_all = [torch.argmax(p).item() for p in preds_all]
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=self.targets, target_names=target_names, zero_division=0, output_dict=False)
            print(report)
            # loss = self.criterion(torch.tensor(preds_all), torch.tensor(labels_all))

        # return loss

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            labels_all = []
            for data, label in self.valid_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                data = torch.tensor(np.expand_dims(data, axis=0))
                data = data.cuda()
                if self.gpu != 1:
                    data = data.cpu()
                labels_all += [label]
                preds_all += [self.model(data).cpu().numpy().squeeze()]
            if self.ordinal:
                labels_orig = labels_all
                labels_all = torch.tensor([[1.0]*l+[0.0]*(self.num_classes-l) for l in labels_all])

            labels_all = torch.tensor(labels_all).cuda()
            preds_all = torch.tensor(preds_all).cuda()
            # print(preds_all)
            # print(preds_all.shape)
            # print(labels_all.shape)
            # sys.exit()
            if self.gpu != 1:
                preds_all = torch.tensor(preds_all).cpu()
                labels_all = torch.tensor(labels_all).cpu()

            loss = self.criterion(preds_all, labels_all)
            # loss = self.criterion(preds_all, labels_all, labels_all == 2)
            target_names = ['class ' + str(i) for i in self.targets]

            labels_all = labels_all.tolist()
            if self.ordinal:
                labels_all = labels_orig
                preds_all = [torch.sum(p>0).item() for p in preds_all]
            else:
                preds_all = [torch.argmax(p).item() for p in preds_all]
            print([preds_all.count(i) for i in self.targets])
            print([labels_all.count(i) for i in self.targets])
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=self.targets, target_names=target_names, zero_division=0, output_dict=False)
            print(report)
            # print(report['accuracy'])
            # loss = -report['accuracy']

        return loss

    def save_checkpoint(self, loss):
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

class CNN_Wrapper_b:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, lr):
        self.gpu = 1

        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.model_name = model_name
        torch.manual_seed(seed)
        # in_size = 167*191*167
        vector_len = 4
        self.targets = list(range(vector_len))
        self.model = _CNN(drop_rate, fil_num=fil_num, out_channels=vector_len).cuda()
        if self.gpu != 1:
            self.model = self.model.cpu()

        self.prepare_dataloader(batch_size, balanced, Data_dir)
        # print(self.imbalanced_ratio)
        # sys.exit()
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        # self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.0)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def load(self, dir):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        del st['l2.weight']
        del st['l2.bias']
        self.model.load_state_dict(st, strict=False)
        print('loaded.')

    def prepare_dataloader(self, b_s, balanced, Data_dir):
        train_data = CNN_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            # sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=b_s, sampler=sampler)
            # if self.cox_local != 1:
                # self.train_dataloader = DataLoader(train_data, batch_size=b_s, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=b_s, drop_last=True)
            # if self.cox_local != 1:
                # self.train_dataloader = DataLoader(train_data, batch_size=b_s, shuffle=True, drop_last=True)
        self.valid_dataloader = CNN_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)

        test_data  = CNN_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = CNN_Data(Data_dir, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=b_s)

    def train_model_epoch(self):
        self.model.train(True)
        s_loss = 0
        num = 0
        for inputs, labels in self.train_dataloader:
            num += 1
            # if self.categorical:
            #     obss = torch.tensor([self.time_class[o.item()] for o in obss])

            inputs, labels = inputs.cuda(), labels.cuda()
            if self.gpu != 1:
                inputs, labels = inputs.cpu(), labels.cpu()

            self.model.zero_grad()
            preds = self.model(inputs)

            loss = self.criterion(preds, labels)
            s_loss += loss
            # loss = self.criterion(preds, labels, labels == 2)
            torch.use_deterministic_algorithms(False)
            loss.backward()
            torch.use_deterministic_algorithms(True)
            self.optimizer.step()
            # print('old loss', loss)
            # print('new loss', self.criterion(self.model(inputs), labels))
            # sys.exit()
        print(self.epoch,'training loss:', (s_loss/num).item())

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        # val_loss = self.valid_model_epoch() # sanity check
        for self.epoch in range(epochs):
            self.train_model_epoch()
            if self.epoch % 1 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                print('{}th epoch validation loss [CE]:'.format(self.epoch), '%.3f' % (val_loss))
                # if self.epoch % (epochs//10) == 0:
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def test(self, trained=False):
        print('testing...')
        if trained:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
            # print(self.model.state_dict())
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            labels_all = []
            for data, label in self.test_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                preds_all += [self.model(data.cuda())]
                labels_all += [label]
            target_names = ['class ' + str(i) for i in self.targets]
            preds_all = [torch.argmax(p).item() for p in preds_all]
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=self.targets, target_names=target_names, zero_division=0, output_dict=False)
            print(report)
            # loss = self.criterion(torch.tensor(preds_all), torch.tensor(labels_all))

        # return loss

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            labels_all = []
            for data, label in self.valid_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                data = torch.tensor(np.expand_dims(data, axis=0))
                data = data.cuda()
                if self.gpu != 1:
                    data = data.cpu()
                labels_all += [label]
                preds_all += [self.model(data).cpu().numpy().squeeze()]

            labels_all = torch.tensor(labels_all).cuda()
            preds_all = torch.tensor(preds_all).cuda()
            # print(preds_all)
            # print(preds_all.shape)
            # print(labels_all.shape)
            # sys.exit()
            if self.gpu != 1:
                preds_all = torch.tensor(preds_all).cpu()
                labels_all = torch.tensor(labels_all).cpu()

            loss = self.criterion(preds_all, labels_all)
            # loss = self.criterion(preds_all, labels_all, labels_all == 2)
            target_names = ['class ' + str(i) for i in self.targets]
            preds_all = [torch.argmax(p).item() for p in preds_all]
            labels_all = labels_all.tolist()
            print([preds_all.count(i) for i in self.targets])
            print([labels_all.count(i) for i in self.targets])
            report = classification_report(y_true=labels_all, y_pred=preds_all, labels=self.targets, target_names=target_names, zero_division=0, output_dict=False)
            print(report)
            # print(report['accuracy'])
            # loss = -report['accuracy']

        return loss

    def save_checkpoint(self, loss):
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

class CNN_Surv_Wrapper_Tra:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, lr, loss_v=0):
        self.gpu = 1
        self.cox_local = 0 #0: global, 1: local
        self.loss_type = 1 # 1 for categorical
        # self.categorical = 0
        self.metric = metric

        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.model_name = model_name
        #'macro avg' or 'weighted avg'
        torch.manual_seed(seed)
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        # self.time_intervals = list(set(self.all_data.time_obs))
        # self.time_intervals.sort()
        # self.time_class = {}
        # self.class_time = {}
        # for i,t in enumerate(self.time_intervals):
        #     self.time_class[t] = i
        #     self.class_time[i] = t

        # in_size = 121*145*121
        vector_len = 3
        self.targets = list(range(vector_len))
        # vector_len = len(self.time_intervals)
        self.model = _CNN_Surv(drop_rate=drop_rate, fil_num=fil_num, out_channels=vector_len).cuda()
        # for n, p in self.model.named_parameters():
            # print(n,p.data)
        # print(sum(p.numel() for p in self.model.parameters()))
        # sys.exit()
        if self.gpu != 1:
            self.model = self.model.cpu()

        self.criterion = sur_loss
        # self.criterion = nn.MSELoss()
        self.lr = lr
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def check(self, dir):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        print( st['l2.weight'])
        print( st['l2.bias'])
        print()
        print( st['l1.weight'])
        print( st['l1.bias'])
        print('loading trained model...')
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        st = torch.load(dir[0])
        print( st['l2.weight'])
        print( st['l2.bias'])
        print()
        print( st['l1.weight'])
        print( st['l1.bias'])

    def load(self, dir, fixed=False):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        del st['l2.weight']
        del st['l2.bias']
        self.model.load_state_dict(st, strict=False)
        if fixed:
            ps = []
            for n, p in self.model.named_parameters():
                if n == 'l2.weight' or n == 'l2.bias' or n == 'l1.weight' or n == 'l1.bias':
                # if n == 'l2.weight' or n == 'l2.bias' :
                    ps += [p]
                    # continue
                else:
                    pass
                    p.requires_grad = False
            self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)
        # for n, p in self.model.named_parameters():
            # print(n, p.requires_grad)
        print('loaded.')

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.train_data = train_data
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            if self.gpu != 1:
                self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            if self.gpu != 1:
                self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
        self.valid_dataloader = AE_Cox_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)

        test_data  = AE_Cox_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=len(train_data))

        Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
        external_data = AE_Cox_Data(Data_dir_NACC, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold, external=True)
        self.external_data = external_data

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, obss, hits in self.train_dataloader:
            # if self.categorical:
            #     obss = torch.tensor([self.time_class[o.item()] for o in obss])

            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.gpu != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()

            # if torch.sum(hits) == 0:
                # continue # because 0 indicates nothing to learn in this batch, we skip it

            self.model.zero_grad()
            preds = self.model(inputs)

            if self.loss_type:
                loss = self.criterion(preds, obss, hits)
            else:
                loss = self.criterion(preds, obss, hits, ver=self.cox_local)

            torch.use_deterministic_algorithms(False)
            loss.backward()
            torch.use_deterministic_algorithms(True)
            clip = 1
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            # print('obss', obss[:10])
            # print('hits', hits[:10])
            # print(preds[:10])
            # print(loss)
            # preds = self.model(inputs)
            # loss = self.criterion(preds, obss, hits)
            # print(preds[:10])
            # print(loss)
            # sys.exit()
        return loss

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1

        cis, bss = self.concord(load=False)
        ci_t, ci_v = cis
        print('initial CI:', 'CI_test vs CI_valid: %.3f : %.3f' % (ci_t[0], ci_v[0]))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for self.epoch in range(epochs):

            train_loss = self.train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//10) == 0:
                val_loss = self.valid_model_epoch()
                cis, bss = self.concord(load=False)
                ci_t, ci_v = cis

                end.record()
                torch.cuda.synchronize()
                print('{}th epoch validation loss [surv] ='.format(self.epoch), '%.3f' % (val_loss), '|| train_loss = %.3f' % (train_loss), '|| CI (test vs valid) = %.3f : %.3f' % (ci_t[0], ci_v[0]), '|| time(s) =', start.elapsed_time(end)//1000)
                # print('{}th epoch validation loss [surv] ='.format(self.epoch), '%.3f' % (val_loss), '|| train_loss = %.3f' % (train_loss), '|| CI (test vs valid) = %.3f : %.3f' % (ci_t[0], ci_v[0]), '|| time(s) =', start.elapsed_time(end)//1000, '|| BS (test vs valid) = %.3f : %.3f' %(bss[0], bss[1]))
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def valid_model_epoch(self):
        if self.metric == 'concord':
            cis, bss = self.concord(load=False)
            ci_t, ci_v = cis
            return -ci_t[0]
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            obss_all = []
            hits_all = []
            # for patches, labels in self.valid_dataloader:
            for data, obss, hits in self.valid_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                data = torch.tensor(np.expand_dims(data, axis=0))
                data = data.cuda()
                if self.gpu != 1:
                    data = data.cpu()
                preds_all += [self.model(data).cpu().numpy().squeeze()]

                # if torch.sum(hits) == 0:
                    # continue # because 0 indicates nothing to learn in this batch, we skip it
                # obss_all += [obss.numpy()[0]]
                # hits_all += [hits.numpy()[0]]
                obss_all += [obss]
                hits_all += [hits]

            # idxs = np.argsort(obss_all, axis=0)[::-1]
            obss_all = np.array(obss_all)
            hits_all = np.array(hits_all)
            # if self.categorical:
            #     obss_all = torch.tensor([self.time_class[o] for o in obss_all])

            if self.gpu != 1:
                # print(preds_all.shape, obss_all.shape, hits_all.shape)
                loss = self.criterion(torch.tensor(preds_all), torch.tensor(obss_all), torch.tensor(hits_all))

            else:
                loss = self.criterion(torch.tensor(preds_all).cuda(), torch.tensor(obss_all).cuda(), torch.tensor(hits_all).cuda())

            # loss = self.criterion(preds_all, torch.tensor(obss_all.reshape(preds_all.shape)).float())
            # loss = self.criterion(preds_all, torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)
            # loss = self.criterion(preds_all, torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)

            # loss = self.criterion(preds_all, obss_all, hits_all, ver=self.cox_local)
        return loss

    def save_checkpoint(self, loss):
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

    def predict_plot(self, id=[10,30], average=False):
        # id: element id in the dataset to plot

        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = self.all_dataloader
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]

        times = train_obss

        data = self.test_data

        fig, ax = plt.subplots()
        self.model = self.model.cpu()
        with torch.no_grad():
            self.model.zero_grad()
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            input, obs, hit = data[id[0]]
            if average:
                inputs = []
                for d in data:
                    item = d[0]
                    if self.csf:
                        item = item.numpy()
                    inputs += [item]
                input = np.mean(inputs,axis=0)
            pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            if average:
                title = 'Average plot'
                ax.plot(times, surv_chances, label=title)
                ax.set(xlabel='time (m)', ylabel='Surv', title='')
                ax.grid()
                ax.legend()
                fig.savefig("likelihood.png")
                return
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred1 = pred

            input, obs, hit = data[id[1]]
            pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred2 = pred
        # print('hit', hit, obs)
        # print(pred)
        # print(sums)
        title = 'ratio (h(x_1)/h(x_2)): %.3f' % (pred1/pred2)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("surv_figs/likelihood.png")
        # plt.show()
        # print(len(surv_chances))
        # print(len(times))
        # print(pred)
        # print(sums[:10])
        # print(surv_chances[:10])
        # print(times[:10])
        # sys.exit()

        return

    def predict_plot_general(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = self.all_dataloader
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]

        times = train_obss

        data = self.test_data

        inputs_0, inputs_1 = [], []
        for d in data:
            item = d[0]
            if d[-1] == 0: #hit
                inputs_0 += [item]
            else:
                inputs_1 += [item]
        input_0 = np.mean(inputs_0,axis=0)
        input_1 = np.mean(inputs_1,axis=0)

        with torch.no_grad():
            self.model.zero_grad()
            input_0 = torch.tensor(input_0).unsqueeze(dim=0)
            input_1 = torch.tensor(input_1).unsqueeze(dim=0)
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            pred_0 = torch.exp(self.model(input_0))
            pred_1 = torch.exp(self.model(input_1))

        fig, ax = plt.subplots()
        event_chances = pred_0 / (sums+pred_0) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 0')

        event_chances = pred_1 / (sums+pred_1) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 1')

        title = 'ratio (AD_0/AD_1): %.3f' % (pred_0/pred_1)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("likelihood_general.png")

        return

    def predict_plot_scatter(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        preds = []
        times = []
        for inputs, obss, hits in self.test_dataloader:

            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.gpu != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()

            preds += [self.model(inputs).item()]
            times += [obss.item()]
        fig, ax = plt.subplots()

        ax.scatter(times, preds)
        title = 'Scatter plot, h_x vs time'
        ax.set(xlabel='time (m)', ylabel='h_x', title=title)
        ax.grid()

        fig.savefig("scatter_plot.png")

        return

    def concord_old(self, load=True, all=False):
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
            train_struc = np.array([(x,y) for x,y in zip(hits_all, obss_all)], dtype=[('hit',bool),('time',float)])
            for dl in dls:
                preds_all, preds_all_brier = [], []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    if self.loss_type: # surv loss
                        interp = interpolate.interp1d(bins, preds, axis=-1, kind='quadratic')
                        concordance_time = 24
                        preds = interp(concordance_time) # approx @ 24
                        preds_all_brier += list(interp(bins))
                        preds_all += [preds[0]]
                    else:
                        preds_all += list(np.array(preds)[0])
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0] == 1]
                # print(preds_all[:10])
                preds_all = [-p for p in preds_all]
                c_index = concordance_index_censored(hits_all, obss_all, preds_all)
                test_struc = np.array([(x,y) for x,y in zip(hits_all, obss_all)], dtype=[('hit',bool),('time',float)])
                bins[-1] = max(obss_all)-1
                bins[0] = min(obss_all)
                # print('train', len(obss_all), obss_all)
                # sys.exit()
                bs = integrated_brier_score(train_struc, test_struc, preds_all_brier, bins)
                cis += [c_index]
                bss += [bs]
            # self.c_index.append(c_index)
        print('done with this 1')
        print(cis, bss)
        return cis, bss

    def concord(self, load=True, all=False, IBS=False):
        # compute CI and BS
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            print(dir)
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        cis = []
        bss = []
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [self.test_dataloader]
            train_dl = DataLoader(self.train_data, batch_size=1, shuffle=False)
            ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)
            if load == False:
                val_dl = DataLoader(self.valid_dataloader, batch_size=1, shuffle=False)
                dls += [val_dl]
            else:
                dls += [train_dl, ext_dl]
            obss_all, hits_all = [], []
            for _, obss, hits in self.train_dataloader:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc_adni = make_struc_array(hits_all, obss_all)
            obss_all, hits_all = [], []
            for _, obss, hits in self.all_dataloader:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc_adni_all = make_struc_array(hits_all, obss_all)
            obss_all, hits_all = [], []
            for _, obss, hits in ext_dl:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc_nacc = make_struc_array(hits_all, obss_all)
            # [test, train, nacc]
            # BS: adni: adni-train~adni-test; adni-full~adni-test
            #     nacc: adni-train~nacc-full; nacc-full~nacc-full
            for i, dl in enumerate(dls):
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
                # print(np.asarray(hits_all) == 1, obss_all, -preds_all[:,1])
                c_index = concordance_index_censored(np.asarray(hits_all) == 1, obss_all, -preds_all[:,1])
                # sys.exit()
                interp = interpolate.PchipInterpolator(bins, preds_all, axis=1)
                new_bins = bins[:-1].copy() + [min(max(obss_all),108)-1]
                preds_all_brier = interp(new_bins)
                test_struc = make_struc_array(hits_all, obss_all)
                if IBS:
                    b_func = retrieve_brier_scores
                else:
                    b_func = _retrieve_brier_scores_fixed_time
                    new_bins = [0,24,48,84]
                if i == 2:
                    bs1 = b_func(train_struc_adni, test_struc, preds_all_brier, new_bins)
                    bs2 = b_func(train_struc_nacc, test_struc, preds_all_brier, new_bins)
                else:
                    bs1 = b_func(train_struc_adni, test_struc, preds_all_brier, new_bins)
                    bs2 = b_func(train_struc_adni_all, test_struc, preds_all_brier, new_bins)
                cis += [c_index]
                bss += [[bs1, bs2]]
        # print(cis, bss)
        return cis, bss

    def overlay_prepare(self, load=True, all=False):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        cis = []
        bss = []
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [self.test_dataloader, DataLoader(self.external_data, batch_size=1, shuffle=False)]
            infos = [self.test_data, self.external_data]
            names = ['ADNI', 'NACC']
            obss_all, hits_all = [], []
            dfs = []

            for dl, info, name in zip(dls, infos, names):
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    # interp = interpolate.interp1d(bins, preds, axis=-1, kind='quadratic')
                    # concordance_time = 24
                    # preds_all_brier += list(interp(bins))
                    # preds_all += [interp(concordance_time)[0]]# approx @ 24
                    preds_all += [list(np.array(preds)[0])]
                    obss_all += [np.array(obss)[0]]
                    hits_all += [float(np.array(hits)[0] == 1)]
                preds_all = np.asarray(preds_all)

                d = {}
                d['RID'] = info.fileIDs
                d['Dataset'] = [name]*len(info.fileIDs)
                d['0'] = preds_all[:,0]
                d['24'] = preds_all[:,1]
                d['48'] = preds_all[:,2]
                d['108'] = preds_all[:,3]
                d['TIMES'] = obss_all
                d['PROGRESSES'] = hits_all
                dfs += [pd.DataFrame(data=d)]
        return dfs

    def overlay_prepare_with_train(self, load=True, all=False):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            print(dir)
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [DataLoader(self.train_data, batch_size=1, shuffle=False, drop_last=True), self.test_dataloader, DataLoader(self.external_data, batch_size=1, shuffle=False)]
            infos = [self.train_data, self.test_data, self.external_data]
            names = ['ADNI_train', 'ADNI', 'NACC']
            obss_all, hits_all = [], []
            dfs = []
            
            for dl, info, name in zip(dls, infos, names):
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    preds_all += [list(np.array(preds)[0])]
                    obss_all += [np.array(obss)[0]]
                    hits_all += [float(np.array(hits)[0] == 1)]
                preds_all = np.asarray(preds_all)
                d = {}
                d['RID'] = info.fileIDs
                d['Dataset'] = [name]*len(info.fileIDs)
                d['0'] = preds_all[:,0]
                d['24'] = preds_all[:,1]
                d['48'] = preds_all[:,2]
                d['108'] = preds_all[:,3]
                d['TIMES'] = obss_all
                d['PROGRESSES'] = hits_all
                dfs += [pd.DataFrame(data=d)]
        return dfs

    def predict_save(self, load=True, bins=[0, 24, 48, 108]):
        # compute CI and BS
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        
        values = []
        with torch.no_grad():
            dls = [self.test_dataloader]
            ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)
            dls += [ext_dl]
            obss_all, hits_all = [], []
            names = ['ADNI_test', 'NACC']
            for i, dl in enumerate(dls):
                preds_all = []
                preds_all_raw = []
                obss_all = []
                hits_all = []
                dname_all = []
                for dname, data, obss, hits in dl:
                    dname_all += [os.path.basename(dname[0])]
                    preds = self.model(data.cuda()).cpu()
                    preds_all_raw.append(preds[0].numpy())
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    preds_all.append(preds[0])
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0]]
                df = pd.DataFrame()
                df['rid'] = dname_all
                # df['pred_raw'] = preds_all_raw
                df[['24', '48', '108']] = np.array(preds_all)[:,1:]
                df['observe'] = obss_all
                df['hit'] = hits_all
                
                dir = './predicts/{}_exp{}_{}.csv'.format(self.model_name, self.exp_idx, names[i])
                df.to_csv(dir)
                # print(pd.read_csv(dir))

    def shap(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        # with torch.no_grad():
        background_dataloader = DataLoader(self.train_data, batch_size=6)
        torch.use_deterministic_algorithms(False)
        for _, data, obss, hits in background_dataloader:
            # preds = self.model(data.cuda()).cpu()
            # preds_prob = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
            # print(preds.shape)
            e = shap.DeepExplainer(self.model, data.cuda())
            out_dir = '/data2/MRI_PET_DATA/shap/'
            bins = [24, 48, 108]
            # note: NACC only!
            ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)

            for rid, data, obss, hits in ext_dl:

                shap_values = e.shap_values(data.cuda())
                shap_values = np.array(shap_values).squeeze()

                aff_mat = np.zeros((4,4))
                aff_mat[0,0] = -1.5
                aff_mat[1,1] = 1.5
                aff_mat[2,2] = 1.5
                aff_mat[3,3] = 1
                aff_mat[0,3] = 90
                aff_mat[1,3] = -126
                aff_mat[2,3] = -72
                # root + exp# + time + rid
                for i in range(len(bins)):
                    fname = out_dir + '{}_{}_{}.nii'.format(self.exp_idx, bins[i], rid[-1])
                    img = nib.Nifti1Image(shap_values[i], aff_mat)
                    # img = nib.load(fname)
                    # print(img.shape)
                    img.to_filename(fname)

            # for now, one background set is used!
            break

class CNN_Surv_Wrapper_Tra_Append:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, lr, l2_w, loss_v=0):
        self.gpu = 1
        self.cox_local = 0 #0: global, 1: local
        self.loss_type = 1 # 1 for categorical
        # self.categorical = 0
        self.metric = metric

        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.model_name = model_name
        #'macro avg' or 'weighted avg'
        torch.manual_seed(seed)
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        # self.time_intervals = list(set(self.all_data.time_obs))
        # self.time_intervals.sort()
        # self.time_class = {}
        # self.class_time = {}
        # for i,t in enumerate(self.time_intervals):
        #     self.time_class[t] = i
        #     self.class_time[i] = t

        # in_size = 121*145*121
        vector_len = 3
        self.targets = list(range(vector_len))
        # vector_len = len(self.time_intervals)
        self.model = _CNN_Surv_Append(drop_rate=drop_rate, fil_num=fil_num, out_channels=vector_len).cuda()
        # for n, p in self.model.named_parameters():
            # print(n,p.data)
        # print(sum(p.numel() for p in self.model.parameters()))
        # sys.exit()
        if self.gpu != 1:
            self.model = self.model.cpu()

        self.criterion = sur_loss
        # self.criterion = nn.MSELoss()
        self.lr = lr
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=l2_w)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def check(self, dir):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        print( st['l2.weight'])
        print( st['l2.bias'])
        print()
        print( st['l1.weight'])
        print( st['l1.bias'])
        print('loading trained model...')
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        st = torch.load(dir[0])
        print( st['l2.weight'])
        print( st['l2.bias'])
        print()
        print( st['l1.weight'])
        print( st['l1.bias'])

    def load(self, dir, fixed=False):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        del st['l2.weight']
        del st['l2.bias']
        self.model.load_state_dict(st, strict=False)
        if fixed:
            ps = []
            for n, p in self.model.named_parameters():
                if n == 'l2.weight' or n == 'l2.bias' or n == 'l1.weight' or n == 'l1.bias':
                # if n == 'l2.weight' or n == 'l2.bias' :
                    ps += [p]
                    # continue
                else:
                    pass
                    p.requires_grad = False
            self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)
        # for n, p in self.model.named_parameters():
            # print(n, p.requires_grad)
        print('loaded.')

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = CNN_Data_Append(Data_dir, self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.train_data = train_data
        if balanced == 1:
            sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            if self.gpu != 1:
                self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            if self.gpu != 1:
                self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
        self.valid_dataloader = CNN_Data_Append(Data_dir, self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)

        test_data  = CNN_Data_Append(Data_dir, self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = CNN_Data_Append(Data_dir, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=len(train_data))

        Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
        external_data = CNN_Data_Append(Data_dir_NACC, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold, external=True)
        self.external_data = external_data

        if True:
            datas = [train_data, self.valid_dataloader, test_data, external_data]
            names = ['ADNI', 'ADNI', 'ADNI', 'NACC']
            for d, n in zip(datas, names):
                fname = 'exp_ids/'+'_'.join([n, d.stage,str(self.exp_idx)])+'.csv'
                df = pd.DataFrame()
                df['rid'] = d.fileIDs
                df.to_csv(fname, index=False)
                # print(pd.read_csv(fname))

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, obss, hits, age, mmse, sex, educ in self.train_dataloader:
            # if self.categorical:
            #     obss = torch.tensor([self.time_class[o.item()] for o in obss])
            inputs_a = torch.tensor([age.tolist(), mmse.tolist(), sex.tolist(), educ.tolist()])
            inputs_a = inputs_a.reshape([-1, 4])
            inputs, inputs_a, obss, hits = inputs.cuda(), inputs_a.cuda(), obss.cuda(), hits.cuda()
            if self.gpu != 1:
                inputs, inputs_a, obss, hits = inputs.cpu(), inputs_a.cpu(), obss.cpu(), hits.cpu()

            # if torch.sum(hits) == 0:
                # continue # because 0 indicates nothing to learn in this batch, we skip it

            self.model.zero_grad()
            preds = self.model([inputs, inputs_a])

            if self.loss_type:
                loss = self.criterion(preds, obss, hits)
            else:
                loss = self.criterion(preds, obss, hits, ver=self.cox_local)

            torch.use_deterministic_algorithms(False)
            loss.backward()
            torch.use_deterministic_algorithms(True)
            clip = 1
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            # preds = self.model(inputs)
            # loss = self.criterion(preds, obss, hits)
            # sys.exit()
        return loss

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1

        cis, bss = self.concord(load=False)
        ci_t, ci_v = cis
        print('initial CI:', 'CI_test vs CI_valid: %.3f : %.3f' % (ci_t[0], ci_v[0]))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for self.epoch in range(epochs):

            train_loss = self.train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//10) == 0:
                val_loss = self.valid_model_epoch()
                cis, bss = self.concord(load=False)
                ci_t, ci_v = cis

                end.record()
                torch.cuda.synchronize()
                if not SWEEP:
                    print('{}th epoch validation loss [surv] ='.format(self.epoch), '%.3f' % (val_loss), '|| train_loss = %.3f' % (train_loss), '|| CI (test vs valid) = %.3f : %.3f' % (ci_t[0], ci_v[0]), '|| time(s) =', start.elapsed_time(end)//1000, '|| BS (test vs valid) = %.3f : %.3f' %(np.mean(bss[0][0][1]), np.mean(bss[0][1][1])))
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def valid_model_epoch(self):
        if self.metric == 'concord':
            cis, bss = self.concord(load=False)
            ci_t, ci_v = cis
            return -ci_t[0]
        print('non-concord need update')
        sys.exit()
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            obss_all = []
            hits_all = []
            # for patches, labels in self.valid_dataloader:
            for data, obss, hits, age, mmse, sex, educ in self.valid_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                data_a = torch.tensor([age.tolist(), mmse.tolist(), sex.tolist(), educ.tolist()]).cuda()


                data = torch.tensor(np.expand_dims(data, axis=0))
                data = data.cuda()
                if self.gpu != 1:
                    data = data.cpu()
                preds_all += [self.model(data).cpu().numpy().squeeze()]

                # if torch.sum(hits) == 0:
                    # continue # because 0 indicates nothing to learn in this batch, we skip it
                # obss_all += [obss.numpy()[0]]
                # hits_all += [hits.numpy()[0]]
                obss_all += [obss]
                hits_all += [hits]

            # idxs = np.argsort(obss_all, axis=0)[::-1]
            obss_all = np.array(obss_all)
            hits_all = np.array(hits_all)
            # if self.categorical:
            #     obss_all = torch.tensor([self.time_class[o] for o in obss_all])

            if self.gpu != 1:
                # print(preds_all.shape, obss_all.shape, hits_all.shape)
                loss = self.criterion(torch.tensor(preds_all), torch.tensor(obss_all), torch.tensor(hits_all))

            else:
                loss = self.criterion(torch.tensor(preds_all).cuda(), torch.tensor(obss_all).cuda(), torch.tensor(hits_all).cuda())

            # loss = self.criterion(preds_all, torch.tensor(obss_all.reshape(preds_all.shape)).float())
            # loss = self.criterion(preds_all, torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)
            # loss = self.criterion(preds_all, torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)

            # loss = self.criterion(preds_all, obss_all, hits_all, ver=self.cox_local)
        return loss

    def save_checkpoint(self, loss):
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

    def predict_plot(self, id=[10,30], average=False):
        # id: element id in the dataset to plot

        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = self.all_dataloader
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]

        times = train_obss

        data = self.test_data

        fig, ax = plt.subplots()
        self.model = self.model.cpu()
        with torch.no_grad():
            self.model.zero_grad()
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            input, obs, hit = data[id[0]]
            if average:
                inputs = []
                for d in data:
                    item = d[0]
                    if self.csf:
                        item = item.numpy()
                    inputs += [item]
                input = np.mean(inputs,axis=0)
            pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            if average:
                title = 'Average plot'
                ax.plot(times, surv_chances, label=title)
                ax.set(xlabel='time (m)', ylabel='Surv', title='')
                ax.grid()
                ax.legend()
                fig.savefig("likelihood.png")
                return
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred1 = pred

            input, obs, hit = data[id[1]]
            pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred2 = pred
        # print('hit', hit, obs)
        # print(pred)
        # print(sums)
        title = 'ratio (h(x_1)/h(x_2)): %.3f' % (pred1/pred2)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("likelihood.png")
        # plt.show()
        # print(len(surv_chances))
        # print(len(times))
        # print(pred)
        # print(sums[:10])
        # print(surv_chances[:10])
        # print(times[:10])
        # sys.exit()

        return

    def predict_plot_general(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = self.all_dataloader
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]

        times = train_obss

        data = self.test_data

        inputs_0, inputs_1 = [], []
        for d in data:
            item = d[0]
            if d[-1] == 0: #hit
                inputs_0 += [item]
            else:
                inputs_1 += [item]
        input_0 = np.mean(inputs_0,axis=0)
        input_1 = np.mean(inputs_1,axis=0)

        with torch.no_grad():
            self.model.zero_grad()
            input_0 = torch.tensor(input_0).unsqueeze(dim=0)
            input_1 = torch.tensor(input_1).unsqueeze(dim=0)
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            pred_0 = torch.exp(self.model(input_0))
            pred_1 = torch.exp(self.model(input_1))

        fig, ax = plt.subplots()
        event_chances = pred_0 / (sums+pred_0) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 0')

        event_chances = pred_1 / (sums+pred_1) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 1')

        title = 'ratio (AD_0/AD_1): %.3f' % (pred_0/pred_1)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("likelihood_general.png")

        return

    def predict_plot_scatter(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        preds = []
        times = []
        for inputs, obss, hits in self.test_dataloader:

            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.gpu != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()

            preds += [self.model(inputs).item()]
            times += [obss.item()]
        fig, ax = plt.subplots()

        ax.scatter(times, preds)
        title = 'Scatter plot, h_x vs time'
        ax.set(xlabel='time (m)', ylabel='h_x', title=title)
        ax.grid()

        fig.savefig("scatter_plot.png")

        return

    def concord_old(self, load=True, all=False):
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
            train_struc = np.array([(x,y) for x,y in zip(hits_all, obss_all)], dtype=[('hit',bool),('time',float)])
            for dl in dls:
                preds_all, preds_all_brier = [], []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    if self.loss_type: # surv loss
                        interp = interpolate.interp1d(bins, preds, axis=-1, kind='quadratic')
                        concordance_time = 24
                        preds = interp(concordance_time) # approx @ 24
                        preds_all_brier += list(interp(bins))
                        preds_all += [preds[0]]
                    else:
                        preds_all += list(np.array(preds)[0])
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0] == 1]
                # print(preds_all[:10])
                preds_all = [-p for p in preds_all]
                c_index = concordance_index_censored(hits_all, obss_all, preds_all)
                test_struc = np.array([(x,y) for x,y in zip(hits_all, obss_all)], dtype=[('hit',bool),('time',float)])
                bins[-1] = max(obss_all)-1
                bins[0] = min(obss_all)
                # print('train', len(obss_all), obss_all)
                # sys.exit()
                bs = integrated_brier_score(train_struc, test_struc, preds_all_brier, bins)
                cis += [c_index]
                bss += [bs]
            # self.c_index.append(c_index)
        print('done with this 1')
        print(cis, bss)
        return cis, bss

    def concord(self, load=False, all=False, IBS=False):
        # compute CI and BS
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        cis = []
        bss = []
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [self.test_dataloader]
            train_dl = DataLoader(self.train_data, batch_size=1, shuffle=False)
            ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)
            val_dl = DataLoader(self.valid_dataloader, batch_size=1, shuffle=False)
            if not load:
                dls += [val_dl]
            else:
                dls += [train_dl, ext_dl]
            obss_all, hits_all = [], []
            for _, obss, hits, age, mmse, sex, educ in self.train_dataloader:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc_adni = make_struc_array(hits_all, obss_all)
            obss_all, hits_all = [], []
            for _, obss, hits, age, mmse, sex, educ in self.all_dataloader:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc_adni_all = make_struc_array(hits_all, obss_all)
            obss_all, hits_all = [], []
            for _, obss, hits, age, mmse, sex, educ in dls[-1]:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc_nacc = make_struc_array(hits_all, obss_all)
            # [test, train, nacc]
            # BS: adni: adni-train~adni-test; adni-full~adni-test
            #     nacc: adni-train~nacc-full; nacc-full~nacc-full
            for i, dl in enumerate(dls):
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits, age, mmse, sex, educ in dl:
                    preds = self.model([data.cuda(), torch.cat((age, mmse, sex, educ), dim=0).cuda()]).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    preds_all.append(preds)
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0]]
                preds_all = np.concatenate(preds_all,axis=0)
                # print(np.asarray(hits_all) == 1, obss_all, -preds_all[:,1])
                c_index = concordance_index_censored(np.asarray(hits_all) == 1, obss_all, -preds_all[:,1])
                # sys.exit()
                interp = interpolate.PchipInterpolator(bins, preds_all, axis=1)
                new_bins = bins[:-1].copy() + [min(max(obss_all),108)-1]
                preds_all_brier = interp(new_bins)
                test_struc = make_struc_array(hits_all, obss_all)
                if IBS:
                    b_func = retrieve_brier_scores
                else:
                    b_func = _retrieve_brier_scores_fixed_time
                    new_bins = [0,24,48,84]
                if i == 2:
                    bs1 = b_func(train_struc_adni, test_struc, preds_all_brier, new_bins)
                    bs2 = b_func(train_struc_nacc, test_struc, preds_all_brier, new_bins)
                else:
                    bs1 = b_func(train_struc_adni, test_struc, preds_all_brier, new_bins)
                    bs2 = b_func(train_struc_adni_all, test_struc, preds_all_brier, new_bins)
                cis += [c_index]
                bss += [[bs1, bs2]]
        # print(cis, bss)
        return cis, bss

    def overlay_prepare(self, load=True, all=False):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        cis = []
        bss = []
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [self.test_dataloader, DataLoader(self.external_data, batch_size=1, shuffle=False)]
            infos = [self.test_data, self.external_data]
            names = ['ADNI', 'NACC']
            obss_all, hits_all = [], []
            dfs = []

            for dl, info, name in zip(dls, infos, names):
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    # interp = interpolate.interp1d(bins, preds, axis=-1, kind='quadratic')
                    # concordance_time = 24
                    # preds_all_brier += list(interp(bins))
                    # preds_all += [interp(concordance_time)[0]]# approx @ 24
                    preds_all += [list(np.array(preds)[0])]
                    obss_all += [np.array(obss)[0]]
                    hits_all += [float(np.array(hits)[0] == 1)]
                preds_all = np.asarray(preds_all)

                d = {}
                d['RID'] = info.fileIDs
                d['Dataset'] = [name]*len(info.fileIDs)
                d['0'] = preds_all[:,0]
                d['24'] = preds_all[:,1]
                d['48'] = preds_all[:,2]
                d['108'] = preds_all[:,3]
                d['TIMES'] = obss_all
                d['PROGRESSES'] = hits_all
                dfs += [pd.DataFrame(data=d)]
        return dfs

    def overlay_prepare_with_train(self, load=True, all=False):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [DataLoader(self.train_data, batch_size=1, shuffle=False, drop_last=True), self.test_dataloader, DataLoader(self.external_data, batch_size=1, shuffle=False)]
            infos = [self.train_data, self.test_data, self.external_data]
            names = ['ADNI_train', 'ADNI', 'NACC']
            obss_all, hits_all = [], []
            dfs = []
            
            for dl, info, name in zip(dls, infos, names):
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    preds_all += [list(np.array(preds)[0])]
                    obss_all += [np.array(obss)[0]]
                    hits_all += [float(np.array(hits)[0] == 1)]
                preds_all = np.asarray(preds_all)
                d = {}
                d['RID'] = info.fileIDs
                d['Dataset'] = [name]*len(info.fileIDs)
                d['0'] = preds_all[:,0]
                d['24'] = preds_all[:,1]
                d['48'] = preds_all[:,2]
                d['108'] = preds_all[:,3]
                d['TIMES'] = obss_all
                d['PROGRESSES'] = hits_all
                dfs += [pd.DataFrame(data=d)]
        return dfs

    def shap(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        with torch.no_grad():
            background_dataloader = DataLoader(self.train_data, batch_size=len(self.train_data))
            for data, obss, hits, age, mmse, sex, educ in background_dataloader:
                data = np.array(data)
                add_data = np.array([age.numpy(), mmse.numpy(), sex.numpy(), educ.numpy()]).T
                print(add_data.shape)
                print(data.shape)
                # item = np.array(data, add_data)
                item = np.array(data, add_data)
                print(item.shape)
                sys.exit()
                item = np.append(data, add_data, axis=1)
                print(item.shape)
                sys.exit()
                print(np.array(item)[0].shape)
                print(np.array(item)[1].shape)
                # preds = self.model([data.cuda(), torch.cat((age, mmse, sex, educ), dim=0).cuda()]).cpu()
                sys.exit()
            # e = shap.DeepExplainer(model, background)
            shap_values = e.shap_values(test_images)

            for dname, data, obss, hits, age, mmse, sex, educ in dl:
                dname_all += [os.path.basename(dname[0])]
                preds = self.model([data.cuda(), torch.cat((age, mmse, sex, educ), dim=0).cuda()]).cpu()
                preds_all_raw.append(preds[0].numpy())
                preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                preds_all.append(preds[0])
                obss_all += [np.array(obss)[0]]
                hits_all += [np.array(hits)[0]]
            for data, obss, hits in self.test_dataloader:
                print(data.shape)
                sys.exit()
                shap_vals = exer(data)
                plt.clf()
                shap.plots.waterfall(shap_vals[0],matplotlib=True,show=False)
                plt.savefig('shap.png')
                break

    def binary_and_roc(self, load=True, thres=0.5):
        # compute CI and BS
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        bins = [0, 24, 48, 108]
        values = []
        with torch.no_grad():
            dls = [self.test_dataloader]
            ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)
            dls += [ext_dl]
            obss_all, hits_all = [], []
            for i, dl in enumerate(dls):
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits, age, mmse, sex, educ in dl:
                    preds = self.model([data.cuda(), torch.cat((age, mmse, sex, educ), dim=0).cuda()]).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    preds_all.append(preds)
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0]]
                preds_all = np.concatenate(preds_all,axis=0)
                preds_all = (preds_all[:, -1]>thres).astype('int')
                # print('thres', thres)
                # print(classification_report(hits_all, preds_all))
                # sys.exit()
                values += [[hits_all, preds_all]]
                # interp = interpolate.PchipInterpolator(bins, preds_all, axis=1)
                # preds_all_brier = interp(new_bins)
        return values

    def predict_save(self, load=True, bins=[0, 24, 48, 108]):
        # compute CI and BS
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        
        values = []
        with torch.no_grad():
            dls = [self.test_dataloader]
            ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)
            dls += [ext_dl]
            obss_all, hits_all = [], []
            names = ['ADNI_test', 'NACC']
            for i, dl in enumerate(dls):
                preds_all = []
                preds_all_raw = []
                obss_all = []
                hits_all = []
                dname_all = []
                for dname, data, obss, hits, age, mmse, sex, educ in dl:
                    dname_all += [os.path.basename(dname[0])]
                    preds = self.model([data.cuda(), torch.cat((age, mmse, sex, educ), dim=0).cuda()]).cpu()
                    preds_all_raw.append(preds[0].numpy())
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    preds_all.append(preds[0])
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0]]
                df = pd.DataFrame()
                df['rid'] = dname_all
                # df['pred_raw'] = preds_all_raw
                df[['24', '48', '108']] = np.array(preds_all)[:,1:]
                df['observe'] = obss_all
                df['hit'] = hits_all
                
                dir = './predicts/{}_exp{}_{}.csv'.format(self.model_name, self.exp_idx, names[i])
                df.to_csv(dir)
                # print(pd.read_csv(dir))

    def calc_aucs(self, load=True, bins=[0, 24, 48, 108]):
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        auc_datasets = []
        with torch.no_grad():
            dls = [self.test_dataloader]
            ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)
            dls += [ext_dl]
            obss_all, hits_all = [], []
            for i, dl in enumerate(dls):
                # if i == 0:
                #     continue
                preds_all = []
                obss_all = []
                hits_all = []
                for data, obss, hits, age, mmse, sex, educ in dl:
                    preds = self.model([data.cuda(), torch.cat((age, mmse, sex, educ), dim=0).cuda()]).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    preds_all.append(preds)
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0]]
                preds_all = np.concatenate(preds_all,axis=0)
                interp = interpolate.PchipInterpolator(bins, preds_all, axis=1)
                new_bins = list(range(bins[-1]))
                preds_all = interp(new_bins)
                aucs = []
                print(i, 'dl')
                for t in new_bins[13:]:
                    # print(t)
                    # auc = self.time_auc(preds_all[:, i], obss_all, hits_all, t)
                    auc = self.auc_star(preds_all[:, t], obss_all, hits_all, t)
                    aucs += [auc]
                    if t > 20:
                        break
                auc_datasets += [aucs]
        return auc_datasets

    def time_auc(self, preds, obss, hits, t):
        # https://click.endnote.com/viewer?doi=10.1002%2Fsim.5958&token=WzI0MjQ5NywiMTAuMTAwMi9zaW0uNTk1OCJd.SG6Zx7AcHzxCXg7IcJZq19ufSAI
        # formula (8)
        # AUC(t) = (1)/(2)
        # (1) = sum_i(sum_j( Ind(min(T_i,C_i)<=t & i_progressed) * 1/P(min(T_i,C_i)>t) * Ind(min(T_j,C_j)>t) * 1/P(C>t) * Ind(M_i>M_j) ))
        # (2) = sum_i( Ind(min(T_i,C_i)<=t & i_progressed) * 1/P(min(T_i,C_i)>t) ) * sum_j( Ind(min(T_j,C_j)>t) * 1/P(C>t) )
        upper = 0
        lower_l = 0
        lower_r = 0
        for i in range(len(preds)):
            for j in range(len(hits)):
                ind_i = (obss[i] <= t) & hits[i]
                W_i = w_estimator(obss, hits, i)
                ind_j = obss[j] > t
                W_t = w_estimator(obss, hits, t)
                ind_m = preds[i] > preds[j]
                upper += ind_i * W_i * ind_j * W_t * ind_m
        for i in range(len(preds)):
            ind_i = (obss[i] <= t) & hits[i]
            W_i = w_estimator(obss, hits, i)
            lower_l += ind_i * W_i
        for j in range(len(preds)):
            ind_j = obss[j] > t
            W_t = w_estimator(obss, hits, t)
            lower_r += ind_j * W_t
        return upper/(lower_l*lower_r)

    def auc_star(self, predicted, time_to_progression, hit, time):
        # probability that surv_i > surv_j | Ti <= t, hit = 1, Tj > t
        # with i and j the indices of two independent subjects
        num = 0
        denom_i = 0
        denom_j = 0
        for i in range(len(hit)):
            hit_i = (time_to_progression[i] <= time) & (hit[i] == 1)
            # print(time_to_progression, hit, i)
            try:
                weight_i = w_estimator(time_to_progression, hit, i)
            except:
                continue
            for j in range(len(hit)):
                surv_j = (time_to_progression[j] > time)
                try:
                    weight_j = w_estimator(time_to_progression, hit, time)
                except:
                    continue
                i_gt_j = predicted[i] > predicted[j]
                i_gt_j += 0.5*(predicted[i] == predicted[j])
                num += hit_i*weight_i*surv_j*weight_j*i_gt_j
            denom_i += hit_i*weight_i
        for j in range(len(hit)):
            surv_j = (time_to_progression[j] > time)
            try:
                weight_j = w_estimator(time_to_progression, hit, time)
            except:
                continue
            denom_j += surv_j*weight_j
        return num/(denom_i*denom_j+num/1e10)

class CNN_Surv_Wrapper_Res(CNN_Surv_Abstract):
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, lr, loss_v=0):
        self.gpu = 1
        self.cox_local = 0 #0: global, 1: local
        self.loss_type = 1 # 1 for categorical
        # self.categorical = 0
        self.metric = metric

        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.model_name = model_name
        #'macro avg' or 'weighted avg'
        torch.manual_seed(seed)
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        # self.time_intervals = list(set(self.all_data.time_obs))
        # self.time_intervals.sort()
        # self.time_class = {}
        # self.class_time = {}
        # for i,t in enumerate(self.time_intervals):
        #     self.time_class[t] = i
        #     self.class_time[i] = t

        # in_size = 121*145*121
        vector_len = 3
        self.targets = list(range(vector_len))
        # vector_len = len(self.time_intervals)
        self.model = _CNN_Surv_Res(drop_rate=drop_rate, fil_num=fil_num, out_channels=vector_len).cuda()
        # for n, p in self.model.named_parameters():
            # print(n,p.data)
        # print(sum(p.numel() for p in self.model.parameters()))
        # sys.exit()
        if self.gpu != 1:
            self.model = self.model.cpu()

        # self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        self.criterion = sur_loss
        # self.criterion = nn.MSELoss()
        self.lr = lr
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def check(self, dir):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        print( st['l2.weight'])
        print( st['l2.bias'])
        print()
        print( st['l1.weight'])
        print( st['l1.bias'])
        print('loading trained model...')
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        st = torch.load(dir[0])
        print( st['l2.weight'])
        print( st['l2.bias'])
        print()
        print( st['l1.weight'])
        print( st['l1.bias'])

    def load(self, dir, fixed=False):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        del st['l2.weight']
        del st['l2.bias']
        self.model.load_state_dict(st, strict=False)
        if fixed:
            ps = []
            for n, p in self.model.named_parameters():
                if n == 'l2.weight' or n == 'l2.bias' or n == 'l1.weight' or n == 'l1.bias':
                    ps += [p]
                    # continue
                else:
                    pass
                    p.requires_grad = False
            self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)
        # for n, p in self.model.named_parameters():
            # print(n, p.requires_grad)
        print('loaded.')

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        train_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.train_data = train_data
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            if self.gpu != 1:
                self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            if self.gpu != 1:
                self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
        self.valid_dataloader = AE_Cox_Data(Data_dir, self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)

        test_data  = AE_Cox_Data(Data_dir, self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=len(train_data))

        Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
        external_data = AE_Cox_Data(Data_dir_NACC, self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold, external=True)
        self.external_data = external_data

    def train_model_epoch(self):
        self.model.train(True)
        for inputs, obss, hits in self.train_dataloader:
            # if self.categorical:
            #     obss = torch.tensor([self.time_class[o.item()] for o in obss])

            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.gpu != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()

            # if torch.sum(hits) == 0:
                # continue # because 0 indicates nothing to learn in this batch, we skip it

            self.model.zero_grad()
            preds = self.model(inputs)

            if self.loss_type:
                loss = self.criterion(preds, obss, hits)
            else:
                loss = self.criterion(preds, obss, hits, ver=self.cox_local)

            torch.use_deterministic_algorithms(False)
            loss.backward()
            torch.use_deterministic_algorithms(True)
            clip = 1
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            # print('obss', obss[:10])
            # print('hits', hits[:10])
            # print(preds[:10])
            # print(loss)
            # preds = self.model(inputs)
            # loss = self.criterion(preds, obss, hits)
            # print(preds[:10])
            # print(loss)
            # sys.exit()
        return loss

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1

        cis, bss = self.concord(load=False)
        ci_t, ci_v = cis
        print('initial CI:', 'CI_test vs CI_valid: %.3f : %.3f' % (ci_t[0], ci_v[0]))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for self.epoch in range(epochs):

            train_loss = self.train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//10) == 0:
                val_loss = self.valid_model_epoch()
                cis, bss = self.concord(load=False)
                ci_t, ci_v = cis

                end.record()
                torch.cuda.synchronize()
                print('{}th epoch validation loss [surv] ='.format(self.epoch), '%.3f' % (val_loss), '|| train_loss = %.3f' % (train_loss), '|| CI (test vs valid) = %.3f : %.3f' % (ci_t[0], ci_v[0]), '|| time(s) =', start.elapsed_time(end)//1000, '|| BS (test vs valid) = %.3f : %.3f' %(bss[0], bss[1]))
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def valid_model_epoch(self):
        if self.metric == 'concord':
            cis, bss = self.concord(load=False)
            ci_t, ci_v = cis
            return -ci_t[0]
        with torch.no_grad():
            self.model.train(False)
            # loss_all = 0
            preds_all = []
            obss_all = []
            hits_all = []
            # for patches, labels in self.valid_dataloader:
            for data, obss, hits in self.valid_dataloader:
                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                data = torch.tensor(np.expand_dims(data, axis=0))
                data = data.cuda()
                if self.gpu != 1:
                    data = data.cpu()
                preds_all += [self.model(data).cpu().numpy().squeeze()]

                # if torch.sum(hits) == 0:
                    # continue # because 0 indicates nothing to learn in this batch, we skip it
                # obss_all += [obss.numpy()[0]]
                # hits_all += [hits.numpy()[0]]
                obss_all += [obss]
                hits_all += [hits]

            # idxs = np.argsort(obss_all, axis=0)[::-1]
            obss_all = np.array(obss_all)
            hits_all = np.array(hits_all)
            # if self.categorical:
            #     obss_all = torch.tensor([self.time_class[o] for o in obss_all])

            if self.gpu != 1:
                # print(preds_all.shape, obss_all.shape, hits_all.shape)
                loss = self.criterion(torch.tensor(preds_all), torch.tensor(obss_all), torch.tensor(hits_all))

            else:
                loss = self.criterion(torch.tensor(preds_all).cuda(), torch.tensor(obss_all).cuda(), torch.tensor(hits_all).cuda())

            # loss = self.criterion(preds_all, torch.tensor(obss_all.reshape(preds_all.shape)).float())
            # loss = self.criterion(preds_all, torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)
            # loss = self.criterion(preds_all, torch.tensor(obss_all).squeeze(), torch.tensor(hits_all).squeeze(), ver=self.cox_local)#,debug=True)

            # loss = self.criterion(preds_all, obss_all, hits_all, ver=self.cox_local)
        return loss

    def save_checkpoint(self, loss):
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

    def predict_plot(self, id=[10,30], average=False):
        # id: element id in the dataset to plot

        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = self.all_dataloader
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]

        times = train_obss

        data = self.test_data

        fig, ax = plt.subplots()
        self.model = self.model.cpu()
        with torch.no_grad():
            self.model.zero_grad()
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            input, obs, hit = data[id[0]]
            if average:
                inputs = []
                for d in data:
                    item = d[0]
                    if self.csf:
                        item = item.numpy()
                    inputs += [item]
                input = np.mean(inputs,axis=0)
            pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            if average:
                title = 'Average plot'
                ax.plot(times, surv_chances, label=title)
                ax.set(xlabel='time (m)', ylabel='Surv', title='')
                ax.grid()
                ax.legend()
                fig.savefig("likelihood.png")
                return
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred1 = pred

            input, obs, hit = data[id[1]]
            pred = torch.exp(self.model(torch.tensor(input).unsqueeze(dim=0)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            # surv_chances = 1 - event_chances
            # surv_chances = event_chances.numpy()[::-1]
            surv_chances = 1 - event_chances.numpy()
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred2 = pred
        # print('hit', hit, obs)
        # print(pred)
        # print(sums)
        title = 'ratio (h(x_1)/h(x_2)): %.3f' % (pred1/pred2)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("likelihood.png")
        # plt.show()
        # print(len(surv_chances))
        # print(len(times))
        # print(pred)
        # print(sums[:10])
        # print(surv_chances[:10])
        # print(times[:10])
        # sys.exit()

        return

    def predict_plot_general(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = self.all_dataloader
        train_x, train_obss, train_hits = [],[],[]
        for items in train_dataloader:
            train_x, train_obss, train_hits = items
        idxs = torch.argsort(train_obss, dim=0, descending=True)
        train_x = train_x[idxs]
        train_obss = train_obss[idxs]
        train_hits = train_hits[idxs]

        times = train_obss

        data = self.test_data

        inputs_0, inputs_1 = [], []
        for d in data:
            item = d[0]
            if d[-1] == 0: #hit
                inputs_0 += [item]
            else:
                inputs_1 += [item]
        input_0 = np.mean(inputs_0,axis=0)
        input_1 = np.mean(inputs_1,axis=0)

        with torch.no_grad():
            self.model.zero_grad()
            input_0 = torch.tensor(input_0).unsqueeze(dim=0)
            input_1 = torch.tensor(input_1).unsqueeze(dim=0)
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)

            pred_0 = torch.exp(self.model(input_0))
            pred_1 = torch.exp(self.model(input_1))

        fig, ax = plt.subplots()
        event_chances = pred_0 / (sums+pred_0) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 0')

        event_chances = pred_1 / (sums+pred_1) #on Training set, need to add
        surv_chances = 1 - event_chances.numpy()
        ax.plot(times, surv_chances, label='AD status: 1')

        title = 'ratio (AD_0/AD_1): %.3f' % (pred_0/pred_1)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("likelihood_general.png")

        return

    def predict_plot_scatter(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        preds = []
        times = []
        for inputs, obss, hits in self.test_dataloader:

            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()
            if self.gpu != 1:
                inputs, obss, hits = inputs.cpu(), obss.cpu(), hits.cpu()

            preds += [self.model(inputs).item()]
            times += [obss.item()]
        fig, ax = plt.subplots()

        ax.scatter(times, preds)
        title = 'Scatter plot, h_x vs time'
        ax.set(xlabel='time (m)', ylabel='h_x', title=title)
        ax.grid()

        fig.savefig("scatter_plot.png")

        return

    def concord_old(self, load=True, all=False):
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
            train_struc = np.array([(x,y) for x,y in zip(hits_all, obss_all)], dtype=[('hit',bool),('time',float)])
            for dl in dls:
                preds_all, preds_all_brier = [], []
                obss_all = []
                hits_all = []
                for data, obss, hits in dl:
                    preds = self.model(data.cuda()).cpu()
                    preds = np.concatenate((np.ones((preds.shape[0],1)), np.cumprod(preds.numpy(), axis=1)), axis=1)
                    if self.loss_type: # surv loss
                        interp = interpolate.interp1d(bins, preds, axis=-1, kind='quadratic')
                        concordance_time = 24
                        preds = interp(concordance_time) # approx @ 24
                        preds_all_brier += list(interp(bins))
                        preds_all += [preds[0]]
                    else:
                        preds_all += list(np.array(preds)[0])
                    obss_all += [np.array(obss)[0]]
                    hits_all += [np.array(hits)[0] == 1]
                # print(preds_all[:10])
                preds_all = [-p for p in preds_all]
                c_index = concordance_index_censored(hits_all, obss_all, preds_all)
                test_struc = np.array([(x,y) for x,y in zip(hits_all, obss_all)], dtype=[('hit',bool),('time',float)])
                bins[-1] = max(obss_all)-1
                bins[0] = min(obss_all)
                # print('train', len(obss_all), obss_all)
                # sys.exit()
                bs = integrated_brier_score(train_struc, test_struc, preds_all_brier, bins)
                cis += [c_index]
                bss += [bs]
            # self.c_index.append(c_index)
        return cis, bss

    def concord(self, load=True, all=False, IBS=False):
        # compute CI and BS
        if load:
            dir = glob.glob(self.checkpoint_dir + '*.pth')
            self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        cis = []
        bss = []
        bins = [0, 24, 48, 108]
        with torch.no_grad():
            dls = [self.test_dataloader]
            train_dl = DataLoader(self.train_data, batch_size=1, shuffle=False)
            ext_dl = DataLoader(self.external_data, batch_size=1, shuffle=False)
            dls += [train_dl, ext_dl]
            obss_all, hits_all = [], []
            for _, obss, hits in self.train_dataloader:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc_adni = make_struc_array(hits_all, obss_all)
            obss_all, hits_all = [], []
            for _, obss, hits in self.all_dataloader:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc_adni_all = make_struc_array(hits_all, obss_all)
            obss_all, hits_all = [], []
            for _, obss, hits in dls[-1]:
                obss_all += obss.tolist()
                hits_all += hits.tolist()
            train_struc_nacc = make_struc_array(hits_all, obss_all)
            # [test, train, nacc]
            # BS: adni: adni-train~adni-test; adni-full~adni-test
            #     nacc: adni-train~nacc-full; nacc-full~nacc-full
            for i, dl in enumerate(dls):
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
                # print(np.asarray(hits_all) == 1, obss_all, -preds_all[:,1])
                c_index = concordance_index_censored(np.asarray(hits_all) == 1, obss_all, -preds_all[:,1])
                # sys.exit()
                interp = interpolate.PchipInterpolator(bins, preds_all, axis=1)
                new_bins = bins[:-1].copy() + [min(max(obss_all),108)-1]
                preds_all_brier = interp(new_bins)
                test_struc = make_struc_array(hits_all, obss_all)
                if IBS:
                    b_func = retrieve_brier_scores
                else:
                    b_func = _retrieve_brier_scores_fixed_time
                    new_bins = [0,24,48,84]
                if i == 2:
                    bs1 = b_func(train_struc_adni, test_struc, preds_all_brier, new_bins)
                    bs2 = b_func(train_struc_nacc, test_struc, preds_all_brier, new_bins)
                else:
                    bs1 = b_func(train_struc_adni, test_struc, preds_all_brier, new_bins)
                    bs2 = b_func(train_struc_adni_all, test_struc, preds_all_brier, new_bins)
                cis += [c_index]
                bss += [[bs1, bs2]]
        # print(cis, bss)
        return cis, bss

    def shap(self):
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        with torch.no_grad():
            exer = shap.Explainer(self.model)
            for data, obss, hits in self.test_dataloader:
                print(data.shape)
                sys.exit()
                shap_vals = exer(data)
                plt.clf()
                shap.plots.waterfall(shap_vals[0],matplotlib=True,show=False)
                plt.savefig('shap.png')
                break


if __name__ == "__main__":
    print('networks.py')
    o = [[1.,2.,1.],[2.,1.,1.],[3.,4.,2.]]
    output = torch.tensor(o, requires_grad=True)
    target = torch.empty(3).random_(2)
    print(output)
    print(target)
    cox_loss = cox_loss(output, target)
    print(cox_loss)
