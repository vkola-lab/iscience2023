import csv
import re
import numpy as np
import torch
import torch.nn as nn
import os, sys
import json
import glob
import pandas as pd
from scipy.stats import zscore, sem
import random
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from utils import read_csv_cox2 as read_csv_cox
import matplotlib.pyplot as plt

accu_all = []
f1_all = []
sens_all = []
spec_all = []

def sur_loss(preds, obss, hits, bins=torch.Tensor([[0, 12, 24, 36, 108]])):
    l_h_x = torch.log(preds.squeeze())
    n_l_h_x = torch.log(1-preds.squeeze())
    survived_bins = torch.ge(obss.view(-1,1), bins[0,1:])
    event_bins = torch.logical_and(torch.ge(obss.view(-1,1), bins[0,:-1]),
                 torch.lt(obss.view(-1, 1), bins[0,1:]))
    hit_bins = torch.logical_and(event_bins, hits.view(-1,1).bool())
    survived_bins = torch.logical_and(survived_bins, torch.logical_not(
            hit_bins))
    pos_sum = torch.sum(l_h_x[hit_bins], axis=0) + torch.sum(n_l_h_x[survived_bins], axis=0)
    return torch.sum(-pos_sum)

def cox_loss(preds, obss, hits):
    idxs = torch.argsort(obss, dim=0, descending=True)
    h_x = preds[idxs].view(-1,1)
    obss = obss[idxs].view(-1,1)
    hits = hits[idxs].view(-1,1)
    num_hits = torch.sum(hits)
    obss_mat = torch.subtract(obss, obss.T)
    obss_mat = torch.where(obss_mat >= 0, 1., 0.)
    first_term = torch.mul(h_x,hits.float())

    if 1:
        second_term = torch.mul(torch.log(torch.matmul(obss_mat, torch.exp(h_x))),hits.float())
    else:
        second_term = torch.mul(torch.logsumexp(torch.mul(obss_mat, h_x.T), dim=1, keepdims=True), hits)

    loss = -torch.div(torch.sum(torch.sub(first_term, second_term)),
                      num_hits)
    if torch.isnan(loss):
        raise Exception('Loss is nan')
    return loss

def calculate_metrics(y_true, y_pred):
    #Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    #Calculate f1
    f1 = f1_score(y_true, y_pred)

    #Calculate sensitivity & specificity
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]==1:
            true_pos += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
            false_pos += 1
        if y_true[i]==y_pred[i]==0:
            true_neg += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
            false_neg += 1

    sensitivity = true_pos/(true_pos+false_neg)
    specificity = true_neg/(true_neg+false_pos)

    return accuracy, f1, sensitivity, specificity

def sem_196(list):
    return 1.96*sem(list)

def _read_csv_parcellations(filename):
    parcellation_tbl = pd.read_csv(filename)
    col_regex = re.compile(r'corr_.*')
    valid_columns = [col for col in parcellation_tbl.columns if
                     col_regex.match(col)]
    replacement_dict = {x: x.replace('corr_vol_','') for x in valid_columns}
    parcellation_tbl = parcellation_tbl[valid_columns + ['RID']].copy()
    parcellation_tbl = parcellation_tbl.rename(columns=replacement_dict)
    return parcellation_tbl

def _average_hemispheres(dataframe):
    dataframe.rename(lambda x: re.sub('^r', '', x), axis=1, inplace=True)
    dataframe.rename(lambda x: re.sub('^l', '', x), axis=1, inplace=True)
    dataframe = dataframe.groupby(dataframe.columns, axis=1).agg(np.mean)
    return dataframe

def read_csv2(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fileIDs, labels = [], []
        for r in reader:
            fileIDs += [str(int(float(r['RID'])))]
            labels += [int(float(r['PROGRESSES']))]
    fileIDs = ['0'*(4-len(f))+f for f in fileIDs]
    return fileIDs, labels

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def write_raw_score(f, preds, labels):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')

def _retrieve_index_partition(idxs, stage, _l, ratio):
    split1 = int(_l * ratio[0])
    split2 = int(_l * (ratio[0] + ratio[1]))
    if 'train' in stage:
        return idxs[:split1]
    elif 'valid' in stage:
        return idxs[split1:split2]
    elif 'test' in stage:
        return idxs[split2:]
    elif 'all' in stage:
        return idxs
    else:
        raise Exception('Unexpected Stage for FCN_Cox_Data!')

class _MLP(nn.Module):
    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(fil_num)
        # self.bn3 = nn.BatchNorm1d(4)
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 4)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self, X):
        X = self.bn1(X)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        # out = self.bn3(out)
        out = self.sig(out)
        return out

class ParcellationData(Dataset):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2)):
        random.seed(seed)
        self.csv_directory = \
            '/home/mfromano/Research/mri-pet/metadata/data_processed/'
        csvname = self.csv_directory + 'merged_dataframe_cox_pruned_final.csv'
        rids, time_obs, time_hit = read_csv_cox(csvname) #training file
        csvname2 = self.csv_directory + 'mri3_cat12_cox.csv'
        parcellation_df = _read_csv_parcellations(csvname2)
        parcellation_df['RID'] = parcellation_df['RID'].apply(
                lambda x: str(int(x)).zfill(4)
        )
        idx_to_drop = np.where([x not in parcellation_df['RID'].to_numpy() for x in rids])
        # print(len(rids))
        # print(len(idx_to_drop))
        # sys.exit()
        for idx in idx_to_drop[0]:
            try:
                rids.pop(idx)
                time_obs.pop(idx)
                time_hit.pop(idx)
            except:
                pass
                

        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[rids,:].reset_index(drop=True)

        parcellation_df = _average_hemispheres(parcellation_df)
        parcellation_df = parcellation_df.to_numpy()

        self.time_obs = []
        self.time_hit = []
        for obs, hit in zip(time_obs, time_hit):
            self.time_obs += [hit if hit else obs]
            self.time_hit += [1 if hit else 0]

        l = len(rids)
        idxs = list(range(len(rids)))
        random.shuffle(idxs)
        self.index_list = _retrieve_index_partition(idxs, stage, l, ratio)
        self.fileIDs = np.array(rids)[self.index_list]
        self.data_l = parcellation_df

        self.data_l = torch.FloatTensor(self.data_l)

        self.data = self.data_l

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        x = self.data[idx]
        obs = self.time_obs[idx]
        hit = self.time_hit[idx]
        return x, obs, hit

class MLP_Wrapper:
    def __init__(self, fil_num, drop_rate, seed, exp_idx,
                 model_name, lr, weight_decay, choice='count'):
        self.seed = seed
        self.choice = choice
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.Data_dir = './DPMs/mlp_exp{}/'.format(exp_idx)
        self.prepare_dataloader(seed)
        self.criterion = sur_loss
        torch.manual_seed(seed)
        self.model = _MLP(in_size=self.in_size, fil_num=fil_num, drop_rate=drop_rate).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(
                0.5, 0.999), weight_decay=weight_decay)
        self.loss_imp = 0.0
        self.loss_tot = 0.0

    def save_checkpoint(self, loss):
        # if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:

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

    def prepare_dataloader(self,seed):
        train_data = ParcellationData(seed, stage = 'train')
        valid_data = ParcellationData(seed, stage = 'valid')
        test_data = ParcellationData(seed, stage = 'test')
        self.train_dataloader = DataLoader(train_data, batch_size=len(
                train_data), shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.in_size = train_data.data.shape[1]
        self.all_data = ParcellationData(seed, stage = 'all')
        self.train_data = train_data
        self.test_data = test_data

    def train(self, epochs):
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            val_loss = self.valid_model_epoch()
            self.save_checkpoint(val_loss)
            if self.epoch % 30 == 0:
                print('{}th epoch validation score:'.format(self.epoch),
                      '%.4f' % (val_loss))
        print('Best model saved at the {}th epoch; cox-based loss:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)

        for inputs, obss, hits in self.train_dataloader:
            if torch.sum(hits) == 0:
                continue
            self.model.zero_grad()
            preds = self.model(inputs)
            loss = self.criterion(preds, obss, hits)
            loss.backward()
            self.optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            preds_all = []
            obss_all = []
            hits_all = []
            for data, obss, hits in self.valid_dataloader:
                patches, obs, hit = data, obss, hits
                preds = self.model(patches)
                preds_all += [preds.numpy()]
                obss_all += [obss.numpy()]
                hits_all += [hits.numpy()]
            loss = self.criterion(torch.tensor(preds_all), torch.tensor(obss_all), torch.tensor(hits_all))
        return loss

    def predict_plot(self, id=[10,30]):
        # id: element id in the dataset to plot

        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)

        train_dataloader = DataLoader(self.all_data, batch_size=len(self.all_data))
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
        with torch.no_grad():
            self.model.zero_grad()
            preds = torch.exp(self.model(train_x))
            sums = torch.cumsum(preds, dim=0)
            input, obs, hit = data[id[0]]
            pred = torch.exp(self.model(input.view(1, -1)))
            event_chances = pred / (sums+pred) #on Training set, need to add
            surv_chances = 1 - event_chances.numpy()
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred1 = pred

            input, obs, hit = data[id[1]]
            pred = torch.exp(self.model(input.view(1, -1)))
            event_chances = pred / (sums+pred)
            surv_chances = 1 - event_chances.numpy()
            title = 'AD status: ' + str(hit) + ', observation time: ' + str(obs)
            ax.plot(times, surv_chances, label=title)
            pred2 = pred
        title = 'ratio (h(x_1)/h(x_2)): %.3f' % (pred1/pred2)
        ax.set(xlabel='time (m)', ylabel='Surv', title=title)
        ax.grid()
        ax.legend()

        fig.savefig("likelihood_csf.png")

        return


def mlp_main(mlp_repeat, model_name, mlp_setting, weight_decay=[0]):
    for curr_wt in weight_decay:
        for repe_idx in range(mlp_repeat):
            curr_model_name = model_name + f'wt_decay_{curr_wt}'
            mlp = MLP_Wrapper(fil_num         = mlp_setting['fil_num'],
                                drop_rate       = mlp_setting['drop_rate'],
                                exp_idx         = repe_idx,
                                seed            = repe_idx,
                                model_name      = curr_model_name,
                                lr              = mlp_setting['learning_rate'],
                                weight_decay    = curr_wt)
            mlp.train(epochs = mlp_setting['train_epochs'])
        # mlp.predict_plot()

def main():
    mlp_repeat = 5

    print('running MLP classifiers')

    mlp_config = read_json('./cnn_config.json')

    m_name = 'mlp_parcellation'

    mlp_main(mlp_repeat, m_name, mlp_config['mlp_parcellation'])
    print('-'*100)

    sys.exit()

if __name__ == "__main__":
    main()
