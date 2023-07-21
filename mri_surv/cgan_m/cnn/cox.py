# cox.py
# Train a model that fuses the output of 3 FCNs for AD status prediction

import torch
torch.set_deterministic(True)
import torch.nn as nn
import sys
import math
import pandas as pd
import numpy as np

sys.path.insert(1, './plot/')
from plot import report_table, roc_plot_perfrom_table
from networks import MLP_Wrapper, MLP_Wrapper_f1
from dataloader import Cox_Data
from utils import read_json, cross_set
from lifelines import CoxPHFitter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report


# def mlp_main(fcn_repeat, mlp_repeat, model_name, mode, mlp_setting):
#     for exp_idx in range(fcn_repeat):
#         for repe_idx in range(mlp_repeat):
#             mlp = MLP_Wrapper_f1(fil_num         = mlp_setting['fil_num'],
#                                 drop_rate       = mlp_setting['drop_rate'],
#                                 batch_size      = mlp_setting['batch_size'],
#                                 balanced        = mlp_setting['balanced'],
#                                 roi_threshold   = mlp_setting['roi_threshold'],
#                                 exp_idx         = exp_idx,
#                                 seed            = repe_idx*exp_idx,
#                                 mode            = mode,
#                                 model_name      = model_name,
#                                 lr              = mlp_setting['learning_rate'],
#                                 metric          = mlp_setting['metric'],
#                                 yr              = mlp_setting['yr'])
#             mlp.train(epochs = mlp_setting['train_epochs'])
#             mlp.test(repe_idx)

def prepare_data(exp_idx, seed=1000, roi_t=0.6, roi_c=200):
    mode = ['mri', 'amyloid', 'fdg']
    Data_dir = ['./DPMs/fcn_{}_exp{}/'.format(m, exp_idx) for m in mode]
    choice = 'count'
    yr = '' #yr not needed in fact
    seed = 1*exp_idx

    train_data = Cox_Data(Data_dir, mode, exp_idx, stage='train', roi_threshold=roi_t, roi_count=roi_c, choice=choice, seed=seed, yr=yr)
    valid_data = Cox_Data(Data_dir, mode, exp_idx, stage='valid', roi_threshold=roi_t, roi_count=roi_c, choice=choice, seed=seed, yr=yr)
    test_data  = Cox_Data(Data_dir, mode, exp_idx, stage='test', roi_threshold=roi_t, roi_count=roi_c, choice=choice, seed=seed, yr=yr)
    # sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
    train = DataLoader(train_data, batch_size=1, shuffle=True)
    valid = DataLoader(valid_data, batch_size=1, shuffle=False)
    test = DataLoader(test_data, batch_size=1, shuffle=False)
    in_size = train_data.in_size
    return train, valid, test, in_size

if __name__ == "__main__":
    print('running Cox survival regression')
    # mlp_repeat = 5
    # fcn_repeat = 5

    # cnn_config = read_json('./cnn_config.json')

    # mlp_names = ['fcn_mri_amyloid_fdg_mlp']
    # scan_names = ['mri', 'amyloid', 'fdg']
    # modes = mlp_names
    #
    # print('-'*100)
    # mlp_main(fcn_repeat, mlp_repeat, mlp_names[0], scan_names, cnn_config['mlp'])
    # print('-'*100)
    #
    # report_table(mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # report_table(txt_file='report', mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # roc_plot_perfrom_table(mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    train, valid, test, in_size = prepare_data(exp_idx=0)
    train_target = []
    valid_target = []
    test_target = []
    ds = [train, valid, test]
    ts = [[],[],[]]
    for i in range(len(ds)):
        d = []
        t = []
        for inputs, obs, hit in ds[i]:
            inputs, obs, hit = inputs.tolist()[0], obs.item(), hit.item()
            row = inputs+[obs, hit] # duration - month, AD status
            # print(obs,hit)
            # sys.exit()
            d += [row]
            t += [1 if hit else 0]

        # print(np.array(d).shape)
        # print(t)
        ds[i] = pd.DataFrame(data=d)
        ts[i] = t
    [train, valid, test] = ds

    # rossi = load_rossi()
    cph = CoxPHFitter(penalizer=0.1)
    # from lifelines.datasets import load_rossi
    # rossi = load_rossi()
    # print(set(rossi['week']))
    # print(set(train[600]))
    cph.fit(train, duration_col=600, event_col=601)
    # print(cph._central_values)
    # print(cph.predict_survival_function(cph._central_values))
    # cph.print_summary()  # access the individual results using cph.summary
    ax = cph.plot_partial_effects_on_outcome(covariates=1, values=[1], cmap='coolwarm')
    ax.figure.savefig('survival.png')

    sys.exit()

    prefix = ['training', 'validation', 'testing']
    for i in range(len(ds)):
        X = ds[i]
        pred = cph.predict_median(X)
        pred = pred.replace(np.inf, 4)
        target_names = ['class ' + str(i) for i in range(4)]
        print(prefix[i], 'accuracy:', accuracy_score(pred, ts[i]))
        print(classification_report(y_true=ts[i], y_pred=pred, labels=[1,2,3,4], target_names=target_names, zero_division=0))




    # print(cph.predict_survival_function(X))
    # print(cph.predict_partial_hazard(X))


    # problem: since the variance is low, the result may not be optimal
