# training surv-based network with pretrained model
# Created: 7/1/2022
# Status: OK
# Consider merge into 1 main file
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python transfer_append_main.py

# import torch.nn as nn
import sys
import torch
import wandb

import numpy as np
import pandas as pd

sys.path.insert(1, '../plot/')
from networks import CNN_Surv_Wrapper_Tra_Append
from utils import read_json, plot_roc, plot_time_auc
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.8.0"):
    torch.use_deterministic_algorithms(True)
else:
    torch.set_deterministic(True)

def cnn_surv_main(repeat, model_name, setting, Wrapper, thres=0.5):
    print('Evaluation metric: {}'.format(setting['metric']))
    c_te, c_tr, c_ex = [], [], []
    b_ad_tr, b_ad_al, b_na_ad, b_na_na = [], [], [], []
    IBS=False
    test_vals_all = []
    ext_vals_all = []
    for exp_idx in range(repeat):
        cnn = Wrapper(fil_num        = setting['fil_num'],
                        drop_rate       = setting['drop_rate'],
                        batch_size      = setting['batch_size'],
                        balanced        = setting['balanced'],
                        Data_dir        = setting['Data_dir'],
                        lr              = setting['learning_rate'],
                        exp_idx         = exp_idx,
                        num_fold        = repeat,
                        seed            = 1000*exp_idx,
                        model_name      = model_name,
                        metric          = setting['metric'])
        # cnn.load('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre_2', 0), fixed=False)
        # cnn.load('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0), fixed=True)
        # cnn.train(epochs = setting['train_epochs'])
        # cnn.check('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0))
        # 0 - test, 1 - valid, 2 - train, 3 - external
        # out = cnn.concord(all=True)
        # out = cnn.concord(load=True, all=True, IBS=IBS)
        # c_te += [out[0][0][0]]
        # c_tr += [out[0][1][0]]
        # c_ex += [out[0][2][0]]
        # if IBS:
        #     b_ad_tr += [out[1][0][0]]
        #     b_ad_al += [out[1][0][1]]
        #     b_na_ad += [out[1][2][0]]
        #     b_na_na += [out[1][2][1]]
        # else:
        #     b_ad_tr += [out[1][0][0][1]]
        #     b_ad_al += [out[1][0][1][1]]
        #     b_na_ad += [out[1][2][0][1]]
        #     b_na_na += [out[1][2][1][1]]
        # [test_aucs, ext_aucs] = cnn.calc_aucs()
        [test_aucs, ext_aucs] = cnn.calc_aucs()
        if not test_vals_all:
            test_vals_all = test_aucs
            ext_vals_all = ext_aucs
        else:
            test_vals_all = np.concatenate((test_vals_all, test_aucs), axis=1)
            ext_vals_all = np.concatenate((ext_vals_all, ext_aucs), axis=1)
        print(np.array(ext_vals_all).shape)
        # test_vals_all = np.concatenate((test_vals_all, test_vals), axis=1)
        # ext_vals_all = np.concatenate((ext_vals_all, ext_vals), axis=1)
    # plot_roc(test_vals_all, fname='plot/'+str(thres)+'_ADNI_test')
    # plot_roc(ext_vals_all, fname='plot/'+str(thres)+'_NACC')
    print(ext_vals_all.shape)
    plot_time_auc(test_vals_all, fname='plot/'+'time_auc_ADNI_test')
    plot_time_auc(ext_vals_all, fname='plot/'+'time_auc_NACC')
        
        
    # print('CI test: %.3f+-%.3f' % (np.mean(c_te), np.std(c_te)))
    # print('CI train: %.3f+-%.3f' % (np.mean(c_tr), np.std(c_tr)))
    # print('CI external: %.3f+-%.3f' % (np.mean(c_ex), np.std(c_ex)))
    # if IBS:
    #     print('BS (ad_tr): %.3f+-%.3f' % (np.mean(b_ad_tr), np.std(b_ad_tr)))
    #     print('BS (ad_al): %.3f+-%.3f' % (np.mean(b_ad_al), np.std(b_ad_al)))
    #     print('BS (na_ad): %.3f+-%.3f' % (np.mean(b_na_ad), np.std(b_na_ad)))
    #     print('BS (na_na): %.3f+-%.3f' % (np.mean(b_na_na), np.std(b_na_na)))
    # else:
    #     strs = ['ad_tr', 'ad_al', 'na_ad', 'na_na']
    #     outs = [b_ad_tr, b_ad_al, b_na_ad, b_na_na]
    #     for s, o in zip(strs, outs):
    #         print('BS (%s):' % s)
    #         means = np.mean(o,axis=0)
    #         stds = np.std(o,axis=0)
    #         for m, s in zip(means, stds):
    #             print('\t%.3f+-%.3f' % (m, s))
    
def main(train=False):
    config = read_json('./cnn_config.json')

    CWrapper = CNN_Surv_Wrapper_Tra_Append
    cnn_config = config['cnn_surv']
    print('running CNN_surv_tra classifiers')

    cnn_repeat = 2
    c_name = 'cnn_mri'

    print('-'*100)
    if train:
        cnn_surv_main(cnn_repeat, c_name+'_surv_tra_append', cnn_config, Wrapper=CWrapper)
        # for thres in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            # cnn_surv_main(cnn_repeat, c_name+'_surv_tra_append', cnn_config, Wrapper=CWrapper, thres=thres)
    # cnn_dfs_raw_train(cnn_repeat, c_name+'_surv_tra_append', cnn_config, Wrapper=CWrapper)
    print('-'*100)

if __name__ == "__main__":
    main(True)