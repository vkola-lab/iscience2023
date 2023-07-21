# training surv-based network with pretrained model
# Created: 5/21/2021
# Status: OK
# Consider merge into 1 main file
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python transfer_main.py

# import torch.nn as nn
import sys
import torch
import numpy as np
import pandas as pd
sys.path.insert(1, '../plot/')
from networks import CNN_Surv_Wrapper_Tra
from utils import read_json, cross_set
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.8.0"):
    torch.use_deterministic_algorithms(True)
else:
    torch.set_deterministic(True)

def cnn_surv_main(repeat, model_name, setting, Wrapper):
    print('Evaluation metric: {}'.format(setting['metric']))
    c_te, c_tr, c_ex = [], [], []
    b_ad_tr, b_ad_al, b_na_ad, b_na_na = [], [], [], []
    IBS=False
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
        # cnn.predict_plot()
        # cnn.predict_save()#set dataloader to 1 if use this!
        # cnn.shap() #set dataloader to 1 if use this!
        # out = cnn.concord(load=True, all=True, IBS=IBS)
        out = cnn.concord(all=True)
        c_te += [out[0][0][0]]
        c_tr += [out[0][1][0]]
        c_ex += [out[0][2][0]]
        if IBS:
            b_ad_tr += [out[1][0][0]]
            b_ad_al += [out[1][0][1]]
            b_na_ad += [out[1][2][0]]
            b_na_na += [out[1][2][1]]
        else:
            b_ad_tr += [out[1][0][0][1]]
            b_ad_al += [out[1][0][1][1]]
            b_na_ad += [out[1][2][0][1]]
            b_na_na += [out[1][2][1][1]]
        
    print('CI test: %.3f+-%.3f' % (np.mean(c_te), np.std(c_te)))
    print('CI train: %.3f+-%.3f' % (np.mean(c_tr), np.std(c_tr)))
    print('CI external: %.3f+-%.3f' % (np.mean(c_ex), np.std(c_ex)))
    if IBS:
        print('BS (ad_tr): %.3f+-%.3f' % (np.mean(b_ad_tr), np.std(b_ad_tr)))
        print('BS (ad_al): %.3f+-%.3f' % (np.mean(b_ad_al), np.std(b_ad_al)))
        print('BS (na_ad): %.3f+-%.3f' % (np.mean(b_na_ad), np.std(b_na_ad)))
        print('BS (na_na): %.3f+-%.3f' % (np.mean(b_na_na), np.std(b_na_na)))
    else:
        strs = ['ad_tr', 'ad_al', 'na_ad', 'na_na']
        outs = [b_ad_tr, b_ad_al, b_na_ad, b_na_na]
        for s, o in zip(strs, outs):
            print('BS (%s):' % s)
            means = np.mean(o,axis=0)
            stds = np.std(o,axis=0)
            for m, s in zip(means, stds):
                print('\t%.3f+-%.3f' % (m, s))
    
# this is for comparison figure generation, turn off if not needed
# corresponding file is */statistics/survival_plot_xz.py
def cnn_surv_main_dfs(repeat, model_name, setting, Wrapper):
    print('Evaluation metric: {}'.format(setting['metric']))
    dfss = []
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
        dfss += [cnn.overlay_prepare(load=True)]
    # df_adni = pd.DataFrame()
    df_adni = dfss[0][0]
    df_nacc = dfss[0][1]
    for i in range(1, 5):
        df_adni = df_adni.append(dfss[i][0], ignore_index=True)
        df_nacc = df_nacc.append(dfss[i][1], ignore_index=True)
    df_nacc = df_nacc.groupby(['RID', 'Dataset']).mean().reset_index()
    df = df_adni.append(df_nacc, ignore_index=True)
    df.to_csv('SCNN.csv')

def cnn_dfs_raw(repeat, model_name, setting, Wrapper):
    print('Evaluation metric: {}'.format(setting['metric']))
    dfss = []
    exp_idxx = {'ADNI': [], 'NACC': []}
    for exp_idx in range(repeat):
        print('*'*50)
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
        df = cnn.overlay_prepare(load=True)
        dfss += [df]
        exp_idxx['ADNI'] += [[1*exp_idx]*len(df[0])]
        exp_idxx['NACC'] += [[1*exp_idx]*len(df[1])]
    df_adni = dfss[0][0]
    df_adni['Exp'] = exp_idxx['ADNI'][0]
    df_nacc = dfss[0][1]
    df_nacc['Exp'] = exp_idxx['NACC'][0]
    for i in range(1, 5):
        dfss_adni = dfss[i][0].copy()
        dfss_adni['Exp'] = exp_idxx['ADNI'][i]
        dfss_nacc = dfss[i][1].copy()
        dfss_nacc['Exp'] = exp_idxx['NACC'][i]
        df_adni = df_adni.append(dfss_adni, ignore_index=True)
        df_nacc = df_nacc.append(dfss_nacc, ignore_index=True)
    df = df_adni.append(df_nacc, ignore_index=True)
    df.to_csv('SCNN_raw.csv')

def cnn_dfs_raw_train(num_exps, model_name, setting, Wrapper):
    print('Evaluation metric: {}'.format(setting['metric']))
    dfss = []
    exp_idxx = {'ADNI': [], 'NACC': [], 'ADNI_train': []}
    for exp_idx in range(num_exps):
        print('*'*50)
        cnn = Wrapper(fil_num        = setting['fil_num'],
                        drop_rate       = setting['drop_rate'],
                        batch_size      = setting['batch_size'],
                        balanced        = setting['balanced'],
                        Data_dir        = setting['Data_dir'],
                        lr              = setting['learning_rate'],
                        exp_idx         = exp_idx,
                        num_fold        = num_exps,
                        seed            = 1000*exp_idx,
                        model_name      = model_name,
                        metric          = setting['metric'])
        df = cnn.overlay_prepare_with_train(load=True)
        dfss += [df]
        exp_idxx['ADNI'] += [[1*exp_idx]*len(df[1])]
        exp_idxx['NACC'] += [[1*exp_idx]*len(df[2])]
        exp_idxx['ADNI_train'] += [[1*exp_idx]*len(df[0])]
    
    df_adni_train = dfss[0][0]
    df_adni_train['Exp'] = exp_idxx['ADNI_train'][0]
    df_adni = dfss[0][1]
    df_adni['Exp'] = exp_idxx['ADNI'][0]
    df_nacc = dfss[0][2]
    df_nacc['Exp'] = exp_idxx['NACC'][0]
    for i in range(1, 5):
        dfss_adni_train = dfss[i][0].copy()
        dfss_adni_train['Exp'] = exp_idxx['ADNI_train'][i]
        dfss_adni = dfss[i][1].copy()
        dfss_adni['Exp'] = exp_idxx['ADNI'][i]
        dfss_nacc = dfss[i][2].copy()
        dfss_nacc['Exp'] = exp_idxx['NACC'][i]
        df_adni = df_adni.append(dfss_adni, ignore_index=True)
        df_nacc = df_nacc.append(dfss_nacc, ignore_index=True)
        df_adni_train = df_adni_train.append(dfss_adni_train, ignore_index=True)
    
    # print(df_adni.sort_values(by=['RID'], ignore_index=True))
    # sys.exit()
    df = df_adni.append(df_nacc, ignore_index=True).append(df_adni_train, ignore_index=True)
    df.to_csv('SCNN_raw.csv')
    print(df)
    return df

def main(train=False):
    config = read_json('./cnn_config.json')

    CWrapper = CNN_Surv_Wrapper_Tra
    cnn_config = config['cnn_surv']
    print('running CNN_surv_tra classifiers')

    cnn_repeat = 5
    c_name = 'cnn_mri'

    print('-'*100)
    if train:
        cnn_surv_main(cnn_repeat, c_name+'_surv_tra_2_F1', cnn_config, Wrapper=CWrapper)
        # cnn_surv_main(cnn_repeat, c_name+'_surv_tra', cnn_config, Wrapper=CWrapper)
        # cnn_surv_main(cnn_repeat, c_name+'_surv_tra_5', cnn_config, Wrapper=CWrapper) #5 is # of layer
    cnn_dfs_raw_train(cnn_repeat, c_name+'_surv_tra_2_F1', cnn_config, Wrapper=CWrapper) #To run this, udpate fileIDs on dataloader!
    # cnn_dfs_raw_train(cnn_repeat, c_name+'_surv_tra', cnn_config, Wrapper=CWrapper)
    print('-'*100)

if __name__ == "__main__":
    main(True)
    # main()