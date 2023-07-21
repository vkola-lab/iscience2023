# training surv-based network with resnet model
# Created: 5/21/2021
# Status: OK
# Consider merge into 1 main file
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python res_main.py

# import torch.nn as nn
import sys
import torch
import numpy as np
import pandas as pd
sys.path.insert(1, '../plot/')
from networks import CNN_Surv_Wrapper_Res
from utils import read_json, cross_set
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.8.0"):
    torch.use_deterministic_algorithms(True)
else:
    torch.set_deterministic(True)

def cnn_surv_main(repeat, model_name, setting, Wrapper):
    print('Evaluation metric: {}'.format(setting['metric']))
    c_te, c_tr, c_ex = [], [], []
    b_te, b_tr, b_ex = [], [], []
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
        # cnn.load('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0), fixed=False)
        # cnn.load('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0), fixed=True)
        cnn.train(epochs = setting['train_epochs'])
        # cnn.check('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0))
        # 0 - test, 1 - valid, 2 - train, 3 - external
        # out = cnn.concord(all=True)
        out = cnn.concord(load=True, all=True)
        c_te += [out[0][0][0]]
        c_tr += [out[0][2][0]]
        c_ex += [out[0][3][0]]
        b_te += [out[1][0]]
        b_tr += [out[1][2]]
        b_ex += [out[1][3]]
        
        # cnn.shap()
        # print(c_te)
        # print(c_tr)
        # print(c_ex)
        # print('exit')
        # cnn.predict_plot()
        # cnn.predict_plot_general()
        # cnn.predict_plot_scatter()
    print('CI test: %.3f+-%.3f' % (np.mean(c_te), np.std(c_te)))
    print('CI train: %.3f+-%.3f' % (np.mean(c_tr), np.std(c_tr)))
    print('CI external: %.3f+-%.3f' % (np.mean(c_ex), np.std(c_ex)))
    print('BS test: %.3f+-%.3f' % (np.mean(b_te), np.std(b_te)))
    print('BS train: %.3f+-%.3f' % (np.mean(b_tr), np.std(b_tr)))
    print('BS external: %.3f+-%.3f' % (np.mean(b_ex), np.std(b_ex)))
    
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
        cnn.concord(all=True)
        cnn.concord_old(all=True)
        dfss += [cnn.overlay_prepare(load=True)]
    # df_adni = pd.DataFrame()
    df_adni = dfss[0][0]
    df_nacc = dfss[0][1]
    for i in range(1, 5):
        df_adni = df_adni.append(dfss[i][0], ignore_index=True)
        df_nacc = df_nacc.append(dfss[i][1], ignore_index=True)
    df_nacc = df_nacc.groupby(['RID', 'Dataset']).mean().reset_index()
    df = df_adni.append(df_nacc, ignore_index=True)
    # print(df_adni)
    # print(df_nacc)
    # print(df)
    df.to_csv('SCNN.csv')
    # sys.exit()

def main(train=True):
    config = read_json('./cnn_config.json')

    CWrapper = CNN_Surv_Wrapper_Res
    cnn_config = config['cnn_surv_res']
    print('running CNN_surv_res classifiers')

    # cnn_main(repeat, 'cnn', config['cnn'])

    # # Testing model using only MRI scans
    # fcn_main(fcn_repeat, 'fcn_mri_test', False, config['fcn_test'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_test_mlp', 'mri_test_', config['mlp'])

    cnn_repeat = 5
    cnn_names = ['cnn_mri', 'cnn_amyloid', 'cnn_fdg']
    scan_names = ['mri', 'amyloid', 'fdg']
    modes = []

    print('-'*100)
    for c_name in cnn_names:
        modes += [c_name]
        # Model using only MRI scans
        if train:
            cnn_surv_main(cnn_repeat, c_name+'_surv_res', cnn_config, Wrapper=CWrapper)
        # cnn_surv_main_dfs(cnn_repeat, c_name+'_surv_tra', cnn_config, Wrapper=CWrapper)
        print('-'*100)
        break
    sys.exit()
    print(modes)
    # cross_set(mode=modes, fcn_repeat=mlp_repeat, mlp_repeat=fcn_repeat)

if __name__ == "__main__":
    main()
