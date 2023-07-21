# training surv-based network without pretrained model
# Modified: 5/21/2021
# Status: OK
# Consider merge into 1 main file

import torch.nn as nn
import sys
sys.path.insert(1, '../plot/')
from plot import report_table,roc_plot_perfrom_table
from networks import CNN_Surv_Wrapper
from utils import read_json, cross_set
import numpy as np

import torch
torch.set_deterministic(True)

def cnn_surv_main(fcn_repeat, model_name, fcn_setting, Wrapper):
    print('Evaluation metric: {}'.format(fcn_setting['metric']))
    c=[]
    for exp_idx in range(fcn_repeat):
        cnn = Wrapper(fil_num        = fcn_setting['fil_num'],
                        drop_rate       = fcn_setting['drop_rate'],
                        batch_size      = fcn_setting['batch_size'],
                        balanced        = fcn_setting['balanced'],
                        Data_dir        = fcn_setting['Data_dir'],
                        lr              = fcn_setting['learning_rate'],
                        exp_idx         = exp_idx,
                        num_fold        = fcn_repeat,
                        seed            = 1000*exp_idx,
                        model_name      = model_name,
                        metric          = fcn_setting['metric'])
        # cnn.train(epochs = fcn_setting['train_epochs'])
        # cnn.test()
        c += [cnn.concord()[0]]
        # cnn.shap()
        # print('exit')
        # sys.exit()
        # cnn.predict_plot()
        # cnn.predict_plot_general()
        # cnn.predict_plot_scatter()

    print(np.mean(c), '+-', np.std(c))


def main():
    config = read_json('./cnn_config.json')

    CWrapper = CNN_Surv_Wrapper
    cnn_config = config['cnn_surv']
    print('running CNN_surv classifiers')

    cnn_repeat = 1
    cnn_names = ['cnn_mri', 'cnn_amyloid', 'cnn_fdg']
    # scan_names = ['mri', 'amyloid', 'fdg']
    modes = []

    print('-'*100)
    for c_name in cnn_names:
        modes += [c_name]
        cnn_surv_main(cnn_repeat, c_name+'_surv', cnn_config, Wrapper=CWrapper)
        print('-'*100)
        break
    sys.exit()
    print(modes)

    if cox:
        report_table(txt_file='report.txt', mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat, out=2)
    else:
        report_table(txt_file='report.txt', mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # roc_plot_perfrom_table(mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # cross_set(mode=modes, fcn_repeat=mlp_repeat, mlp_repeat=fcn_repeat)

if __name__ == "__main__":
    main()