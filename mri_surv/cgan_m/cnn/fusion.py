# fusion.py
# Train a model that fuses the output of 3 FCNs together for AD status prediction


# classifier baselines: applying the previous cnn (vanilla cnn, with our tuned parameters) for baseline estimation
import torch.nn as nn
import sys
sys.path.insert(1, './plot/')
from plot import report_table,roc_plot_perfrom_table
from networks import MLP_Wrapper, MLP_Wrapper_f1
from utils import read_json, cross_set

import torch
torch.set_deterministic(True)

def mlp_main(fcn_repeat, mlp_repeat, model_name, mode, mlp_setting):
    for exp_idx in range(fcn_repeat):
        for repe_idx in range(mlp_repeat):
            mlp = MLP_Wrapper_f1(fil_num         = mlp_setting['fil_num'],
                                drop_rate       = mlp_setting['drop_rate'],
                                batch_size      = mlp_setting['batch_size'],
                                balanced        = mlp_setting['balanced'],
                                roi_threshold   = mlp_setting['roi_threshold'],
                                exp_idx         = exp_idx,
                                seed            = repe_idx*exp_idx,
                                mode            = mode,
                                model_name      = model_name,
                                lr              = mlp_setting['learning_rate'],
                                metric          = mlp_setting['metric'],
                                yr              = mlp_setting['yr'])
            mlp.train(epochs = mlp_setting['train_epochs'])
            mlp.test(repe_idx)


if __name__ == "__main__":
    mlp_repeat = 5
    fcn_repeat = 5
    print('running FCN classifiers')

    cnn_config = read_json('./cnn_config.json')
    # cnn_main(repeat, 'cnn', cnn_config['cnn'])

    # # Testing model using only MRI scans
    # fcn_main(fcn_repeat, 'fcn_mri_test', False, cnn_config['fcn_test'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_test_mlp', 'mri_test_', cnn_config['mlp'])

    # years = ['1']
    # years = ['2']
    # years = ['3']
    mlp_names = ['fcn_mri_amyloid_fdg_mlp']
    scan_names = ['mri', 'amyloid', 'fdg']
    modes = mlp_names

    print('-'*100)
    mlp_main(fcn_repeat, mlp_repeat, mlp_names[0], scan_names, cnn_config['mlp'])
    print('-'*100)

    # # Model using both MRI & amyloid PET scans
    # fcn_main(fcn_repeat, 'fcn_mri_amyloid', False, cnn_config['fcn_dual'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_amyloid_mlp', 'mri_amyloid_', cnn_config['mlp'])
    #
    # # Model using both MRI & fdg PET scans
    # fcn_main(fcn_repeat, 'fcn_mri_fdg', False, cnn_config['fcn_dual'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_fdg_mlp', 'mri_fdg_', cnn_config['mlp'])
    #
    # # Model using both anyloid & fdg PET scans
    # fcn_main(fcn_repeat, 'fcn_amyloid_fdg', False, cnn_config['fcn_dual'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_amyloid_fdg_mlp', 'amyloid_fdg_', cnn_config['mlp'])
    #
    # # Model using both MRI & amyloid & fdg scans
    # fcn_main(fcn_repeat, 'fcn_mri_amyloid_fdg', False, cnn_config['fcn_tri'])
    # mlp_main(fcn_repeat, mlp_repeat, 'fcn_mri_amyloid_fdg_mlp', 'mri_amyloid_fdg_', cnn_config['mlp'])


    # mlp_names = ['fcn_mri_mlp', 'fcn_amyloid_mlp']
    # mlp_names = ['fcn_mri_mlp']
    # mlp_names = ['fcn_mri_test_mlp']
    # mlp_names = ['fcn_mri_mlp', 'fcn_amyloid_mlp', 'fcn_fdg_mlp', 'fcn_mri_amyloid_mlp', 'fcn_mri_fdg_mlp', 'fcn_amyloid_fdg_mlp', 'fcn_mri_amyloid_fdg_mlp']
    report_table(mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # report_table(txt_file='report', mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # roc_plot_perfrom_table(mode=modes, fcn_repeat=fcn_repeat, mlp_repeat=mlp_repeat)
    # cross_set(mode=modes, fcn_repeat=mlp_repeat, mlp_repeat=fcn_repeat)
