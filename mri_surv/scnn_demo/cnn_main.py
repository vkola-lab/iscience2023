# Same structure as pre-trained network for transfer learning | standard learning
# Created: 5/22/2021
# Status: OK
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cnn_main.py
# Consider merge into 1 main file

import sys
import torch
sys.path.insert(1, '../plot/')
from networks import CNN_Wrapper
from utils import read_json

torch.set_deterministic(True)

def cnn_main(repeat, model_name, setting, Wrapper):
    print('Evaluation metric: {}'.format(setting['metric']))
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
        # cnn.load('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', exp_idx))
        cnn.train(epochs = setting['train_epochs'])
        cnn.test(True)

def main():
    config = read_json('./cnn_config.json')

    CWrapper = CNN_Wrapper
    cnn_config = config['cnn']
    print('training CE-based network for time-to progression...')

    cnn_repeat = 1
    cnn_names = ['cnn_mri', 'cnn_amyloid', 'cnn_fdg']
    # scan_names = ['mri', 'amyloid', 'fdg']
    # modes = []

    print('-'*100)
    for c_name in cnn_names:
        # modes += [c_name]
        cnn_main(cnn_repeat, c_name, cnn_config, Wrapper=CWrapper)
        print('-'*100)
        break
    print('training completed')

if __name__ == "__main__":
    main()