

def generate_shap_values(repeat, model_name, setting, Wrapper):
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
        out = cnn.model(load=True, all=True)