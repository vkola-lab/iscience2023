from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
import sys
sys.path.insert(0, "/home/mfromano/Research/mri-surv-dev/mri_surv")
from simple_mlps.datas import ParcellationDataVentriclesNacc, ParcellationDataVentricles
import pandas as pd
import numpy as np
import functools
# ignore warnings
# python -W ignore cph.py
import warnings
from tqdm import tqdm

def z_score(x: np.ndarray, mn: np.ndarray, sd: np.ndarray) -> np.array:
    assert (x.shape[1] == mn.shape[1]) and (x.shape[1] == sd.shape[1])
    return (x - mn) / sd

CI_train = []
CI_valid = []
CI_test = []
CI_nacc = []
for e_id in range(5):
    adni_train = ParcellationDataVentricles(seed=e_id, stage='train')
    adni_valid = ParcellationDataVentricles(seed=e_id, stage='valid')
    adni_test = ParcellationDataVentricles(seed=e_id, stage='test')

    nacc = ParcellationDataVentriclesNacc(seed=e_id, stage='all')
    df_train = pd.DataFrame(data=adni_train.data[adni_train.index_list].numpy())

    mn = df_train.to_numpy().mean(axis=0, keepdims=True)
    sd = df_train.to_numpy().std(axis=0, keepdims=True)
    zs = functools.partial(z_score, mn = mn, sd = sd)

    df_train = pd.DataFrame(data=zs(adni_train.data[adni_train.index_list].numpy()))

    df_train['observe'] = adni_train.time_obs[adni_train.index_list]
    df_train['hit'] = adni_train.hit[adni_train.index_list]

    df_valid = pd.DataFrame(data=zs(adni_valid.data[adni_valid.index_list].numpy()))
    df_valid['observe'] = adni_valid.time_obs[adni_valid.index_list]
    df_valid['hit'] = adni_valid.hit[adni_valid.index_list]

    df_test = pd.DataFrame(data=zs(adni_test.data[adni_test.index_list].numpy()))
    df_test['observe'] = adni_test.time_obs[adni_test.index_list]
    df_test['hit'] = adni_test.hit[adni_test.index_list]

    df_nacc = pd.DataFrame(data=zs(nacc.data[nacc.index_list]))
    df_nacc['observe'] = nacc.time_obs[nacc.index_list]
    df_nacc['hit'] = nacc.hit[nacc.index_list]

    opt_set = [0,0,0]
    for pe in tqdm(range(100)):
        for ra in range(10):
            cph = CoxPHFitter(penalizer=pe/10, l1_ratio=ra/10)
            cph.fit(df_train, duration_col="observe", event_col="hit")
            ci = cph.score(df_valid, 'concordance_index')
            if ci >= opt_set[0]:
                opt_set = [ci, pe/10, ra/10]
    cph = CoxPHFitter(penalizer=opt_set[1], l1_ratio=opt_set[2])
    cph.fit(df_train, duration_col="observe", event_col="hit")
    print(opt_set)
    # cph.print_summary()
    CI_train.append(cph.score(df_train, 'concordance_index'))
    CI_valid.append(cph.score(df_valid, 'concordance_index'))
    CI_test.append(cph.score(df_test, 'concordance_index'))
    CI_nacc.append(cph.score(df_nacc, 'concordance_index'))
    print('CI train:', cph.score(df_train, 'concordance_index'))
    print('CI valid:', cph.score(df_valid, 'concordance_index'))
    print('CI test: ', cph.score(df_test, 'concordance_index'))
    print('CI nacc: ', cph.score(df_nacc, 'concordance_index'))

    test_pred = cph.predict_survival_function(df_test)
    nacc_pred = cph.predict_survival_function(df_nacc)
    for n, preds, d in zip(['ADNI_test', 'NACC'], [test_pred, nacc_pred], [adni_test, nacc]):
        df_result = pd.DataFrame(preds.T)
        df_result['rid'] = d.rid[d.index_list]
        df_result['observe'] = d.time_obs[d.index_list]
        df_result['hit'] = d.hit[d.index_list]
        dir_ = './cph_predicts/cph_exp{}_{}.csv'.format(e_id, n)
        df_result.to_csv(dir_)

print(CI_train, CI_test, CI_valid, CI_nacc)
print('CI train: %.3f+-%.3f' % (np.mean(CI_train), np.std(CI_train)))
print('CI valid: %.3f+-%.3f' % (np.mean(CI_valid), np.std(CI_valid)))
print('CI test: %.3f+-%.3f' % (np.mean(CI_test), np.std(CI_test)))
print('CI external: %.3f+-%.3f' % (np.mean(CI_nacc), np.std(CI_nacc)))