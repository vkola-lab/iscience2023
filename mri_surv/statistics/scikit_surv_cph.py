from lifelines.datasets import load_rossi
from sksurv.linear_model import CoxnetSurvivalAnalysis
import sys
sys.path.insert(0, "/home/mfromano/Research/mri-surv-dev/mri_surv")
from simple_mlps.datas import ParcellationDataVentriclesNacc, ParcellationDataVentricles
from simple_mlps.model_ibs.ibs import *
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

    df_train = zs(adni_train.data[adni_train.index_list].numpy())

    df_train_struc = make_struc_array(
        hits= adni_train.hit[adni_train.index_list],
        obss = adni_train.time_obs[adni_train.index_list]
        )

    df_valid = zs(adni_valid.data[adni_valid.index_list].numpy())
    
    df_valid_struc = make_struc_array(
        hits = adni_valid.hit[adni_valid.index_list],
        obss = adni_valid.time_obs[adni_valid.index_list]
    )

    df_test = zs(adni_test.data[adni_test.index_list].numpy())
    
    df_test_struc = make_struc_array(
        hits = adni_test.hit[adni_test.index_list],
        obss = adni_test.time_obs[adni_test.index_list]
    )

    df_nacc_struc = make_struc_array(
        hits = nacc.hit[nacc.index_list],
        obss = nacc.time_obs[nacc.index_list]
    )
    df_nacc = zs(nacc.data[nacc.index_list].numpy())

    opt_set = [0,0,0]
    for pe in tqdm(range(100)):
        for ra in range(10):
            cph = CoxnetSurvivalAnalysis(alpha_min_ratio=(pe+1)/100, l1_ratio=(ra+1)/10)
            cph.fit(df_train, df_train_struc)
            ci = cph.score(df_valid, df_valid_struc)
            if ci >= opt_set[0]:
                opt_set = [ci, (pe+1)/10, (ra+1)/10]
    cph = CoxnetSurvivalAnalysis(alpha_min_ratio=opt_set[1], l1_ratio=opt_set[2], fit_baseline_model=True)
    cph.fit(df_train, df_train_struc)

    CI_train.append(cph.score(df_train, df_train_struc))
    CI_valid.append(cph.score(df_valid, df_valid_struc))
    CI_test.append(cph.score(df_test, df_test_struc))
    CI_nacc.append(cph.score(df_nacc, df_nacc_struc))

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