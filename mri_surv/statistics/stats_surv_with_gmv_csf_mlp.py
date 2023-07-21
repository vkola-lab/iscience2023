import numpy as np
import pandas as pd
import os.path as path
import os
from simple_mlps.model_ibs.ibs import retrieve_brier_scores, make_struc_array
from sksurv.metrics import concordance_index_censored
import scipy.stats as statspkg
import statsmodels.api as sm
import itertools

DATA_DIR = "/data2/MRI_PET_DATA/predicts/"

adni_subjects = pd.read_csv(path.join(path.abspath(
    os.getcwd()), "metadata/data_processed/weibull_model_adni.csv"))
adni_subjects = adni_subjects.loc[:, [
    "RID", "hit", "time_obs", "Fold0", "Fold1", "Fold2", "Fold3", "Fold4"]]
adni_subjects['RID'] = adni_subjects['RID'].astype(str)
adni_subjects['RID'] = adni_subjects['RID'].str.zfill(4)

models = ['mlp', 'weibull', 'cph', 'cnn', 'mlp_exp_gmv_csf']
# models = ['cnn']
bins = [0, 24, 48, 108]

ci_adni_df = pd.DataFrame()
ci_nacc_df = pd.DataFrame()
bs_adni_df = pd.DataFrame()
bs_nacc_df = pd.DataFrame()

for i, mod in enumerate(models):
    stats: pd.DataFrame = pd.DataFrame(
        columns=["CI_adni", "BS_adni", "CI_nacc", "BS_nacc"])
    print(mod)

    for idx in range(0, 5):  # for each fold

        preds_adni = pd.read_csv(DATA_DIR + mod + "_exp" +
                                 str(idx) + "_ADNI_test.csv")

        preds_nacc = pd.read_csv(DATA_DIR + mod + "_exp" +
                                 str(idx) + "_NACC.csv")

        truth_adni = pd.read_csv(DATA_DIR + "truth_exp" +
                                 str(idx) + "_ADNI_test.csv")

        truth_nacc = pd.read_csv(DATA_DIR + "truth_exp" +
                                 str(idx) + "_NACC.csv")

        preds_adni['RID'] = preds_adni['RID'].astype(str)
        preds_adni['RID'] = preds_adni['RID'].str.zfill(4)

        preds_nacc['RID'] = preds_nacc['RID'].astype(str)
        preds_nacc['RID'] = preds_nacc['RID'].str.zfill(4)

        truth_adni['RID'] = truth_adni['RID'].astype(str)
        truth_adni['RID'] = truth_adni['RID'].str.zfill(4)

        truth_nacc['RID'] = truth_nacc['RID'].astype(str)
        truth_nacc['RID'] = truth_nacc['RID'].str.zfill(4)

        preds_only_adni = preds_adni.filter(regex="pred\.*")
        preds_only_nacc = preds_nacc.filter(regex="pred\.*")

        foldnum = "Fold{}".format(idx)

        adni_fold_train = adni_subjects[["RID", foldnum, "hit", "time_obs"]]
        adni_fold_train = adni_fold_train[adni_fold_train[foldnum] == 1]

        train_struc = make_struc_array(
            adni_fold_train.hit, adni_fold_train.time_obs)

        test_struc_adni = make_struc_array(truth_adni.hit, truth_adni.observed)
        test_struc_nacc = make_struc_array(truth_nacc.hit, truth_nacc.observed)

        colnames = preds_adni.columns
        colnames = [x for x in colnames if 'pred.prob' in x]

        preds_only_adni_bs = preds_only_adni
        preds_only_nacc_bs = preds_only_nacc
        preds_only_adni_bs = preds_only_adni[[
            'pred.prob.{}'.format(time) for time in bins]].copy()
        preds_only_nacc_bs = preds_only_nacc[[
            'pred.prob.{}'.format(time) for time in bins]].copy()

        ########################
        # Actual stats here
        ########################

        bs_adni = 0
        ci_adni = 0
        bs_nacc = 0
        ci_nacc = 0

        print(preds_only_adni_bs.shape)
        print(train_struc.shape)
        print(test_struc_adni.shape)

        bs_adni = retrieve_brier_scores(
            bins, preds_only_adni_bs, train_struc, test_struc_adni)[0]

        ci_adni = np.mean(
            np.array(
                [
                    concordance_index_censored(
                        truth_adni.hit.astype(bool),
                        truth_adni.observed,
                        1-preds_only_adni['pred.prob.{}'.format(time)]
                    )[0] for time in bins[1:]
                ]
            )
        )

        print(
            np.array(
                [
                    concordance_index_censored(
                        truth_adni.hit.astype(bool),
                        truth_adni.observed,
                        1-preds_only_adni['pred.prob.{}'.format(time)]
                    )[0] for time in bins[1:]
                ]
            )
        )

        # print("NACC")

        bs_nacc = retrieve_brier_scores(
            bins, preds_only_nacc_bs, train_struc, test_struc_nacc)[0]


        print(preds_only_nacc[['pred.prob.{}'.format(time) for time in bins[1:]]])
        ci_nacc = np.mean(
            np.array(
                [
                    concordance_index_censored(
                        truth_nacc.hit.astype(bool),
                        truth_nacc.observed,
                        1-preds_only_nacc['pred.prob.{}'.format(time)]
                    )[0] for time in bins[1:]
                ]
            )
        )

        print(np.array(
                [
                    concordance_index_censored(
                        truth_nacc.hit.astype(bool),
                        truth_nacc.observed,
                        1-preds_only_nacc['pred.prob.{}'.format(time)]
                    )[0] for time in bins[1:]
                ]
            ))
        stats = pd.concat([stats, pd.Series({
            "CI_adni": ci_adni,
            "BS_adni": bs_adni,
            "CI_nacc": ci_nacc,
            "BS_nacc": bs_nacc,
        }).to_frame().T], ignore_index=True)

    # model = pd.Series([mod for i in range(0, 5)])

    ci_adni_df[mod] = stats['CI_adni']
    bs_adni_df[mod] = stats['BS_adni']
    ci_nacc_df[mod] = stats['CI_nacc']
    bs_nacc_df[mod] = stats['BS_nacc']


print("CI ADNI")
print(np.mean(ci_adni_df, axis=0))
print(np.std(ci_adni_df, axis=0))
print()

print("BS ADNI")
print(np.mean(bs_adni_df, axis=0))
print(np.std(bs_adni_df, axis=0))
print()

print("CI NACC")
print(np.mean(ci_nacc_df, axis=0))
print(np.std(ci_nacc_df, axis=0))
print()

print("BS NACC")
print(np.mean(bs_nacc_df, axis=0))
print(np.std(bs_nacc_df, axis=0))


##################
# Stats time
#################


def do_stats(df: pd.DataFrame) -> None:
    results = pd.DataFrame(columns=df.columns, index=df.columns)
    for (label1, column1), (label2, column2) in itertools.combinations(df.items(), 2):
        ttestval = statspkg.ttest_rel(column1, column2)
        results.loc[label1, label2] = results.loc[label2,  # type: ignore
                                                  label1] = ttestval  # type: ignore
        pvals.append(ttestval[1])

        print((label1, label2), ttestval[0], ttestval[1])
    print()


pvals = []
do_stats(ci_adni_df)
print(sm.stats.multipletests(
    pvals, alpha=0.05, method="fdr_bh"))

pvals = []
do_stats(bs_adni_df)
print(sm.stats.multipletests(
    pvals, alpha=0.05, method="fdr_bh"))

pvals = []
do_stats(ci_nacc_df)
print(sm.stats.multipletests(
    pvals, alpha=0.05, method="fdr_bh"))

pvals = []
do_stats(bs_nacc_df)
print(sm.stats.multipletests(
    pvals, alpha=0.05, method="fdr_bh"))

print(statspkg.f_oneway(
    ci_adni_df['mlp'], ci_adni_df['cnn'], ci_adni_df['cph'], ci_adni_df['weibull'], ci_adni_df['mlp_exp_gmv_csf']))

print(statspkg.f_oneway(
    bs_adni_df['mlp'], bs_adni_df['cnn'], bs_adni_df['cph'], bs_adni_df['weibull'], ci_adni_df['mlp_exp_gmv_csf']))

print(statspkg.f_oneway(
    ci_nacc_df['mlp'], ci_nacc_df['cnn'], ci_nacc_df['cph'], ci_nacc_df['weibull'], ci_adni_df['mlp_exp_gmv_csf']))

print(statspkg.f_oneway(
    bs_nacc_df['mlp'], bs_nacc_df['cnn'], bs_nacc_df['cph'], bs_nacc_df['weibull'], ci_adni_df['mlp_exp_gmv_csf']))
