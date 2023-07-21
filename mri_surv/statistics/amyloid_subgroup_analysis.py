import pandas as pd
import numpy as np
import seaborn as sns
import os.path as path
import os
import sys
import tabulate

from simple_mlps.model_ibs.ibs import retrieve_brier_scores, make_struc_array
from sksurv.metrics import concordance_index_censored

DATA_DIR = "/data2/MRI_PET_DATA/predicts/"

def filter_and_dump_amyloid_rids() -> pd.DataFrame:
    df = pd.read_csv("metadata/data_raw/ADNI/ADNIMERGE.csv")
    df = df[["RID", "VISCODE", "AV45"]].dropna()
    df = df.query("AV45 >= 1.11").copy()
    df_met = pd.read_csv("metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv")
    met_index = df_met[["RID", "VISCODE2"]].set_index(["RID", "VISCODE2"]).index
    df = df.assign(VISCODE2=df["VISCODE"]).drop(columns = "VISCODE")
    df.set_index(["RID", "VISCODE2"], inplace=True)
    df = df.reindex(met_index).dropna()
    df.to_csv("./metadata/data_processed/amyloid_positive_rids.csv")
    return stringify_rids(df)

def load_amyloid_rids() -> pd.DataFrame:
    df = pd.read_csv("./metadata/data_processed/amyloid_positive_rids.csv")
    return stringify_rids(df)

def stringify_rids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(RID = list(map(lambda x: str(x).zfill(4), df["RID"])))
    return df

def retrieve_adni_training_test_split() -> pd.DataFrame:
    adni_subjects = pd.read_csv(path.join(path.abspath(
        os.getcwd()), "metadata/data_processed/weibull_model_adni.csv"))
    adni_subjects = adni_subjects.loc[:, [
        "RID", "hit", "time_obs", "Fold0", "Fold1", "Fold2", "Fold3", "Fold4"]]
    adni_subjects['RID'] = adni_subjects['RID'].astype(str)
    adni_subjects['RID'] = adni_subjects['RID'].str.zfill(4)
    return adni_subjects

def read_csv(fname: str) -> pd.DataFrame:
    df = pd.read_csv(fname)
    return stringify_rids(df)

def filter_by_amyloid(truth_adni: pd.DataFrame, adni_subjects: pd.DataFrame) -> pd.DataFrame:
    rids = adni_subjects["RID"]
    truth_adni = truth_adni.set_index("RID")
    truth_adni = truth_adni.filter(items = rids, axis=0).copy()
    assert all([(x in adni_subjects["RID"].to_numpy()) for x in truth_adni.index])
    return truth_adni

def retrieve_brier_and_ci_scores():
    adni_subjects = retrieve_adni_training_test_split()
    amyloid_ids = load_amyloid_rids()

    models = ['mlp', 'weibull', 'cph', 'cnn']

    bins = [0, 24, 48, 108]

    ci_adni_df = pd.DataFrame()
    bs_adni_df = pd.DataFrame()

    for mod in models:
        stats = pd.DataFrame(
            columns=["CI_adni", "BS_adni"])
        print(mod)

        for idx in range(0, 5):  # for each fold

            preds_adni = read_csv(DATA_DIR + mod + "_exp" +
                                    str(idx) + "_ADNI_test.csv")

            truth_adni = read_csv(DATA_DIR + "truth_exp" +
                                    str(idx) + "_ADNI_test.csv")
            
            truth_adni = filter_by_amyloid(truth_adni, amyloid_ids)

            preds_adni = filter_by_amyloid(preds_adni, amyloid_ids)

            print(len(preds_adni))

            preds_only_adni = preds_adni.filter(regex="pred\.*").copy()

            foldnum = "Fold{}".format(idx)

            adni_fold_train = adni_subjects[["RID", foldnum, "hit", "time_obs"]]

            adni_fold_train = adni_fold_train[adni_fold_train[foldnum] == 1]

            train_struc = make_struc_array(
                adni_fold_train.hit, adni_fold_train.time_obs)
            
            test_struc_adni = make_struc_array(truth_adni.hit, truth_adni.observed)

            colnames = preds_adni.columns
            colnames = [x for x in colnames if 'pred.prob' in x]

            preds_only_adni_bs = preds_only_adni[[
                'pred.prob.{}'.format(time) for time in bins]]

            ########################
            # Actual stats here
            ########################

            bs_adni = 0
            ci_adni = 0

            preds_only_adni_bs = preds_only_adni_bs.mask(preds_only_adni_bs < 0, 0)

            curr_bins = [6, 24, 48, 84]

            bs_adni = retrieve_brier_scores(
                curr_bins, preds_only_adni_bs, train_struc, test_struc_adni)[0]

            ci_adni = np.mean(
                np.array(
                    [
                        concordance_index_censored(
                            truth_adni.hit.astype(bool),
                            truth_adni.observed,
                            1-preds_only_adni['pred.prob.{}'.format(time)]
                        )[0] for time in bins[1:]  # N.B. we are averaging CI's across different time bins!
                    ]
                )
            )


            stats = pd.concat([stats, pd.Series({
                "CI_adni": ci_adni,
                "BS_adni": bs_adni,
            }).to_frame().T], ignore_index=True)

        ci_adni_df[mod] = stats['CI_adni']
        bs_adni_df[mod] = stats['BS_adni']
    return ci_adni_df, bs_adni_df


def stats_for_amyloid():
    ci, bs = retrieve_brier_and_ci_scores()
    "#s: 51, 47, 52, 56, 48 "

    tbl = pd.DataFrame(columns=["Integrated Brier Score", "Concordance Index"])
    pm = u'\u00b1'
    for col in bs.columns:
        tbl.loc[col, "Integrated Brier Score"] = f"{bs[col].mean():0.3f} {pm} {bs[col].std():0.3f}"
        tbl.loc[col, "Concordance Index"] = f"{ci[col].mean():0.3f} {pm} {ci[col].std():0.3f}"
    print(tbl)

if __name__ == "__main__":
    stats_for_amyloid()