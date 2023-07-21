from typing import Tuple

import numpy as np
import pandas as pd
from scipy import interpolate


def inter(preds_raw: np.ndarray, time: np.ndarray):
    bins = [0, 24, 48, 108]
    interp = interpolate.PchipInterpolator(bins, preds_raw, axis=1)
    return interp(time)


def pred_reversions(ds) -> Tuple[np.ndarray, np.ndarray]:
    """
    pred_reversions

    Computes the predicted survival probability at time T for different
    patients using our CNN model

    Parameters
    ----------
    ds : Dataset
        NACC and ADNI are valid here

    Returns
    -------
    Tuple[ndarray, ndarray ]
        return times and inputs
    """
    df = load_preds_and_reverters(ds)
    df = (
        df.dropna()
    )  # drop cases where we revert but the data is not included in the ds
    input_mat = df[
        ["24", "48", "108"]
    ].to_numpy()  # take only predicted values at these bins
    input_mat = np.concatenate([np.ones((input_mat.shape[0], 1)), input_mat], axis=1)
    times = df["T"].to_numpy()
    return times, input_mat  # here, times is the time of reversion/last visit c dx


def load_reverters():
    return pd.read_csv("./metadata/data_processed/reverted_rids.csv")


def load_preds(ds="ADNI"):
    dir_ = f"./metadata/data_processed/predicts_all/predicts_cnn/predicts/cnn_predicted_survival_{ds.lower()}.txt"
    df = pd.read_csv(dir_, dtype={"rid": str})
    df = df.rename(columns={"rid": "RID"})
    df = df.drop(columns=["Unnamed: 0", "observe", "hit"])
    df = df.groupby("RID").agg(np.mean)
    return df


def load_preds_and_reverters(ds="ADNI") -> pd.DataFrame:
    df = load_preds(ds)  # load predicted survival from CNN
    reverters = (
        load_reverters()
    )  # load the survival probabilities computed in statistics/other_dementia/reversion_stats.py
    reverters = reverters.query("DS == @ds").copy().set_index("RID")  # get dataset
    df = reverters.merge(
        df, how="left", on="RID"
    )  # merge reverters with predictions for each reverter
    return df


def main():
    t, df = pred_reversions("ADNI")
    g = inter(df, t)  # now, interpolate at given time point
    print(np.mean(t))  # average time at final dx
    print(np.mean(g[np.eye(43) == 1]))  # now, average survival prob
    print(np.mean(g[np.eye(43) == 1] > 0.5))  # now, proportion with surv prob > 0.5

    t, df = pred_reversions("NACC")
    print(np.mean(t))
    g = inter(df, t)
    print(np.mean(g[np.eye(len(t)) == 1]))
    print(np.mean(g[np.eye(len(t)) == 1] > 0.5))
