import json
import csv

import numpy as np
from typing_extensions import Literal


def read_json(config_file: str) -> dict:
    """
    Reads json file given file name

    Args:
        config_file (str): file name

    Returns:
        dict: dictionary corresponding to json file
    """
    with open(config_file, "r") as config_buffer:
        config = json.loads(config_buffer.read())
    return config


def read_csv_cox(filename, skip_ids: list = None) -> tuple:
    """
    Takes a demographics file (
            './metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv'
        ) or (
            './metadata/data_processed/merged_dataframe_cox_test_pruned_final.csv'
        )
    and returns columns corresponding to RID, TIMES, PROGRESSES, AGE, MMSE (if available)

    Args:
        filename (_type_): string filename as above
        skip_ids (list, optional): ids to skip when extracting data. Defaults to None.

    Returns:
        tuple: RID, times, hits, age, mmse
    """
    with open(filename, "r") as fi:
        reader = csv.DictReader(fi)
        file_ids, time_obs, hit, age, mmse = [], [], [], [], []
        for r in reader:
            if skip_ids is not None:
                if r["RID"] in skip_ids:
                    continue
            file_ids += [str(r["RID"])]
            time_obs += [float(r["TIMES"])]
            hit += [int(float(r["PROGRESSES"]))]
            age += [float(r["AGE"])]
            if "MMSCORE_mmse" in r.keys():
                mmse += [float(r["MMSCORE_mmse"])]
            else:
                mmse += [np.nan if r["MMSE"] == "" else float(r["MMSE"])]
    return file_ids, np.asarray(time_obs), np.asarray(hit), age, mmse


def retrieve_kfold_partition(
    idxs: list,
    stage: Literal["all", "train", "test", "valid"],
    folds: int = 5,
    exp_idx: int = 0,
    shuffle: bool = True,
    random_state: float = 120) -> np.ndarray:
    """
    retrieves partition for a given stage and # of folds, in addition to exp_idx

    Args:
        idxs (list): list of idxs to partition
        stage (Literal): Literal; either all, train, trest, or valid
        folds (int, optional): number of folds. Defaults to 5.
        exp_idx (int, optional): exp_idx. Defaults to 1. Max should be folds - 1
        shuffle (bool, optional): whether or not to shuffle idxs. Defaults to True.
        random_state (float, optional). Defaults to 120.

    Raises:
        ValueError: errors out if ndims idxs > 1 or if stage is not as specified above

    Returns:
        np.ndarray: idx list corresponding to the appropriate fold
    """
    idxs = np.asarray(idxs).copy()  # do NOT perform this operation in place
    assert(exp_idx < folds)
    if shuffle:  # permute idxs
        np.random.seed(random_state)
        idxs = np.random.permutation(idxs)
    if "all" in stage:  # retrieve all items
        return idxs
    if len(idxs.shape) > 1:
        raise ValueError
    fold_len = len(idxs) // folds  # establish length of each partition given # folds
    folds_stitched = []  # going to physically stitch folds together
    for fold in range(folds):  # go in order to stitch folds together
        folds_stitched.append(idxs[fold * fold_len : (fold + 1) * fold_len])
    test_idx = exp_idx  # id for test data (0-# folds - 1)
    valid_idx = (exp_idx + 1) % folds  # validation data is whichever of 0- # folds - 1 follows
    train_idx = np.setdiff1d(np.arange(0, folds, 1), [test_idx, valid_idx])  # set diff
    if "test" in stage:
        return folds_stitched[test_idx]
    if "valid" in stage:
        return folds_stitched[valid_idx]
    if "train" in stage:
        return np.concatenate([folds_stitched[x] for x in train_idx], axis=0)
    raise ValueError
