from torch.utils.data import Dataset
from weibull_mlp.utilities import read_csv_cox, retrieve_kfold_partition
from simple_mlps.datas import ParcellationDataVentricles, ParcellationDataVentriclesNacc
import random
import pandas as pd
import numpy as np



def prep_data(dataset='ADNI', csvname='./metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv'):
    parcellation_file = './metadata/data_processed/parcellation_volumes_raw.csv'
    parcellation_df = pd.read_csv(parcellation_file)
    parcellation_df = parcellation_df.query(
        f'Dataset == \'{dataset}\'').drop(columns='Dataset').copy()
    rids, time_obs, hit, age, mmse = read_csv_cox(csvname)

    if dataset == 'ADNI':
        parcellation_df['RID'] = parcellation_df['RID'].apply(
            lambda x: str(int(x)).zfill(4)
        )

    parcellation_df.set_index('RID', inplace=True)

    parcellation_df = parcellation_df.loc[rids, :]
    parcellation_df["age"] = age
    parcellation_df["mmse"] = mmse
    parcellation_df["time_obs"] = time_obs
    parcellation_df["hit"] = hit

    idxs = list(range(len(rids)))

    for fold in range(5):
        train_vec = np.zeros_like(age, dtype="int")
        train_index = retrieve_kfold_partition(idxs, "train", 5, exp_idx=fold)
        test_index = retrieve_kfold_partition(idxs, "test", 5, exp_idx=fold)
        valid_index = retrieve_kfold_partition(idxs, "valid", 5, exp_idx=fold)
        assert len(np.intersect1d(train_index, test_index)) == 0
        assert len(np.intersect1d(valid_index, train_index)) == 0
        assert len(np.intersect1d(valid_index, test_index)) == 0
        train_vec[train_index] = 1
        train_vec[valid_index] = 2
        train_vec[test_index] = 3
        parcellation_df[f'Fold{fold}'] = train_vec

    return parcellation_df


def generate_csvs():
    df = prep_data(dataset='ADNI')
    df.to_csv('metadata/data_processed/weibull_model_adni.csv')

    df = prep_data(dataset='NACC',
                   csvname='./metadata/data_processed/merged_dataframe_cox_test_pruned_final.csv')
    df.to_csv('metadata/data_processed/weibull_model_nacc.csv')


class ParcellationData(Dataset):
    """
    Dataset specifically for parcellation data
    """

    MAX_YEARS = 10

    def __init__(
        self,
        exp_idx: int,
        add_age: bool=False,
        add_mmse: bool=False,
        stage: str = "train",
        dataset: str = "ADNI",
    ) -> None:
        random.seed(1000)

        self.exp_idx = exp_idx
        self._df = pd.read_csv(
            f'metadata/data_processed/weibull_model_{dataset.lower()}.csv')

        if dataset == 'ADNI':
            self._df['RID'] = self._df['RID'].apply(
                lambda x: str(int(x)).zfill(4)
            )
        self._df.set_index('RID', drop=True, inplace=True)

        if not add_age:
            self._df.drop(columns="age", inplace=True)
        if not add_mmse:
            self._df.drop(columns="mmse", inplace=True)

        col = f"Fold{exp_idx}"

        stage_map = {"train": 1, "valid": 2, "test": 3}

        if stage is not "all":
            self._df = self._df.loc[self._df[col] == stage_map[stage], :]

        self.time_obs = self._df["time_obs"].to_numpy()
        self.hit = self._df["hit"].to_numpy()

        fold_cols = [f"Fold{x}" for x in range(5)]

        self._df.drop(columns=['time_obs', 'hit',
                      'TIV']+fold_cols, inplace=True)

        self.labels = self._df.columns
        self.data = self._df.to_numpy()
        self.rid = np.array(self._df.index)
        self.hit_matrix = self._generate_hit_matrix()

    def __len__(self) -> int:
        return len(self.rid)

    def __getitem__(self, idx) -> tuple:
        x = self.data[idx]
        obs = self.time_obs[idx]
        hit = self.hit[idx]
        rid = self.rid[idx]
        hit_matrix_row = self.hit_matrix[idx]
        return x, obs, hit, rid, hit_matrix_row

    def get_features(self) -> np.ndarray:
        return self.labels

    def get_data(self) -> np.ndarray:
        return self.data

    def _generate_hit_matrix(self) -> np.ndarray:
        """Generates a matrix of shape (n_samples, MAX_YEARS) for the loss function

        Raises:
            ValueError: hits must be either 0 or 1

        Returns:
            np.ndarray: a hit matrix of shape (n_samples, MAX_YEARS) for the loss function with each row representing the hits for a given subject
                0 if not converted, 1 if converted, -1 if censored at particular year
        """
        final_matrix = np.ndarray(shape=(0, self.MAX_YEARS))

        for idx, (h, t) in enumerate(zip(self.hit, self.time_obs)):
            yr = t // 12
            baseline = []

            if h == 1:  # converter
                baseline = [0 if i < yr else 1 for i in range(self.MAX_YEARS)]
            elif h == 0 and yr < self.MAX_YEARS:  # nonconverter and censored
                baseline = [0 if i < yr else -1 for i in range(self.MAX_YEARS)]
            elif h == 0 and yr >= self.MAX_YEARS:  # nonconverter and not censored
                baseline = [0] * self.MAX_YEARS
            else:
                raise ValueError
            if t <= 12:  # censored before 1yr mark, b
                baseline[0] = 0  # we know that baseline they are MCI

            final_matrix = np.vstack((final_matrix, baseline))

        return final_matrix
