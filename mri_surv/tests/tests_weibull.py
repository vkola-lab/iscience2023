import unittest

import pandas as pd
import numpy as np

from weibull_mlp import datas, make_imaging_sheet_full


class WeibullTest(unittest.TestCase):
    def test_datagen(self) -> None:
        df2 = pd.read_csv("./metadata/data_processed/weibull_model_adni.csv").set_index(
            "RID", drop=True
        )
        df = datas.prep_data(
            dataset="ADNI",
            csvname="./metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv",
        )
        self.assertTrue(np.allclose(df.to_numpy(), df2.to_numpy(), equal_nan=True))

        df2 = pd.read_csv("./metadata/data_processed/weibull_model_nacc.csv").set_index(
            "RID", drop=True
        )
        df = datas.prep_data(
            dataset="NACC",
            csvname="./metadata/data_processed/merged_dataframe_cox_test_pruned_final.csv",
        )
        diff_ = df.to_numpy() - df2.to_numpy()
        locs = np.where(np.isnan(diff_))
        self.assertTrue(all(locs[1] == 144))
        self.assertTrue(np.allclose(df.to_numpy(), df2.to_numpy(), equal_nan=True))


def _process():
    raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
