import pandas as pd
import numpy as np
from statistics.survival_plot_xz import load_metadata_mlp_pivot_raw, load_metadata_mlp_pivot
import unittest


def _simplify_dataframe(*args):
    output_dfs = []
    for df in args:
        df = df.copy()
        df['Cluster Idx'] = df['Cluster Idx'].astype(int)
        df = df.groupby(['Dataset', 'RID']).agg(np.mean).reset_index()
        output_dfs.append(df)
    return output_dfs

def _compare_dataframes(df1, df2):
    df1 = df1.copy().sort_values(by=['Dataset','RID'])
    df2 = df2.copy().sort_values(by=['Dataset','RID'])
    column_order = ['0', '108', '24', '48', 'TIMES', 'PROGRESSES', 'Cluster Idx']
    df1 = df1[column_order]
    df2 = df2[column_order]
    df1['Cluster Idx'] = df1['Cluster Idx'].astype(int)
    df2['Cluster Idx'] = df2['Cluster Idx'].astype(int)
    return np.allclose(df1.to_numpy(),df2.to_numpy(), equal_nan=True)

class TestSurvivalPlot(unittest.TestCase):
    def test_unused_nii_files(self):
        mlp, cnn, vit =load_metadata_mlp_pivot()
        mlp_raw, cnn_raw, vit_raw = load_metadata_mlp_pivot()
        mlp_raw, cnn_raw, vit_raw = _simplify_dataframe(mlp_raw, cnn_raw, vit_raw)
        self.assertTrue(_compare_dataframes(mlp, mlp_raw))
        self.assertTrue(_compare_dataframes(cnn, cnn_raw))
        self.assertTrue(_compare_dataframes(vit, vit_raw))

if __name__ == '__main__':
    unittest.main()