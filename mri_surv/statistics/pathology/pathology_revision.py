import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statistics.clustered_mlp_output_wrappers import \
    load_metadata_pathology_mlp_pivot

# Desired columns
COLS = ('ClusterIdx', 'PROGRESSES', 'BRAAK', 'CERAD', 'NIA_ADNC')

PATH = ['BRAAK', 'CERAD', 'NIA_ADNC']

# for patients who are have progressed, plot different stages of BRAAK, CERAD, NIA_ADNC
# for patients who are have not progressed, plot different stages of BRAAK, CERAD, NIA_ADNC

def _df_for_path(col_name) -> pd.DataFrame:
    df = load_metadata_pathology_mlp_pivot()
    df = df[['PROGRESSES', col_name]]
    df = df.dropna(axis=0).reset_index(drop=True)
    df['PROGRESSES'] = df['PROGRESSES'].replace({0: 'No', 1: 'Yes'}).astype('category')
    df.rename(columns={'PROGRESSES': 'Clinical progression?'}, inplace=True)
    new_df = pd.crosstab(df['Clinical progression?'],df[col_name])
    if col_name is 'NIA_ADNC':
        order = np.flip(['High', 'Intermediate', 'Low', 'Not AD'])
        new_df = new_df[order]
    return new_df

def main() -> None:
    for p in PATH:
        df = _df_for_path(p)
        df.to_pickle(f'./metadata/data_processed/heatmap_data_{p.lower()}.pkl')
