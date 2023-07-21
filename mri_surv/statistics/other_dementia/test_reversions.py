
from scipy import interpolate
import numpy as np
import pandas as pd

def inter(preds_raw, time):
    bins = [0, 24, 48, 108]
    interp = interpolate.PchipInterpolator(bins, preds_raw, axis=1)
    return interp(time)

def pred_reversions(ds):
    df = load_preds_and_reverters(ds)
    df = df.dropna()
    input_mat = df[['24', '48', '108']].to_numpy()
    input_mat = np.concatenate([np.ones((input_mat.shape[0],1)),input_mat], axis=1)
    times = df['T'].to_numpy()
    return times, input_mat

def load_reverters():
    return pd.read_csv('./metadata/data_processed/reverted_rids.csv')

def load_preds(ds='ADNI'):
    dir_ = f'./metadata/data_processed/predicts_cnn_{ds.lower()}.csv'
    df = pd.read_csv(dir_, dtype={'rid': str })
    df = df.rename(columns={'rid': 'RID'})
    df = df.drop(columns=['Unnamed: 0', 'observe','hit'])
    df = df.groupby('RID').agg(np.mean)
    return df

def load_preds_and_reverters(ds='ADNI'):
    df = load_preds(ds)
    reverters = load_reverters()
    reverters = reverters.query('DS == @ds').copy().set_index('RID')
    df = reverters.merge(df, how='left', on='RID')
    return df

def main():
    t, df = pred_reversions('ADNI')
    g = inter(df, t)
    print(np.mean(t))
    print(np.mean(g[np.eye(43) == 1]))
    print(np.mean(g[np.eye(43) == 1] >0.5))

    t, df = pred_reversions('NACC')
    print(np.mean(t))
    g = inter(df, t)
    print(np.mean(g[np.eye(len(t)) == 1]))
    print(np.mean(g[np.eye(len(t)) == 1] > 0.5))