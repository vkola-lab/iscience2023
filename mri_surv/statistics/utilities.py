import json
import pandas as pd
import numpy as np
import datetime
from pingouin import partial_corr
from typing import Union, Dict, Iterable, Tuple, List
from scipy import interpolate
from statsmodels.stats.multitest import multipletests

__all__ = ['load_json',
'upper_and_lower_quartiles',
'deabbreviate_parcellation', 
'bootstrap_ci', 
'CONFIG', 
'to_datetime',
'time_and_progress',
'jaccard_sim']

def load_json(fname):
    with open(fname, 'r') as fi:
        return json.loads(fi.read())

JSON_FI = './statistics/config/statistics_config.json'

CONFIG = load_json(JSON_FI)

def time_and_progress(metadata: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if 'adni' not in metadata.keys() or 'nacc' not in metadata.keys():
        raise ValueError('keys ADNI and NACC must be in metadata!')
    adni_cols = metadata['adni'][['RID','TIMES','PROGRESSES']].copy()
    adni_cols['Dataset'] = 'ADNI'
    nacc_cols = metadata['nacc'][['RID', 'TIMES', 'PROGRESSES']].copy()
    nacc_cols['Dataset'] = 'NACC'
    return pd.concat([adni_cols, nacc_cols], axis=0, ignore_index=True)

def get_interpolator(bins, bin_values: Union[tuple, list, np.ndarray]):
    return interpolate.interp1d(bins,
            bin_values.astype(float),
            kind='quadratic',
            bounds_error=False,
            fill_value=(np.nan, 0)
    )

def pchip_interpolator(bins: list, bin_values: Union[Tuple, List, np.ndarray]):
    return interpolate.PchipInterpolator(
            np.asarray(bins),
            np.asarray(bin_values.astype(float)),
            extrapolate=True
    )

def sliding_auc(risk_at_times, times, hits, censoring_pattern):
    raise NotImplementedError

def benjamini_hochberg_correct(p_val: list):
    p_val = multipletests(p_val, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]
    return p_val

def to_datetime(df: pd.DataFrame, column: str):
    """Converts specified column to a datetime date from string

    Args:
        df (pd.DataFrame): df to convert to datetime
        column (str): column of df
    """
    df.loc[:,column] = df.loc[:,column].apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
    )

def upper_and_lower_quartiles(df, colname):
    low_quartile, high_quartile = df[colname].quantile([0.25, 0.75])
    low_quartile_df = df.loc[df[colname] <= low_quartile,:].copy(
    ).reset_index(drop=True)
    low_quartile_df['Quartile'] = 'Bottom'
    high_quartile_df = df.loc[df[colname] >= high_quartile, :].copy(
    ).reset_index(drop=True)
    high_quartile_df['Quartile'] = 'Top'
    return pd.concat([low_quartile_df, high_quartile_df], ignore_index=True,
                     axis=0)

def deabbreviate_parcellation(df, axis=0):
    json = load_json(JSON_FI)
    df_dict = pd.read_csv(json['neuromorphometrics'], delimiter=';',
                          usecols=['ROIabbr','ROIname'])
    df_dict = df_dict.loc[[x[0] == 'l' for x in df_dict['ROIabbr']],:]
    df_dict['ROIabbr'] = df_dict['ROIabbr'].apply(
            lambda x: x[1:]
    )
    df_dict['ROIname'] = df_dict['ROIname'].apply(
            lambda x: x.replace('Left ', '')
    )
    df_dict = df_dict.set_index('ROIabbr').to_dict()['ROIname']
    if axis==0:
        df.rename(index=df_dict, inplace=True)
    else:
        df.rename(columns=df_dict, inplace=True)
    return df

def bootstrap_ci(
        nd_array: np.ndarray,
        quantile: Union[float, np.array, Iterable],
        nreps:np.long=10000):
    np.random.seed(1000)
    stat = []
    for _ in range(nreps):
        sample_idx = np.random.choice(np.arange(0,nd_array.shape[0],1), nd_array.shape[0], replace=True)
        sample_ndarray = nd_array[sample_idx,:]
        stat.append(np.mean(sample_ndarray, axis=0))
    return np.quantile(stat, quantile, axis=0, interpolation='linear')

def bootstrap_ci_median(
        nd_array: np.ndarray,
        quantile: Union[float, np.array, Iterable],
        nreps:np.long=10000):
    np.random.seed(1000)
    stat = []
    for _ in range(nreps):
        sample_idx = np.random.choice(np.arange(0,nd_array.shape[0],1), nd_array.shape[0], replace=True)
        sample_ndarray = nd_array[sample_idx,:]
        stat.append(np.expand_dims(np.quantile(sample_ndarray, quantile, axis=0, interpolation='linear'), 2))
    stat = np.concatenate(stat, axis=2)
    mn = np.mean(stat, axis=2, keepdims=False)
    return mn[0,:], mn[1, :]

def parcorr(df1: pd.DataFrame) -> pd.DataFrame:
    pcorr = df1.pcorr() - np.eye(df1.shape[1])
    return pcorr

def jaccard_sim(x,y):
    val = len(np.intersect1d(x[0],y[0]))/len(np.union1d(x[0],y[0]))
    return val