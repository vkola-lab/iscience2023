# supplement file for classifier's survival plot, will overlay on original image
# Created: 11/2/2021
# Status: in progress
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python survival_plot_xz.py

from collections import defaultdict
from statistics.mlp_output_wrappers import load_mlp, load_demographics, convert_to_pivot, convert_to_pivot_preserve_exp
from statistics.clustered_mlp_output_wrappers import load_all_clusters
# from statistics.dataframe_validation import MlpPivotSurvSchema, MlpPivotSurvSchemaRaw
from statistics.utilities import time_and_progress, CONFIG, pchip_interpolator, bootstrap_ci, bootstrap_ci_median
from statistics.statistics_formatters import kaplan_meier_estimator
from typing import Tuple, Union
from typing_extensions import Literal
from sksurv.metrics import concordance_index_censored, integrated_brier_score, brier_score
import sksurv.nonparametric as sksurvnp
import numpy as np
from scipy import interpolate
import pandas as pd
import os, sys
import colorcet as cc
import matplotlib.pyplot as plt
import pandera as pa
plt.style.use('./statistics/styles/style.mplstyle')

Dataset = Literal['ADNI', 'NACC']

def make_struc_array(hits, obss):
    return np.array([(x,y) for x,y in zip(hits == 1, obss)], dtype=[('hit',bool),('time',float)])

# TODO: update this so that MLP data returns train data as well as test data in the dataframe
@pa.check_types
def load_metadata_mlp_pivot_raw():
    df_long = load_mlp()
    metadata = load_demographics()
    metadata = time_and_progress(metadata)
    df_pivot = convert_to_pivot_preserve_exp(df_long)
    pivot = df_pivot.reset_index().copy()
    mlp_pivot = pivot.merge(
        metadata, how='inner', on=['RID','Dataset'],
        validate='many_to_one'
        )
    mlp_pivot.rename(columns={'Experiment': 'Exp'}, inplace=True)
    clusters = load_all_clusters()
    mlp_pivot = _add_clusters_and_format_df(mlp_pivot, clusters)

    cnn_pivot = pd.read_csv('./cgan_m/cnn/SCNN_raw.csv', index_col=0).sort_values(by=['RID'], ignore_index=True)
    cnn_pivot = _add_clusters_and_format_df(cnn_pivot, clusters)

    return mlp_pivot, cnn_pivot

@pa.check_types
def load_metadata_mlp_pivot() -> Tuple:
    df_long = load_mlp()
    metadata = load_demographics()
    metadata = time_and_progress(metadata)
    df_pivot = convert_to_pivot(df_long)
    pivot = df_pivot.reset_index().copy()
    mlp_pivot = pivot.merge(
        metadata, how='inner', on=['RID','Dataset'],
        validate='one_to_one'
        )
    clusters = load_all_clusters()
    mlp_pivot['Cluster Idx'] = clusters.loc[mlp_pivot['RID'], 'Cluster Idx'].to_numpy()
    mlp_pivot['TIMES'] = mlp_pivot['TIMES'].astype(float)

    cnn_pivot = pd.read_csv('./metadata/data_processed/SCNN_raw.csv', index_col=0).sort_values(by=['RID'], ignore_index=True)
    cnn_pivot = cnn_pivot.groupby(["RID", "Dataset"]).agg(np.mean).reset_index()
    cnn_pivot['Cluster Idx'] = clusters.loc[cnn_pivot['RID'], 'Cluster Idx'].to_numpy()
    cnn_pivot['TIMES'] = cnn_pivot['TIMES'].astype(float)

    print(cnn_pivot)

    return mlp_pivot, cnn_pivot


@pa.check_types
def _add_clusters_and_format_df(pivot_tbl: pd.DataFrame, clusters: pd.DataFrame):
    pivot_tbl = pivot_tbl.copy()
    clusters = clusters.copy()
    if 'Cluster Idx' not in pivot_tbl.columns:
        pivot_tbl['Cluster Idx'] = clusters.loc[pivot_tbl['RID'], 'Cluster Idx'].to_numpy()
    pivot_tbl['TIMES'] = pivot_tbl['TIMES'].astype(float)
    pivot_tbl['Exp'] = pivot_tbl['Exp'].astype(float)
    return pivot_tbl

@pa.check_types
def compute_brier_score_pivot(
            pivot, 
            cluster: Union[str,int], 
            test_fold: int,
            test_set: Dataset
        ):
    pivot_train = pivot.query('Dataset == \'ADNI\'').copy()
    pivot_test = pivot.query('Dataset == @test_set').copy()
    if type(cluster) == int:
        cluster = str(cluster)
        if cluster not in ['0','1','2','3']:
            raise TypeError
    pivot_train.rename(columns={'Cluster Idx': 'Cluster'}, inplace=True)
    pivot_test.rename(columns={'Cluster Idx': 'Cluster'}, inplace=True)

    training_data = _parse_test_train(pivot_train, cluster, np.setdiff1d(list(range(5)), [test_fold]))
    test_data = _parse_test_train(pivot_test, cluster, np.asarray([test_fold]))
    train_struc = make_struc_array(training_data.PROGRESSES, training_data.TIMES)
    test_struc = make_struc_array(test_data.PROGRESSES, test_data.TIMES)
    bs = retrieve_brier_scores([0,24,48,108], test_data[['0','24','48','108']], train_struc, test_struc)
    ci = concordance_index_censored(test_struc['hit'], test_struc['time'], 1-test_data['24'])
    return bs, ci

@pa.check_types
def compute_brier_score_pivot_static(
            pivot, 
            cluster: Union[str,int], 
            test_fold: int,
            test_set: Dataset
        ):
    pivot_train = pivot.query('Dataset == \'ADNI\'').copy()
    pivot_test = pivot.query('Dataset == @test_set').copy()
    if type(cluster) == int:
        cluster = str(cluster)
        if cluster not in ['0','1','2','3']:
            raise TypeError
    pivot_train.rename(columns={'Cluster Idx': 'Cluster'}, inplace=True)
    pivot_test.rename(columns={'Cluster Idx': 'Cluster'}, inplace=True)

    training_data = _parse_test_train(pivot_train, cluster, np.setdiff1d(list(range(5)), [test_fold]))
    test_data = _parse_test_train(pivot_test, cluster, np.asarray([test_fold]))
    train_struc = make_struc_array(training_data.PROGRESSES, training_data.TIMES)
    test_struc = make_struc_array(test_data.PROGRESSES, test_data.TIMES)
    ci = retrieve_brier_scores_static(train_struc, test_struc, test_data[['0','24','48','108']], '24', cluster, test_fold, test_set)
    return ci

def retrieve_brier_scores_static(train_struc, test_struc, preds_raw, time, cluster, fold, test_set):
    bins = [0, 24, 48, 108]
    new_max = min(float(max(test_struc['time'])),108)
    new_min = min(test_struc['time'])

    train_max = float(max(train_struc['time']))
    if new_max > train_max:
        new_max = train_max
        test_struc = test_struc.copy()
        bad_idx = test_struc['time'] >= new_max
        test_struc = np.delete(test_struc, bad_idx)
        preds_raw = preds_raw.copy()
        preds_raw.drop(preds_raw[bad_idx].index, axis=0, inplace=True)
        new_max = max(test_struc['time'])

    x,y = sksurvnp.kaplan_meier_estimator(train_struc['hit'], train_struc['time'])

    plt.plot(x,y)
    plt.plot(np.array([0, 24, 48, 108]), preds_raw.T)
    plt.savefig(f"figures/test_figure_{cluster}_{fold}_{test_set}.png")
    plt.close()

    truncated_bins = np.concatenate([[new_min], bins[1:-1],[new_max-1]], axis=-1)
    interp = interpolate.PchipInterpolator(bins, preds_raw, axis=1)
    preds_brier = interp(truncated_bins)

    brier_scores = brier_score(train_struc, test_struc, preds_brier[:,1], time)
    return brier_scores[1]

def retrieve_brier_scores(bins, preds_raw, train_struc, test_struc):
    bins = bins.copy()
    new_max = min(float(max(test_struc['time'])),108)
    new_min = min(test_struc['time'])

    train_max = float(max(train_struc['time']))
    if new_max > train_max:
        new_max = train_max
        test_struc = test_struc.copy()
        bad_idx = test_struc['time'] >= new_max
        test_struc = np.delete(test_struc, bad_idx)
        preds_raw = preds_raw.copy()
        preds_raw.drop(preds_raw[bad_idx].index, axis=0, inplace=True)
        new_max = max(test_struc['time'])

    truncated_bins = np.concatenate([[new_min], bins[1:-1],[new_max-1]], axis=-1)
    interp = interpolate.PchipInterpolator(bins, preds_raw, axis=1)
    preds_brier = interp(truncated_bins)

    brier_scores = integrated_brier_score(train_struc, test_struc, preds_brier, truncated_bins)
    return brier_scores, interp

def _parse_test_train(ds_data, cluster: str, fold: np.array):
    ds_data = ds_data.query(
        'Exp in @fold').query(
        'Cluster == @cluster').copy()
    return ds_data


def survival_probability_smooth(pivots):
    """Up-samples the survival probability data for plotting purposes

    Args:
        step (int, optional): step size for interpolation in units of months. Defaults to 1.
    """
    for pivot in pivots:
        survival_values = []
        bins = ['0', '24', '48', '108']
        bins_float = [np.float(bin) for bin in bins]
        new_axis = np.arange(0,109,1)
        new_bins = np.asarray([str(int(x)) for x in new_axis])
        for _, row in pivot.iterrows():
            bin_values = row[bins]
            interpolator = pchip_interpolator(bins_float, bin_values)
            interp_vals = interpolator(new_axis)
            survival_values.append(np.clip(interp_vals, a_min=0, a_max=1))
        survival_values = np.asarray(survival_values)
        pivot.drop(columns=bins, inplace=True)
        for bin in new_bins:
            pivot[bin] = np.nan
        pivot.loc[:, new_bins] = survival_values
    return pivots

def predict_survival_by_group(df, group_variable='Cluster Idx', pred_top_and_bot=True) -> defaultdict:
    survival_data = defaultdict(dict)
    for group, sub_df in df.groupby(group_variable):
        kmf = kaplan_meier_estimator(sub_df, label=group)
        survival_data[group]['kmf'] = kmf
        if pred_top_and_bot:
            pred = sub_df[eval(CONFIG['new_bins']).astype(str)].to_numpy()
            top_pred = bootstrap_ci(pred, [0.025, 0.975], 10000)
            survival_data[group]['pred_top_mlp'] = top_pred[0]
            survival_data[group]['pred_bot_mlp'] = top_pred[1]
    return survival_data

def predict_survival_by_group_extra(df, group_variable, survival_data, name):
    for group, sub_df in df.groupby(group_variable):
        pred = sub_df[eval(CONFIG['new_bins']).astype(str)].to_numpy()
        top_pred = bootstrap_ci(pred, [0.025, 0.975], 10000)
        survival_data[group]['pred_top_'+name] = top_pred[1]
        survival_data[group]['pred_bot_'+name] = top_pred[0]
    return survival_data

def predict_survival_nested_group(pivots, first_group='Dataset', second_group='Cluster Idx') -> dict:
    survival_data = {}
    names = ['mlp', 'cnn']
    for i, pivot in enumerate(pivots):
        if i == 0:
            for group, sub_df in pivot.groupby(first_group):
                survival_data[group] = predict_survival_by_group(sub_df.copy(), second_group)
        else:
            for group, sub_df in pivot.groupby(first_group):
                if group in ('NACC','ADNI'):
                    print(sub_df.shape)
                    survival_data[group] = predict_survival_by_group_extra(sub_df.copy(), second_group, survival_data[group], names[i])
    return survival_data

def plot_survival_data_overlay_by_group(survival_data_dict: dict, label='NACC'):
    step = 255 // len(survival_data_dict.items())
    for key, value in survival_data_dict.items():
        fig, ax = plt.subplots()
        polys_mlp = plt.fill_between(np.arange(0,109,1), value['pred_bot_mlp'], value['pred_top_mlp'], alpha=0.5, axes=ax, facecolor=cc.glasbey[step*0])
        polys_cnn = plt.fill_between(np.arange(0,109,1), value['pred_bot_cnn'], value['pred_top_cnn'], alpha=0.5, axes=ax, facecolor=cc.glasbey[step*1])
        line = value['kmf'].plot_survival_function(ax=ax, ci_alpha=0.5, color=cc.glasbey[step*3])
        
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Probability of survival')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 108])
        ax.legend([polys_mlp, polys_cnn, line.get_lines()[0]], ['Predicted survival (MLP)','Predicted survival (CNN)', 'Kaplan Meier Estimate'], fontsize='large')
        os.makedirs(os.path.split(CONFIG['survival_plots_overlay'][label])[0], exist_ok=True)
        plt.savefig('{}{}.svg'.format(CONFIG['survival_plots_overlay'][label], key), dpi=300)
        plt.close()

def iterate_summary_stats():
    mlp, cnn = load_metadata_mlp_pivot_raw()
    pivots = {'MLP': mlp, 'CNN': cnn}
    cluster_map = {'0': 'H', '1': 'IH', '2': 'IL', '3': 'L'}
    df_collection = []
    for model_name, model in pivots.items():
        for DS in ('ADNI','NACC'):
            for cluster in [str(x) for x in range(4)]:
                bs_list = []
                ci_list = []
                ci_list_24 = []
                for exp in range(5):
                    bs, ci = compute_brier_score_pivot(model, cluster, test_fold=exp, test_set=DS)
                    ci24 = compute_brier_score_pivot_static(model, cluster, test_fold=exp, test_set=DS)
                    if bs is not None:
                        bs_list.append(bs[0])
                        ci_list.append(ci[0])
                        ci_list_24.append(ci24[0])
                df_collection.append(pd.Series({
                        'Model': model_name,
                        'Dataset': DS,
                        'Subtype': cluster_map[cluster],
                        'Brier': f'{np.mean(bs_list)} +/- {np.std(bs_list)}',
                        'CI': f'{np.mean(ci_list)} +/- {np.std(ci_list)}',
                        'CI24': f'{np.mean(ci_list_24)} +/- {np.std(ci_list_24)}'
                    }))
    return pd.concat(df_collection, axis=1, ignore_index=True)

def main():
    mlp_pivot, cnn_pivot = load_metadata_mlp_pivot()
    pivots = [mlp_pivot, cnn_pivot]
    pivots_smoothed = survival_probability_smooth(pivots)
    surv_predictions = predict_survival_nested_group(pivots_smoothed)
    plot_survival_data_overlay_by_group(surv_predictions['NACC'], 'NACC')
    plot_survival_data_overlay_by_group(surv_predictions['ADNI'], 'ADNI')
    print(': ok')
    st = iterate_summary_stats()
    print(st)
    return st

if __name__ == "__main__":
    print('do not run this directly, instead import main() from root and call main().')
