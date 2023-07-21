# supplement file for classifier's survival plot, will overlay on original image
# Created: 11/2/2021
# Status: in progress
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python survival_plot.py

import numpy as np
import pandas as pd
import os
from collections import defaultdict
# from typing import Union, Dict, Tuple, List
# from statistics.utilities import CONFIG, time_and_progress, bootstrap_ci, get_interpolator, to_datetime
# from statistics.mlp_output_wrappers import load_mlp, load_demographics, convert_to_pivot, load_all_clusters
# from statistics.statistics_formatters import kaplan_meier_estimator, KaplanMeierPairwise, KaplanMeierFitter
# from lifelines.statistics import survival_difference_at_fixed_point_in_time_test as surv_fixed_point
# from icecream import ic
# from tabulate import tabulate
# import matplotlib.pyplot as plt
# import seaborn as sns
# import colorcet as cc
# import itertools
# import lifelines
# from statsmodels.stats.multitest import multipletests

# plt.style.use('./statistics/styles/style.mplstyle')

def load_metadata_mlp_pivot():
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
    return mlp_pivot

def survival_probability_smooth(mlp_pivot):
    """Up-samples the survival probability data for plotting purposes

    Args:
        step (int, optional): step size for interpolation in units of months. Defaults to 1.
    """
    survival_values = []
    bins = CONFIG['bins']
    bins_float = [np.float(bin) for bin in bins]
    new_axis = eval(CONFIG['new_bins'])
    new_bins = np.asarray([str(int(x)) for x in new_axis])
    for _, row in mlp_pivot.iterrows():
        bin_values = row[bins]
        interpolator = get_interpolator(bins_float, bin_values)
        interp_vals = interpolator(new_axis)
        survival_values.append(np.clip(interp_vals, a_min=0, a_max=1))
    survival_values = np.asarray(survival_values)
    mlp_pivot.drop(columns=bins, inplace=True)
    for bin in new_bins:
        mlp_pivot[bin] = np.nan
    mlp_pivot.loc[:, new_bins] = \
        survival_values
    return mlp_pivot

def predict_survival_by_group(df, group_variable='Cluster Idx', pred_top_and_bot=True) -> defaultdict:
    survival_data = defaultdict(dict)
    for group, sub_df in df.groupby(group_variable):
        kmf = kaplan_meier_estimator(sub_df, label=group)
        survival_data[group]['kmf'] = kmf
        if pred_top_and_bot:
            pred = sub_df[eval(CONFIG['new_bins']).astype(str)].to_numpy()
            top_pred = bootstrap_ci(pred, [0.025, 0.975], 10000)
            survival_data[group]['pred_top'] = top_pred[0]
            survival_data[group]['pred_bot'] = top_pred[1]
    return survival_data

def predict_survival_nested_group(mlp_pivot, first_group='Dataset', second_group='Cluster Idx') -> dict:
    survival_data = {}
    for group, sub_df in mlp_pivot.groupby(first_group):
        survival_data[group] = predict_survival_by_group(sub_df.copy(), second_group)
    return survival_data

def plot_survival_data_overlay_by_group(mlp_pivot, survival_data_dict: dict, label='NACC'):
    for key, value in survival_data_dict.items():
        fig, ax = plt.subplots()
        polys = plt.fill_between(eval(CONFIG['new_bins']), value['pred_bot'], value['pred_top'], alpha=0.2, axes=ax,
            facecolor=sns.color_palette('pastel')[2])
        line = value['kmf'].plot_survival_function(ax=ax, ci_alpha=0.2, color=sns.color_palette('pastel')[3])
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Probability of survival')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 108])
        ax.legend([polys, line.get_lines()[0]], ['Predicted survival','Kaplan Meier Estimate'])
        os.makedirs(os.path.split(CONFIG['survival_plots_overlay'][label])[0], exist_ok=True)
        plt.savefig(
                '{}{}.svg'.format(CONFIG['survival_plots_overlay'][label], key),
                dpi=300)
        plt.close()

def main():
    mlp_pivot = load_metadata_mlp_pivot()
    mlp_pivot_smoothed = survival_probability_smooth(mlp_pivot)
    surv_predictions = predict_survival_nested_group(mlp_pivot_smoothed)
    plot_survival_data_overlay_by_group(mlp_pivot_smoothed, surv_predictions['NACC'], 'NACC')
    plot_survival_data_overlay_by_group(mlp_pivot_smoothed, surv_predictions['ADNI'], 'ADNI')

if __name__ == "__main__":
    main()
    print('testing ok.')
