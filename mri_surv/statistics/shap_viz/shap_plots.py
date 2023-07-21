from statistics.mlp_output_wrappers import load_roiabbr_to_roiname_map, map_neuromorph_nm_to_lobe
from statistics.clustered_mlp_output_wrappers import load_any_long
from statistics.utilities import parcorr, jaccard_sim
from statistics.dataframe_validation import SwarmPlotSchema, pa, DataFrame
from typing import Tuple, Dict
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import os

def bin_dataframe(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, list]:
    df[f'{col} Bin'], bins = pd.qcut(df[col], 10, labels=False, retbins=True)
    return df, bins

def _shap_gm_corr(df_long) -> pd.DataFrame:
    """Takes as input

    Args:
        shap_long_df (pd.DataFrame): dataframe in long format with shap values
        for each region and corresponding zscored gm volumes

    Raises:
        TypeError: If the required columns are not in the df

    Returns:
        pd.DataFrame: entry w/ correlation coefficient for each region
    """
    if not len(np.intersect1d(
        ['Gray Matter Vol', 'Shap Value', 'Region'], df_long.columns)) == 3:
        raise TypeError
    corr_values = {}
    for region, region_df in df_long.groupby('Region'):
        shap_values = region_df['Shap Value'].to_numpy()
        gm_values = region_df['Gray Matter Vol'].to_numpy()
        corr_values[region], _ = stats.spearmanr(shap_values, gm_values)
    return pd.DataFrame(data={'Region': list(corr_values.keys()),
                              'Corr': list(corr_values.values())})

def _shap_mean(df_long) -> pd.DataFrame:
    """Takes as input

    Args:
        df_long (pd.DataFrame): dataframe in long format with shap values
        for each region

    Raises:
        TypeError: If the required columns are not in the df

    Returns:
        pd.DataFrame: entry w/ correlation coefficient for each region
    """
    if not len(np.intersect1d(
        ['Shap Value', 'Region'], df_long.columns)) == 2:
        raise TypeError
    values = {}
    for region, region_df in df_long.groupby('Region'):
        shap_values = region_df['Shap Value'].to_numpy()
        values[region] = np.mean(np.abs(shap_values))
    return pd.DataFrame(data={'Region': list(values.keys()),
                              'Val': list(values.values())})

def get_region_order(tbl, by='corr', nhead=5,
                     _return_vals=False):
    if by == 'corr':
        shap_gm_value_corr = _shap_gm_corr(tbl)
        shap_gm_value_corr = shap_gm_value_corr.sort_values(by='Corr', ascending=False)
        top_indices = shap_gm_value_corr.head(nhead)
        bottom_indices = shap_gm_value_corr.tail(nhead)
    elif by == 'val':
        shap_gm_value_abs = _shap_mean(tbl)
        shap_gm_value_abs = shap_gm_value_abs.sort_values(by='Val', ascending=False)
        top_indices = shap_gm_value_abs.head(nhead)
        bottom_indices = shap_gm_value_abs.tail(nhead)
    else:
        raise NotImplementedError
    top_and_bottom_regions = {}
    top_and_bottom_regions['top'] = top_indices['Region'].to_numpy()
    top_and_bottom_regions['bottom'] = np.flip(bottom_indices['Region'].to_numpy())
    if _return_vals:
        top_and_bottom_regions['vals'] = {}
        top_and_bottom_regions['vals']['Top'] = top_indices
        top_and_bottom_regions['vals']['Bottom'] = bottom_indices
    return top_and_bottom_regions

@pa.check_types
def _retrieve_binned_shap_data() -> Dict[str, DataFrame[SwarmPlotSchema]]:
    roiabbr_to_roiname = load_roiabbr_to_roiname_map()
    roiname_to_roiabbr = {y: x for x, y in roiabbr_to_roiname.items()}
    df_adni = load_any_long('ADNI')
    df_adni['Region'] = df_adni['Region'].replace(roiname_to_roiabbr)
    df_adni, bins = bin_dataframe(df_adni, "Gray Matter Vol")
    df_nacc = load_any_long('NACC')
    df_nacc['Region'] = df_nacc['Region'].replace(roiname_to_roiabbr)
    df_nacc[f'Gray Matter Vol Bin'] = pd.cut(df_nacc['Gray Matter Vol'],
                                             bins=bins, labels=False)
    df_nacc['Dataset'] = 'NACC'
    df_adni['Dataset'] = 'ADNI'
    datas = {'NACC': df_nacc.copy(), 'ADNI': df_adni.copy()}
    return datas

def retrieve_swarm_plot_data(_dump=False):
    datas = _retrieve_binned_shap_data()
    if _dump:
        nacc_df = datas['NACC'].copy()
        abbr_to_lobe_map = map_neuromorph_nm_to_lobe()
        nacc_df['Cortex'] = nacc_df['Region'].copy().replace(abbr_to_lobe_map)
        nacc_df.to_csv('./results/shap_dataframe.csv')
    return datas

def swarm_plot(by='corr', _plt=True) -> None:
    plt.style.use('./statistics/styles/style.mplstyle')
    data = retrieve_swarm_plot_data()
    df = []
    os.makedirs('figures/figure5', exist_ok=True)
    for dataset_name, dataset in data.items():
        for cluster_idx, cluster_df in dataset.groupby('Cluster Idx'):
            top_and_bottom_regions = get_region_order(cluster_df, by=by)
            for key, regions in top_and_bottom_regions.items():
                print(f'Beginning {dataset_name}, {cluster_idx}, {regions}')
                sub_df = cluster_df.query('Region in @regions').copy()
                if key == 'top':
                    df.append(sub_df)
                sub_df['Region'] = sub_df['Region'].astype('category')
                sub_df['Region'] = sub_df['Region'].cat.reorder_categories(
                        regions)
                if _plt:
                    _, ax = plt.subplots()
                    sns.swarmplot(
                        x="Region", y="Shap Value", hue="Gray Matter Vol Bin",
                        dodge=False,
                        size=3,
                        palette='rocket',
                        data=sub_df,
                        ax=ax
                    )
                    for j in ax.xaxis.get_majorticklabels():
                        j.set_horizontalalignment('left')

                    #Help from : https://stackoverflow.com/questions/62884183/trying-to-add-a-colorbar-to-a-seaborn-scatterplot
                    norm = plt.Normalize(0,9)
                    sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
                    sm.set_array(np.ndarray([]))
                    ##############################

                    ax.get_legend().remove()
                    ax.figure.colorbar(sm)
                    ax.set_ylim([-0.05, 0.05])
                    plt.xticks(rotation=-45)
                    plt.savefig(f'figures/figure5/shap_stripplot_cluste'
                                 f'r{cluster_idx}_{key}_{dataset_name}.svg',
                                dpi=300)
                    plt.savefig(f'figures/figure5/shap_stripplot_cluste'
                                 f'r{cluster_idx}_{key}_{dataset_name}.png',
                                    dpi=300)
    df = pd.concat(df, axis=0, ignore_index=True)
    df.to_csv('metadata/data_processed/swarm_plot_data_raw.csv',
              index=False)

def jaccard_similarity_salient_regions():
    df = pd.read_csv('metadata/data_processed/swarm_plot_data_raw.csv',
                     dtype={'RID': str})
    df_by_region = df.groupby(['Dataset', 'Cluster Idx']).apply(
            lambda x: pd.unique(x['Region'])).reset_index()
    df_by_region = df_by_region.query('Dataset == \'NACC\'').copy().drop(
            columns=['Dataset'])
    df_by_region.set_index('Cluster Idx', inplace=True)
    similarity = []
    for pair in itertools.combinations(
            list(range(df_by_region.shape[0])), 2):
        similarity.append(
                jaccard_sim(df_by_region.loc[pair[0],:].to_numpy(),
                            df_by_region.loc[pair[1],:].to_numpy())
        )
    print(f'{np.mean(similarity)} +/- {np.std(similarity)}')
    return similarity

def main():
    swarm_plot('corr')  # verified
    jaccard_similarity_salient_regions()  #
    retrieve_swarm_plot_data(_dump=True)
