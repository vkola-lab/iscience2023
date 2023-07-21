from statsmodels.stats import multitest
from statistics.shap_viz.utils import sig
from scipy.stats import wilcoxon

import pandas as pd
import numpy as np
import itertools

__all__ = ['bootstrap_pvals']


def bootstrap_pvals(mod: str) -> pd.DataFrame:
    if mod == 'mlp':
        df = pd.read_csv('./metadata/data_processed/shap_mlp_barplot.csv')
    elif mod == 'cnn':
        df = pd.read_csv('./metadata/data_processed/shap_cnn_barplot.csv')
    df = df.drop(columns='Cortex')
    p_vals = _bs_pvals(df, reps=10000, seed=10)
    p_vals.to_csv(f'./metadata/data_processed/shap_{mod}_pvals.csv')
    return p_vals


def pairwise_pvals_abs_all() -> pd.DataFrame:
    df = pd.read_csv('./metadata/data_processed/shap_cnn_abs_barplot.csv')
    df = df.drop(columns='Cortex')
    p_vals = pairwise_pvals_abs(df)
    p_vals.to_csv(f'./metadata/data_processed/shap_cnn_abs_pvals.csv')
    return p_vals


def _bs_pvals(df: pd.DataFrame, reps=100000, seed=10) -> pd.DataFrame:
    p_vals = pd.DataFrame(columns=['Cluster','Lobe','p_uncorr', 'p_corr'])
    for cluster, cluster_df in df.groupby('Cluster'):
        p_vals_cluster = pd.DataFrame(columns=['Cluster','Lobe','p_uncorr'])
        for lobe, lobe_cluster_df in cluster_df.groupby('Lobe'):
            p = bootstrap_mean_pval(
                lobe_cluster_df['Shap'].to_numpy(),
                reps=reps,
                seed=seed)
            se = pd.Series({
                'Cluster': cluster,
                'Lobe': lobe,
                'p_uncorr': p
            }).to_frame().T
            p_vals_cluster = pd.concat([p_vals_cluster,se], axis=0, ignore_index=True)
        _, p_vals_cluster['p_corr'],  _, _ = multitest.multipletests(p_vals_cluster['p_uncorr'], method='fdr_bh')
        p_vals = pd.concat([p_vals, p_vals_cluster], axis=0, ignore_index=True)
    return p_vals


def bootstrap_mean_pval(arr: np.array, reps: int, seed: int) -> np.float:
    rng = np.random.default_rng(seed=seed)
    val_list = np.full((reps,1), fill_value=np.nan)
    samp = arr
    for r in range(reps):
        samp_last = samp
        samp = rng.choice(arr, size=arr.shape)
        assert(not all(x == y for x,y in zip(samp_last, samp)))
        assert(np.array_equal(samp.shape,arr.shape))
        val_list[r] = np.mean(samp)
    return 2*min(np.mean(val_list <= 0), np.mean(val_list >= 0))


def pairwise_pvals_abs(df: pd.DataFrame) -> pd.DataFrame:
    p_vals = pd.DataFrame(columns=['Cluster','Comparison','stat','z', 'p_uncorr', 'p_corr'])
    for cluster, cluster_df in df.groupby('Cluster'):
        p_vals_cluster = pd.DataFrame(columns=['Cluster','Comparison','stat','z', 'p_uncorr'])
        regions = cluster_df.Lobe.unique()
        region_pairs = itertools.combinations(regions, 2)
        print(region_pairs)
        for pair in region_pairs:
            lobe1 = cluster_df.query(f"Lobe == \"{pair[0]}\"").sort_values("RID")
            lobe2 = cluster_df.query(f"Lobe == \"{pair[1]}\"").sort_values("RID")
            assert all(lobe1["RID"].to_numpy() == lobe2["RID"].to_numpy())
            st, p = wilcoxon(lobe1["Shap"].to_numpy(), lobe2["Shap"].to_numpy(), alternative="two-sided")
            
            _, p_lt = wilcoxon(lobe1["Shap"].to_numpy(), lobe2["Shap"].to_numpy(), alternative="less")
            _, p_gt = wilcoxon(lobe1["Shap"].to_numpy(), lobe2["Shap"].to_numpy(), alternative="greater")
            print(f"{cluster}, {pair}: {len(lobe1.Shap)}, {len(lobe2.Shap)}")
            se = pd.Series({
                    'Cluster': cluster,
                    'Comparison': f"{pair[0]}-{pair[1]}",
                    'stat': st,
                    'z': "x_<_y" if (p_lt < 0.05 and p_gt > 0.05) else "x_>_y",
                    'p_uncorr': p
                }).to_frame().T
            p_vals_cluster = pd.concat([p_vals_cluster,se], axis=0, ignore_index=True)
        _, p_vals_cluster['p_corr'],  _, _ = multitest.multipletests(p_vals_cluster['p_uncorr'], method='fdr_bh')
        p_vals = pd.concat([p_vals, p_vals_cluster], axis=0, ignore_index=True)
    return p_vals
