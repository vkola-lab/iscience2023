from statistics.shap_viz.utils import *
from statistics.shap_viz.shap_brains_cnn import *
from statistics.shap_viz.shap_stats_cnn import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_brain_diffs() -> None:
    map_ = {0: 'H', 1: 'IH', 2: 'IL', 3: 'L'}
    cuts = None
    for cluster in range(3):
        sbca_orig = ShapBrainClusterAvg.load_brain(cluster)
        sbca = ShapBrainClusterAvg.load_brain(cluster+1)
        cuts = sbca_orig.plot_brain_diff(map_[cluster+1], sbca=sbca, cut_coords=cuts)

def plot_brain_diffs_abs() -> None:
    map_ = {0: 'H', 1: 'IH', 2: 'IL', 3: 'L'}
    cuts = None
    for cluster in range(3):
        sbca_orig = ShapBrainClusterAvgAbs.load_brain(cluster)
        sbca = ShapBrainClusterAvgAbs.load_brain(cluster+1)
        cuts = sbca_orig.plot_brain_diff(map_[cluster+1], sbca=sbca, cut_coords=cuts)

def stats_post_r() -> None:
    bootstrap_pvals('cnn')

def plt_legend():
    # now plot bar
    fig, ax = plt.subplots(figsize=(1, 2))
    fig.subplots_adjust(left=0.1, right=0.5)
    cmap = mpl.cm.RdBu_r
    bounds = [-2, 2]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax, orientation='vertical', extendfrac='auto',
                label="Z-score")

    plt.savefig('figures/shap_legend.svg')

def main() -> None:
    dump_series()
    # average_over_folds_and_time()
    average_abs_over_folds_and_time()
    # mask_brain_all_lobe_mn()
    mask_brain_all_lobe_mn_abs()
    # plot_brain_diffs()
    plot_brain_diffs_abs()
    # plt_legend()
