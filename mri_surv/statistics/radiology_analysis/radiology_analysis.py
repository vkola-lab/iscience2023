import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats import multitest
from statistics.clustered_mlp_output_wrappers import load_shap_with_parcellations
from statistics.mlp_output_wrappers import name_to_lobe_map

plt.style.use("./statistics/styles/radstyle.mplstyle")

FNAME = "./metadata/data_raw/MCIADSubtypeAssessment_weighted_nosub.csv"

COLTYPES = {
    "cingulate_avg": float,
    "mesial_temp_avg": float,
    "temporal_lobe_other_avg": float,
    "insula_avg": float,
    "frontal_avg": float,
    "parietal_avg": float,
    "occipital_avg": float,
    "id": str,
    "id2": str,
    "rev_initials": str,
}

COLMAP = {
    "cingulate_avg": "Cingulate",
    "mesial_temp_avg": "Mesial Temporal",
    "temporal_lobe_other_avg": "Other Temporal",
    "insula_avg": "Insula",
    "frontal_avg": "Frontal",
    "parietal_avg": "Parietal",
    "occipital_avg": "Occipital",
}

PARCELLATION_RADIOLOGY_MAP = {
    "Limbic-Cing": "cingulate_avg",
    "TL-M": "mesial_temp_avg",
    "TL": "temporal_lobe_other_avg",
    "Insula": "insula_avg",
    "FL": "frontal_avg",
    "PL": "parietal_avg",
    "OL": "occipital_avg",
}

RADIOLOGY_COLMAP = {
    "Limbic-Cing": "Cingulate",
    "TL-M": "Mesial Temporal",
    "TL": "Other Temporal",
    "Insula": "Insula",
    "FL": "Frontal",
    "PL": "Parietal",
    "OL": "Occipital",
}


def shap_and_gmv_by_lobe() -> pd.DataFrame:
    """
    Loads Z-scored normalized gray matter volumes for parcellated brain regions,
    maps them to the appropriate lobe, and averages across the brain regions
    within each cluster

    Returns:
        pd.DataFrame: dataframe with averaged SHAP values and z-scored,
            normalized gray-matter volume values corresponding to each lobe
            for each patient/RID
    """
    shap = load_shap_with_parcellations()
    map_ = name_to_lobe_map()
    shap["Region"] = shap["Region"].apply(lambda x: map_[x] if x in map_.keys() else "")
    shap = shap.query("Dataset == 'ADNI'").copy()
    dropped = shap.drop(columns=["Cluster Idx", "Gray Matter Vol Raw", "Dataset"])
    dropped = dropped.loc[shap["Region"] != "", :]
    return dropped.groupby(["RID", "Region"]).agg(np.mean).reset_index()


def read_hemisphere_average() -> pd.DataFrame:
    """
    Loads hemisphere-averaged values

    Returns:
        pd.DataFrame: dataframe with columns COLTYPES.keys()
    """
    return pd.read_csv(FNAME, dtype=COLTYPES, usecols=list(COLTYPES.keys()))


def _load_decoder() -> dict:
    df = pd.read_csv(
        "./metadata/data_processed/shuffled_mri_names.csv",
        dtype={"Cluster Idx": str, "Unnamed: 0": str},
        usecols=["Cluster Idx", "Unnamed: 0"],
    )
    df = df.rename(columns={"Unnamed: 0": "id", "Cluster Idx": "ClusterIdx"})
    df = df.set_index("id", drop=True)
    df = df.to_dict("index")
    decoder = {x: y["ClusterIdx"] for x, y in df.items()}
    return decoder


def _load_decoder_with_rid() -> tuple:
    df = pd.read_csv(
        "./metadata/data_processed/shuffled_mri_names.csv",
        dtype={"Cluster Idx": str, "Unnamed: 0": str, "RID": str},
        usecols=["Cluster Idx", "Unnamed: 0", "RID"],
    )
    df = df.rename(columns={"Unnamed: 0": "id", "Cluster Idx": "ClusterIdx"})
    df = df.set_index("id", drop=True)
    df = df.to_dict("index")
    dict_cluster = {x: y["ClusterIdx"] for x, y in df.items()}
    dict_rid = {x: y["RID"].zfill(4) for x, y in df.items()}
    return dict_cluster, dict_rid


def load_decoded_df() -> pd.DataFrame:
    """
    loads hemisphere-averaged radiologist gradings, with the appropriate "cluster"
        assigned to each id

    Returns:
        pd.DataFrame: hemisphere-averaged radiologist gradings
    """
    df = read_hemisphere_average()
    decoder = _load_decoder()
    df["Subtype"] = df["id"].replace(decoder)
    return df


def load_decoded_df_rid() -> pd.DataFrame:
    """
    Loads hemisphere-averaged radiologist gradings with the appropriate RID and cluster assigned

    Returns:
        pd.DataFrame: Dataframe with grading, id, Subtype, and RID as columns
    """
    df = read_hemisphere_average()
    dict_cluster, dict_rid = _load_decoder_with_rid()
    df["Subtype"] = df["id"].replace(dict_cluster)
    df["RID"] = df["id"].replace(dict_rid)
    return df


def mean_grading() -> pd.DataFrame:
    """
    Loads radiologist scores, averaged across radiologists and hemispheres

    Returns:
        pd.DataFrame: wide-form dataframe containing average subtypes across
        hemispheres and radiologists
    """
    df = load_decoded_df()
    df.drop(["id2", "rev_initials"], axis=1, inplace=True)
    df["SubtypeLabels"] = df["Subtype"].replace(
        {"0": "H", "1": "IH", "2": "IL", "3": "L"}
    )
    df["Subtype"] = 3 - df["Subtype"].apply(np.float)
    df = df.groupby(["id", "Subtype", "SubtypeLabels"]).agg(np.mean).reset_index()
    return df


def mean_grading_vs_gmv() -> pd.DataFrame:
    """
    Generates a dataframe with mean radiologist grade, averaged across RID, in addition
        to mean Z-scored gray matter volume for each patient

    Returns:
        pd.DataFrame: long-form dataframe with mean grade, region, RID, and z-scored gray matter
            volume for each patient
    """
    df = load_decoded_df_rid()
    shap = shap_and_gmv_by_lobe()
    df.drop(
        ["id2", "rev_initials", "Subtype", "id"], axis=1, inplace=True
    )  # don't care about subtypes or id now that we have RID
    map_ = {y: x for x, y in PARCELLATION_RADIOLOGY_MAP.items()}
    df = df.melt(id_vars=["RID"], var_name="Region", value_name="Mean Grade")
    df = df.groupby(["RID", "Region"]).agg(np.mean).reset_index()
    df["Region"] = df["Region"].replace(map_)
    df = df.merge(
        shap, left_on=["RID", "Region"], right_on=["RID", "Region"], how="left"
    )
    return df


def melt_it() -> pd.DataFrame:
    """
    Retrieves the average grading for each brain region across hemispheres and radiologists
    for each patient. Transforms it into long-form from wide form.

    Returns:
        pd.DataFrame: long-form dataframe with mean grades for each brain region averaged
        across each patient.
    """
    df = mean_grading()
    df = df.melt(
        id_vars=["Subtype", "SubtypeLabels", "id"],
        var_name="Region",
        value_name="Mean Grade",
    )
    df["Region"] = df["Region"].replace(COLMAP)
    return df


# The below from https://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid
def create_annotater(rho, p):
    def annotate(data, **kws) -> None:
        ax = plt.gca()
        region = pd.unique(data["Region"])
        assert len(region) == 1
        r = rho[region[0]]
        p_ = p[region[0]]
        ax.text(
            0.05,
            0.9,
            r"$\rho$" + f"={r:.2f}, p={p_:.2E}",
            transform=ax.transAxes,
            size=24,
        )

    return annotate


def plot_stripplot() -> None:
    """
    Plots strip plots for all Radiology-graded regions. This utilizes the mean
    grade for each radiologist
    """
    df = melt_it()
    df["SubtypeLabels"] = df["SubtypeLabels"].astype("category")
    df["SubtypeLabels"] = df["SubtypeLabels"].cat.reorder_categories(
        ["L", "IL", "IH", "H"]
    )
    df["Region"] = df["Region"].astype("category")
    df["Region"] = df["Region"].cat.reorder_categories(list(RADIOLOGY_COLMAP.values()))
    g_ = (
        sns.catplot(
            data=df,
            seed=0,
            kind="strip",
            x="SubtypeLabels",
            y="Mean Grade",
            col_wrap=4,
            col="Region",
        ),
    )
    for ax_, title in zip(g_[0].axes.flatten(), list(RADIOLOGY_COLMAP.values())):
        ax_.set_title(title)
        ax_.set_yticks([0, 1, 2, 3])
        ax_.set_yticklabels(["0", "1", "2", "3"])
    g_[0].set_xlabels("Subtype")
    g_[0].set_ylabels("Grading")
    _dump_plot("radiologist_grading")


def plot_lm() -> None:
    """
    Plots a linear model comparing gray matter volume with
    """
    df = mean_grading_vs_gmv()
    df["Region"] = df["Region"].replace(RADIOLOGY_COLMAP)
    df["Region"] = df["Region"].astype("category")
    df["Region"] = df["Region"].cat.reorder_categories(list(RADIOLOGY_COLMAP.values()))
    df, rho, p = get_spearman_gmv(df)
    p_vals = [x for x in p.values()]
    _, p_vals, _, _ = multitest.multipletests(p_vals, method="fdr_bh")
    p = {key: x for key, x in zip(p.keys(), p_vals)}
    g_ = (
        sns.lmplot(
            data=df,
            x="Mean Grade",
            y="Gray Matter Vol",
            hue="Region",
            col_wrap=4,
            col="Region",
        ),
    )
    g_[0].set(xlim=(-0.1, 3.1))
    annotate = create_annotater(rho, p)
    g_[0].map_dataframe(annotate)
    g_[0].set_ylabels("Gray Matter Vol")
    for ax_, title in zip(g_[0].axes.flatten(), list(RADIOLOGY_COLMAP.values())):
        ax_.set_title(title)
        ax_.set_xticks([0, 1, 2, 3])
        ax_.set_xticklabels(["0", "1", "2", "3"])
    g_[0].set_xlabels("Mean Grading")
    _dump_plot("gmv_vs_grading")


def get_spearman_gmv(df) -> tuple:
    spearman_and_p = (
        df[["Region", "Mean Grade", "Gray Matter Vol"]]
        .groupby("Region")
        .apply(lambda x: spearmanr(x["Mean Grade"], x["Gray Matter Vol"]))
    )
    rho, p = _dictify(spearman_and_p)
    return df, rho, p


def get_spearman() -> tuple:
    df = melt_it()
    spearman_and_p = (
        df[["Region", "Subtype", "Mean Grade"]]
        .groupby("Region")
        .apply(lambda x: spearmanr(x["Subtype"], x["Mean Grade"]))
    )
    rho, p = _dictify(spearman_and_p)
    return df, rho, p


def _dictify(df: pd.Series) -> tuple:
    rho = {row: x[0] for row, x in zip(df.index, df)}
    p = {row: x[1] for row, x in zip(df.index, df)}
    return rho, p


def _dump_plot(ti_) -> None:
    for fmt in (".png", ".svg"):
        plt.savefig(f"figures/radiologist_plots_{ti_}{fmt}", dpi=300)


def melt_stats() -> pd.DataFrame:
    """
    Computes mannwhitneyu/ranksum statistic for higher and lower-grade path

    Returns:
        pd.DataFrame: dataframe with p-values comparing high- and low- risk subgroups
    """
    df = melt_it()
    df["SubtypeLabels"] = df["SubtypeLabels"].replace({"IH": "H", "IL": "L"})
    df.drop(columns=["id", "Subtype"], inplace=True)
    df = df.groupby("Region").apply(_wilcox)
    df = _correct_pvals(df)
    return df


def _wilcox(df_group) -> tuple:
    high, low = df_group.groupby("SubtypeLabels").apply(
        lambda x: x["Mean Grade"].to_numpy()
    )
    res = mannwhitneyu(high, low)
    return res.statistic, res.pvalue, len(high), len(low)


def _correct_pvals(df) -> pd.DataFrame:
    print(df)
    p_val_list = [x[1] for x in df.to_numpy()]
    statistic = [x[0] for x in df.to_numpy()]
    _, p_val_list, _, _ = multitest.multipletests(p_val_list, method="fdr_bh")
    index = df.index
    return pd.DataFrame({"Region": index, "p-val": p_val_list, "statistic": statistic})


def main() -> None:
    """
    Plots strip plot, linear model, and mann-u whitney stats for radiology
    
    """
    plot_stripplot()
    plot_lm()
    melt_stats().to_csv("results/pairwise_ranksum_radiology.csv")
