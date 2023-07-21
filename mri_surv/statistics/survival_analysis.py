import numpy as np
import pandas as pd
import os
from collections import defaultdict
from typing import Union, Tuple, List
from statistics.utilities import (
    CONFIG,
    bootstrap_ci,
    pchip_interpolator,
    benjamini_hochberg_correct,
)
from statistics.clustered_mlp_output_wrappers import (
    load_metadata_survival,
    load_metadata_mlp_pivot,
)
from statistics.statistics_formatters import kaplan_meier_estimator, KaplanMeierPairwise
from statistics.dataframe_validation import (
    MlpPivotClusterSchema,
    DataFrame,
    MlpPivotSchema,
    MlpSurvivalSchema,
    pa,
)
from lifelines.statistics import (
    survival_difference_at_fixed_point_in_time_test as surv_fixed_point,
)
from lifelines.statistics import pairwise_logrank_test
from lifelines import CoxPHFitter
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import lifelines

plt.style.use("./statistics/styles/style.mplstyle")


@pa.check_types
def survival_probability_smooth(
    mlp_pivot: DataFrame[MlpPivotClusterSchema],
) -> pd.DataFrame:
    """Up-samples the survival probability data for plotting purposes

    Args:
        step (int, optional): step size for interpolation in units of months. Defaults to 1.
    """
    survival_values = []
    bins = CONFIG["bins"]
    bins_float = [np.float(bin) for bin in bins]
    new_axis = eval(CONFIG["new_bins"])
    new_bins = np.asarray([str(int(x)) for x in new_axis])
    for _, row in mlp_pivot.iterrows():
        bin_values = row[bins]
        interpolator = pchip_interpolator(bins_float, bin_values)
        interp_vals = interpolator(new_axis)
        survival_values.append(np.clip(interp_vals, a_min=0, a_max=1))
    survival_values = np.asarray(survival_values)
    mlp_pivot.drop(columns=bins, inplace=True)
    for bin in new_bins:
        mlp_pivot[bin] = np.nan
    mlp_pivot.loc[:, new_bins] = survival_values
    return mlp_pivot


@pa.check_types
def predict_survival_by_group(
    df: pd.DataFrame, group_variable="Cluster Idx", pred_top_and_bot=True
) -> defaultdict:
    survival_data = defaultdict(dict)
    for group, sub_df in df.groupby(group_variable):
        kmf = kaplan_meier_estimator(sub_df, label=group)
        survival_data[group]["kmf"] = kmf
        if pred_top_and_bot:
            pred = sub_df[eval(CONFIG["new_bins"]).astype(str)].to_numpy()
            top_pred = bootstrap_ci(pred, (0.025, 0.9750), 10000)
            survival_data[group]["pred_top"] = top_pred[0]
            survival_data[group]["pred_bot"] = top_pred[1]
    return survival_data


class EventSchema(pa.SchemaModel):
    at_risk: pa.typing.Series[float]
    removed: pa.typing.Series[float]
    observed: pa.typing.Series[float]
    censored: pa.typing.Series[float]
    entrance: pa.typing.Series[float]
    event_at: pa.typing.Index[float]


@pa.check_types
def at_risk_row(event_table: EventSchema, time_list: list) -> pd.DataFrame:
    event_table = event_table["at_risk"].copy()
    event_dict = {"At Risk": [], "Times": []}
    for t in range(len(time_list) - 1):
        # following lines with help from /lifelines/blob/master/lifelines/plotting.py
        start = time_list[t]
        stop = time_list[t + 1]
        sub_tbl = event_table.loc[start:stop].copy()
        max_val = sub_tbl.agg(max)
        event_dict["At Risk"].append(max_val)
        event_dict["Times"].append(start)
    event_series = pd.DataFrame(event_dict)
    return event_series


def dump_at_risk_df(df: pd.DataFrame, dataset: str):
    df = df.copy()
    df["Subtype"] = df["Subtype"].replace({"0": "H", "1": "IH", "2": "IL", "3": "L"})
    df["Times"] = df["Times"].apply(lambda x: str(int(x)))
    df = df.pivot(index="Subtype", columns="Times", values="At Risk")
    os.makedirs(os.path.split(CONFIG["survival_plots"][dataset])[0], exist_ok=True)
    df.to_csv(CONFIG["survival_plots"][dataset][:-4] + ".csv")


def at_risk_row_full(event_table: EventSchema, time_list: list) -> pd.DataFrame:
    obs = event_table["observed"].copy()
    cens = event_table["censored"].copy()
    event_table = event_table["at_risk"].copy()
    event_dict = {"At Risk": [], "Times": [], "Observed": [], "Censored": []}
    max_val = np.nan
    obs_curr = np.nan
    cens_curr = np.nan
    for t in range(len(time_list) - 1):
        # following lines with help from /lifelines/blob/master/lifelines/plotting.py
        start = time_list[t]
        stop = time_list[t + 1]
        sub_tbl = event_table.loc[start : (stop - 1 / 365.25)].copy()
        prev_max = max_val - obs_curr - cens_curr
        max_val = sub_tbl.agg(max)
        obs_curr = obs.loc[start : (stop - 1 / 365.25)]

        if pd.isna(max_val):
            max_val = prev_max

        cens_curr = cens.loc[start : (stop - 1 / 365.25)]

        event_dict["At Risk"].append(max_val)
        event_dict["Times"].append(start)
        event_dict["Observed"].append(obs_curr)
        event_dict["Censored"].append(cens_curr)
    event_series = pd.DataFrame(event_dict)
    return event_series


def dump_at_risk_df_full(df: pd.DataFrame):
    df = df.copy()
    df["Times"] = df["Times"].apply(lambda x: str(int(x)))
    df["Subtype"] = df["Subtype"].replace({"0": "H", "1": "IH", "2": "IL", "3": "L"})
    df = df.melt(
        id_vars=["Dataset", "Subtype", "Times"], var_name="Type", value_name="Count"
    )
    df.set_index(["Dataset", "Subtype", "Type"], inplace=True, drop=True)
    df = df.pivot(columns="Times")
    df.to_csv("results/survival_data_whole_cohort_clustered.csv")


@pa.check_types
def plot_raw_kmf_by_group(
    mlp_pivot: DataFrame[MlpPivotClusterSchema],
    group_variable="Cluster Idx",
    dataset="ADNI",
):
    datas = mlp_pivot.query("Dataset == @dataset").copy()
    survival_data = predict_survival_by_group(
        datas, group_variable, pred_top_and_bot=False
    )
    plt.style.use("./statistics/styles/kmfstyle.mplstyle")
    fig, ax = plt.subplots()
    line = {}
    ncolors = len(cc.coolwarm)
    time_list = (
        0,
        24,
        48,
        96,
        np.inf,
    )
    at_risk_array = []
    for cluster, value in survival_data.items():
        color = cc.bmy[ncolors // (int(cluster) + 1) - 1]
        line[int(cluster)] = value["kmf"].plot_survival_function(
            ax=ax, ci_alpha=0.2, color=color
        )
        new_df = at_risk_row(value["kmf"].event_table, time_list)
        new_df["Subtype"] = cluster
        at_risk_array.append(new_df)
    at_risk_df = pd.concat(at_risk_array, ignore_index=True, axis=0)
    dump_at_risk_df(at_risk_df, dataset)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Probability of survival")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 108])
    ax.legend(
        [line[i].get_lines()[i] for i in line.keys()],
        [f"Subtype {i}" for i in line.keys()],
    )
    os.makedirs(os.path.split(CONFIG["survival_plots"][dataset])[0], exist_ok=True)
    plt.tight_layout()
    plt.savefig(CONFIG["survival_plots"][dataset])
    plt.close()
    plt.style.use("./statistics/styles/style.mplstyle")


@pa.check_types
def get_raw_kmf_by_group_full(
    mlp_pivot: DataFrame[MlpPivotClusterSchema],
    group_variable="Cluster Idx",
):

    time_list = np.arange(0, 108, 12)
    at_risk_array = []
    for dataset in (
        "ADNI",
        "NACC",
    ):
        datas = mlp_pivot.query("Dataset == @dataset").copy()
        survival_data = predict_survival_by_group(
            datas, group_variable, pred_top_and_bot=False
        )
        for cluster, value in survival_data.items():

            new_df = at_risk_row_full(value["kmf"].event_table, time_list)
            new_df["Subtype"] = cluster
            new_df["Dataset"] = dataset
            at_risk_array.append(new_df)
        at_risk_df = pd.concat(at_risk_array, ignore_index=True, axis=0)
    print(at_risk_df)
    dump_at_risk_df_full(at_risk_df)


def predict_survival_nested_group(
    mlp_pivot: pd.DataFrame, first_group="Dataset", second_group="Cluster Idx"
) -> dict:
    survival_data = {}
    for group, sub_df in mlp_pivot.groupby(first_group):
        survival_data[group] = predict_survival_by_group(sub_df.copy(), second_group)
    return survival_data


def plot_survival_data_overlay_by_group(survival_data_dict: dict, label="NACC"):
    for key, value in survival_data_dict.items():
        fig, ax = plt.subplots()
        polys = plt.fill_between(
            eval(CONFIG["new_bins"]),
            value["pred_bot"],
            value["pred_top"],
            alpha=0.2,
            axes=ax,
            facecolor=sns.color_palette("pastel")[2],
        )
        line = value["kmf"].plot_survival_function(
            ax=ax, ci_alpha=0.2, color=sns.color_palette("pastel")[3]
        )
        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Probability of survival")
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 108])
        ax.legend(
            [polys, line.get_lines()[0]],
            ["Predicted survival", "Kaplan Meier Estimate"],
        )
        os.makedirs(
            os.path.split(CONFIG["survival_plots_overlay"][label])[0], exist_ok=True
        )
        plt.savefig(
            "{}{}.svg".format(CONFIG["survival_plots_overlay"][label], key), dpi=300
        )
        plt.close()


def kmf_statistics_by_group(
    mlp_pivot: DataFrame[MlpPivotSchema], group_variable="Cluster Idx", dataset="ADNI"
):
    datas = mlp_pivot.query("Dataset == @dataset").copy()
    datas["Cluster Idx"] = datas["Cluster Idx"].astype("category")
    survival_data = KaplanMeierPairwise(datas, dataset, group_variable)
    return survival_data


def _dataframe_bh_correct_wrapper(df: pd.DataFrame):
    p_value_corrected = benjamini_hochberg_correct(df["p"].to_numpy())
    df["p_correct"] = p_value_corrected
    return df


def write_pairwise_comparisons(tbl: pd.DataFrame):
    with open(CONFIG["survival_statistics"], "w") as fi:
        tbl = tbl.reset_index()
        tbl = (
            tbl.groupby(["Dataset", "Time"])
            .apply(_dataframe_bh_correct_wrapper)
            .reset_index(drop=True)
        )
        tbl.sort_values(["Dataset", "Time"], inplace=True)
        tbl = tbl[
            [
                "Dataset",
                "Time",
                "Clusters",
                "test_statistic",
                "p",
                "p_correct",
                "n_censored_A",
                "n_obs_A",
                "n_censored_B",
                "n_obs_B",
                "test_name",
                "point_estimate",
            ]
        ]
        fi.write(
            tabulate(tbl, headers=tbl.columns, showindex=False, tablefmt="fancygrid")
        )


@pa.check_types
def compare_surv_different_times(
    km_dataset: DataFrame[MlpPivotClusterSchema], time
) -> defaultdict:
    clog = lambda s: np.log(
        -np.log(s)
    )  # from https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/statistics.py
    survival_data = defaultdict(dict)
    for ds, sub_df_dataset in km_dataset.groupby("Dataset"):
        for cluster, sub_df in sub_df_dataset.groupby("Cluster Idx"):
            kmf = kaplan_meier_estimator(sub_df, label=cluster)
            survival_data[ds][cluster] = kmf
    results = defaultdict(lambda: defaultdict(dict))
    for dataset, dataset_dict in survival_data.items():
        for t in time:
            # iterate through pairs right here
            clusters = list(dataset_dict.keys())
            for idx1 in range(len(clusters)):
                for idx2 in range(idx1):
                    pair = f"{clusters[idx2]}x{clusters[idx1]}"
                    sA_t = dataset_dict[clusters[idx1]].predict(t)
                    sB_t = dataset_dict[clusters[idx2]].predict(t)
                    results[dataset][pair][t] = surv_fixed_point(
                        t,
                        dataset_dict[clusters[idx1]],
                        dataset_dict[clusters[idx2]],
                        point_estimate=f"log(-log(x)) difference "
                        + f"{clusters[idx1]}-{clusters[idx2]}: "
                        f"{clog(sA_t)-clog(sB_t)}",
                    )
    return results


@pa.check_types
def surv_different_times_cis(
    km_dataset: DataFrame[MlpPivotClusterSchema], time
) -> defaultdict:
    cluster_map = {"0": "H", "1": "IH", "2": "IL", "3": "L"}

    survival_data = defaultdict(dict)
    for ds, sub_df_dataset in km_dataset.groupby("Dataset"):
        for cluster, sub_df in sub_df_dataset.groupby("Cluster Idx"):
            kmf = kaplan_meier_estimator(sub_df, label=cluster)
            survival_data[ds][cluster] = kmf
    results = defaultdict(lambda: defaultdict(dict))
    for dataset, dataset_dict in survival_data.items():
        for t in time:
            # iterate through pairs right here
            clusters = list(dataset_dict.keys())
            for idx1 in range(len(clusters)):

                p = dataset_dict[clusters[idx1]].survival_function_.asof(t).to_numpy()
                ci = (
                    dataset_dict[clusters[idx1]].confidence_interval_.asof(t).to_numpy()
                )
                for idx2 in range(idx1):
                    p2 = (
                        dataset_dict[clusters[idx2]]
                        .survival_function_.asof(t)
                        .to_numpy()
                    )
                    ci2 = (
                        dataset_dict[clusters[idx2]]
                        .confidence_interval_.asof(t)
                        .to_numpy()
                    )

                    pair = f"{clusters[idx2]}x{clusters[idx1]}"
                    results[dataset][pair][t] = surv_fixed_point(
                        t,
                        dataset_dict[clusters[idx1]],
                        dataset_dict[clusters[idx2]],
                        point_estimate=f"{cluster_map[clusters[idx1]]}: "
                        f"{p[0]:0.2f} [{ci[0]:0.2f}, {ci[1]:0.2f}],"
                        f"{cluster_map[clusters[idx2]]}: {p2[0]:0.2f} [{ci2[0]:0.2f}, {ci2[1]:0.2f}]",
                    )
    return results


@pa.check_types
def cph_subgroups(km_dataset: DataFrame[MlpPivotClusterSchema]) -> defaultdict:

    cluster_map = {"0": "H", "1": "IH", "2": "IL", "3": "L"}

    with open("results/cph_subgroup_stats.txt", "w") as fi:
        fi.write("-" * 20 + "\n")
        df = pd.DataFrame(
            columns=[
                "Comparison",
                "Dataset",
                "exp(coef)",
                "exp(coef) lower 95%",
                "exp(coef) upper 95%",
                "p",
            ]
        )
        for ds, sub_df_dataset in km_dataset.groupby("Dataset"):
            fi.write(f"{ds}" + "-" * 20 + "\n")
            for idx1 in range(4):
                for idx2 in range(idx1):
                    subtbl = sub_df_dataset.loc[
                        (sub_df_dataset.loc[:, "Cluster Idx"] == str(idx1)).to_numpy()
                        | (
                            sub_df_dataset.loc[:, "Cluster Idx"] == str(idx2)
                        ).to_numpy(),
                        :,
                    ]
                    subtbl = subtbl[["Cluster Idx", "PROGRESSES", "TIMES"]]
                    fi.write(f"{idx1} vs {idx2}" + " " + "-" * 18 + "\n")
                    cph = CoxPHFitter()
                    cph.fit(subtbl, duration_col="TIMES", event_col="PROGRESSES")
                    sub_df = cph.summary[
                        ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
                    ]
                    sub_df["Comparison"] = f"{cluster_map[str(idx1)]} vs {cluster_map[str(idx2)]}"
                    sub_df["Dataset"] = ds
                    df = pd.concat([df, sub_df], axis=0, ignore_index=True)
        df = df.groupby('Dataset').apply(_dataframe_bh_correct_wrapper).reset_index(drop=True)
        df.to_csv(fi)

@pa.check_types
def compare_surv_logrank(km_dataset: DataFrame[MlpPivotClusterSchema]):
    results = {}
    km_dataset = km_dataset[["TIMES", "PROGRESSES", "Dataset", "Cluster Idx"]]
    km_dataset.rename(columns={"Cluster Idx": "Cluster"}, inplace=True)
    for ds, sub_df_dataset in km_dataset.groupby("Dataset"):
        sub_df_dataset = sub_df_dataset.copy().drop(columns="Dataset")
        results[ds] = pairwise_logrank_test(
            sub_df_dataset[["TIMES"]].to_numpy(),
            groups=sub_df_dataset[["CLUSTER"]].to_numpy(),
            event_observed=sub_df_dataset[["PROGRESSES"]].to_numpy(),
        )
    return results


class FormattedStatisticalResult(lifelines.statistics.StatisticalResult):
    def __init__(self, results: lifelines.statistics.StatisticalResult, indices: Tuple):
        self._kwargs = dict(
            test_name=results.test_name,
            null_distribution=results.null_distribution,
            n_censored_A=sum(results.fitterA.event_table["censored"]),
            n_obs_A=results.fitterA.event_table.loc[0, "at_risk"],
            n_censored_B=sum(results.fitterB.event_table["censored"]),
            n_obs_B=results.fitterB.event_table.loc[0, "at_risk"],
            point_estimate=results.point_estimate,
        )
        super().__init__(
            p_value=results.p_value,
            test_statistic=results.test_statistic,
            name=results.name,
            **self._kwargs,
        )
        self.specs = indices

    def to_table(self):
        mi = pd.MultiIndex.from_tuples(
            (self.specs,), names=("Dataset", "Clusters", "Time")
        )
        tbl = self.summary.copy()
        df = pd.DataFrame(index=mi, data=tbl.to_numpy(), columns=tbl.columns)
        for kwarg, value in self._kwargs.items():
            df.loc[mi, kwarg] = value
        return df


def formatted_statistical_result_stack(
    nested_dict: Union[defaultdict, dict], labels: Tuple = ()
):
    tables = []
    for key, value in nested_dict.items():
        if type(value) == lifelines.statistics.StatisticalResult:
            result = FormattedStatisticalResult(
                value, indices=tuple(list(labels) + [key])
            )
            tables.append(result.to_table())
        else:
            f = formatted_statistical_result_stack(value, tuple(list(labels) + [key]))
            tables.append(f)
    tables = pd.concat(tables, axis=0)
    return tables


def run_kmfs():
    mlp_pivot: DataFrame[MlpPivotClusterSchema] = load_metadata_mlp_pivot()
    plot_raw_kmf_by_group(mlp_pivot, dataset="ADNI")
    plot_raw_kmf_by_group(mlp_pivot, dataset="NACC")


def main():
    mlp_pivot: DataFrame[MlpPivotClusterSchema] = load_metadata_mlp_pivot()
    # results = surv_different_times_cis(mlp_pivot, [24, 48, 96])
    # df = formatted_statistical_result_stack(results)
    # write_pairwise_comparisons(df)

    cph_subgroups(mlp_pivot)

    # plot_raw_kmf_by_group(mlp_pivot, dataset='ADNI')
    # plot_raw_kmf_by_group(mlp_pivot, dataset='NACC')

    # get_raw_kmf_by_group_full(mlp_pivot)

    # results = compare_surv_different_times(mlp_pivot, [24, 48, 96])
    # df = formatted_statistical_result_stack(results)
    # write_pairwise_comparisons(df)
    # mlp_pivot_smoothed = survival_probability_smooth(mlp_pivot)

    # surv_predictions = predict_survival_nested_group(mlp_pivot_smoothed)
    # plot_survival_data_overlay_by_group(surv_predictions['NACC'], 'NACC')
    # plot_survival_data_overlay_by_group(surv_predictions['ADNI'], 'ADNI')
