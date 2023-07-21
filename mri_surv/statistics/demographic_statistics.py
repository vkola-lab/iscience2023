from statistics.utilities import (
    load_json,
    deabbreviate_parcellation,
    upper_and_lower_quartiles,
)
from statistics.statistics_formatters import *
from icecream import ic
from collections import defaultdict
from scipy import stats
from scikit_posthocs import posthoc_dunn

# from sksurv.nonparametric import CensoringDistributionEstimator
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import (
    survival_difference_at_fixed_point_in_time_test as surv_fixed_point,
)
from tabulate import tabulate
import logging
import os
import pandera as pa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("./statistics/styles/style.mplstyle")
JSON_FI = "./statistics/config/statistics_config.json"
logger_name = os.path.join(os.path.abspath("."), "logs/demographic_statistics.log")

with open(logger_name, "w") as fi:
    fi.write("")

hdlr = logging.FileHandler(logger_name)
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)
logger.setLevel("INFO")

__all__ = ["TrainAndTestStats"]

CONFIG_ = load_json(JSON_FI)


class _DemographicStatistics(object):
    def __init__(self, dataset="ADNI"):
        self.__assign_props(json_fi=JSON_FI, dataset=dataset)
        self.__recode_progression_categories()
        self.__recode_categorical_values()
        self.__cross_tabulate_columns()
        self.__summarize_continuous_columns_and_values()

    def __recode_progression_categories(self):
        self.df["PROGRESSION_CATEGORY"] = self.df["PROGRESSION_CATEGORY"].map(
            {2: "\u003C4 years", 1: "\u003C2 years", -1: "\u22654 years"}
        )
        self.df["PROGRESSION_CATEGORY"] = self.df["PROGRESSION_CATEGORY"].fillna(
            "Censored"
        )
        self.df["PROGRESSION_CATEGORY"] = self.df["PROGRESSION_CATEGORY"].astype(
            "category"
        )
        self.df["PROGRESSION_CATEGORY"] = self.df[
            "PROGRESSION_CATEGORY"
        ].cat.reorder_categories(
            ["\u003C2 years", "\u003C4 years", "\u22654 years", "Censored"]
        )

    def __assign_props(self, json_fi, dataset):
        props = load_json(json_fi)[f"{dataset}_config"]
        for key, value in props.items():
            setattr(self, key, value)
        self.df = pd.read_csv(
            self.metadata_csv,
            usecols=self.demographics_columns["raw"],
            dtype={"RID": str},
        )

    def __recode_categorical_values(self):
        for col in self.demographics_columns["categorical"]:
            self.df[col] = [
                str(int(x)) if isinstance(x, float) and not np.isnan(x) else x
                for x in self.df[col]
            ]
            self.df[col] = self.df[col].astype("category")

    def __cross_tabulate_columns(self):
        self.df_crosstab = [
            pd.crosstab(
                self.df["PROGRESSION_CATEGORY"],
                self.df[col],
                margins=True,
            )
            for col in self.demographics_columns["categorical"]
        ]

    def __summarize_continuous_columns_and_values(self):
        self.df_continuous = {}
        for col in self.demographics_columns["continuous"]:
            self.df[col] = self.df[col].astype("float")
            df = self.df[[col] + ["PROGRESSION_CATEGORY"]].copy()
            df.rename(columns={col: self.column_maps[col]}, inplace=True)
            self.df_continuous[self.column_maps[col]] = df.groupby(
                "PROGRESSION_CATEGORY"
            ).describe()
            self.df_continuous[self.column_maps[col]] = self.df_continuous[
                self.column_maps[col]
            ].T
            self.df_continuous[self.column_maps[col]].index = self.df_continuous[
                self.column_maps[col]
            ].index.droplevel(level=0)

    def write_summary_stats(self):
        for f in self.df_crosstab:
            with open(self.output_file, "a") as fi:
                fi.write(f"\n------------\n{f.columns.name}\n")
            f.to_csv(self.output_file, sep=",", index=True, mode="a")
        for key in self.df_continuous:
            with open(self.output_file, "a") as fi:
                fi.write(f"\n" + "-" * 30 + f"\n{key}\n")
                fi.write(
                    tabulate(
                        self.df_continuous[key], headers=self.df_continuous[key].columns
                    )
                )
                fi.write(f"\n\n")


class TrainAndTestStats(object):
    def __init__(
        self,
        train_stats: _DemographicStatistics = None,
        test_stats: _DemographicStatistics = None,
    ):
        if train_stats is None:
            train_stats = _DemographicStatistics("ADNI")
        if test_stats is None:
            test_stats = _DemographicStatistics("NACC")
        os.makedirs("figures/figure_1", exist_ok=True)
        self.props = load_json(JSON_FI)
        self._train_stats = train_stats
        self._test_stats = test_stats
        self._load_parcellations()
        self.merged_df = _merge_df(train_stats, test_stats)
        self.merged_df_parcellations = self.merged_df.merge(
            self.parcellation, left_on=["RID"], right_on=["RID"]
        )
        self.stats_output = self.props["demographics_output_file"]
        self.shared_columns = self.props["shared_columns"]
        self.csf_columns = self.props["csf_columns"]
        self.crosstab_columns = self.props["crosstab_columns"]
        self._cross_tabulate_columns()
        self.stats = {}
        self.kmf = defaultdict(lambda: defaultdict(lambda x: x))

    def _load_parcellations(self):
        self.parcellation = pd.read_csv(self.props["parcellation_csv"])
        self.parcellation.drop(
            columns=self.props["ventricles"] + ["Dataset"] + ["PROGRESSION_CATEGORY"],
            inplace=True,
        )

    def _cross_tabulate_columns(self):
        df_crosstab = {}
        df_crosstab_normalized = {}
        for col in self.crosstab_columns:
            df_crosstab[col] = pd.crosstab(
                self.merged_df["Dataset"], self.merged_df[col], margins=True
            )
            df_crosstab_normalized[col] = df_crosstab[col].copy()
            df_crosstab_normalized[col] = df_crosstab_normalized[col].divide(
                df_crosstab_normalized[col]["All"], axis="rows"
            )
            df_crosstab_normalized[col].drop(
                columns=["All"], index=["All"], inplace=True
            )
            logger.info(col)
            logger.info(df_crosstab)
            logger.info(
                f"Total length: {len(self.merged_df[col])}-Total nans: {np.sum(pd.isna(self.merged_df[col]))}"
            )
        self.df_crosstab = df_crosstab
        self.df_crosstab_normalized = df_crosstab_normalized

    def _retrieve_parcellation_response(self):
        parcellation = self.parcellation.set_index("RID").copy()
        response = self.merged_df.query("Dataset == 'ADNI'").copy()
        response = response[["RID", "TIME_TO_PROGRESSION"]].copy()
        response.set_index("RID", inplace=True)
        parcellation = parcellation.loc[response.index, :]
        labels = parcellation.columns
        logger.info(parcellation.head())
        logger.info(response.head())
        return parcellation.to_numpy(), response.to_numpy(), labels

    def _create_subplot(self, df, fig, ax, strip=False):
        if strip:
            sns.swarmplot(
                x="Time to progress to AD",
                y="Values_Test",
                hue="Dataset",
                dodge=True,
                size=5,
                data=df,
                ax=ax,
            )
            sns.violinplot(
                x="Time to progress to AD",
                y="Values_Train",
                hue="Dataset",
                split=True,
                inner="quartile",
                data=df,
                ax=ax,
                cut=0,
            )
            hdl, lbl = ax.get_legend_handles_labels()  #
            # https://stackoverflow.com/questions/51579215/remove-seaborn
            # -lineplot-legend-title
            ax.legend(handles=hdl[:2], labels=lbl[:2])
            sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)
        else:
            sns.violinplot(
                x="Time to progress to AD",
                y="Value",
                hue="Dataset",
                split=True,
                inner="quartile",
                data=df,
                ax=ax,
                cut=0,
            )
            sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)
        # add_stat_annotation(ax, data=df, x="Time to progress to AD",
        #                     y='Value', hue='Dataset')

    def generate_stacked_bar(self, columns=None):
        if columns == None:
            columns = self.crosstab_columns
        for i, col in enumerate(columns):
            fig, ax = plt.subplots()
            self.df_crosstab_normalized[col].T.plot.bar(stacked=False, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("Proportion")
            ax.set_title("")
            ax.legend().set_title("")
            plt.xticks(rotation=-45)
            # https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
            for sp in ax.spines.values():
                sp.set_visible(False)
            plt.savefig(
                f"figures/figure_1/demographic_statistics_stacked_bar_"
                f"{str(i).zfill(2)}" + ".svg",
                dpi=300,
            )
            plt.savefig(
                f"figures/figure_1/demographic_statistics_stacked_bar_"
                f"{str(i).zfill(2)}" + ".png",
                dpi=300,
            )
            plt.close()

    def generate_violin_plots(self):
        df = _get_melted_df(
            self.merged_df,
            self.shared_columns,
            ["Time to progress to AD", "RID", "Dataset"],
        )
        for i, col in enumerate(self.shared_columns):
            fig, ax = plt.subplots()
            if col in self.csf_columns:
                swarm = True
            else:
                swarm = False
            self._create_subplot(df.query("Test == @col").copy(), fig, ax, swarm)
            ax.set_xlabel("Years since diagnosis")
            ax.set_ylabel(col)
            ax.legend().set_title("")
            plt.xticks(rotation=-45)

            plt.savefig(
                f"figures/figure_1/demographic_statistics_violin_plots_"
                f"{str(i).zfill(2)}" + ".svg",
                dpi=300,
            )
            plt.savefig(
                f"figures/figure_1/demographic_statistics_violin_plots_"
                f"{str(i).zfill(2)}" + ".png",
                dpi=300,
            )
            plt.close()

    def compare_surv_different_times(self, time):
        clog = lambda s: np.log(
            -np.log(s)
        )  # from https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/statistics.py
        self._nacc_mdl = KaplanMeierFitter().fit(
            self.km_dataset["NACC"]["TIMES"],
            self.km_dataset["NACC"]["PROGRESSES"],
            label="NACC",
        )
        self._adni_mdl = KaplanMeierFitter().fit(
            self.km_dataset["ADNI"]["TIMES"],
            self.km_dataset["ADNI"]["PROGRESSES"],
            label="ADNI",
        )
        ci_adni = self._adni_mdl.confidence_interval_
        p_adni = self._adni_mdl.survival_function_
        ci_nacc = self._nacc_mdl.confidence_interval_
        p_nacc = self._nacc_mdl.survival_function_
        

        with open('results/CI_for_survival_curves.txt', 'w') as fi:
            fi.write(f"\tSurvival estimate for ADNI: \n")
            fi.write(f"\t{ci_adni.reset_index().to_numpy()}")
            fi.write(f"{p_adni.reset_index().to_numpy()}")
            fi.write(f"\tSurvival estimate for NACC: \n")
            fi.write(f"{ci_nacc.reset_index().to_numpy()}")
            fi.write(f"{p_nacc.reset_index().to_numpy()}")

        with open(self.stats_output, "a") as fi:
            fi.write(
                "-----------------\nComparing NACC and ADNI at different " "times\n"
            )
            for t in time:
                sA_t = self._adni_mdl.predict(t)

                sB_t = self._nacc_mdl.predict(t)

                sB_t = self._nacc_mdl.survival_function_at_times(t)

                results = surv_fixed_point(
                    t,
                    self._adni_mdl,
                    self._nacc_mdl,
                    point_estimate=f"log(-log(x)) difference " + f"ADNI-NACC: "
                    f"{clog(sB_t)-clog(sA_t)}",
                )
                fi.write(f"\t{t} months: \n")
                fi.write(results.to_ascii() + "\n")

    def generate_kaplan_meier(self):
        fig, ax = plt.subplots()
        self.km_dataset = {}
        kmf = {}
        at_risk_array = []
        time_list = (
            0,
            24,
            48,
            96,
            np.inf,
        )
        query_times = list(range(0, 120, 12))
        query_times[-1] = np.inf
        print(query_times)
        for grp, tbl in self.merged_df.groupby("Dataset"):
            kmf[grp] = kaplan_meier_estimator(tbl, grp)
            self.km_dataset[grp] = tbl.copy()[["TIMES", "PROGRESSES"]]
            ax = kmf[grp].plot_survival_function(
                ax=ax, ci_alpha=0.2, at_risk_counts=True
            )
            new_df = at_risk_row(kmf[grp].event_table, query_times)
            new_df["Dataset"] = grp
            at_risk_array.append(new_df)
        at_risk_df = pd.concat(at_risk_array, ignore_index=True, axis=0)
        dump_at_risk_df(at_risk_df)
        with open(self.stats_output, "a") as fi:
            self.dataset_kms = KaplanMeierStatistics(
                self.km_dataset, "Datasets", key_1="ADNI", key_2="NACC"
            )
            fi.write(str(self.dataset_kms))
        ax.set_xlabel("Time from MCI visit (months)")
        ax.set_ylabel("Survival rate")
        plt.savefig(f"figures/figure_1/baseline_kaplan_meier.svg", dpi=300)
        plt.savefig(f"figures/figure_1/baseline_kaplan_meier.png", dpi=300)
        plt.close()

    def get_kruskal_wallis(self, col_list):
        with open(self.stats_output, "a") as fi:
            for dataset, df in self.merged_df.groupby("Dataset"):
                for col in col_list:
                    grouper = "Time to progress to AD"
                    unique_vals = pd.unique(df[grouper])
                    x = [
                        df.loc[df[grouper] == _x, col].to_numpy() for _x in unique_vals
                    ]
                    kws = KruskalWallis(x, col)
                    fi.write(f"------------\n{dataset}, {col}:\n")
                    fi.write(str(kws))
                    # dunn post hoc
                    df = df.copy()
                    sub_df = df.loc[df[col].notna(), :].copy()
                    p = posthoc_dunn(
                        sub_df, val_col=col, group_col=grouper, p_adjust="bonferroni"
                    )
                    sub_df["ranks"] = sub_df[col].rank()
                    mean_ranks = sub_df.groupby(grouper)["ranks"].mean()
                    # https://github.com/maximtrp/scikit-posthocs/blob
                    # /6c2bd5c17710cf9d4a2d55ba0c36b41132f20e79/scikit_posthocs/_posthocs.py
                    p_dunn = _format_dunn_p(p)
                    fi.write("\n\t Dunn test w/ bonferroni correction\n")
                    fi.write(f"\tMean ranks: {mean_ranks.to_string()}\n")
                    [
                        fi.write("\t {}\n".format(p_dunn[i, :]))
                        for i in range(p_dunn.shape[0])
                    ]
                    p_bh = posthoc_dunn(
                        sub_df, val_col=col, group_col=grouper, p_adjust="fdr_bh"
                    )
                    p_bh = _format_dunn_p(p_bh)
                    fi.write("\n\t Dunn test w/ BH correction\n")
                    fi.write(f"\tMean ranks: {mean_ranks.to_string()}\n")
                    [
                        fi.write("\t {}\n".format(p_bh[i, :]))
                        for i in range(p_bh.shape[0])
                    ]

    def get_pooled_mwu(self, col_list):
        with open(self.stats_output, "a") as fi:
            for col in col_list:
                datasets = []
                labels = []
                for label, df in self.merged_df.groupby("Dataset"):
                    datasets.append(df[col].to_numpy())
                    labels.append(label)
                assert len(datasets) == 2
                mwu = MannWhitneyU(*datasets, *labels)
                fi.write(
                    f"------------\nPooled MWU for {col}, "
                    f"{labels[0]}x{labels[1]}\n-----------\n"
                )
                fi.write(str(mwu))

    def print_lm(self, columns):
        for col in columns:
            gl = ChiSquare(self.df_crosstab[col])
            with open(self.stats_output, "a") as fi:
                fi.write(str(gl))

    def retrieve_all_stats(self):
        with open(self.stats_output, "w") as fi:
            fi.write("Statistics\n")
        self.generate_stacked_bar()
        self.generate_violin_plots()
        self.generate_kaplan_meier()
        self.compare_surv_different_times([24, 48])
        self.get_kruskal_wallis(
            [
                "MMSE",
                "Age (yrs)",
                "Aβ42 (pg/mL)",
                "t-tau (pg/mL)",
                "p-tau (pg/mL)",
                "Education (yrs)",
            ]
        )
        self.get_pooled_mwu(["MMSE", "Age (yrs)", "Education (yrs)"])
        self.print_lm(["Time to progress to AD", "#APOE \u03B54 Alleles", "Sex"])
        self._train_stats.write_summary_stats()
        self._test_stats.write_summary_stats()


def _find_csf_assay_type(tbl) -> dict:
    csf_codes = CONFIG_["NACC_config"]["csf_assay_md"]
    # "1": "ELISA",
    # "2": "LUMINEX",
    # "8": "Other"
    csf_codes["-1"] = "Missing"
    val_counts = {col: dict() for col in CONFIG_["NACC_config"]["csf_cols"]}
    for col in CONFIG_["NACC_config"]["csf_cols"]:
        counts = tbl[col].value_counts(dropna=False)
        for key, value in counts.to_dict().items():
            if np.isnan(key):
                key = -1
            val_counts[col][csf_codes[str(int(key))]] = value
    return val_counts


def make_struc_array(hits, obss):
    return np.array(
        [(x, y) for x, y in zip(hits == 1, obss)],
        dtype=[("hit", bool), ("time", float)],
    )


def _format_dunn_p(p_df):
    # help from https://stackoverflow.com/questions/34417685/melt-the-upper-triangular-matrix-of-a-pandas-dataframe
    df = p_df.where(np.triu(np.ones(p_df.shape), k=1) == 1)
    df = df.stack().reset_index()
    return df.to_numpy()


def _get_melted_df(df, continuous_vars, id_variables):
    df = df[continuous_vars + list(id_variables)].copy()
    df = pd.melt(
        df,
        id_vars=id_variables,
        value_name="Value",
        var_name="Test",
        value_vars=continuous_vars,
    )
    train_index = df.query("Dataset == 'ADNI'").index
    test_index = df.query("Dataset == 'NACC'").index
    df.loc[:, "Values_Train"] = np.nan
    df.loc[:, "Values_Test"] = np.nan
    df.loc[test_index, "Values_Test"] = df.loc[test_index, "Value"]
    df.loc[train_index, "Values_Train"] = df.loc[train_index, "Value"]
    return df


def _merge_df(train, test):
    train_df = train.df.copy().rename(columns=train.column_maps)
    train_df.loc[:, "Dataset"] = "ADNI"
    test_df = test.df.copy().rename(columns=test.column_maps)
    test_df.loc[:, "Dataset"] = "NACC"
    new_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    new_df["Dataset"] = (
        new_df["Dataset"].astype("category").cat.reorder_categories(["ADNI", "NACC"])
    )
    return new_df


class EventSchema(pa.SchemaModel):
    at_risk: pa.typing.Series[float]
    removed: pa.typing.Series[float]
    observed: pa.typing.Series[float]
    censored: pa.typing.Series[float]
    entrance: pa.typing.Series[float]
    event_at: pa.typing.Index[float]


def at_risk_row(event_table: EventSchema, time_list: list) -> pd.DataFrame:
    print(event_table)
    obs = event_table["observed"].copy()
    cens = event_table["censored"].copy()
    event_table = event_table["at_risk"].copy()
    event_dict = {"At Risk": [], "Times": [], "Observed": [], "Censored": []}
    for t in range(len(time_list) - 1):
        # following lines with help from /lifelines/blob/master/lifelines/plotting.py
        start = time_list[t]
        stop = time_list[t + 1]
        sub_tbl = event_table.loc[start:stop].copy()
        max_val = sub_tbl.agg(max)
        obs_curr = obs.loc[start : (stop - 1 / 365.25)].sum()
        cens_curr = cens.loc[start : (stop - 1 / 365.25)].sum()
        event_dict["At Risk"].append(max_val)
        event_dict["Times"].append(start)
        event_dict["Observed"].append(obs_curr)
        event_dict["Censored"].append(cens_curr)
    event_series = pd.DataFrame(event_dict)
    return event_series


def dump_at_risk_df(df: pd.DataFrame):
    df = df.copy()
    df["Times"] = df["Times"].apply(lambda x: str(int(x)))
    df = df.melt(id_vars=["Dataset", "Times"], value_name="Count", var_name="Type")
    df.set_index(["Dataset", "Type"], inplace=True, drop=True)
    df = df.pivot(
        columns="Times",
    )
    df.to_csv("results/survival_data_whole_cohort.csv")


def main():
    tats = TrainAndTestStats()
    tats.generate_violin_plots()
    tats.generate_stacked_bar()
    tats.generate_kaplan_meier()
    tats.retrieve_all_stats()
    dict_ = _find_csf_assay_type(tats._test_stats.df.copy())
    return dict_, tats._test_stats


def kaplan_meier_test():
    tats = TrainAndTestStats()
    tats.generate_kaplan_meier()
    tats.compare_surv_different_times([24, 48, 96])

if __name__ == "__main__":
    main()
