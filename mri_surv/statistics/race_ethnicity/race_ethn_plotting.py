""""""

import os

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from statistics.race_ethnicity.race_ethn_metadata import OUTPUT_FNAME
from statistics.demographic_statistics import _DemographicStatistics
from statistics.race_ethnicity.race_ethn_metadata import *


plt.style.use("./statistics/race_ethnicity/ethstyle.mplstyle")


def load_race_ethn_df() -> pd.DataFrame:
    if not os.path.isfile(OUTPUT_FNAME):
        generate_race_ethn_df()
    return pd.read_pickle(OUTPUT_FNAME)


def _summarize_df(df: pd.DataFrame) -> None:
    df.groupby("Dataset").apply(lambda x: x.describe)


def _cross_tabulate_columns(merged_df: pd.DataFrame) -> dict:
    """
    cross_tabulate_columns

    Gets counts for Race and Ethnicity

    It is worth noting that this function replaces a value of "Unknown"
        for "MultipleValues"

    Parameters
    ----------
    merged_df : pd.DataFrame
        output of load_race_ethn_df

    Returns
    -------
    dict[Dataset: pd.DataFrame]
        cross-tabulated dataframes for race and ethnicity
    """
    df_crosstab = {}
    merged_df["RACE"] = merged_df["RACE"].replace({"MultipleValues": "Unknown"})
    merged_df["ETHNICITY"] = merged_df["ETHNICITY"].replace(
        {"MultipleValues": "Unknown"}
    )
    for col in (
        "RACE",
        "ETHNICITY",
    ):
        df_crosstab[col] = pd.crosstab(
            merged_df["Dataset"], merged_df[col], margins=True
        )
    return df_crosstab


def _stats(df_crosstab: pd.DataFrame) -> None:
    """
    Retrieves Chi2 statistics from ethnicity and race dataframes, comparing
    "white" and "non-white" or hispanic and non-hispanic.

    Of note,

    Args:
        df_crosstab (pd.DataFrame): output of _cross_tabulate_columns
    """
    for key, df in df_crosstab.items():
        df: pd.Series = df.copy()
        if key == "RACE":
            df.drop(columns=["Unknown", "MoreThanOneRace", "All"], inplace=True)
            df.drop(index="All", inplace=True)
            non_white = [
                "AmericanIndianOrAlaskanNative",
                "Asian",
                "BlackOrAfricanAmerican",
                "NativeHawaiianOrPacificIslander",
            ]
            non_white_map = {x: "NonWhite" for x in non_white}
            df.rename(columns=non_white_map, inplace=True)
            df = df.groupby(level=0, axis=1).agg(sum)
            chi2, p, dof, _ = chi2_contingency(df)
            n = df.sum(axis=1)
            with open("results/ethn_race_summary_stats.txt", "a") as fi:
                fi.write("Nonwhite proportion between cohorts: \n\n")
                fi.write(f"Chi2={chi2},p={p},dof={dof},n={n.to_numpy()}")
                fi.write("\n" + "-" * 20 + "\n")
        elif key == "ETHNICITY":
            df.drop(columns=["Unknown", "All"], inplace=True)
            df.drop(index="All", inplace=True)
            df = df.groupby(level=0, axis=1).agg(sum)
            chi2, p, dof, _ = chi2_contingency(df)
            n = df.sum(axis=1)
            with open("results/ethn_race_summary_stats.txt", "a") as fi:
                fi.write("HispanicOrLatino proportion between cohorts: \n\n")
                fi.write(f"Chi2={chi2},p={p},dof={dof},n={n.to_numpy()}")
                fi.write("\n" + "-" * 20 + "\n")


def generate_pi_chart(crosstab: dict) -> None:
    """
    generate_pi_chart

    Creates pie charts for race and ethnicity

    Parameters
    ----------
    crosstab : dict
        output of cross_tabulate_columns

    Returns
    -------
    None
    """
    columns = (
        "RACE",
        "ETHNICITY",
    )
    rows = (
        "ADNI",
        "NACC",
    )

    def make_labels(co):
        labs = []
        for row, n in co.iterrows():
            labs.append(f"{row}, n=" + str(n.values[0]))
        return labs

    for col in columns:
        curr_df = crosstab[col].T
        curr_df = curr_df.drop(index="All", columns="All")
        for row in rows:
            curr_df.loc[:, row].plot.pie(
                subplots=True,
                labels=make_labels(curr_df[[row]]),
                title=row,
                labeldistance=None,
                legend=True,
            )
            plt.savefig(
                f"figures/figure_1/statistics_pichart_" f"{col}_{row}" + ".svg", dpi=300
            )
            plt.savefig(
                f"figures/figure_1/statistics_pichart_" f"{col}_{row}" + ".png", dpi=300
            )
            plt.close()


def load_demographic_sheet(dataset: str) -> pd.DataFrame:
    """
    load_demographic_sheet

    Loads demographics dataframe containing RID, RACE, and ETHNICITY info

    Parameters
    ----------
    dataset : str
        'NACC' or 'ADNI'

    Returns
    -------
    pd.DataFrame
        Dataframe with respective columns for dataset, ethn, and race
    """
    fi = DIR[dataset]
    df = pd.read_csv(
        fi, usecols=list(COL_NAMES[dataset].keys()), dtype=COL_NAMES[dataset]
    )
    df.rename(columns=COL_NAMES_MAP[dataset], inplace=True)
    if dataset == "ADNI":
        df["RID"] = df["RID"].apply(lambda x: str(x).zfill(4))
    return df


def _load_and_recode_ethnicity_race(dataset: str) -> pd.DataFrame:
    """
    _load_and_recode_ethnicity_race

    Replaces numeric values for Race and Ethnicity with their true values
    conditioned on dataset

    Parameters
    ----------
    dataset : str
        either 'NACC' or 'ADNI'

    Returns
    -------
    pd.DataFrame
        dataframe with RID, RACE and ETHNICITY encoded
    """
    df = load_demographic_sheet(dataset)
    df["RACE"] = df["RACE"].replace(RACE_MAP[dataset])
    df["ETHNICITY"] = df["ETHNICITY"].replace(ETHN_MAP[dataset])
    df = (
        df.groupby("RID").agg(lambda x: list(pd.unique(x[~pd.isna(x)]))).reset_index()
    )  # RACE and ETHNICITY may be recorded multiple times
    return df


def _merge_ethnicity_and_race(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    merge_ethnicity_and_race

    takes a demographics dataframe and adds ethnicity and race info

    Parameters
    ----------
    df : pd.DataFrame
        df attribute of _DemographicStatistics
    dataset : str
        either "NACC" or "ADNI"

    Returns
    -------
    pd.DataFrame
        dataframe df with ethnicity and race added
    """
    df = df.set_index("RID", drop=True)  # index by RID instead of arbitrary
    df = df[["PROGRESSION_CATEGORY"]].copy()
    ds = _load_and_recode_ethnicity_race(dataset)
    ds = ds.set_index("RID", drop=True)
    df = pd.merge(df, ds, how="left", left_index=True, right_index=True)
    return df

def _merge_covariates(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    _merge_covariates

    takes a demographics dataframe and adds ethnicity and race info to other covariates
        age, mmse, educational status, APOE status, sex

    Parameters
    ----------
    df : pd.DataFrame
        df attribute of _DemographicStatistics
    dataset : str
        either "NACC" or "ADNI"

    Returns
    -------
    pd.DataFrame
        dataframe df with ethnicity and race added
    """
    df = df.set_index("RID", drop=True)  # index by RID instead of arbitrary
    if dataset == "ADNI":
        df = df[["PROGRESSION_CATEGORY", "TIMES", "PROGRESSES", "MMSCORE_mmse", "AGE", "PTEDUCAT_demo", "PTGENDER_demo", "APOE"]].copy()
        df = df.rename(
            columns={
                "MMSCORE_mmse": "MMSE",
                "PTEDUCAT_demo": "EDUC",
                "PTGENDER_demo": "SEX"
            }
        )
    else:
        df = df[["PROGRESSION_CATEGORY", "TIMES", "PROGRESSES", "MMSE", "AGE", "EDUC", "SEX", "APOE"]].copy()
    ds = _load_and_recode_ethnicity_race(dataset)
    ds = ds.set_index("RID", drop=True)
    df = pd.merge(df, ds, how="left", left_index=True, right_index=True)
    return df

def _drop_multi_ethn_race(df: pd.DataFrame):
    """
    drop_multi_ethn_race

    Takes a dataframe (df field from _DemographicStatistics),
        merged with race and ethnicity, and replaces RIDs with
        multiple values with MultipleValues. It is important to note that
        MultipleValues is different than MoreThanOneRace; MultipleValues

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    """

    def drop_multi_ethn_race_row(se: pd.Series):
        for col in (
            "RACE",
            "ETHNICITY",
        ):
            unique_vals = se[col].values[0]
            if len(unique_vals) > 1:
                print(f"dropping {unique_vals} from {se.index}")
                unique_vals = "MultipleValues"
            else:
                unique_vals = unique_vals[0]
            se[col] = unique_vals
        return se

    return df.copy().groupby("RID").apply(drop_multi_ethn_race_row)


def _merge_df(adni_df: pd.DataFrame, nacc_df: pd.DataFrame) -> pd.DataFrame:
    """
    merge_df

    Merges output of _drop_multi_ethn_race, performed for NACC and ADNI

    Parameters
    ----------
    adni_df : pd.DataFrame

    nacc_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Merged dataframe with new column "Dataset"
    """
    adni_df = adni_df.copy()
    nacc_df = nacc_df.copy()
    adni_df.loc[:, "Dataset"] = "ADNI"
    nacc_df.loc[:, "Dataset"] = "NACC"
    new_df = pd.concat([adni_df, nacc_df], axis=0, ignore_index=False)
    new_df["Dataset"] = (
        new_df["Dataset"].astype("category").cat.reorder_categories(["ADNI", "NACC"])
    )
    new_df["RACE"] = new_df["RACE"].astype("category")
    new_df["ETHNICITY"] = new_df["ETHNICITY"].astype("category")
    return new_df


def generate_race_ethn_df() -> None:
    adni = _DemographicStatistics("ADNI")
    nacc = _DemographicStatistics("NACC")

    adni_merged = _merge_ethnicity_and_race(adni.df, "ADNI")
    nacc_merged = _merge_ethnicity_and_race(nacc.df, "NACC")

    adni = _drop_multi_ethn_race(adni_merged)
    nacc = _drop_multi_ethn_race(nacc_merged)

    df = _merge_df(adni, nacc)
    df.to_pickle(OUTPUT_FNAME)

def generate_covariate_df() -> None:
    adni = _DemographicStatistics("ADNI")
    nacc = _DemographicStatistics("NACC")

    adni_merged = _merge_covariates(adni.df, "ADNI")
    nacc_merged = _merge_covariates(nacc.df, "NACC")

    adni = _drop_multi_ethn_race(adni_merged)
    nacc = _drop_multi_ethn_race(nacc_merged)

    df = _merge_df(adni, nacc)
    df.to_csv(OUTPUT_FNAME[:-4] + ".csv")


def main():
    df = load_race_ethn_df()
    generate_covariate_df()
    df = _cross_tabulate_columns(df)
    generate_pi_chart(df)
    _stats(df)
