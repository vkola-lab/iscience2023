import random
import argparse
import os

import pandas as pd

from preprocessing.cohorts import ADNICollection, NaccCollection
from preprocessing.move_nii_files import (
    find_and_move_mri,
    find_and_move_mri_ad,
    find_and_move_unused_mri,
)
from preprocessing import move_nacc_files
from process_parcellations import make_imaging_sheet
from statistics import (
    survival_analysis,
    mlp_analysis,
    clustering,
    demographic_statistics,
    brain_visualization,
    shap_plots,
    plotting,
    pathology_revision,
)

from statistics.other_dementia import reversion_stats, reversion_predict
from statistics.race_ethnicity import race_ethn_plotting
from statistics.radiology_analysis import radiology_analysis
from statistics.shap_viz import brain_viz_shap
from weibull_mlp import make_imaging_sheet_full
from weibull_mlp.run import train_weibull
from simple_mlps import mlp_wrappers

######################################


def create_csv_time(suffix="", CollectionClass=ADNICollection) -> pd.DataFrame:
    """
    creates initial csv file with merged data, curated for MCI visits with CSF and MRI
    and also with time to progression
    """
    _p = CollectionClass()
    _table = _p.get_progression_data_time_to_progress_ad()
    for col in _table.columns:
        if any([type(x) == list for x in _table[col]]):
            _table.loc[:, col] = _p.stringify(_table[col])
    _table.to_csv("metadata/data_processed" "/merged_dataframe_cox" + suffix + ".csv")
    return _table


def create_csv_time_unused(suffix="") -> None:
    _p = ADNICollection()
    _table = _p.tbl_merged.copy()
    _table = _table[
        [
            "RID",
            "DX",
            "EXAMDATE",
            "VISCODE2",
            "VISCODE3",
            "Phase",
            "EXAMDATE_mri3",
            "PTGENDER_demo",
            "AGE",
            "MMSCORE_mmse",
            "abeta",
            "tau",
            "ptau",
        ]
    ]
    not_nan_inds = [
        not pd.isna(x) if type(x) is not list else True for x in _table.EXAMDATE_mri3
    ]
    _table = _table.loc[not_nan_inds, :]
    for col in _table.columns:
        if any([type(x) == list for x in _table[col]]):
            _table.loc[:, col] = _p.stringify(_table[col])
    _table.set_index("RID", drop=True, inplace=True)
    _table.to_csv(
        "metadata/data_processed" "/merged_dataframe_all_cox" + suffix + ".csv",
        index_label="RID",
    )


def consolidate_images_noqc(move=True) -> None:
    find_and_move_mri(move=move)


def consolidate_dummy_data(move=True) -> None:
    dummy_df = find_and_move_unused_mri(move=move)
    dummy_df.to_csv("metadata/data_processed/merged_dataframe_unused_cox_pruned.csv")


def consolidate_images_ad(move=True) -> None:
    find_and_move_mri_ad(move=move)


def consolidate_images_nacc(move=True) -> None:
    move_nacc_files.find_and_move_mri(move, save=True)

##########################################


class ProcessImagesMRI(object):
    def __init__(self, suffix="cox_noqc") -> None:
        self.prefix = "matlab -nodisplay -r \"addpath(genpath('.'));"
        self.prefix += "addpath(genpath('/usr/local/spm'));"
        self.suffix = suffix
        self.basedir = "/data2/MRI_PET_DATA/processed_images_final_{}/".format(
            self.suffix
        )
        self.realign = self.prefix + "realign_all_niftis('{}'," "'_{}');exit\"".format(
            self.basedir, suffix
        )
        self.process = self.prefix + "process_files_cox('_{}');exit\"".format(suffix)
        self.parcellate = self.prefix + "batch_mrionly_job('_{}');exit\"".format(suffix)

    def __call__(self) -> None:
        os.system(self.realign)
        os.system(self.process)
        os.system(self.parcellate)


class ProcessImagesMRIAD(object):
    def __init__(self, suffix="cox_noqc_AD") -> None:
        self.prefix = "matlab -nodisplay -r \"addpath(genpath('.'));"
        self.prefix += "addpath(genpath('/usr/local/spm'));"
        self.suffix = suffix
        self.basedir = "/data2/MRI_PET_DATA/processed_images_final_{}/".format(
            self.suffix
        )
        self.realign = self.prefix + "realign_all_niftis('{}'," "'_{}');exit\"".format(
            self.basedir, suffix
        )
        self.parcellate = self.prefix + "batch_mrionly_job('_{}');exit\"".format(suffix)

    def __call__(self) -> None:
        os.system(self.realign)
        # os.system(self.process)
        os.system(self.parcellate)


class ProcessImagesMRIDummy(object):
    def __init__(self, suffix="unused_cox") -> None:
        self.prefix = "matlab -nodisplay -r \"addpath(genpath('.'));"
        self.prefix += "addpath(genpath('/usr/local/spm'));"
        self.suffix = suffix
        self.basedir = "/data2/MRI_PET_DATA/processed_images_final_{}/".format(
            self.suffix
        )
        self.realign = (
            self.prefix + "realign_all_niftis_unused('{}',"
            "'_{}');exit\"".format(self.basedir, suffix)
        )
        self.process = self.prefix + "process_files_cox_unused('_{}');exit\"".format(
            suffix
        )
        self.parcellate = self.prefix + "batch_mrionly_unused_job('_{}');exit\"".format(
            suffix
        )
        self.parcellate_errors = (
            self.prefix + "batch_mrionly_job_errors('_{}');exit\"".format(suffix)
        )

    def move_nii(self) -> None:
        orig_dir = self.basedir + "ADNI_MRI_nii_recenter_" + self.suffix + "/"
        new_dir = self.basedir + "ADNI_MRI_nii_recenter_NL_" + self.suffix
        os.makedirs(new_dir, exist_ok=True)
        dat = pd.read_csv(
            "metadata/data_processed/merged_dataframe_unused_cox_pruned.csv",
            dtype={"RID": str},
        )
        for _, row in dat.iterrows():
            if row.DX == "NL":
                os.system(f"rsync -v {orig_dir}{row.FILE_CODE}.nii {new_dir}/")

    def process_errors(self) -> None:
        os.system(self.parcellate_errors)

    def __call__(self) -> None:
        os.system(self.realign)
        os.system(self.process)
        self.move_nii()
        os.system(self.parcellate)


class ProcessImagesMRINacc(object):
    def __init__(self, suffix="cox_test") -> None:
        self.prefix = "matlab -nodisplay -r \"addpath(genpath('.'));"
        self.prefix += "addpath(genpath('/usr/local/spm'));"
        self.suffix = suffix
        self.basedir = "/data2/MRI_PET_DATA/processed_images_final_{}/".format(
            self.suffix
        )
        self.realign = (
            self.prefix + "realign_all_niftis_nacc('{}',"
            "'_{}');exit\"".format(self.basedir, suffix)
        )
        self.process = (
            self.prefix
            + "process_files_cox('_{}', '^(NACC[0-9]+).*\.nii$',1);exit\"".format(
                suffix
            )
        )
        self.parcellate = self.prefix + "batch_mrionly_job_nacc('_{}');exit\"".format(
            suffix
        )

    def __call__(self) -> None:
        os.system(self.realign)
        os.system(self.process)
        os.system(self.parcellate)


def make_imaging_sheet_main() -> None:
    make_imaging_sheet.main()


def dem_stats() -> None:
    demographic_statistics.main()


def cluster() -> None:
    clustering.main()


def survival_for_subtype() -> None:
    survival_analysis.main()


def mlp_plots_and_stats() -> None:
    mlp_analysis.main()

def make_stats_and_plots(_plot=False) -> None:
    make_imaging_sheet.main()
    demographic_statistics.main()
    clustering.main()
    survival_analysis.main()
    mlp_analysis.main()
    if _plot:
        brain_visualization.main()
        shap_plots.main()
        plotting.main()

def dump_reversion_stats() -> None:
    reversion_stats.main()
    reversion_predict.main()


def dump_race_ethn_stats() -> None:
    race_ethn_plotting.main()

def radiologist_stats() -> None:
    radiology_analysis.main()

def make_path_stats() -> None:
    pathology_revision.main()

def shap_brains() -> None:
    brain_viz_shap.main()

def shap_stats_post_r() -> None:
    brain_viz_shap.stats_post_r()

def make_imaging_sheet_weibull() -> None:
    make_imaging_sheet_full.main_alt()

def train_weibull_mlp() -> None:
    train_weibull()

def train_gmv_and_csf_mlp() -> None:
    mlp_wrappers.gmv_csf_sur_loss()

if __name__ == "__main__":
    random.seed(10)
    parser = argparse.ArgumentParser("Enter in the time frame in y")
    parser.add_argument("--makecsv", dest="csv", default=0, type=bool)
    parser.add_argument("--moverawimages", dest="moveraw", default=0, type=bool)
    parser.add_argument("--extractimg", dest="extractimg", default=0, type=bool)
    parser.add_argument("--plot_examples", dest="plot_examples", default=0, type=bool)
    parser.add_argument("--stats", dest="stats", default=0, type=bool)
    parser.add_argument("--process_image", dest="process_image", default=0, type=bool)
    parser.add_argument("--test", dest="test", default=0, type=bool)
    parser.add_argument("--run_mlp", dest="run_mlp", default=0, type=bool)
    parser.add_argument("--reversions", dest="reversions", default=0, type=bool)
    args = parser.parse_args()

    if args.csv == 1:
        if args.test == 1:
            SUFFIX = "_test"
            _class = NaccCollection
        else:
            _class = ADNICollection
            SUFFIX = "_noqc"
        table = create_csv_time(SUFFIX, _class)
    if args.extractimg == 1:
        if args.test == 1:
            consolidate_images_nacc(args.moveraw == 1)
        else:
            consolidate_images_noqc(args.moveraw == 1)

    if args.process_image == 1:
        SUFFIX = "cox"
        if args.test == 1:
            SUFFIX += "_test"
            p = ProcessImagesMRINacc(SUFFIX)
        else:
            SUFFIX += "_noqc"
            p = ProcessImagesMRI(SUFFIX)
        p()

    if args.reversions == 1:
        dump_reversion_stats()
