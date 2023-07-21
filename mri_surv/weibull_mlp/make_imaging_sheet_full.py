import os
from typing import Dict, Tuple
import multiprocessing as mp
import pandas as pd
import json
from process_parcellations.cat12reader import Cat12Reader
from process_parcellations.make_imaging_sheet import uniq_rids

with open("process_parcellations/parcellation_config.json") as fi:
    CONFIG = json.loads(fi.read())


def process_subj_uncorr(
    rid: int, path: str, statspath: str, total_subjs: int, pos: int
) -> Dict:
    """Process the Cat12 output for a single subject

    Extract cortical thickness and ROI-based volumetrics for an individual subject.

    Args:
        rid (int): ADNI RID
        path (str): base path for Cat12 output
        statspath (str): base path for Cat12 statistics
        total_subjs (int): total number of subjects to process
        conversion_sheet (pd.DataFrame): dataframe containing conversion info
        pos (int): index of subject in processing pipeline

    Returns:
        Dict: dictionary containing RID, Modality, total intracranial volume,
        total surface area, conversion to AD, and all volumetric/cortical thickness data
    """
    print(f"[{pos+1}/{total_subjs}]")

    xmlreader = Cat12Reader()

    # TIV and TSA
    xmlreader.filename = (
        f"{statspath}cat_{rid}_mri.xml"
        if "mri" not in rid
        else f"{statspath}cat_{rid}.xml"
    )
    tiv = xmlreader.parseImageStats("vol_TIV")

    # subcortical volumes
    xmlreader.filename = (
        f"{path}/catROI_{rid}_mri.xml"
        if "mri" not in rid
        else f"{path}/catROI_{rid}.xml"
    )

    subcort = xmlreader.parseXML("neuromorphometrics", "Vgm")
    labels, data = zip(*subcort)
    labels = [f"vol_{lab}" for lab in labels]
    subcort = dict(zip(labels, data))

    merged_data = {"RID": rid, "TIV": tiv, **subcort}
    return merged_data


def retrieve_dataframe_for_datadir_uncorr(
    datadir: str, statspath: str, df_re_strings: str
) -> pd.DataFrame:
    """
    Given a directory containing parcellation data in xml format,
    a path for statistics (GMVs and TIVs), and a regular expression to parse the files,
    return a dataframe containing GMVs, TIVs

    Args:
        datadir (str): _description_
        statspath (str): _description_
        df_re_strings (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    for path, _, files in os.walk(datadir):
        abs_paths = [f"{path}/{f}" for f in files if f.endswith(".xml")]
        rids = uniq_rids(abs_paths, df_re_strings)
        for rid in CONFIG["rids_to_drop"]:
            if rid in rids:
                rids.remove(rid)
        pool = mp.Pool(processes=15)
        # process subjects in PARALLEL
        results = [
            pool.apply_async(
                process_subj_uncorr, args=(subj, path, statspath, len(rids), ind)
            )
            for ind, subj in enumerate(rids)
        ]
        merged_data = [p.get() for p in results]
        df = pd.DataFrame(merged_data)
        df = df.set_index("RID")
        df = df.sort_index()
    return df


def retrieve_subcort_volumes_from_suffix_lateral(
    suffix: str, df_re_string: str
) -> pd.DataFrame:
    """
    With an dataset specifier suffix and a regular expression to use to search, find 
    the parcellation file for a given patient and retrieve parcellated GMVs and TIV

    Args:
        suffix (str): dataset specifier
        df_re_string (str): regex to use to identify files for each patient, given the dataset

    Returns:
        pd.DataFrame: dataframe with TIV and GMV information for each RID in 
    """
    basedir = CONFIG["basedir"] + suffix + CONFIG["parcellation_path"] + suffix
    datadir = basedir + CONFIG["datadir"] + "/"
    statspath = basedir + CONFIG["statspath"] + "/"
    df = retrieve_dataframe_for_datadir_uncorr(datadir, statspath, df_re_string)
    return df


def main_alt(dev=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Acquires GMVs for all of the neuromorphometrics brain regions for all of the patients
    with processed scans via CAT12. Also retrieves their TIV.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    df_subcort_all = []
    df_subcort_labels = ["ADNI", "NACC"]
    df_re_strings = [
        r".*(?P<name>[0-9]{4}).*\.xml$",  # regex for adni patients (4 digits)
        r".*_(?P<name>NACC[0-9]+)_.*\.xml$",  # regex for nacc patients (NACC and then numbers)
    ]
    for i, suffix in enumerate(["cox_noqc", "cox_test"]):  # for adni (noqc) and test
        df_subcort = retrieve_subcort_volumes_from_suffix_lateral(
            suffix, df_re_strings[i]
        )
        df_subcort["Dataset"] = df_subcort_labels[i]
        df_subcort_all += [df_subcort]
    df_subcort_all = pd.concat(df_subcort_all, axis=0)
    if dev:
        return df_subcort_all
    df_subcort_all.to_csv(
        "./metadata/data_processed/parcellation_volumes_raw.csv", index_label="RID"
    )


if __name__ == "__main__":
    main_alt()
