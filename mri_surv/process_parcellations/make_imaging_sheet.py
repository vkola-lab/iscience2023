# Author: Akshara Balachandra
# Date: Saturday 21 Nov 2020
# Description:

import os
from typing import Dict, Tuple, NewType
import re
import multiprocessing as mp
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from process_parcellations.cat12reader import Cat12Reader
from icecream import ic

VERBOSE = False

with open("process_parcellations/parcellation_config.json") as fi:
    CONFIG = json.loads(fi.read())


def uniq_rids(paths: list, re_str: str=r'.*(?P<name>[0-9]{4}).*\.xml$') -> list:
    regex = re.compile(re_str)
    matches = [regex.match(p).group('name') for p in paths]
    return list(set(matches))

def _read_csv_parcellations(parcellation_tbl):
    col_regex = re.compile(r'corr_vol_.*')
    valid_columns = [col for col in parcellation_tbl.columns if
                     col_regex.match(col)]
    replacement_dict = {x: x.replace('corr_vol_','') for x in valid_columns}
    parcellation_tbl = parcellation_tbl[valid_columns].copy()
    parcellation_tbl = parcellation_tbl.rename(columns=replacement_dict)
    return parcellation_tbl

def _read_csf_parcellations(parcellation_tbl):
    col_regex = re.compile(r'corr_csf_vol_.*')
    valid_columns = [col for col in parcellation_tbl.columns if
                     col_regex.match(col)]
    replacement_dict = {x: x.replace('corr_csf_vol_','') for x in
                        valid_columns}
    parcellation_tbl = parcellation_tbl[valid_columns].copy()
    parcellation_tbl = parcellation_tbl.rename(columns=replacement_dict)
    return parcellation_tbl

def _average_hemispheres(dataframe):
    dataframe.rename(lambda x: re.sub('^[rl]{1}', '', x), axis=1, inplace=True)
    dataframe = dataframe.groupby(dataframe.columns, axis=1).agg(np.mean)
    return dataframe

def process_subj(rid: int, path: str, statspath: str, total_subjs: int, pos: int) -> Dict:
    """Process the Cat12 output for a single subject

    Extract cortical thickness and ROI-based volumetrics for and individual subject.

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
    print(f'[{pos+1}/{total_subjs}]')

    xmlreader = Cat12Reader()

    # TIV and TSA
    xmlreader.filename = f'{statspath}cat_{rid}_mri.xml' if 'mri' not in rid else f'{statspath}cat_{rid}.xml'
    tiv = xmlreader.parseImageStats('vol_TIV')
    tsa = xmlreader.parseImageStats('surf_TSA')

    # subcortical volumes
    xmlreader.filename = f'{path}/catROI_{rid}_mri.xml' if 'mri' not in rid else f'{path}/catROI_{rid}.xml'

    subcort = xmlreader.parseXML('neuromorphometrics', 'Vgm')
    labels, data = zip(*subcort)
    labels = [f'vol_{lab}' for lab in labels]
    subcort = dict(zip(labels, data))

    corr_vollabs = [f'corr_{lab}' for lab in labels]
    corr_vol = [float(val/tiv) for val in data]
    corr_vol = dict(zip(corr_vollabs, corr_vol))
    merged_data = {'RID': rid, 'TIV': tiv, 'TSA': tsa,
                   **corr_vol}
    return merged_data

def _rename_progression_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['PROGRESSION_CATEGORY'] = df['PROGRESSION_CATEGORY'].map(
            {int(x): y for x,y in CONFIG['progression_category_map'].items()})
    df['PROGRESSION_CATEGORY'] = df['PROGRESSION_CATEGORY'].fillna('Censored')
    df['PROGRESSION_CATEGORY'] = df['PROGRESSION_CATEGORY'].astype('category')
    if len(df.query('PROGRESSION_CATEGORY == \'Censored\'')) == 0:
        categories = CONFIG['progression_category_order'][:-1]
    else:
        categories = CONFIG['progression_category_order']
    df['PROGRESSION_CATEGORY'] = \
        df['PROGRESSION_CATEGORY'].cat.reorder_categories(categories)
    return df

def format_df_subcort(df_subcort: pd.DataFrame, suffix: str, df_subcort_label: str) -> pd.DataFrame:
    df_metadata = pd.read_csv(CONFIG['input_csv_pref'] + f'{suffix}_pruned_final.csv')
    df_metadata['RID'] = df_metadata['RID'].apply(
            lambda x: str(int(x)).zfill(4) if not type(x) == str else x
    )
    df_metadata.set_index('RID', inplace=True)
    df_subcort = df_subcort.merge(df_metadata[['PROGRESSION_CATEGORY']],
                                  left_index=True,
                                  right_index=True,
                                  validate='one_to_one')
    df_subcort = _rename_progression_categories(df_subcort)
    df_subcort['Dataset'] = df_subcort_label
    return df_subcort

def average_hemisphere_volumes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_subcort = _read_csv_parcellations(df)
    df_subcort = _average_hemispheres(df_subcort)
    df_subcort = df_subcort.sort_index().copy()
    df_csf = _read_csf_parcellations(df)
    df_csf = _average_hemispheres(df_csf)
    df_csf = df_csf.sort_index()
    return df_subcort, df_csf

def retrieve_dataframe_for_datadir(datadir: str, statspath: str, df_re_strings: str) -> pd.DataFrame:
    for path, _, files in os.walk(datadir):
        absPaths = [f'{path}/{f}' for f in files if f.endswith('.xml')]
        rids = uniq_rids(absPaths, df_re_strings)
        for rid in CONFIG['rids_to_drop']:
            if rid in rids:
                rids.remove(rid)
        pool = mp.Pool(processes=15)
        # process subjects in PARALLEL
        results = [pool.apply_async(process_subj,
                                    args=(subj,
                                            path,
                                            statspath,
                                            len(rids),
                                            ind))
                    for ind, subj in enumerate(rids)]
        merged_data = [p.get() for p in results]
        df = pd.DataFrame(merged_data)
        df = df.set_index('RID')
        df = df.sort_index()
    return df

def retrieve_subcort_volumes_from_suffix(suffix: str, df_re_string: str) -> pd.DataFrame:
    basedir = CONFIG['basedir'] + \
            suffix + CONFIG['parcellation_path'] + suffix
    datadir = basedir + CONFIG['datadir'] + '/'
    statspath = basedir + CONFIG['statspath'] + '/'
    df = retrieve_dataframe_for_datadir(datadir, statspath, df_re_string)
    df_subcort, _ = average_hemisphere_volumes(df)
    return df_subcort

def generate_parcellations_for_cn() -> pd.DataFrame:
    df = retrieve_subcort_volumes_from_suffix('unused_cox', r'.*(?P<name>([0-9]{4}_mri_I[0-9]+))\.xml$')
    return df

def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_subcort_all = []
    df_subcort_labels = ['ADNI', 'NACC']
    df_re_strings = [r'.*(?P<name>[0-9]{4}).*\.xml$',
                     r'.*_(?P<name>NACC[0-9]+)_.*\.xml$']
    for i, suffix in enumerate(['cox_noqc','cox_test']):
        df_subcort = retrieve_subcort_volumes_from_suffix(suffix, df_re_strings[i])
        df_subcort_all += [format_df_subcort(df_subcort, suffix,
                                             df_subcort_labels[i])]
    df_subcort_all = pd.concat(df_subcort_all, axis=0)
    df_subcort_all.to_csv(CONFIG['output_csv_combo'], index_label='RID')

    df_ad_visit = retrieve_subcort_volumes_from_suffix('cox_noqc_AD', df_re_strings[0])
    df_ad_visit_formatted = format_df_subcort(df_ad_visit, 'cox_noqc_AD', 'ADNI_AD')
    df_ad_visit_formatted.to_csv(CONFIG['output_csv_adni_ad'], index_label='RID')
    return df_subcort_all, df_ad_visit_formatted

if __name__ == '__main__':
    main()
