# Author: Akshara Balachandra / Michael Romano
# Date: 02-19-23
# Description: Revisions #2

import os
from typing import Dict, Tuple
import multiprocessing as mp
import pandas as pd
import json
from process_parcellations.cat12reader import Cat12Reader
from process_parcellations.make_imaging_sheet import uniq_rids


with open("process_parcellations/parcellation_config.json") as fi:
    CONFIG = json.loads(fi.read())

def process_subj(rid: int, path: str, statspath: str, total_subjs: int, pos: int) -> Dict:
    """Process the Cat12 output for a single subject

    Extract ROI-based volumetrics for an individual subject.

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

    # TIV
    xmlreader.filename = f'{statspath}cat_{rid}_mri.xml' if 'mri' not in rid else f'{statspath}cat_{rid}.xml'
    tiv = xmlreader.parseImageStats('vol_TIV')

    # subcortical volumes
    xmlreader.filename = f'{path}/catROI_{rid}_mri.xml' if 'mri' not in rid else f'{path}/catROI_{rid}.xml'

    subcort = xmlreader.parseXML('neuromorphometrics', 'Vgm')
    gmv_labels, gmv_data = zip(*subcort)
    gmv_labels = [f'vol_{lab}' for lab in gmv_labels]
    gmv_vol = dict(zip(gmv_labels, gmv_data))

    corr_vollabs = [f'corr_{lab}' for lab in gmv_labels]
    corr_vol = [float(val/tiv) for val in gmv_data]
    corr_vol = dict(zip(corr_vollabs, corr_vol))

    csf = xmlreader.parseXML('neuromorphometrics', 'Vcsf')
    csf_labels, csf_data = zip(*csf)
    csf_labels = [f'csf_{lab}' for lab in csf_labels]
    csf_vol = dict(zip(csf_labels, csf_data))

    corr_csflabs = [f'corr_{lab}' for lab in csf_labels]
    corr_csf = [float(val/tiv) for val in csf_data]
    corr_csf = dict(zip(corr_csflabs, corr_csf))

    merged_data = {'RID': rid, 'TIV': tiv,
                   **corr_vol, **corr_csf, **gmv_vol, **csf_vol}
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
    df_subcort_all.to_csv("./metadata/data_processed/mri3_cat12_combined_csf_gmv.csv", index_label='RID')
    return df_subcort_all

if __name__ == "__main__":
    main()