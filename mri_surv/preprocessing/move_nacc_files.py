import logging
import os
import re
import json
import nibabel as nib
import numpy as np
import pandas as pd
from subprocess import run
import datetime
import zipfile
import matplotlib.pyplot as plt
import abc
import warnings
from datetime import time
from nilearn.plotting import plot_img
from icecream import ic
from preprocessing.move_nii_files import save_new_file
from dataclasses import dataclass


with open('preprocessing/metadata_files.json') as fi:
    CONFIG = json.loads(fi.read())

hdlr = logging.FileHandler(os.path.join(os.path.abspath('.'),
                                        'logs/consolidate_nifti_nacc.log'))
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)

def warning(_str):
    logger.warning(_str)

ic.configureOutput(outputFunction=warning)

with open('logs/consolidate_nifti_nacc.log','w') as fi:
    fi.write(str(datetime.datetime.now()))

@dataclass
class NaccConfig:
    MRI_DIR = CONFIG['NACC']['MRI_DIR']
    MRI_DIR2 = CONFIG['NACC']['MRI_DIR2']
    TARGET_PATH = CONFIG['NACC']['TARGET_PATH']
    TMP_PATH = CONFIG['NACC']['TMP_PATH']
    os.makedirs(TARGET_PATH, exist_ok=True)
    json_regex = re.compile(r'^.*\.json$')
    mprage_regex = re.compile(r'(?i).*mp[-]{0,1}rage.*')
    spgr_regex = re.compile(r'(?i).*spgr.*')
    unzipped_files_dataframe = CONFIG['NACC']['unzipped_df']
    _mpr_regex = re.compile(r'(?i)^.*accel.*$')
    _grappa_regex2 = re.compile(r'(?i)^.*grappa.*$')
    _sense2_regex = re.compile(r'(?i)^.*sense.*$')
    regex_list = [_mpr_regex, _grappa_regex2, _sense2_regex]

NC = NaccConfig()

def n_participants_wrapper(fn):
    def n_participants(*args, **kwargs):
        output = fn(*args, **kwargs)
        logger.warning(f'{fn.__name__}: n participants is {self.n_subjects()}')
        return output
    return n_participants

def _load_raw_df_nacc():
    return pd.read_csv(CONFIG['NACC']['csv_original'])

def _unzip_nacc_mri(mri_file_name: str) -> dict:
    full_file_name = os.path.join(NC.MRI_DIR, mri_file_name)
    try:
        zf = zipfile.ZipFile(full_file_name)
    except FileNotFoundError:
        try:
            zf = zipfile.ZipFile(os.path.join(NC.MRI_DIR2, mri_file_name))
        except FileNotFoundError as e2:
            print(f'{e2}: can\'t find file {mri_file_name}!')
            return {'json': [], 'nii': []}
    prefix = mri_file_name.split('/')[-1][:-4]
    files = [zf.extract(fi, path=os.path.join(NC.TMP_PATH, prefix))
                for fi in
                zf.infolist()]
    json_files = [x for x in files if NC.json_regex.match(x)]
    nii_files = [x[:-5] + '.nii' for x in json_files]
    return {'json': json_files, 'nii': nii_files}

def _unzip_all_nii_files() -> pd.DataFrame:
    df_raw = _load_raw_df_nacc()
    rid_files = []
    for _, row in df_raw.iterrows():
        mri_zip_filename = row['File_MRI'].replace('.zip', 'ni.zip')
        current_files = _unzip_nacc_mri(mri_zip_filename)
        n_files = len(current_files['json'])
        current_files['RID'] = [row['RID']] * n_files
        rid_files.append(pd.DataFrame.from_dict(current_files))
    unzipped_files = pd.concat(rid_files, axis=0, ignore_index=True)
    unzipped_files.to_csv(NC.unzipped_files_dataframe, index=False)
    return unzipped_files

def _condition_series(serie):
    if 'MPRAGE' in serie or 'mprage' in serie or 'mp-rage' in serie or \
            'MP-RAGE' in serie:
        return True
    if 'SPGR' in serie or 'spgr' in serie:
        return True
    return False

def _retrieve_series_description(json_filename: str) -> str:
    with open(json_filename, 'rb') as fi:
        json_file = json.load(fi)
    return json_file['SeriesDescription'] if 'SeriesDescription' in \
                                                json_file.keys() else ''

def _to_iso(acq_time: str):
    time_parts = acq_time.split(':')
    time_parts[2] = str(time_parts[2]).zfill(9)
    time_parts = ':'.join(time_parts)
    return time_parts

def _retrieve_acquisition_time(json_filename: str) -> str:
    with open(json_filename, 'rb') as fi:
        json_file = json.load(fi)
    acq_time = json_file['AcquisitionTime'] if 'AcquisitionTime' in \
                                                json_file.keys() else pd.NaT
    try:
        acq_time = _to_iso(acq_time)
        acq_time = time.fromisoformat(acq_time)
    except AttributeError as e:
        logger.info(f'{e}')
    except TypeError as e:
        logger.info(f'{e}: {acq_time}')
    except ValueError as e:
        logger.info(f'{e}: {acq_time}')
    except Exception as e:
        raise ValueError(f'{e}: {acq_time}')
    return acq_time

def _retrieve_imagetype(json_filename: str) -> str:
    with open(json_filename, 'rb') as fi:
        json_file = json.load(fi)
    img_type = json_file['ImageType'] if 'ImageType' in \
                                            json_file.keys() else []
    return ','.join(img_type)

def _find_t1_files_for_all(rid_dataframe):
    rid_dataframe.loc[:, 'Description'] = rid_dataframe['json'].apply(
            _retrieve_series_description)
    rid_dataframe.loc[:, 'Is_T1'] = rid_dataframe['Description'].apply(
            lambda x: bool(
                    NC.mprage_regex.match(x) or
                    NC.spgr_regex.match(x))
    )
    return rid_dataframe

def _find_t1_files_for_all2(rid_dataframe):
    rid_dataframe.loc[:, 'Description'] = rid_dataframe['json'].apply(
            _retrieve_series_description)
    rid_dataframe.loc[:, 'Is_T1'] = rid_dataframe['Description'].apply(
            lambda x: bool(
                    NC.mprage_regex.match(x)) or
                    bool(NC.spgr_regex.match(x)))
    return rid_dataframe

def _find_imagetype_for_all(rid_dataframe):
    rid_dataframe.loc[:, 'ImageType'] = rid_dataframe['json'].apply(
            _retrieve_imagetype
    )
    rid_dataframe.loc[:, 'Is_Original'] = rid_dataframe['ImageType'].apply(
            lambda x: ('ORIGINAL' in x) and ('PRIMARY' in x)
    )

def _find_acquisitiontime_for_all(rid_dataframe):
    rid_dataframe.loc[:, 'AcquisitionTime'] = rid_dataframe['json'].apply(
            _retrieve_acquisition_time)

def _has_correct_dims(nifti_filename: str):
    img = nib.load(nifti_filename).get_data().squeeze()
    return len(img.shape) == 3 and min(img.shape) >= 80

def _has_correct_dims_for_all(rid_dataframe):
    warnings.filterwarnings('ignore')
    rid_dataframe.loc[:, 'Has_Correct_Dims'] = rid_dataframe['nii'].apply(
            _has_correct_dims
    )
    warnings.filterwarnings('default')

def n_unique_rids(df):
    return len(pd.unique(df['RID']))

def _remove_files_without_mri(rid_dataframe):
    idx_to_drop = rid_dataframe.query('Is_T1 == False')
    idx_to_drop2 = rid_dataframe.query('Has_Correct_Dims == False')
    idx_to_drop3 = rid_dataframe.query('Is_Original == False')
    idx_to_drop = np.union1d(list(idx_to_drop.index),
                                list(idx_to_drop2.index))
    idx_to_drop = np.union1d(list(idx_to_drop), list(idx_to_drop3.index))
    rid_dataframe.drop(index=idx_to_drop, inplace=True)
    rid_dataframe.reset_index(drop=True, inplace=True)
    rid_dataframe.drop(columns=['Is_T1', 'Has_Correct_Dims'], inplace=True)

def _retrieve_zip_files(move=False) -> pd.DataFrame:
    if move:
        df = _unzip_all_nii_files()
    else:
        try:
            df = pd.read_csv(NC.unzipped_files_dataframe)
        except:
            df = _unzip_all_nii_files()
    return df

def _remove_accel_for_rid(rid_sub_dataframe) -> pd.DataFrame:
    is_accel = []
    if len(rid_sub_dataframe) == 1:
        return rid_sub_dataframe
    for idx, row in rid_sub_dataframe.iterrows():
        is_accel.append(any([x.match(row['Description']) for x in
                                NC.regex_list]))
    if all(is_accel):
        return rid_sub_dataframe
    else:
        idx_to_keep = np.setdiff1d(
                list(rid_sub_dataframe.index), list(
                        rid_sub_dataframe.index[is_accel])
        )
        return rid_sub_dataframe.loc[idx_to_keep, :]

def _identify_most_recent(rid_sub_dataframe) -> pd.DataFrame:
    if len(rid_sub_dataframe) == 1:
        return rid_sub_dataframe
    assert (not any([pd.isna(x) for x in rid_sub_dataframe[
        'AcquisitionTime']]))
    rid_sub_dataframe = rid_sub_dataframe.sort_values('AcquisitionTime')
    return rid_sub_dataframe.iloc[[-1], :]

def _remove_accel_for_all(rid_dataframe) -> pd.DataFrame:
    return rid_dataframe.groupby('RID').apply(
            _remove_accel_for_rid
    ).reset_index(drop=True)

def _choose_most_recent(rid_dataframe) -> pd.DataFrame:
    return rid_dataframe.groupby('RID').apply(
            _identify_most_recent
    ).reset_index(drop=True)

def _plot_mri(mri, fname, cut_coords=(0, 0, 0),
                ):
    fig, ax = plt.subplots(1, 1)
    img1 = plot_img(mri, axes=ax, draw_cross=False, cut_coords=cut_coords)
    img1.title('MRI' + ' ' + fname)
    fig.savefig(os.path.join(
            NC.TARGET_PATH, 'figures/', fname
    ),
            orientation='landscape')
    plt.close()

def _plot_all_brains(rid_dataframe: pd.DataFrame):
    rid_dataframe.sort_values('RID', inplace=True)
    os.makedirs(os.path.join(NC.TARGET_PATH, 'figures'), exist_ok=True)
    for _, row in rid_dataframe.iterrows():
        mri = nib.load(row['nii'])
        fname = row['RID'] + '_' + str(row['AcquisitionTime']) + \
            '_' + row['Description'].strip(' ') + '.pdf'
        _plot_mri(mri, fname)

def merge_dfs(rid_dataframe: pd.DataFrame):
    df_raw = _load_raw_df_nacc()
    rid_dataframe.set_index('RID', inplace=True)
    full_dataframe = df_raw.set_index('RID')
    df_final = rid_dataframe.merge(
            full_dataframe, how='left', left_index=True, right_index=True
    )
    df_final.reset_index(inplace=True)
    return df_final

def move_nii_files(df_final: pd.DataFrame, move: bool=False):
    os.makedirs(NC.TARGET_PATH, exist_ok=True)
    df_final.loc[:, 'NEW_MRI'] = ''
    for idx, row in df_final.iterrows():
        img = nib.load(row['nii'])
        nm = os.path.join(NC.TARGET_PATH, row['RID'])
        df_final.loc[idx, 'NEW_MRI'] = nm
        orig_descrip = img.header['descrip']
        if move:
            save_new_file(img, nm, orig_descrip)

def find_and_move_mri(move=False, pl_ot=False, save=False) -> pd.DataFrame:
    df = _retrieve_zip_files(move)
    n_rids = len(pd.unique(df['RID']))
    logger.warning(f'beginning with {n_rids} rids')
    print('Loaded df')
    _has_correct_dims_for_all(df)
    n_rids = len(pd.unique(df['RID']))
    logger.warning(f'Dopped bad dims, now {n_rids} rids')
    print('Correct dims')
    _find_t1_files_for_all(df)
    n_rids = len(pd.unique(df['RID']))
    logger.warning(f'Assigned t1 files, now {n_rids} rids')
    _find_acquisitiontime_for_all(df)
    n_rids = len(pd.unique(df['RID']))
    logger.warning(f'Acquisition time assigned, now {n_rids} rids')
    _find_imagetype_for_all(df)
    n_rids = len(pd.unique(df['RID']))
    logger.warning(f'Determined image type, now {n_rids} rids')
    _remove_files_without_mri(df)
    n_rids = len(pd.unique(df['RID']))
    logger.warning(f'Removed files w/o mri, now {n_rids} rids')
    df = _remove_accel_for_all(df)
    n_rids = len(pd.unique(df['RID']))
    logger.warning(f'Removed accelerated images, now {n_rids} rids')
    df = _choose_most_recent(df)
    n_rids = len(pd.unique(df['RID']))
    logger.warning(f'Selected most recent images, now {n_rids} rids')
    if pl_ot:
        _plot_all_brains(df)
    df_final = merge_dfs(df)
    move_nii_files(df_final, move)
    if save:
        df_final.to_csv(CONFIG['NACC']['csv_original'][:-4] + '_pruned_final.csv',
                            index=False)
    return df_final
