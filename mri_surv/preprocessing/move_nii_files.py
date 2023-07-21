import logging
import os
import re
import json
import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import numpy as np
import pandas as pd
import sys
from subprocess import run
import datetime
import zipfile
import matplotlib.pyplot as plt
import abc
from datetime import time
from nilearn.plotting import plot_img
from icecream import ic
from dataclasses import dataclass
from typing import Dict, List, NewType
from tqdm import tqdm


with open('preprocessing/metadata_files.json') as fi:
    CONFIG = json.loads(fi.read())

with open(CONFIG['ADNI']['logs'],'w') as fi:
    fi.write(str(datetime.datetime.now()))

hdlr = logging.FileHandler(os.path.join(os.path.abspath('.'),CONFIG['ADNI']['logs']))
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)

def warning(_str):
    logger.warning(_str)

ic.configureOutput(outputFunction=warning)


class T1MetadataDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return T1MetadataDataFrame

class ClinicalDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return ClinicalDataFrame

class ImageDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return ImageDataFrame

class _AdniConfig:
    def __init__(self):
        self.attrs = {}
        self._assign_regex()
        self.attrs['t1_mris'] = _get_t1_files()

    def _assign_regex(self):
        mpr_regex = re.compile(r'(?i)^.*accel.*$')
        grappa_regex2 = re.compile(r'(?i)^.*grappa.*$')
        sense2_regex = re.compile(r'(?i)^.*sense.*$')
        regex_list = [mpr_regex, grappa_regex2, sense2_regex]
        self.attrs['date_regex'] = re.compile(r'^(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{'
                                     r'2}).*$')
        self.attrs['subj_regex'] = re.compile(r'^.*_S_(?P<rid>[0-9]{4}).*\.nii$')
        self.attrs['id_regex'] = re.compile(r'^.*(?P<id>I[0-9]+)\.nii$')
        self.attrs['regex_mriaccel'] = regex_list
        self.attrs['repeat_regex'] = re.compile(r'(?i)^.*repeat.*$')
    
    def __getitem__(self, key):
        return self.attrs[key]

class _QualityControl:
    def __init__(self):
        """
            This creates a quality control sheet using the
            image ids for each of the images we have downloaded
            in conjunction with the QC data from
            Mayo Clinic. Useful for ignoring "accelerated"
            scans and serving as tiebreaker in the event
            that mulitiple scans were conducted on the
            same day.

            Properties:
            BASE_DIR: location of raw data
            MRI_FILES: names of raw image files
            FIELD_CONVERSION_LIST: column names -> desired column names
                    for uniformity across sheets

            """
        self.BASE_DIR = os.path.join(
                os.path.abspath('.'),
                'metadata', 'data_raw','ADNI'
        )
        self.MRI_FILES = [
                'MPRAGE_all.csv',
                'SPGR_all.csv',
        ]

        self.FIELD_CONVERSION_LIST = (
                {
                        'RID': 'RID',
                        'PTID': 'PTID',
                        'LONI_IMAGE': 'LONI_IMAGE',
                        'SERIES_DATE': 'EXAMDATE',
                        'STUDY_OVERALLPASS': 'PASS',
                        'SERIES_SELECTED': 'SELECTED',
                        'T1_ACCELERATED': 'T1_ACCELERATED'
                },
                {
                        'RID': 'RID',
                        'loni_image': 'LONI_IMAGE',
                        'series_date': 'EXAMDATE',
                        'study_overallpass': 'PASS',
                        'series_selected': 'SELECTED',
                        'T1_ACCELERATED': 'T1_ACCELERATED'
                },
        )
        self.DATE_FORMATS = (
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y%m%d',
        )
        self.QC_FILES = (
                'MAYOADIRL_MRI_QUALITY_ADNI3.csv',
                'MAYOADIRL_MRI_IMAGEQC_12_08_15.csv',
        )
        qc_mri_images = self.get_qc_sheet(self.QC_FILES,
                                          self.FIELD_CONVERSION_LIST,
                                          self.DATE_FORMATS)

        mprage_list = [self.create_mprage_list(fi) for fi in self.MRI_FILES]
        mprage_list = pd.concat(mprage_list, ignore_index=True)
        mprage_list.set_index('LONI_IMAGE', inplace=True)
        mprage_list = mprage_list.to_dict()
        qc_mri_images = self.merge_qc_and_mris(mprage_list, qc_mri_images)
        self.qc_mri = qc_mri_images.to_dict()

    def __getitem__(self, key):
        return self.qc_mri[key]

    @staticmethod
    def merge_qc_and_mris(mprage_list, qc_mri_images):
        qc_mri_images['Modality'] = qc_mri_images['LONI_IMAGE'].apply(
                lambda x: mprage_list['Modality'][x] if x in mprage_list[
                    'Modality'].keys() else '')
        qc_mri_images['Sequence'] = qc_mri_images['LONI_IMAGE'].apply(
                lambda x: mprage_list['Sequence'][x] if x in mprage_list[
                    'Sequence'].keys() else '')
        qc_mri_images.set_index('LONI_IMAGE', inplace=True)
        return qc_mri_images

    def get_qc_sheet(self, fi_list: tuple, field_conversion: tuple,
                     dateformat: tuple):
        final_field_list = [list(x.values()) for x in field_conversion]
        final_field_list = list(set(sum(final_field_list, [])))
        all_qc = pd.DataFrame(data=None, columns=final_field_list)

        '''
            iterate through all of the QC sheets. Keep all of the columns
                    to be used, convert them to a homogenized format, and
                    then append
            '''
        for index, fi in enumerate(fi_list):
            qc_file = pd.read_csv(os.path.join(self.BASE_DIR, fi),
                                  low_memory=False)
            qc_file = qc_file[list(field_conversion[index].keys())]
            qc_file.rename(columns=field_conversion[index], inplace=True)
            qc_file['EXAMDATE'] = [datetime.datetime.strptime(
                    str(x), dateformat[index]).date()
                                   for x in qc_file['EXAMDATE']]
            if 'T1_ACCELERATED' not in qc_file.columns:
                qc_file['T1_ACCELERATED'] = -1
            all_qc = all_qc.append(qc_file, ignore_index=True)
        '''
            Some images don't have an 'I' appended to the beginning. For these
                    images, append an I. Also, if an RID is missing, try to 
                    extract from PTID
        '''
        all_qc['LONI_IMAGE'] = ['I' + str(x) if str(x)[0] != 'I' else x
                                for x in all_qc['LONI_IMAGE']]
        all_qc['RID'] = [str(int(x)).zfill(4) if ~np.isnan(x) else y[-4:] for
                         x, y in zip(all_qc['RID'], all_qc['PTID'])]
        all_qc.drop(columns='PTID', inplace=True)
        return all_qc

    def create_mprage_list(self, fi: str) -> pd.DataFrame:
        """Create an MRI dataframe from information on ADNI website.
            Homogenize columns.

            Args:
                fi (str): filename obtained from ADNI website

            Returns:
                pd.DataFrame: dataframe with patient parsed. Used to generate
                the modality
                    and sequence for each scan
            """
        mprage_list = pd.read_csv(os.path.join(self.BASE_DIR, fi))
        mprage_dict = {
                'Image Data ID': 'LONI_IMAGE',
                'Subject': 'Subject',
                'Description': 'Sequence',
                'Modality': 'Modality',
                'Type': 'Type',
                'Acq Date': 'EXAMDATE'
        }
        mprage_list = mprage_list[list(mprage_dict.keys())]
        mprage_list = mprage_list.rename(columns=mprage_dict)
        mprage_list['RID'] = [x[-4:] for x in mprage_list['Subject']]
        mprage_list.drop(columns='Subject', inplace=True)
        mprage_list['EXAMDATE'] = [datetime.datetime.strptime(str(x),
                                                              '%m/%d/%Y')
                                   for x in mprage_list['EXAMDATE']]
        mprage_list.drop(
                mprage_list[mprage_list['Type'] != 'Original'].index,
                inplace=True)
        return mprage_list

def _retrieve_dates_from_csv(df: ClinicalDataFrame, column: str='EXAMDATE_mri3') -> dict:
    """returns a dictionary containing RIDs as keys, mri dates as values for all non-OD RIDs

    Args:
        df (pd.DataFrame): dataframe containing columns TIME_TO_OD, OD, RID, and @column
        column (str, optional): column to use for examdates. Defaults to 'EXAMDATE_mri3'.

    Returns:
        dict: {RID: date}
    """
    data = df.copy()
    print(len(data))
    data = _drop_od_indices(data)
    print(len(data))
    data = _drop_empty_rows(data, column=column)
    print(len(data))
    rids = data['RID'].to_numpy()
    dates = data[column].apply(
            lambda x: list(x.split(','))).to_numpy()
    mri_rid_date = {
            str(int(x)).zfill(4): y for x, y in zip(rids, dates)
    }
    return mri_rid_date

def _drop_empty_rows(data: ClinicalDataFrame, column: str) -> ClinicalDataFrame:
    """Takes a dataframe with column @column. Drops rows with empty values (empty == '' or np.nan)

    Args:
        data (pd.DataFrame): dataframe with column @column
        column (str): name of column to use for empty value query

    Returns:
        pd.DataFrame: dataframe (copy of dataframe) with missing rows
    """
    data = data.copy()
    empty_idx = data[[(isinstance(x,str) and x == '') or (isinstance(x,float) and np.isnan(x)) for x in data[column]]]
    data.drop(empty_idx.index, inplace=True)
    logger.warning('Dropping empty rows: {}'.format(empty_idx[['RID',column]]))
    return data

def _drop_od_indices(data: ClinicalDataFrame) -> ClinicalDataFrame:
    """Takes a dataframe with columns TIME_TO_OD, RID, OD and drops rows where patients have an OD dx

    Args:
        data (pd.DataFrame): df with above mentioned columns

    Returns:
        pd.DataFrame: dataframe without OD rows
    """
    data = data.copy()
    od_indices = data[[not pd.isna(x) for x in data['TIME_TO_OD']]]
    data.drop(od_indices.index,
              inplace=True)
    logger.warning('Dropping other dementias: {}'.format(od_indices[
                                                             ['RID', 'OD']]))
    return data

def _select_largest_iid(image_df: ImageDataFrame) -> pd.Series:
    """Takes a dataframe image_df; returns row corresponding to largest IID value (corresponds to last scan taken)

    Args:
        image_df (pd.DataFrame): dataframe to query

    Returns:
        pd.Series: a row from the dataframe corresponding to largest IID
    """
    iids = image_df['IID'].to_numpy()
    logger.warning(f'Filtering: {iids}')
    iids = np.asarray([int(x[1:]) for x in iids])
    i_max = iids.argmax()
    image_df_out = image_df.iloc[i_max,:]
    logger.warning(f'Picked: {image_df_out.IID}')
    return image_df_out

def _images_at_last_date(image_and_date_df: ImageDataFrame) -> ImageDataFrame:
    """Takes a dataframe. Selects rows of dataframe at most recent date (using DATE column)

    Args:
        image_and_date_df (pd.DataFrame): dataframe with column "DATE"

    Returns:
        pd.DataFrame: trimmed dataframe with only scans on given date
    """
    image_and_date_df = image_and_date_df.copy()
    image_and_date_df['DATE'] = image_and_date_df['DATE'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%d').date())
    image_and_date_df = image_and_date_df.reset_index().set_index(
            'DATE')
    latest_date = max(list(image_and_date_df.index))  # find last date
    image_choice = image_and_date_df.loc[[latest_date], :]  # find images
    image_choice.reset_index(inplace=True)
    return image_choice

def _selected_images(image_choice: ImageDataFrame) -> ImageDataFrame:
    """Takes a dataframe, image_choice, and returns all selected scans only if this doesn't eliminate all scans

    Args:
        image_choice (pd.DataFrame): dataframe with column "selected"

    Returns:
        pd.DataFrame: as specified above
    """
    if len(image_choice) > 1 and not all(image_choice.selected != 1):
        selected_image = image_choice.query('selected == 1')
    else:
        selected_image = image_choice
    return selected_image.copy()

def _filter_by_date(df: ImageDataFrame, desired_dates: dict) -> ImageDataFrame:
    """Filters a DataFrame by rids and desired dates for each rid

    Args:
        df (pd.DataFrame): dataframe with columns DATE, RID
        desired_dates (dict): {RID: [DATE,]}

    Returns:
        pd.DataFrame: filtered DataFrame
    """
    image_dataframe = df.copy()
    idx_to_keep = []
    n_missing = 0
    for rid, values in desired_dates.items():
        rid_dates = image_dataframe.query('RID == @rid')['DATE']
        dates_to_keep = [x in values for x in rid_dates]
        if len(dates_to_keep) == 0 or all([x == False for x in
                                           dates_to_keep]):
            logger.warning(f'scan for {rid} at dates {values} '
                           f'is '
                           f'unavailable')
            n_missing += 1
        idx_to_keep += list(rid_dates.loc[dates_to_keep].index)
    print(f'Scans missing: {n_missing}')
    image_dataframe = image_dataframe.loc[idx_to_keep, :].reset_index(drop=True)
    return image_dataframe

def retrieve_date_dx_from_csv(data: ClinicalDataFrame, column='EXAMDATE_mri3') -> ClinicalDataFrame:
    """Takes a dataframe with diagnosis information and a column to use to query diagnosis code

    Args:
        data (pd.DataFrame): dataframe with columns RID, DX, and @column
        column (str, optional): a column in data w/ dates corresponding to MRIs. Defaults to 'EXAMDATE_mri3'.

    Returns:
        pd.DataFrame: dataframe with index RID, date of mri exam, and entries corresponding to diagnosis
    """
    rids = data['RID'].to_numpy()
    dates = data[column].apply(
                lambda x: list(x.split(','))
            ).to_numpy()
    dx = data['DX'].to_numpy()
    mri_rid_date = {'RID': [], column: [], 'DX': []}
    for rid, date, d in zip(rids, dates, dx):
        for dat in date:  # multiple dates for each rid/dx, so make sure at least one entry for each date
            if d in ['AD','MCI','NL']:  # ensure that diagnosis is one of AD, MCI, or NL/CN
                mri_rid_date['RID'].append(str(int(rid)).zfill(4))
                mri_rid_date[column].append(dat)
                mri_rid_date['DX'].append(d)
            else:
                logger.warning(f'{rid, dat} is not in DX codes, dx: {d}')
    mri_rid_date = pd.DataFrame.from_dict(mri_rid_date)
    return ClinicalDataFrame(mri_rid_date.set_index(['RID', column]))

def _assign_dx_values(df: ImageDataFrame, dx_df: ClinicalDataFrame, column: str='EXAMDATE_mri3') -> ImageDataFrame:
    """Takes a dataframe with image information (RID and DATE) and assigns diagnosis to each image using dx dataframe

    Args:
        df (pd.DataFrame): dataframe with image information / columns RID and DATE
        dx_df (pd.DataFrame): dataframe with diagnosis information (DX and @column)
        column (str, optional): column to use to query date of dx. Defaults to 'EXAMDATE_mri3'.

    Returns:
        pd.DataFrame: dataframe containing only rows with a valid diagnosis
    """
    df = df.copy()
    df.set_index(['RID','DATE'], inplace=True)
    df['DX'] = ''
    idx_to_drop = []
    rid_date_dx_df = retrieve_date_dx_from_csv(dx_df, column=column)
    for idx, row in df.iterrows():
        if idx in rid_date_dx_df.index:
                df.loc[idx,'DX'] = rid_date_dx_df.loc[idx, 'DX']
        else:
            idx_to_drop += [idx]
    logger.warning(f'dropping {idx_to_drop}')
    return df.drop(index=idx_to_drop, axis=0).reset_index()

def assign_all_col_values(df: ImageDataFrame, dx_df: ClinicalDataFrame, column_name:
list) -> ImageDataFrame:
    """
    assign_all_col_values

    Creates new columns @column_name in df using values from dx_df

    Parameters
    ----------
    df : pd.DataFrame
        Target dataframe where new columns are desired
    dx_df : pd.DataFrame
        Source dataframe where new values are sourced
    column_name : list
        Columns for mapping

    Returns
    -------
    pd.DataFrame
        DataFrame with these new columns
    """
    for col in column_name:
        df = _assign_col_values(df, dx_df, col)
    return df

def _assign_col_values(df: ImageDataFrame, dx_df: ClinicalDataFrame, column_name: str ='') -> ImageDataFrame:
    """
    _assign_col_values assigns values in column @column_name from dx_df to df

    Parameters
    ----------
    df : pd.DataFrame
        destination df
    dx_df : pd.DataFrame
        source df
    column_name : str, optional
        column to use for value mapping, by default ''

    Returns
    -------
    pd.DataFrame
        df, but with added column
    """
    df = df.copy()
    df.set_index(['RID','DATE'], inplace=True)
    df[column_name] = ''
    rid_date_dx_df = _retrieve_column_from_csv(dx_df, column=column_name)
    for idx, _ in df.iterrows():
        if idx in rid_date_dx_df.index:
            df.loc[idx,column_name] = rid_date_dx_df.loc[idx, column_name]
        else:
            logger.warning(f'{idx} is not in DX codes, empty DX')
    return df.reset_index()

def _retrieve_column_from_csv(data: ClinicalDataFrame, column: str='') -> ClinicalDataFrame:
    """
    _retrieve_column_from_csv returns a dataframe with RID and EXAMDATE_mri3 as indices; column is @column

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with RID and EXAMDATE_mri3 and @column as columns
    column : str, optional
        column to use for value assignments, by default ''

    Returns
    -------
    pd.DataFrame
        dataframe with indices RID, EXAMDATE_mri3 and column of values equal to @column values
    """
    rids = data['RID'].to_numpy()
    dates = data['EXAMDATE_mri3'].apply(
            lambda x: list(x.split(','))).to_numpy()
    col = data[column].to_numpy()
    mri_rid_date = {'RID': [], 'EXAMDATE_mri3': [], column: []}
    for rid, date, d in zip(rids, dates, col):
        for dat in date:
            mri_rid_date['RID'].append(str(int(rid)).zfill(4))
            mri_rid_date['EXAMDATE_mri3'].append(dat)
            mri_rid_date[column].append(d)
    mri_rid_date = pd.DataFrame.from_dict(mri_rid_date)
    return mri_rid_date.set_index(['RID', 'EXAMDATE_mri3'])

def flatten(nested_list: list) -> list:
    """
    flatten unflattens a list of lists or list-likes

    Parameters
    ----------
    nested_list : list or list-like
        list of lists

    Returns
    -------
    list
        list
    """
    return [y for x in list(nested_list) for y in x]

def save_new_file(img: nib.Nifti1Image, nm: str, orig_description: str) -> bool:
    """
    save_new_file takes an img, name for file, and the description of the image to move

    Parameters
    ----------
    img : nib.Nifti1Image
        image to save
    nm : str
        name to use to save file
    orig_description : str
        description of initial file before move

    Returns
    -------
    bool
        True if file exists in new location (either moved or already exists at destination), False o.w.
    """
    if os.path.isfile(nm + '.nii'):
        old_fi = nib.load(nm + '.nii')
        dat1 = img.get_data()
        dat2 = old_fi.get_data()
        if dat1.dtype == np.float64:
            print(f'{nm} is a float!')  # note if it is a float, will always overwrite destination file
        if not np.allclose(np.squeeze(dat1), np.squeeze(dat2), equal_nan=True):
            try:
                logger.warning('Overwriting file: {}'.format(orig_description))
                print('OVERWRITING FILE {}'.format(nm))
                nib.save(img, "{}.nii".format(nm))
            except Exception as e:
                logger.error(str(e))
                return False
        return True
    else:
        try:
            nib.save(img, "{}.nii".format(nm))
            logger.warning('Saving file: {}'.format(orig_description))
            print('saving file {}'.format(nm))
            return True
        except Exception as e:
            logger.error(str(e))
            return False

def plot_mri(mri: nib.Nifti1Image, fname: str, target_dir: str, cut_coords: tuple=(0, 0, 0)):
    """
    plot_mri takes a nifti1image and plots it using the cut_coords, saving with the fname at the target_dir
    
    Parameters
    ----------
    mri : nib.Nifti1Image
        MRI to plot
    fname : str
        file name to use for image title and file name
    target_dir : str
        destination directory to save
    cut_coords : tuple, optional
        coordinates to use for axial, sagittal, and coronal cuts, by default (0, 0, 0)
    """
    fig, ax = plt.subplots(1, 1)
    img1 = plot_img(mri, axes=ax, draw_cross=False, cut_coords=cut_coords)
    img1.title('MRI' + ' ' + fname)
    fig.savefig(os.path.join(target_dir, fname),
                orientation='landscape')
    plt.close()

def plot_all_brains(folder: str, target_dir='.'):
    """
    plot_all_brains Plots all mris in folder @folder at target directory @target_dir

    Parameters
    ----------
    folder : str
        source folder for MRIs
    target_dir : str, optional
        directory to save all mris, by default '.'
    """
    os.makedirs(os.path.join(os.path.abspath(target_dir), 'figures'),
                exist_ok=True)
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.nii'):
                mri = nib.load(os.path.join(root, file))
                fname = file.replace('.nii', '') + '.png'
                print(fname)
                plot_mri(mri, fname,
                         os.path.join(os.path.abspath(target_dir), 'figures'))

def _drop_rois(roi_dict: Dict[str, List], rois_to_drop: List) -> Dict[str, List]:
    """
    _drop_rois drops keys in roi_dict listed in rois_to_drop

    Parameters
    ----------
    roi_dict : Dict[str, List]
        list of rids and data corresponding to each
    rois_to_drop : List
        list of rois to drop from roi_dict

    Returns
    -------
    Dict[str, List]
        dictionary without rois from rois_to_drop
    """
    for roi in rois_to_drop:
        roi_dict.pop(roi, None)
    return roi_dict

def _extract_fileparts(filename: str, file_directory: str,
                        ac: _AdniConfig) -> tuple:
    """
    _extract_fileparts Takes a filename for an MRI and parses the RID, date of scan, and image id (IID)

    Parameters
    ----------
    filename : str
        name of file to be parsed
    file_directory : str
        file directory in which file is located (this is where the date of the scan is listed)

    Returns
    -------
    tuple
        (rid, date, _id) of scan
    """
    rid = re.match(ac['subj_regex'], filename)
    if rid:
        rid = rid['rid']
    else:
        logger.warning('no RID match for ' + filename + ' continuing')
        rid = None
    date = re.match(ac['date_regex'], file_directory)['date']
    _id = re.match(ac['id_regex'], filename)['id']
    return rid, date, _id

def move_and_rename_all_files(filelist: List[str], modality: str,
                                orig_dir: str, foldersuffix: str = '',
                                keep_original_name=False
                                ) -> None:
    """
    move_and_rename_all_files takes a list of files, suffix to use for naming, the directory in which filelist files are located,
        and a suffix to use for the new folder, and moves/renames all files

    Parameters
    ----------
    filelist : List[str]
        a list of files to move
    modality : str
        a suffix, such as "MRI", to use for naming newly moved files and their respective folders
    orig_dir : str
        parent directory in which files are initially located
    foldersuffix : str, optional
        suffix to use for new folder, by default ''
    keep_original_name : bool, optional
        whether or not to keep the file's original name, by default False
    """
    ac = _AdniConfig()
    i = 0
    for filepath in filelist:
        root, file = os.path.split(filepath)
        folder_nm = os.path.join(orig_dir, '..',
                                    modality + '_nii' + foldersuffix)
        _, fpath = os.path.split(os.path.abspath(os.path.join(root, '..')))
        if not os.path.isdir(folder_nm):
            os.mkdir(folder_nm)
        if not keep_original_name:
            rid, date, _id = _extract_fileparts(file, fpath, ac=ac)
            name = rid + '_' + date.strip('-') + '_' + _id + '_' + modality

            img = nib.load(os.path.join(root, file))
            orig_descrip = img.header['descrip']
            img.header['descrip'] = file + ',' + re.match(
                    ac['date_regex'], fpath)['date']
            nm = f'{folder_nm}/{name}'
            save_new_file(img, nm, orig_descrip)
        else:
            _, pth = os.path.split(root)
            new_root = os.path.join(folder_nm, pth)
            if not os.path.isdir(new_root):
                os.mkdir(new_root)
            run(['rsync', '-av', filepath, f'{new_root}/{file}'])
            print(i)

def get_file_df(root_dir: str, force=False) -> ImageDataFrame:
    """
    get_file_df takes the name of a directory and returns a dataframe with the RID, Date, IID, and name of each MRI file therein

    [extended_summary]

    Parameters
    ----------
    root_dir : str
        root directory to interrogate

    Returns
    -------
    pd.DataFrame
        dataframe with above mentioned columns
    """
    if not force:
        fi = CONFIG['ADNI']['raw_mri_dataframe_adni']
        try:
            file_df = ImageDataFrame(pd.read_csv(fi, dtype={'RID': str, 'IID': str, 'DATE': str, 'Name': str}))
        except:
            print(f'File {fi} does not exist! Will generate now...')
            file_df = _get_file_df(root_dir)
    else:
        file_df = _get_file_df(root_dir)
    return file_df

def _get_file_df(root_dir: str) -> ImageDataFrame:
    file_dict = {
            'RID': [],
            'DATE': [],
            'IID': [],
            'Name': [],
    }
    ac = _AdniConfig()
    for root, _, files in tqdm(os.walk(root_dir)):
        for file in files:
            if file.endswith('.nii'):
                _, file_directory = os.path.split(
                        os.path.abspath(os.path.join(root, '../')))
                rid, date, _id = _extract_fileparts(file, file_directory, ac)
                if rid is None:
                    continue
                file_dict['RID'].append(rid)
                file_dict['DATE'].append(date)
                file_dict['IID'].append(_id)
                file_dict['Name'].append(os.path.join(root, file))
    file_df = ImageDataFrame(pd.DataFrame.from_dict(file_dict))
    file_df.to_csv(CONFIG['ADNI']['raw_mri_dataframe_adni'], index=False)
    return file_df

def _apply_auto_qc(image_dataframe: ImageDataFrame=None) -> ImageDataFrame:
    """
    filter_by_date_and_selected retrieves a filtered dataframe with only a single MRI for each patient

    Filters by RID, first looks for most recent date for MRI, then filters by images selected for further processing, then
    last scan in series (largest IID)

    Parameters
    ----------
    image_dataframe : pd.DataFrame, optional
        image dataframe with columns RID, DATE, SELECTED, IID, by default None

    Returns
    -------
    pd.DataFrame
        new dataframe with columns RID, DATE, MODALITY (MRI), IID, and Name of file containing one row per patient

    Raises
    ------
    Exception
        If function accidentally prunes out all images will raise an error
    """
    image_list = ImageDataFrame(pd.DataFrame(data=None, columns=['RID',
                                    'DATE',
                                    'IID',
                                    'Name']))
    for rid, image_series in image_dataframe.groupby('RID'):
        image_series = _drop_accel_conditional_for_rid(image_series)
        image_series = _drop_failqc_conditional_for_rid(image_series)
        image_series = _images_at_last_date(image_series)
        if len(image_series) > 1:
            image_series = _selected_images(image_series)
        if len(image_series) > 1:
            image_series = _select_largest_iid(image_series)
        if len(image_series) == 0:
            raise Exception('No images left!')
        image = image_series.T.squeeze()
        image_list = \
            image_list.append({
                    'RID': rid,
                    'DATE': image['DATE'],
                    'IID': image['IID'],
                    'Name': image['Name']
            }, ignore_index=True)
    return image_list

def _drop_accel_conditional_for_rid(sub_df: ImageDataFrame) -> ImageDataFrame:
    """
    _drop_accel_conditional_for_rid takes a dataframe and returns only scans that were unaccelerated, if not all scans are unaccelerated

    Parameters
    ----------
    sub_df : pd.DataFrame
        dataframe with column corresponding to whether or not scan was fully sampled (unaccel=True)

    Returns
    -------
    pd.DataFrame
        filtered dataframe
    """
    if all(sub_df.unaccel == False):
        out_df = sub_df.copy()
    else:
        out_df = sub_df.query('unaccel == True').copy()
    return out_df

def _drop_failqc_conditional_for_rid(sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    _drop_failqc_conditional_for_rid drops images that didn't pass quality control, unless none of the images passed quality control

    Parameters
    ----------
    sub_df : pd.DataFrame
        dataframe with scans from a single patient

    Returns
    -------
    pd.DataFrame
        dataframe with scans that passed QC for a single patient
    """
    if all(sub_df.pass_qc == False):
        out_df = sub_df
    else:
        out_df = sub_df.query('pass_qc == True')
    return out_df

def _is_selected(imageid: str, qc: _QualityControl) -> bool:
    """
    _is_selected returns true if an image was "selected" for further processing

    Parameters
    ----------
    imageid : str
        an image id, IID

    Returns
    -------
    bool
        as above
    """
    if imageid in list(qc['SELECTED'].keys()):
        return True if qc['SELECTED'][imageid] == 1 else False
    else:
        return True

def _passes_qc(imageid: str, qc: _QualityControl) -> bool:
    """
    _passes_qc returns true if an image passes quality control

    Parameters
    ----------
    imageid : str
        image id, IID

    Returns
    -------
    bool
        true if an image passes quality control
    """
    if imageid in list(qc['PASS'].keys()):
        return False if qc['PASS'][imageid] == 0 else True
    else:
        return True

def _filter_scans(fname: str, regex_list: list) -> bool:
    """
    _filter_scans Takes the name of a scan and filters by a list of regular expressions. If it matches any, returns true

    Parameters
    ----------
    fname : str
        name of file
    regex_list : list
        list of regular expressions as returned by re.compile()

    Returns
    -------
    bool
        true if filename matches anything in regex_list
    """
    _, last = os.path.split(os.path.abspath(fname))
    is_valid = any([x.match(last) for x in regex_list])
    return is_valid

def _is_non_accel_mri(imageid: str, qc: _QualityControl, ac: _AdniConfig) -> bool:
    """
    _is_non_accel_mri determines whether or not imageid is accelerated

    Parameters
    ----------
    t1files : pd.DataFrame
        dataframe indexed by imageid and with a "description" column that we can query for ?Acceleration information
    imageid : str
        image id

    Returns
    -------
    bool
        whether or not image is accelerated
    """
    t1files = ac['t1_mris']
    if imageid in qc['T1_ACCELERATED'].keys():
        accelerated = qc['T1_ACCELERATED'][imageid] == 1
        accelerated = accelerated or _filter_scans(
                qc['Sequence'][imageid], ac['regex_mriaccel'])
        if accelerated:
            logger.warning('{} is accelerated'.format(imageid))
            return False
        else:
            return True
    else:
        accelerated = _filter_scans(t1files.loc[imageid, 'Description'],
                                        _AdniConfig()['regex_mriaccel'])
        val = t1files.loc[imageid,'Description']
        logger.warning(f'trying to filter {imageid}: '
                        f'{val}:'
                        f'{accelerated}')
        if accelerated:
            return False
        else:
            return True

def _add_qc_columns(image_dataframe: ImageDataFrame) -> ImageDataFrame:
    qc = _QualityControl()
    ac = _AdniConfig()
    image_dataframe['selected'] = image_dataframe['IID'].apply(lambda x:
        _is_selected(x,qc)
    )
    print('Added column \'selected\'')
    image_dataframe['unaccel'] = image_dataframe['IID'].apply(
            lambda x: _is_non_accel_mri(x, qc, ac)
    )
    print('Added column \'unaccel\'')
    image_dataframe['pass_qc'] = image_dataframe['IID'].apply(
            lambda x: _passes_qc(x, qc)
    )
    print('Added column \'pass_qc\'')
    return image_dataframe

def _get_t1_files() -> T1MetadataDataFrame:
    """
    _get_t1_files retrieves a list of all scans that were downloaded for MPRAGE and SPGR

    Returns
    -------
    DataFrame with image info for all T1 MRIs in ADNI
    """
    t1_files = [
            os.path.join(
                    CONFIG['ADNI']['BASEDIR'],
                    CONFIG['ADNI']['MPRAGEFILE']
            ),
            os.path.join(
                    CONFIG['ADNI']['BASEDIR'],
                    CONFIG['ADNI']['SPGRFILE']
            )
    ]
    t1_files = T1MetadataDataFrame(pd.concat(
            [pd.read_csv(x) for x in t1_files],
            axis=0,
            ignore_index=True
    ))
    t1_files.rename(columns={'Image Data ID': 'IID'}, inplace=True)
    t1_files.set_index('IID', inplace=True)
    return t1_files

def _load_raw_df() -> ClinicalDataFrame:
    df_raw = ClinicalDataFrame(pd.read_csv(CONFIG['ADNI']['csv_original']))
    df_raw['RID'] = df_raw['RID'].apply(lambda x: str(int(x)).zfill(4))
    return df_raw

def _load_all_df() -> ClinicalDataFrame:
    df_all_raw = ClinicalDataFrame(pd.read_csv(CONFIG['ADNI']['csv_original_all']))
    df_all_raw['RID'] = df_all_raw['RID'].apply(lambda x: str(int(x)).zfill(4))
    return df_all_raw

def _prune_csv_file(df: ClinicalDataFrame, image_list_final: ImageDataFrame, rids: np.array, 
addl_suffix: str='') -> ClinicalDataFrame:
    df = df.loc[[x in rids for x in df.RID], :]
    df = df.reset_index(drop=True)
    df = df[CONFIG['ADNI']['FINAL_COLUMNS']]
    image_list_final = image_list_final.copy().set_index('RID', drop=True)
    image_list_final.sort_index(level=[0, 1], inplace=True)
    df['MRI_IID'] = df['RID'].apply(
            lambda x: image_list_final.loc[x, 'IID']
    )
    df['MRI_fname'] = df['RID'].apply(
            lambda x: image_list_final.loc[x, 'Name']
    )
    df.to_csv(CONFIG['ADNI']['csv_original'][:-4] + addl_suffix + '_pruned_final.csv',
                index=False)
    return df

def find_and_move_mri_ad(move: bool=False, addl_suffix='_AD') -> None:
    df_raw = _load_raw_df()
    mri_dict_ad = _retrieve_dates_from_csv(df_raw, column='AD_MRI_DATE')
    mri_dict_ad = _drop_rois(mri_dict_ad, CONFIG['ADNI']['rois_to_drop'])
    image_dataframe = get_file_df(CONFIG['ADNI']['MRI_DIR'])
    image_dataframe_ad = _filter_by_date(image_dataframe, mri_dict_ad)
    image_dataframe_ad = _add_qc_columns(image_dataframe_ad)
    image_list_final_ad = _apply_auto_qc(image_dataframe_ad)
    filenames_mri_ad = image_list_final_ad[['RID', 'Name']].set_index(
        'RID').to_dict()
    print(len(list(filenames_mri_ad['Name'].keys())))
    df = _prune_csv_file(df_raw, image_list_final_ad,
                    rids=list(filenames_mri_ad['Name'].keys()), addl_suffix=addl_suffix)
    if move is True:
        move_and_rename_all_files(
            list(filenames_mri_ad['Name'].values()), 'MRI', foldersuffix='_' + CONFIG['ADNI']['foldersuffix'] + addl_suffix,
            orig_dir=CONFIG['ADNI']['MRI_DIR']
        )
    return df

def find_and_move_mri(move: bool=False, addl_suffix='') -> None:
    df_raw = _load_raw_df()
    mri_dict = _retrieve_dates_from_csv(df_raw)
    print(len(mri_dict.keys()))
    mri_dict = _drop_rois(mri_dict, CONFIG['ADNI']['rois_to_drop'])
    image_dataframe = get_file_df(CONFIG['ADNI']['MRI_DIR'])
    image_dataframe = _filter_by_date(image_dataframe, mri_dict)
    print(len(pd.unique(image_dataframe['RID'])))
    image_dataframe = _add_qc_columns(image_dataframe)
    image_list_final = _apply_auto_qc(image_dataframe)
    print(len(image_list_final))
    filenames_mri = image_list_final[['RID','Name']].set_index('RID').to_dict()
    df = _prune_csv_file(df_raw, image_list_final, list(filenames_mri['Name'].keys()), addl_suffix='')
    if move is True:
        move_and_rename_all_files(
                list(filenames_mri['Name'].values()), 'MRI',
                foldersuffix='_' + CONFIG['ADNI']['foldersuffix'] + addl_suffix,
                orig_dir=CONFIG['ADNI']['MRI_DIR']
                )
    return df

def find_and_move_unused_mri(move: bool=False) -> pd.DataFrame:
    df_raw = _load_raw_df()
    df_all = _load_all_df()
    mri_dict = _retrieve_dates_from_csv(df_raw)
    image_dataframe = get_file_df(CONFIG['ADNI']['MRI_DIR']).set_index('RID')
    intersection = np.intersect1d(list(mri_dict.keys()),
                                    list(image_dataframe.index))
    image_dataframe = image_dataframe.drop(index=intersection, axis=0).reset_index()
    image_dataframe = _assign_dx_values(image_dataframe, df_all)
    image_dataframe = assign_all_col_values(image_dataframe,
                                            df_all,
                                            ['PTGENDER_demo', 'AGE',
                                                'MMSCORE_mmse','abeta',
                                                'tau','ptau'])
    filenames_mri = image_dataframe[['RID','Name','IID']].set_index(
            ['RID','IID']).to_dict()
    if move is True:
        move_and_rename_all_files(list(filenames_mri['Name'].values()), 'MRI',
                                        foldersuffix='_unused_'
                                                    + CONFIG['ADNI']['foldersuffix'],
                                        orig_dir=CONFIG['ADNI']['MRI_DIR'])
    image_dataframe = image_dataframe[['RID', 'IID', 'DATE', 'DX',
                                        'PTGENDER_demo','AGE',
                                        'MMSCORE_mmse', 'abeta',
                                                'tau','ptau']]
    image_dataframe.loc[:,'FILE_CODE'] = image_dataframe[['RID',
                                                        'IID']].apply(
            lambda x: x.RID + '_mri_' + x.IID, axis=1
    )
    return image_dataframe