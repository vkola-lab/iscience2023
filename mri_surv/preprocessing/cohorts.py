#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:52:45 2020

@author: mfromano
"""

import pandas as pd
import numpy as np
import sys
from preprocessing.metadata_locations import DXTRANS, \
    DX_CONVERSION_DICT, MetadataCSV, FILE_LIST, NACC_FILE_LIST
from icecream import ic
import re
import datetime
import abc
import logging
import os
from typing import List, Dict

MERGE_CODES = ['RID', 'VISCODE', 'VISCODE2']
MEDHX_DICT = {
        1: 'Psych', 2: 'Neuro', 3: 'HEENT', 4: 'Cardio', 5: 'Resp',
        6: 'Hepatic', 7: 'Derm', 8: 'MSK', 9: 'Endo', 10: 'GI',
        11: 'Heme', 12: 'Renal', 13: 'Allergies', 14: 'EtOH',
        15: 'Drugs', 16: 'Smoking', 17: 'Malignancy', 18: 'SurgicalHx',
        19: 'Other'
}
log_file = os.path.join(os.path.abspath("./logs"),
                                        "cohort.log")
with open(log_file, 'w') as fi:
    fi.write(f'Logs {datetime.datetime.now()}')
hdlr = logging.FileHandler(log_file)
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)
logger.setLevel('INFO')

def log_wrapper(fun):
    def logger_fun(*args, **kwargs):
        # try:
        logger.warning(f'------------------------\nRunning: {fun.__name__}\n-----------------------------------\n')
        return fun(*args, **kwargs)
        # except Exception as e:
        #     logger.error(e)
    return logger_fun

class Cohort(abc.ABC):
    def __init__(self, file_list: dict = FILE_LIST, **kwargs) -> \
            None:
        """
        Args:
            file_list (dict):
            **kwargs:
        """
        self.tbl_merged = pd.DataFrame(data=None)
        self.FILE_LIST = file_list
        self.tables = {}
        self.tbl_merged = pd.DataFrame()
        for key in file_list.keys():
            csv_dict = file_list[key]
            logger.warning(f'Loading file {key}')
            self.tables[key] = self.__data_from_csv(csv_dict)
            self.tables[key] = self._modify_data_from_csv(self.tables[key],
                                                         csv_dict)
        self._pre_process()
        n_subs = len(pd.unique(self.tables['reg']['RID']))
        logger.warning(f'Pre merge, {n_subs}')
        self._merge_tables()
        logger.warning(f'Post merge, {self.n_subjects()}')
        self._post_process(**kwargs)
        logger.warning(f'Post-process, {self.n_subjects()}')

    def __data_from_csv(self, csv_dict: dict) -> pd.DataFrame:
        """
        Args:
            csv_dict (dict):
        """
        fi = csv_dict['loc']
        fields = csv_dict['fields']
        data_fi = pd.read_csv(fi, low_memory=False,
                              usecols=fields, dtype=csv_dict['dtype'])
        data_fi = data_fi.rename(columns=csv_dict['colmap'])
        n_pts = len(pd.unique(data_fi['RID'])) if 'RID' in data_fi.columns else None
        logger.warning(f'{n_pts} patients\n')
        data_fi = self.reformat_examdate_col(data_fi, csv_dict['date_fmt'])
        data_fi = self.prune_nan_rows(data_fi, csv_dict['drop_nan_row'])
        n_pts = len(pd.unique(data_fi['RID'])) if 'RID' in data_fi.columns else None
        logger.warning(f'{n_pts} patients after dropping empties\n')
        data_fi = self.recode_columns(data_fi, csv_dict['value_recode'])
        return data_fi

    @abc.abstractmethod
    def n_subjects(self):
        pass

    @abc.abstractmethod
    def _modify_data_from_csv(self, data_fi: pd.DataFrame,
                             csv_dict: MetadataCSV) -> pd.DataFrame:
        """
        Args:
            data_fi (pd.DataFrame):
            csv_dict (MetadataCSV):
        """
        return data_fi

    @abc.abstractmethod
    def _pre_process(self):
        pass

    @abc.abstractmethod
    def _merge_tables(self):
        pass

    @abc.abstractmethod
    def _post_process(self, **kwargs):
        """
        Args:
            **kwargs:
        """
        pass

    @abc.abstractmethod
    def get_progression_data_time_to_progress(self):
        pass

    @staticmethod
    def reformat_examdate_col(data_fi: pd.DataFrame, datefmt: str) -> \
            pd.DataFrame:
        """Replaces np.nan and -4 with valid date formats. Also converts to
        'DATE' format

        Args:
            data_fi (pd.DataFrame):
            datefmt (str):

        Returns:
            pd.DataFrame: data frame with reformatted dates
        """
        if 'EXAMDATE' not in data_fi.columns:
            return data_fi
        assert (not any([x == '-4' for x in data_fi['EXAMDATE']]))
        data_fi['EXAMDATE'] = data_fi['EXAMDATE'].replace(
                to_replace=[pd.NA], value=[pd.NaT])
        data_fi['EXAMDATE'] = pd.to_datetime(data_fi['EXAMDATE'],
                                             format=datefmt).dt.date
        return data_fi

    @staticmethod
    def recode_columns(data_fi: pd.DataFrame, col_conversion_dict: Dict):
        if col_conversion_dict is None:
            return data_fi
        for col in col_conversion_dict.keys():
            data_fi[col] = data_fi[col].replace(
                    col_conversion_dict[col]
            )
        return data_fi

    @staticmethod
    def prune_nan_rows(tbl: pd.DataFrame, nan_row: dict) -> pd.DataFrame:
        """takes as input a dataframe and a dictionary with handles to
        definitions for rows to drop.

            drops these rows.

        Args:
            tbl (pd.DataFrame): dataframe to prune
            nan_row (dict): a dictionary containing keys = column names and

        Returns:
            pd.DataFrame: return pruned DataFrame
        """
        for column in nan_row.keys():
            nan_inds = tbl[column].apply(nan_row[column])
            tbl.drop(index=tbl[nan_inds].index, axis=0, inplace=True)
            tbl.reset_index(drop=True, inplace=True)
        return tbl

    @staticmethod
    def stringify(lis_t: pd.Series) -> List:
        """takes a Series and converts the lists in each entry to strings

        Args:
            lis_t (pd.Series): series to listify

        Returns:
            list: returns list of strings
        """
        return [','.join([str(y) for y in x]) if type(x) == list else x for x
                in lis_t]

    @staticmethod
    def rid_dif(tbl1: pd.DataFrame, tbl2: pd.DataFrame) -> pd.DataFrame:
        """Obtains the RIDs/patients that are not in tbl2

        Args:
            tbl1 (pd.DataFrame): tbl to prune
            tbl2 (pd.DataFrame): reference to obtain RIDs to exclude

        Returns:
            pd.DataFrame: pruned dataframe
        """
        rid_tbl2 = pd.unique(tbl2['RID'])
        tbl_dif = tbl1[not tbl1['RID'].isin(rid_tbl2)].reset_index(drop=True)
        return tbl_dif

    @staticmethod
    def get_progression_category(time_to_progression_and_dx: np.array):
        if 24 > time_to_progression_and_dx[0] >= 0:
            category = 1
        elif 48 > time_to_progression_and_dx[0] >= 24:
            category = 2
        elif time_to_progression_and_dx[0] >= 48:
            category = -1
        else:
            category = np.nan
        return category

    @staticmethod
    def round_progression_time(times: np.array, hits: np.array) -> np.array:
        """takes a 1D array and rounds everything to the nearest year

        Args:
            times (np.array): 1D array with times to progression or final dx

        Returns:
            times (np.array): times rounded to nearest 12 months

        Raises:
            ValueError: if array has more than 1 dimension
        """
        times = [np.ceil(x / 12)*12 if y else np.floor(x / 12)*12 for x,y in zip(times, hits)]
        return times

    @staticmethod
    def get_2yr_category(time_to_progression_and_dx: np.array):
        if 24 >= time_to_progression_and_dx[0]:
            category = 1
        elif time_to_progression_and_dx[1] > 24:
            category = 0
        else:
            category = np.nan
        return category

    @staticmethod
    def steps_under_one_year(dx_dates, progresses, time_to_progression):
        if progresses:
            vals = dx_dates.split(',')
            vals = np.asarray(list(map(int, vals)))
            vals = vals[vals <= round(time_to_progression)]
            vals.sort()
            if vals[-1]-vals[-2] <= 12:
                return True
            else:
                return False
        return True

    def retrieve_times(self, df):
        times = [x[1].TIME_TO_FINAL_DX if not x[1].PROGRESSES else x[
            1].TIME_TO_PROGRESSION for x in df.iterrows()]
        times = np.asarray(times)
        df.loc[:,'TIMES'] = times
        df.loc[:,'TIMES_ROUNDED'] = self.round_progression_time(times, df['PROGRESSES'].to_numpy())
        df.loc[:,'STEPS_UNDER_1_YR'] = [
            self.steps_under_one_year(
                x[1].DX_DATES, x[1].PROGRESSES, x[1].TIME_TO_PROGRESSION
                ) for x in df.iterrows()
        ]

    def merge_columns(self, col_names: list, new_key: str):
        """Takes 2 columns and merges them

        Args:
            col_names (list): column names to draw from (length should be 2)
            new_key (str): new column key to place these columns under
        """
        self.listify(col_names[0])
        self.listify(col_names[1])
        tbl = self.tbl_merged
        tbl[new_key] = [x + y for x, y in zip(tbl[col_names[0]],
                                              tbl[col_names[1]])]

    def unlist(self, col: str or pd.Series) -> list:
        """Takes a series of mixed lists/non-lists, and returns a list. If
            a string is given as argument, the column Series is used and set

        Args:
            col (str or pd.Series): takes a string column name or a list

        Returns:
            list: returns an 'unlisted' version of input i.e. no nested lists
        """
        if type(col) == str:
            series = self.tbl_merged[col]
        else:
            series = col
        list_inds = [type(x) == list for x in series]
        assert (all(len(x) == 1 for x in series[list_inds]))
        vals = [x if type(x) != list else x[0] for x in series]
        if type(col) == str:
            self.tbl_merged[col] = vals
        return vals

    def listify(self, col) -> None:
        """Converts an un-nested or mixed Series from tbl_merged and
            creates a nested version containing the set

        Args:
            col ([type]): string column name
        """
        self.tbl_merged[col] = [list(set(x)) if type(x) == list else [x] for
                                x in self.tbl_merged[col]]

    def drop_nans(self, col: str, is_time=False) -> None:
        """for a nested list, iterate through items in said list and drop nan
        values.

        Args:
            col (str): Description
            is_time (bool, optional): if a time value, use NaT for nan values
                else np.nan
        """
        pruned_list = []
        series = self.tbl_merged[col]
        for row in series:
            new_row = []
            if type(row) != list:
                new_row = [row]
            else:
                for element in row:
                    if not pd.isna(element):
                        new_row.append(element)
            if len(new_row) == 0:
                if is_time:
                    new_row = [pd.NaT]
                else:
                    new_row = [np.nan]
            pruned_list.append(new_row)
        self.tbl_merged[col] = pruned_list

    def retrieve_column_intersections(self, columns: dict) -> dict:
        """For all of the columns listed, find and return the intersection of
        'True' values defined by 'columns

        Args:
            columns (dict): dictionary where keys=columns and the 'True'

        Returns:
            dict: return a dictionary containing intersection indices where
            these intersections occur
        """
        pc_table = self.tbl_merged
        rid_loc = {}
        for rid, subject_vals in pc_table.groupby('RID'):
            has_values = np.ones(len(subject_vals))  # begin with array of ones
            for key in columns.keys():
                has_values = np.logical_and(has_values,
                                            subject_vals[key] == columns[key])
            if any(has_values):
                rid_loc[rid] = subject_vals.index[np.where(has_values)]
        return rid_loc

class ADNICollection(Cohort):
    def __init__(self, file_list: Dict = FILE_LIST, cumulative: bool = True) \
            -> None:
        """
        Args:
            file_list (Dict):
            cumulative (bool):
        """
        super(ADNICollection, self).__init__(file_list, cumulative=cumulative)

    def _modify_data_from_csv(self, data_fi: pd.DataFrame,
                             csv_dict: MetadataCSV) -> pd.DataFrame:

        """
        Args:
            data_fi (pd.DataFrame):
            csv_dict (MetadataCSV):
        """
        def remove_screening_duplicate(tbl: pd.DataFrame) -> pd.DataFrame:
            if len(np.intersect1d(['Phase', 'VISCODE2', 'RID'],
                                  tbl.columns)) < 3:
                return tbl
            subtbl = tbl.query('Phase == \'ADNIGO\' and VISCODE2 == \'sc\' and '
                               'RID < 2000')
            tbl.drop(index=subtbl.index, inplace=True)
            tbl.reset_index(drop=True, inplace=True)
            return tbl

        def consolidate_duplicates(tbl: pd.DataFrame,
                                   data_fi_dict: MetadataCSV) -> pd.DataFrame:
            if data_fi_dict['duplicates'] is True:
                data_fi_grouped = tbl.groupby(MERGE_CODES)
                tbl = data_fi_grouped.agg(lambda x:
                                          list(set(x))).reset_index()
                tbl.reset_index(drop=True, inplace=True)
                if 'DONE' in tbl.columns:
                    tbl['DONE'] = self.unlist(tbl['DONE'])
            return tbl
        data_fi = remove_screening_duplicate(data_fi)
        if 'VISCODE2' not in data_fi.columns:
            data_fi['VISCODE2'] = data_fi['VISCODE'].copy()
        data_fi = consolidate_duplicates(data_fi, csv_dict)
        return data_fi

    def _pre_process(self):
        pass

    def n_subjects(self):
        return len(pd.unique(self.tbl_merged['RID']))

    def _merge_tables(self):
        self.tbl_merged = self.tables['reg']
        self.__merge_apoe(self.tables['apoe'])
        self.tables.pop('apoe', None)
        self.tables.pop('reg', None)
        for key in self.tables.keys():
            curr_table = self.tables[key]
            assert (not np.any(curr_table.duplicated(subset=MERGE_CODES,
                                                     keep=False)))
            curr_table.columns = [
                    x + '_' + key if str(x) not in MERGE_CODES else x
                    for x in curr_table.columns]
            self.tbl_merged = pd.merge(self.tbl_merged, curr_table,
                                       on=MERGE_CODES,
                                       how="left", validate='one_to_one')
            self.tbl_merged.reset_index(inplace=True, drop=True)
        self.tables = {}

    def _post_process(self, cumulative: bool = False):
        """Summary

        Args:
            cumulative (bool):
        """
        logger.warning(f'Pre-fail: {self.n_subjects()}')
        self.__drop_fails()
        logger.warning(f'Dropped fails: {self.n_subjects()}')
        self.__dx_entry_from_phase()
        logger.warning(f'Added DX entry: {self.n_subjects()}')
        self.__combine_bl_visits()
        logger.warning(f'Combined BL visit: {self.n_subjects()}')
        self.__convert_monthly_visit()
        logger.warning(f'Converted monthly visit: {self.n_subjects()}')
        self.__add_age()
        logger.warning(f'Converted age: {self.n_subjects()}')
        self.__add_demographics()
        logger.warning(f'Add demographics: {self.n_subjects()}')
        self.__add_medical_hx()
        logger.warning(f'Add medical hx: {self.n_subjects()}')
        if cumulative:
            self.progress_to_ad_cumulative()
        else:
            self.progress_to_ad()
        logger.warning(f'Progress to AD: {self.n_subjects()}')
        self.has_csf()
        logger.warning(f'Added CSF: {self.n_subjects()}')
        self.merge_csf_columns()
        logger.warning(f'Combined CSF: {self.n_subjects()}')
        self.combine_tau()
        logger.warning(f'combined tau: {self.n_subjects()}')
        self.combine_amyloid()
        logger.warning(f'Combined amyloid: {self.n_subjects()}')
        self.combine_fdg()
        logger.warning(f'combined FDG: {self.n_subjects()}')
        self.tbl_merged.to_pickle('metadata/data_processed/dataframe_adni_postprocessed.pkl')

    def __drop_fails(self) -> None:
        """This removes from tbl_merged all of the visits w/ the specified
        invalid_code entries
        """
        invalid_codes = ['f', 'nv', 'uns1', 'tau']
        for code in invalid_codes:
            self.tbl_merged.drop(
                    index=self.tbl_merged[
                        self.tbl_merged.VISCODE2 == code].index,
                    inplace=True)
        self.tbl_merged.reset_index(inplace=True, drop=True)

    def __dx_entry_from_phase(self) -> None:
        """Writes dx for each visit"""

        def other_dementia(prelim_dx, dxad):
            """Takes two list like arguments, one with a preliminary
            diagnosis, and the other
            with the diagnosis code. returns a converted diagnosis code
            containing a "1" where patients
            have 'Other Dementia'

            Args:
                prelim_dx (list-like): Diagnosis from choice of NL, MCI, or AD
                dxad (list-like): diagnosis code for AD vs other pathology

            Returns:
                list: boolean array where '1' corresponds to a diagnosis of AD
            """
            dx = []
            for x, y in zip(prelim_dx, dxad):
                if (not pd.isna(y)) and (y != 1) and (x == 'AD'):
                    dx.append(1)
                else:
                    dx.append(0)
            return dx

        # Set dx for each visit as np.nan and set OD to boolean value "0"
        self.tbl_merged['DX'] = np.nan
        self.tbl_merged['OD'] = 0

        for phase, subtbl in self.tbl_merged.groupby('Phase'):
            dx_values = subtbl[DX_CONVERSION_DICT[phase]['col']]
            converted_values = [DXTRANS[phase][int(x)] if not pd.isna(x)
                                else np.nan for x in
                                dx_values]
            od = other_dementia(
                    converted_values, subtbl[DX_CONVERSION_DICT[phase][
                        'dx']])
            self.tbl_merged.loc[subtbl.index, 'DX'] = converted_values
            self.tbl_merged.loc[subtbl.index, 'OD'] = od

        self.tbl_merged.drop(columns=[x for x in self.tbl_merged.columns if
                                      '_dxsum' in x], inplace=True)

    def __convert_monthly_visit(self) -> None:
        """This establishes a time course for observation of patients Using
        VISCODE2, the month # of study enrollment is obtained. Should be run
        following combine_bl_visits
        """

        def translate_month_code(val: str):
            """Takes as argument a VISCODE2 value in the form 'sc','bl',
            or r'm[0-9]+'
            and tells you the time point in the study

            Args:
                val (str): value from VISCODE2

            Returns:
                [type]: returns time into study
            """
            if val == 'bl':
                return 0
            elif val == 'sc':
                return np.nan
            else:
                return int(val.replace('m', ''))

        self.tbl_merged['VISCODE3'] = self.tbl_merged['VISCODE2'].apply(
                translate_month_code)

    def __combine_bl_visits(self):
        """Summary"""
        # RIDs are divided based on when they enrolled: <2000 = ADNI1,
        # 2000 - < 4000 = GO, 4000 - <6000 = ADNI2 \
        # and >= 6000 = ADNI3. Each has a different "value" for baseline.
        # for ADNI1 visits: baseline visit contains all that we need.
        # We can simply combine these visits
        participant_table = self.tbl_merged
        participant_table_new = pd.DataFrame(data=None,
                                             columns=participant_table.columns)
        n_dropped = 0
        for rid, sub_df in participant_table.groupby('RID'):
            sub_df = sub_df.copy()
            if (6000 > rid >= 2000) and not np.any(sub_df['VISCODE2']
                                                   == 'scmri'):
                n_dropped += 1
                continue
            if np.sum(sub_df['VISCODE2'] == 'bl') != 1 or \
                    np.sum(sub_df['VISCODE2'] == 'sc') != 1:
                n_dropped += 1
                continue
            sub_df = self.__combine_sc_bl(sub_df)
            participant_table_new = participant_table_new.append(
                    sub_df, ignore_index=True)
        participant_table_new.reset_index(inplace=True, drop=True)
        logger.warning(f'\tDropped {n_dropped} patients')
        self.tbl_merged = participant_table_new

    @staticmethod
    def __combine_sc_bl(sub_df: pd.DataFrame) -> pd.DataFrame:
        """Given a dataframe for a single subject, combine bl, sc, and scmri in
        the order:

            bl > sc > scmri

        Args:
            sub_df (pd.DataFrame): should be a dataframe for a single subject

        Returns:
            pd.DataFrame: dataframe with bl, sc, and scmri combined
        """
        """
        obtain rows, each of length 1, and set bl_visit's examdate to nat
        """
        bl_visit = sub_df[sub_df.VISCODE2 == 'bl'].copy()
        bl_visit.loc[:, 'EXAMDATE'] = pd.NaT
        sc_visit = sub_df[sub_df.VISCODE2 == 'sc'].copy()
        scmri_visit = sub_df[sub_df.VISCODE2 == 'scmri'].copy()

        # combine these rows, remembering to reset index
        bl_visit_new = bl_visit.reset_index(drop=True).combine_first(
                sc_visit.reset_index(drop=True)).combine_first(
                scmri_visit.reset_index(drop=True))

        # drop the indices with the initial visits
        sub_df.drop(index=sc_visit.index, inplace=True)
        sub_df.drop(index=scmri_visit.index, inplace=True)
        sub_df.drop(index=bl_visit.index, inplace=True)

        # append the new, merged bl_visit and reset the indices for the
        # dataframe
        sub_df = sub_df.append(bl_visit_new, ignore_index=True)
        sub_df.reset_index(inplace=True, drop=True)
        return sub_df

    def __merge_apoe(self, apoe_tbl: pd.DataFrame) -> None:
        """Merges apoe with registration data. Assigns single
            APOE1 and APOE2 types to every registration entry

        Args:
            apoe_tbl (pd.DataFrame): a pandas dataframe from the ApoE datasheet
        """
        table = self.tbl_merged
        assert (len(apoe_tbl) == len(pd.unique(apoe_tbl.RID)))
        assert all([type(x) != list for x in apoe_tbl])
        for rid, tbl in table.groupby('RID'):
            apoe1 = apoe_tbl[apoe_tbl.RID == rid]['APGEN1'].to_numpy()
            apoe2 = apoe_tbl[apoe_tbl.RID == rid]['APGEN2'].to_numpy()
            table.loc[tbl.index, 'APOE1'] = apoe1[
                0] if apoe1.size != 0 else np.nan
            table.loc[tbl.index, 'APOE2'] = apoe2[
                0] if apoe2.size != 0 else np.nan
        self._make_apoe(table)

    @staticmethod
    def _make_apoe(table: pd.DataFrame) -> None:
        table.loc[:,'APOE'] = np.sum(
            [
                table.APOE1 == 4,
                table.APOE2 == 4
                ], axis=0
            )
        table.drop(columns=['APOE1', 'APOE2'], inplace=True)

    def __add_age(self) -> None:
        """Adds a column corresponding to the patient's age to tbl_merged
        assumes birthdayis the first of the month
        """
        tbl = self.tbl_merged
        tbl.loc[:, 'AGE'] = np.nan
        day = 1
        _slice = ['PTDOBMM_demo', 'PTDOBYY_demo', 'EXAMDATE']
        """
        Iterate through all of the patients. Get the birthday information
        stored in each patient's baseline visit and compute time at each
        visit using exam date
        """
        for _, subtbl in tbl.groupby('RID'):
            bl = subtbl.loc[subtbl.VISCODE2 == 'bl', _slice].to_numpy()
            assert (len(bl) == 1)
            month, year, _ = bl[0]
            if np.any(pd.isna(bl[0])):
                tbl.loc[subtbl.index, 'AGE'] = np.nan
            else:
                birthday = datetime.date(year=int(year), month=int(month),
                                         day=day)
                date_dif = subtbl.EXAMDATE - birthday
                tbl.loc[subtbl.index, 'AGE'] = [
                        np.nan if pd.isna(x) else round(x.days / 365.25) for x
                        in date_dif]

    def __add_demographics(self) -> None:
        rgp = re.compile(r'.*_demo')
        demo_fields = list(filter(rgp.match, self.tbl_merged.columns))
        for _, tbl in self.tbl_merged.groupby(['RID']):
            tbl = tbl.sort_values('VISCODE3')
            bl_values = tbl.iloc[0, :][demo_fields].copy().to_dict()
            self.tbl_merged.loc[tbl.index, :] = tbl.assign(**bl_values)

    def __add_medical_hx(self) -> None:
        rgp = re.compile(r'.*_medhx')
        med_fields = list(filter(rgp.match, self.tbl_merged.columns))
        for _, tbl in self.tbl_merged.groupby(['RID', 'Phase']):
            tbl = tbl.sort_values('VISCODE3')
            tbl.loc[:, med_fields] = tbl.loc[:,med_fields].ffill(axis=0)
            self.tbl_merged.loc[tbl.index, :] = tbl

    def progress_to_ad(self) -> None:
        """This function creates three columns: whether or not a subject
        progresses to dementia, the time it takes for the subject to progress,
        and, for patients with 'other dementias' the time it takes to reach
        other dementias NOTE: we ignore reversion cases
        """
        """First, initialize all columns: PROGRESSES, TIME_TO_PROGRESSION, 
        and TIME_TO_OD
        """
        tbl = self.tbl_merged
        tbl['PROGRESSES'] = np.nan
        tbl['TIME_TO_PROGRESSION'] = np.nan
        tbl['TIME_TO_OD'] = np.nan
        reversions = 0
        reversion_rids = []

        """Iterate through each subject.
        If a patient has an AD diagnosis, determine if all subsequent visits 
        had an AD diagnosis
            if this holds, set time to progression
            if the patient 'reverts', set to np.nan
        If a patient does not have an AD diagnosis, they do not 'progress'
        If a patient has an OD diagnosis somewhere, find the first time that 
        they have this diagnosis.
            set OD to 1/true
        """
        for rid, subtbl in tbl.groupby('RID'):
            if all([pd.isna(x) for x in subtbl.VISCODE3]):
                continue
            subtbl = subtbl.sort_values(by='VISCODE3').copy()
            assert (subtbl.iloc[0, :].VISCODE2 == 'bl')
            all_dates = subtbl['VISCODE3']
            if any(subtbl.DX == 'AD'):
                initial_ad_date = min(all_dates[subtbl.DX == 'AD'])
                dx_after_initial = subtbl.DX[[x > initial_ad_date
                                              for x in all_dates]]
                if all([(x == 'AD' or pd.isna(x)) for x in dx_after_initial]):
                    tbl.loc[subtbl.index, 'PROGRESSES'] = 1
                    datedif = initial_ad_date - all_dates
                    tbl.loc[subtbl.index, 'TIME_TO_PROGRESSION'] = datedif
                else:  # reversion cases
                    reversions += 1
                    reversion_rids.append(rid)
                    tbl.loc[subtbl.index, 'PROGRESSES'] = np.nan
                    tbl.loc[subtbl.index, 'TIME_TO_PROGRESSION'] = np.nan
            else:
                tbl.loc[subtbl.index, 'PROGRESSES'] = 0
                tbl.loc[subtbl.index, 'TIME_TO_PROGRESSION'] = np.nan
            if any(subtbl.OD == 1):
                initial_od_date = min(all_dates[subtbl.OD == 1])
                datedif = initial_od_date - all_dates
                tbl.loc[subtbl.index, 'TIME_TO_OD'] = datedif
                tbl.loc[subtbl.index, 'OD'] = 1

    def progress_to_ad_cumulative(self) -> None:
        """This function creates three columns: whether or not a subject
        progresses to dementia, the time it takes for the subject to progress,
        and, for patients with 'other dementias' the time it takes to reach
        other dementias NOTE: we ignore reversion cases
        """
        """First, initialize all columns: PROGRESSES, TIME_TO_PROGRESSION, 
        and TIME_TO_OD
        """
        tbl = self.tbl_merged
        tbl['PROGRESSES'] = np.nan
        tbl['TIME_TO_PROGRESSION'] = np.nan
        tbl['TIME_TO_OD'] = np.nan

        """Iterate through each subject.
        If a patient has an AD diagnosis, determine if all subsequent visits 
        had an AD diagnosis
            if this holds, set time to progression
            if the patient 'reverts', set to np.nan
        If a patient does not have an AD diagnosis, they do not 'progress'
        If a patient has an OD diagnosis somewhere, find the first time that 
        they have this diagnosis.
            set OD to 1/true
        """
        for rid, subtbl in tbl.groupby('RID'):
            if all([pd.isna(x) for x in subtbl.VISCODE3]):
                continue
            subtbl = subtbl.sort_values(by='VISCODE3').copy()
            assert (subtbl.iloc[0, :].VISCODE2 == 'bl')
            all_dates = subtbl['VISCODE3']
            if any(subtbl.DX == 'AD'):
                initial_ad_date = min(all_dates[subtbl.DX == 'AD'])
                tbl.loc[subtbl.index, 'PROGRESSES'] = 1
                datedif = initial_ad_date - all_dates
                tbl.loc[subtbl.index, 'TIME_TO_PROGRESSION'] = datedif
            else:
                tbl.loc[subtbl.index, 'PROGRESSES'] = 0
                tbl.loc[subtbl.index, 'TIME_TO_PROGRESSION'] = np.nan
            if any(subtbl.OD == 1):
                initial_od_date = min(all_dates[subtbl.OD == 1])
                datedif = initial_od_date - all_dates
                tbl.loc[subtbl.index, 'TIME_TO_OD'] = datedif
                tbl.loc[subtbl.index, 'OD'] = 1

    def has_csf(self):
        """Determine whether or not each patient has CSF data, and whether or
        not it is from the first spreadsheet or second
        """
        tbl = self.tbl_merged
        csf_columns1 = tbl[['ABETA_csf_1', 'TAU_csf_1',
                            'PTAU_csf_1']]
        csf_columns2 = tbl[['PTAU_csf_2', 'ABETA42_csf_2',
                            'TAU_csf_2']]
        has_csf1 = csf_columns1.apply(lambda x: not pd.isna(x).all(), axis=1)
        has_csf2 = csf_columns2.apply(lambda x: not pd.isna(x).all(), axis=1)
        tbl.loc[:, 'has_csf'] = np.logical_or(has_csf1, has_csf2)
        tbl.loc[:, 'has_csf1'] = has_csf1
        tbl.loc[:, 'has_csf2'] = has_csf2

    def merge_csf_columns(self):
        """Merge csf values from both csf columns. Drop initial columns and
        unlist the merged columns
        """
        self.replace_abeta_vals()
        csf_columns2 = ['PTAU_csf_2', 'ABETA42_csf_2',
                        'TAU_csf_2']
        self.tbl_merged.loc[self.tbl_merged['has_csf1'].to_numpy(),
                            csf_columns2] = np.nan
        csf_merge_columns = {
                'tau': ['TAU_csf_1', 'TAU_csf_2'],
                'abeta': ['ABETA_csf_1', 'ABETA42_csf_2'],
                'ptau': ['PTAU_csf_1', 'PTAU_csf_2']
        }
        for key in csf_merge_columns.keys():
            cols = csf_merge_columns[key]
            self.merge_columns(cols, key)
            self.tbl_merged.drop(columns=cols, inplace=True)
            self.drop_nans(key)
            self.unlist(key)

    def replace_abeta_vals(self):
        """Examines ABETA_csf_1 for entries matching the regular expression that
        follows. Replace the values that currently exist here with the
        'recalculated' ABETA result
        """
        tbl = self.tbl_merged
        abeta_val = re.compile(
                '^Recalculated ABETA result = (?P<abeta_val>[0-9]+) pg/mL$')
        output_values = []
        for loc, x in enumerate(tbl.COMMENT_csf_1):
            match = abeta_val.match(str(x))
            if match:
                output_values.append(match['abeta_val'])
            else:
                output_values.append(tbl.ABETA_csf_1.iloc[loc])
        tbl['ABETA_csf_1'] = output_values

    def combine_tau(self) -> None:
        """Combine tau entries from both tau2 and tau3"""
        tbl = self.tbl_merged
        columns = ['DONE_tau2', 'DONE_tau3', 'EXAMDATE_tau3', 'EXAMDATE_tau2']
        [self.listify(col) for col in columns]
        tbl['DONE_tau'] = [x + y for x, y in
                           zip(tbl.DONE_tau2,
                               tbl.DONE_tau3)]
        tbl['DATE_tau'] = [x + y for x, y in
                           zip(tbl.EXAMDATE_tau2,
                               tbl.EXAMDATE_tau3)]
        self.drop_nans('DONE_tau')
        self.unlist('DONE_tau')
        self.drop_nans('DATE_tau')
        tbl.drop(columns=columns, inplace=True)

    def combine_fdg(self) -> None:
        """Performs the same function as combine_tau"""
        tbl = self.tbl_merged
        columns = ['DONE_fdg1', 'EXAMDATE_fdg1',
                   'DONE_fdg2go', 'EXAMDATE_fdg2go',
                   'DONE_fdg3', 'EXAMDATE_fdg3']
        [self.listify(col) for col in columns]
        tbl['DONE_fdg'] = [x + y + z for x, y, z in zip(tbl.DONE_fdg1,
                                                        tbl.DONE_fdg2go,
                                                        tbl.DONE_fdg3)]
        tbl['DATE_fdg'] = [x + y + z for x, y, z in zip(tbl.EXAMDATE_fdg1,
                                                        tbl.EXAMDATE_fdg2go,
                                                        tbl.EXAMDATE_fdg3)]
        self.drop_nans('DONE_fdg')
        self.unlist('DONE_fdg')
        self.drop_nans('DATE_fdg')
        tbl.drop(columns=columns, inplace=True)

    def combine_amyloid(self) -> None:
        """Merges amyloid columns"""
        tbl = self.tbl_merged
        columns = ['DONE_amyloid', 'DONE_amyloid2', 'EXAMDATE_amyloid',
                   'EXAMDATE_amyloid2']
        self.unlist('TRACERTYPE_amyloid2')
        self.unlist('DONE_amyloid2')
        """Determine whether or not the Tracertype value was 2
        """
        done_amy2 = [1 if (x == 1 and y == 2) else np.nan for x, y in
                     zip(self.tbl_merged[
                             'DONE_amyloid2'].to_numpy(),
                         self.tbl_merged[
                             'TRACERTYPE_amyloid2'].to_numpy())]
        test = [1 if x == 2 else np.nan for x in
                np.multiply(self.tbl_merged[
                                'DONE_amyloid2'].to_numpy(),
                            self.tbl_merged[
                                'TRACERTYPE_amyloid2'].to_numpy())]
        assert (np.array_equal(done_amy2, test, equal_nan=True))
        self.tbl_merged['DONE_amyloid2'] = done_amy2
        [self.listify(col) for col in columns]
        tbl['DONE_amyloid'] = [x + y for x, y in
                               zip(tbl.DONE_amyloid,
                                   tbl.DONE_amyloid2)]
        tbl['DATE_amyloid'] = [x + y for x, y in
                               zip(tbl.EXAMDATE_amyloid,
                                   tbl.EXAMDATE_amyloid2)]
        self.drop_nans('DONE_amyloid')
        self.drop_nans('DATE_amyloid')
        self.unlist('DONE_amyloid')
        tbl.drop(columns=['DONE_amyloid2', 'EXAMDATE_amyloid2',
                          'EXAMDATE_amyloid'], inplace=True)


    def get_progression_data_time_to_progress(self):
        """Currently unused; gets the time to progression

        Returns:
            pd.DataFrame: time to progress
        """
        rid_dict = self.retrieve_column_intersections(
                {
                        'DONE_mri3': 1,
                        'has_csf': 1,
                        'DX': 'MCI'
                }
        )
        n_sub = len(rid_dict.keys())
        logger.warning(f'number subjects prior to longitudinal: {n_sub}')
        imaging_table = self.has_longitudinal_info_time(
                rid_dict)
        logger.warning(f'number final subjects: {len(imaging_table)}')
        return imaging_table

    def get_progression_data_time_to_progress_nocsf(self):
        """Currently unused; gets the time to progression

        Returns:
            pd.DataFrame: time to progress
        """
        rid_dict = self.retrieve_column_intersections(
                {
                        'DONE_mri3': 1,
                        'DX': 'MCI'
                }
        )
        n_sub = len(rid_dict.keys())
        print(rid_dict.keys())
        imaging_table = self.has_longitudinal_info_time(
                rid_dict)
        return imaging_table

    def has_longitudinal_info(self, rid_image_dict: dict,
                              n_months=24) -> pd.DataFrame:
        """
        Args:
            rid_image_dict (dict):
            n_months:
        """
        new_table = pd.DataFrame(data=None)
        for rid, tbl in self.tbl_merged.groupby('RID'):
            if rid not in rid_image_dict.keys():
                continue
            imaging_visits = tbl.loc[rid_image_dict[rid], :].sort_values(
                    'VISCODE3')
            last_visit = max(tbl.VISCODE3)
            for _, row in imaging_visits.iterrows():
                current_visit = row.VISCODE3
                if (last_visit - current_visit) < n_months:
                    break
                if (row['PROGRESSES'] == 1) & (0 < row['TIME_TO_PROGRESSION'] <=
                                               n_months):
                    new_table = new_table.append(row, ignore_index=True)
                    break
                elif row['PROGRESSES'] == 0:
                    new_table = new_table.append(row, ignore_index=True)
                    break
        return new_table

    def get_progression_data_time_to_progress_ad(self):
        """Currently unused; gets the time to progression

        Returns:
            pd.DataFrame: time to progress
        """
        rid_dict = self.retrieve_column_intersections(
                {
                        'DONE_mri3': 1,
                        'has_csf': 1,
                        'DX': 'MCI'
                }
        )
        imaging_table = self.has_longitudinal_info_time(
                rid_dict)
        rid_dict_ad = self.retrieve_column_intersections(
            {
                'DONE_mri3': 1,
                'DX': 'AD'
            }
        )
        imaging_table = self.merge_imaging_table_with_ad_imaging(imaging_table, rid_dict_ad)
        return imaging_table

    def has_longitudinal_info_time(self, rid_image_dict: dict) -> pd.DataFrame:
        """looks for time to progression for each image in rid_image_dict
            -iterate through all rids -get first visit with all of these
            modalities and and append to new table

        Args:
            rid_image_dict (dict): dict with keys=patient ids, values=idx for

        Returns:
            pd.DataFrame: [description]
        """

        new_table = pd.DataFrame(data=None)
        for rid, tbl in self.tbl_merged.groupby('RID'):
            if rid not in rid_image_dict.keys():
                continue
            tbl = tbl.sort_values('VISCODE3')
            imaging_visits = tbl.loc[rid_image_dict[rid], :].sort_values(
                    'VISCODE3')
            if imaging_visits.iloc[0, :]['PROGRESSES'] == 1 and \
                    imaging_visits.iloc[0, :]['TIME_TO_PROGRESSION'] <= 0:  # would occur if progresses before first imaging visit
                continue
            dx_visits = tbl.loc[[not pd.isna(x) for x in tbl['DX']], :]  # visits where there is a dx

            imaging_visits['TIME_TO_FINAL_DX'] = \
                max(dx_visits['VISCODE3']) - imaging_visits.iloc[0, :]['VISCODE3']  # length of time under observation

            imaging_visits['PROGRESSION_CATEGORY'] = \
                self.get_progression_category(
                    imaging_visits.iloc[0, :][['TIME_TO_PROGRESSION',
                                              'TIME_TO_FINAL_DX']].to_numpy()
                )
            imaging_visits['PROGRESSION_CATEGORY_2YR'] = \
                self.get_2yr_category(
                    imaging_visits.iloc[0, :][['TIME_TO_PROGRESSION',
                                              'TIME_TO_FINAL_DX']].to_numpy()
                )
            imaging_visits_final = imaging_visits.iloc[0, :].copy()

            imaging_visits_final['DX_VALUES'] = ','.join(
                    [str(x) for x in dx_visits['DX']]
            )
            imaging_visits_final['DX_DATES'] = ','.join(
                    [str(x) for x in dx_visits['VISCODE3']]
            )
            visit_diffs = np.cumsum(np.diff([int(x) for x in dx_visits['VISCODE3']]))
            imaging_visits_final['DX_STEPS'] = '0,' + ','.join(
                    [str(x) for x in visit_diffs]
            )
            new_table = new_table.append(imaging_visits_final,
                                         ignore_index=True)
        self.retrieve_times(new_table)
        return new_table

    def merge_imaging_table_with_ad_imaging(self, imaging_table: pd.DataFrame, rid_image_dict: dict) -> pd.DataFrame:
        """looks for time to progression for each image in rid_image_dict
            -iterate through all rids -get first visit with all of these
            modalities and and append to new table

        Args:
            rid_image_dict (dict): dict with keys=patient ids, values=idx for

        Returns:
            pd.DataFrame: [description]
        """
        imaging_table['AD_MRI_DATE'] = ''
        for rid, _ in imaging_table.groupby('RID'):
            if rid not in rid_image_dict.keys():
                continue
            tbl = self.tbl_merged.query('RID == @rid').copy()
            ad_visit = tbl.query('TIME_TO_PROGRESSION == 0')
            if len(ad_visit) < 1:
                continue
            if len(ad_visit) > 1:
                raise ValueError
            ad_visit = ad_visit.iloc[0,:]
            if type(ad_visit['EXAMDATE_mri3']) == float and np.isnan(ad_visit['EXAMDATE_mri3']):
                continue
            imaging_table.loc[imaging_table['RID'] == rid,'AD_MRI_DATE'] = ','.join([str(x) for x in ad_visit['EXAMDATE_mri3']])
        return imaging_table

def _retrieve_ad_mri_dates(imaging_visits: pd.DataFrame):
    mri_visit = imaging_visits.query('TIME_TO_PROGRESSION == 0')
    print(mri_visit)
    print(imaging_visits.TIME_TO_PROGRESSION)
    if len(mri_visit) > 0:
        return ','.join([str(x) for x in mri_visit['EXAMDATE_mri3']])
    else:
        return ''

def n_participants_wrapper(fn):
    def n_participants(self, *args, **kwargs):
        output = fn(self, *args, **kwargs)
        logger.warning(f'{fn.__name__}: n participants is {self.n_subjects()}')
        return output
    return n_participants

class NaccCollection(Cohort):
    def __init__(self, file_list: Dict = NACC_FILE_LIST, threshold=183) -> None:
        """
        Args:
            file_list (Dict):
            threshold:
        """
        self.threshold = threshold
        super(NaccCollection, self).__init__(
                file_list=file_list
        )

    def _modify_data_from_csv(self, data_fi: pd.DataFrame,
                             csv_dict: MetadataCSV) -> pd.DataFrame:
        """
        For each file in the file_list, creates an EXAMDATE column upon reading.
        Args:
            data_fi (pd.DataFrame): dataframe loaded and parsed per metaclass
            csv_dict (MetadataCSV): takes this as argument per metaclass
        """
        data_fi['EXAMDATE'] = data_fi[
            ['VISITMO', 'VISITDAY', 'VISITYR']
        ].apply(
                lambda x: datetime.date(year=int(x.VISITYR), month=int(
                        x.VISITMO), day=x.VISITDAY), axis=1
        )
        data_fi.drop(
                columns=['VISITMO', 'VISITDAY', 'VISITYR'],
                inplace=True
        )
        return data_fi

    def _pre_process(self):
        """
        Code run prior to merging the reg and mri dataframes
        """
        self._get_dx_values()
        self._merge_csf_values()
        self._identify_first_mci_visit()
        self._identify_first_alzheimers_dementia_visit()
        self._identify_time_to_progression()
        self._identify_final_obs_date()
        self._compute_time_to_final_visit()
        self.df = self._find_all_optimal_imaging_visits('MCI')
        self.df_ad = self._find_all_optimal_imaging_visits('AD', query_visit=self.df)
        self._identify_date_of_death()

    def _post_process(self, **kwargs):
        """
        Args:
            **kwargs: This is a dummy argument. Overrides _post_process from
            metaclass. Code to be run after merging.
        """
        self._identify_all_dx_visits()
        self._drop_idx_without_mri()

    @n_participants_wrapper
    def _get_dx_values(self) -> None:
        """
        For the reg table, pass in a sub dataframe with diagnosis columns,
        assign to DX
        """
        self.tables['reg']['DX'] = self.tables['reg'][
            ['NACCUDSD', 'NACCTMCI', 'NACCALZD']
        ].apply(
                lambda x: self._dx_dictionary(x), axis=1
        )
        self.tables['reg'].drop(
                columns=['NACCUDSD', 'NACCTMCI', 'NACCALZD'], inplace=True
        )

    @staticmethod
    def _dx_dictionary(dx_dataframe: pd.DataFrame) -> str:
        """
        Args:
            dx_dataframe (pd.DataFrame): a dictionary with a single row
            containing the three columns 'NACCUDSD', 'NACCTMCI', 'NACCALZD'.
            Assigns a diagnosis based on RDD.
        """
        if dx_dataframe['NACCUDSD'] == 1:
            return 'NC'
        elif (dx_dataframe['NACCUDSD'] == 3) or (dx_dataframe['NACCTMCI']
                                                 in [1, 2, 3, 4]):
            return 'MCI'
        elif dx_dataframe['NACCUDSD'] == 4:
            if dx_dataframe['NACCALZD'] == 1:
                return 'AD'
            else:
                return 'OD'
        elif dx_dataframe['NACCUDSD'] == 2:
            return 'IMP'
        else:
            raise ValueError(f'No dementia specified: {dx_dataframe}')

    @n_participants_wrapper
    def _identify_first_mci_visit(self) -> None:
        """
        For each patient, identifies the first visit at which they had a
        diagnosis of MCI. If they haven't received a diagnosis of MCI at all,
        that patient is dropped.
        """
        self.tables['reg'].loc[:, 'FIRST_MCI'] = pd.NaT
        idx_to_drop = []
        n_rids_dropped = 0
        for rid, table in self.tables['reg'].groupby('RID'):
            mci_visits = table.query('DX == \'MCI\'')
            if len(mci_visits) == 0:
                idx_to_drop += list(table.index)
                n_rids_dropped += 1
                continue
            first_mci_date = min(mci_visits['EXAMDATE'])
            self.tables['reg'].loc[table.index, 'FIRST_MCI'] = first_mci_date
        logger.warning(f'\tDropping {n_rids_dropped} with no MCI visits')
        self.tables['reg'].drop(index=idx_to_drop, inplace=True)

    def n_subjects(self):
        if hasattr(self, 'tbl_merged'):
            if 'RID' in self.tbl_merged.index.names:
                tbl_copy = self.tbl_merged.copy().reset_index()
                return len(pd.unique(tbl_copy['RID']))
            elif 'RID' in self.tbl_merged.columns:
                return len(pd.unique(self.tbl_merged['RID']))
        return len(pd.unique(self.tables['reg']['RID']))

    @n_participants_wrapper
    def _identify_first_alzheimers_dementia_visit(self) -> None:
        """
        For each patient, identify first visit with AD, if it exists. If
        first visit with AD occurs before first MCI visit, drop the patient.
        Adds columns 'FIRST_AD_VISIT', 'PROGRESSES'
        """
        self.tables['reg'].loc[:, 'FIRST_AD_VISIT'] = pd.NaT
        self.tables['reg'].loc[:, 'FIRST_AD_VISIT_NO'] = np.nan
        self.tables['reg'].loc[:, 'PROGRESSES'] = 0
        idx_to_drop = []
        ad_rid_to_drop = 0
        for rid, table in self.tables['reg'].groupby('RID'):
            ad_visits = table.query('DX == \'AD\'')
            # logger.info(f'\n\t{table.DX.to_numpy()} =====> {ad_visits.DX.to_numpy()},{ad_visits.EXAMDATE.to_numpy()}\n')
            if len(ad_visits) == 0:
                continue
            visit_date = min(ad_visits['EXAMDATE'])
            visit_idx_temp = ad_visits.query('EXAMDATE == @visit_date')['Visit']
            visit_idx = min(ad_visits['Visit'])
            assert(all(visit_idx == visit_idx_temp) and len(visit_idx_temp) == 1)
            assert(len(pd.unique(ad_visits['FIRST_MCI'])) == 1)
            if visit_date > ad_visits.iloc[0, :]['FIRST_MCI']:
                self.tables['reg'].loc[table.index, 'PROGRESSES'] = 1
                self.tables['reg'].loc[
                    table.index, 'FIRST_AD_VISIT'
                ] = visit_date
                self.tables['reg'].loc[table.index, 'FIRST_AD_VISIT_NO'] = visit_idx
                # logger.info(self.tables['reg'].loc[table.index,['PROGRESSES','FIRST_AD_VISIT']])
            else:
                idx_to_drop += list(table.index)
                ad_rid_to_drop += 1
        logger.warning(f'\tDropping {ad_rid_to_drop} patients with an AD visit before MCI visit')
        self.tables['reg'].drop(index=idx_to_drop, inplace=True)

    @n_participants_wrapper
    def _identify_time_to_progression(self) -> None:
        """
        Identifies time to progression for patients with AD. Adds column
        "TIME_TO_PROGRESSION"
        """
        self.tables['reg'].loc[:, 'TIME_TO_PROGRESSION'] = pd.NaT
        for rid, table in self.tables['reg'].groupby('RID'):
            time_to_progression = table['FIRST_AD_VISIT'] - table['EXAMDATE']
            time_to_progression = time_to_progression.apply(
                    lambda x: x.days * 12 / 365.25 if not type(x) == float
                    else x
            )
            self.tables['reg'].loc[table.index, 'TIME_TO_PROGRESSION'] = \
                time_to_progression
            # logger.info(self.tables['reg'].loc[table.index][['DX','EXAMDATE','TIME_TO_PROGRESSION']])

    @n_participants_wrapper
    def _identify_final_obs_date(self) -> None:
        """
        Identifies date of final diagnostic visit, "FINAL_OBS_DATE", for each
        RID.
        """
        self.tables['reg'].loc[:, 'FINAL_OBS_DATE'] = pd.NaT
        self.tables['reg'].loc[:, 'AGE_AT_FINAL_DX'] = np.nan
        for rid, table in self.tables['reg'].groupby('RID'):
            dx_visits = table.loc[[not pd.isna(x) for x in table['DX']], :]
            visit_date = max(dx_visits['EXAMDATE'])
            visit_age = max(dx_visits['AGE'])
            self.tables['reg'].loc[table.index, 'FINAL_OBS_DATE'] = visit_date
            self.tables['reg'].loc[table.index, 'AGE_AT_FINAL_DX'] = visit_age
            # logger.info(self.tables['reg'].loc[table.index,['DX','EXAMDATE','FINAL_OBS_DATE','AGE_AT_FINAL_DX','AGE']])

    @n_participants_wrapper
    def _identify_date_of_death(self) -> None:
        """
        Identifies date of final diagnostic visit, "FINAL_OBS_DATE", for each
        RID.
        """
        self.tables['reg'].loc[:, 'DATE_OF_DEATH'] = self.tables['reg'][
            ['NACCMOD', 'NACCYOD']
        ].apply(
                lambda x: datetime.date(year=int(x.NACCYOD), month=int(
                        x.NACCMOD), day=1) if not np.isnan(x.NACCYOD) else
                pd.NaT, axis=1
        )

    @n_participants_wrapper
    def _compute_time_to_final_visit(self):
        self.tables['reg']['TIME_TO_FINAL_DX'] = \
            self.tables['reg']['FINAL_OBS_DATE'] - \
            self.tables['reg']['EXAMDATE']
        self.tables['reg']['TIME_TO_FINAL_DX'] = self.tables['reg'][
            'TIME_TO_FINAL_DX'].apply(
                lambda x: x.days * 12 / 365.25 if not pd.isna(x) else x
        )
        # for rid, tbl in self.tables['reg'].groupby('RID'):
        #     logger.info(tbl.loc[:,['DX','EXAMDATE','FINAL_OBS_DATE','TIME_TO_FINAL_DX']])

    @staticmethod
    def _image_dx_combinations(
            reg_tbl: pd.DataFrame,
            mri_tbl: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            reg_tbl (pd.DataFrame): A registration table containing only MCI
            visits for a particular RID
            mri_tbl (pd.DataFrame): An MRI table containing all MRIs for a
            particular RID

        Returns a dataframe containing all combinations of reg-mri visits and
        the associated time between the visits
        """
        idx = pd.MultiIndex.from_product(
                [reg_tbl['Visit'],
                 mri_tbl['Visit']]
        )
        reg_tbl.set_index('Visit', inplace=True)
        mri_tbl.set_index('Visit', inplace=True)
        df = pd.DataFrame(data=None, index=idx)
        df.loc[(slice(None), slice(None)),
               'DATE_RANGE'] = pd.NaT
        for index, row in df.iterrows():
            sub_list = [
                    reg_tbl.loc[index[0], 'EXAMDATE'],
                    mri_tbl.loc[index[1], 'EXAMDATE'],
            ]
            df.loc[index, 'DATE_RANGE'] = (max(sub_list) - min(sub_list)
                    ).days
        return df

    def __optimal_visit_for_rid(self, rid: str,
                                image_dx_combinations_for_rid: pd.DataFrame,
                                visit_idx: Dict, mci=True, query_visit=None) -> Dict:
        """
        Args:
            rid (str): RID for a patient
            image_dx_combinations_for_rid (pd.DataFrame): dataframe
                output of _image_dx_combinations
            visit_idx (Dict): previous output of this function to be appended
        """
        image_dx_combinations_for_rid = image_dx_combinations_for_rid.copy()
        if query_visit is not None:
            if query_visit in image_dx_combinations_for_rid.index.get_level_values(0):
                image_dx_combinations_for_rid = image_dx_combinations_for_rid.loc[(query_visit, slice(None)),:]
            else:
                return visit_idx

        image_dx_combinations_for_rid['DATE_RANGE_ABS'] = \
            image_dx_combinations_for_rid['DATE_RANGE'].apply(abs)
        min_length_of_time = min(image_dx_combinations_for_rid[
                                     'DATE_RANGE_ABS'])
        idx = image_dx_combinations_for_rid['DATE_RANGE_ABS'].idxmin()
        if min_length_of_time < self.threshold:
            visit_idx['RID'].append(rid)
            visit_idx['Visit_REG'].append(idx[0])
            visit_idx['Visit_MRI'].append(idx[1])
            visit_idx['Time_between'].append(
                    image_dx_combinations_for_rid.loc[idx, 'DATE_RANGE']
            )
            visit_idx['MCI_MRI_VISIT'].append(mci)
        return visit_idx

    def _find_all_optimal_imaging_visits(self, dx='MCI', query_visit=None) -> None:
        """
        Creates a dataframe, df. For each RID, identifies the MCI-MRI visit
        combination closest in timing.
        """
        visit_idx = {
                'RID': [],
                'Visit_REG': [],
                'Visit_MRI': [],
                'Time_between': [],
                'MCI_MRI_VISIT': []
        }
        for rid, subtbl in self.tables['reg'].groupby('RID'):
            subtbl = subtbl.query('DX == @dx').copy()
            if len(subtbl) == 0:
                continue
            mri_tbl = self.tables['mri'].query('RID == @rid')
            if len(mri_tbl) == 0:
                continue
            image_dx_combinations_for_rid = self._image_dx_combinations(
                    subtbl, mri_tbl
            )
            if query_visit is not None:
                if rid not in query_visit.index:
                    continue
                idx = subtbl['FIRST_AD_VISIT_NO'].iloc[0].copy()
                visit_idx = self.__optimal_visit_for_rid(
                        rid, image_dx_combinations_for_rid, visit_idx, dx == 'MCI', query_visit=idx
                )
            else:
                visit_idx = self.__optimal_visit_for_rid(
                        rid, image_dx_combinations_for_rid, visit_idx, dx == 'MCI'
                )
        return pd.DataFrame.from_dict(visit_idx).set_index('RID')

    def _merge_tables(self) -> None:
        """
        Merged reg and mri. Overrides method from metaclass.
        """
        def add_columns(row: pd.Series, suffix: str, row_setter: pd.Series):
            """
            Args:
                row (pd.Series): row from _find_all_optimal_imaging_visits output.
            """
            self.tbl_merged.loc[(row['RID'], row['Visit_REG']),
                                'Time_between_visits' + suffix] = row_setter[
                'Time_between']
            self.tbl_merged.loc[(row['RID'], row['Visit_REG']),
                                'File_MRI' + suffix] = \
                self.tables['mri'].loc[(row_setter['RID'], row_setter['Visit_MRI']),
                                    'File_MRI']
            self.tbl_merged.loc[(row['RID'], row['Visit_REG']), 'DATE_MRI' + suffix] \
                = self.tables['mri'].loc[(row_setter['RID'], row_setter['Visit_MRI']),
                                        'EXAMDATE']
            if suffix != '_AD':
                self.tbl_merged.loc[(row['RID'], row['Visit_REG']), 'MCI_MRI_VISIT'] = True

        for key in ['mri', 'reg']:
            self.tables[key] = self.tables[key].reset_index(
                    drop=True).set_index(
                    ['RID', 'Visit']
            )
        self.tbl_merged = self.tables['reg'].copy()
        self.df.reset_index(inplace=True)
        self.df_ad.reset_index(inplace=True)

        file_column_name = 'File_MRI'
        date_column_name = 'DATE_MRI'
        time_between_name = 'Time_between_visits'

        self.tbl_merged.loc[:, file_column_name] = np.nan
        self.tbl_merged.loc[:, date_column_name] = pd.NaT
        self.tbl_merged.loc[:, time_between_name] = np.nan
        self.tbl_merged.loc[:, 'MCI_MRI_VISIT'] = False

        file_column_name = 'File_MRI_AD'
        date_column_name = 'DATE_MRI_AD'
        time_between_name = 'Time_between_visits_AD'

        self.tbl_merged.loc[:, file_column_name] = np.nan
        self.tbl_merged.loc[:, date_column_name] = pd.NaT
        self.tbl_merged.loc[:, time_between_name] = np.nan
        self.tbl_merged.loc[:, 'MCI_MRI_VISIT'] = False

        for _, row in self.df.iterrows():
            add_columns(row, '', row)
        
        for _, row_ad in self.df_ad.iterrows():
            rid = row_ad['RID']
            row = self.df.query('RID == @rid')
            assert(len(row) <= 1)
            row = row.iloc[0]
            if row['Visit_MRI'] >= row_ad['Visit_MRI']:
                continue
            add_columns(row, '_AD', row_ad)
        tmp_tbl = self.tbl_merged.copy().reset_index()
        tbl_len = len(pd.unique(tmp_tbl['RID']))
        logger.warning(f'Length of merged table is {tbl_len}')

    @n_participants_wrapper
    def _identify_all_dx_visits(self):
        self.tbl_merged['DX_DATES'] = ''
        self.tbl_merged['DX_VALUES'] = ''
        self.tbl_merged['DX_STEPS'] = ''
        for rid, tbl in self.tbl_merged.groupby('RID'):
            if any(tbl.MCI_MRI_VISIT == True):
                tbl = tbl.sort_values('EXAMDATE')
                assert(~any([np.isnan(x) for x in tbl.DX if type(x) is not str]))
                first_mci_mri_visit = tbl.query('MCI_MRI_VISIT == True').iloc[0,:]['EXAMDATE']
                visits_after_initial = tbl.loc[tbl['EXAMDATE'] >= first_mci_mri_visit, ['DX','EXAMDATE']]
                visits_after_initial_np = np.concatenate([
                        [visits_after_initial.iloc[0]['EXAMDATE']],visits_after_initial['EXAMDATE'].to_numpy()
                    ])
                diff_months = np.diff(visits_after_initial_np)
                diff_months = np.vectorize(lambda x: x.days*12/365.25, otypes=[float])(diff_months)
                diff_months = np.cumsum(diff_months)
                self.tbl_merged.loc[tbl.index, 'DX_DATES'] = ','.join([str(round(x)) for x in diff_months])
                self.tbl_merged.loc[tbl.index, 'DX_STEPS'] = self.tbl_merged.loc[tbl.index, 'DX_DATES']
                self.tbl_merged.loc[tbl.index, 'DX_VISITS'] = ','.join(visits_after_initial['DX'].to_numpy())
    
    @n_participants_wrapper
    def _drop_idx_without_mri(self):
        """
        Drop each index of tbl_merged t does not have an mri
        """
        self.tbl_merged.reset_index(inplace=True)
        # no_mri = [
        #     pd.isna(x) and pd.isna(y) for x,y in
        #     zip(self.tbl_merged['File_MRI'].to_numpy(), self.tbl_merged['File_MRI_AD'])
        #     ]
        no_mri = [
            pd.isna(x) for x in self.tbl_merged['File_MRI'].to_numpy()
            ]
        self.tbl_merged.drop(index=self.tbl_merged.loc[no_mri, :].index,
                             inplace=True)
        self.tbl_merged.reset_index(drop=True, inplace=True)
        self.tbl_merged.set_index(['RID', 'Visit'], inplace=True)

    @staticmethod
    def __ad_before_mri(rid_df: pd.DataFrame):
        """
        Args:
            rid_df (pd.DataFrame): dataframe with a single row. Returns true if
            AD visit occured prior to ad
        """
        return rid_df['FIRST_AD_VISIT'] < rid_df['DATE_MRI']

    def get_progression_data_time_to_progress(self) -> pd.DataFrame:
        """
        Finds MCI visits. identifies first visit w/ corresponding MRI,
        assuming visit and MRI are both prior to AD conversion. Returns tbl.
        """
        self.tbl_merged.reset_index(inplace=True)
        final_visits = []
        n_no_mci_visit = 0
        n_ad_before_mci = 0
        n_ad_before_mri = 0
        for rid, rid_tbl, in self.tbl_merged.groupby('RID'):
            rid_tbl = rid_tbl.sort_values('EXAMDATE')
            mci_visits = rid_tbl.query('DX == \'MCI\'').copy()
            if len(mci_visits) == 0:
                n_no_mci_visit += 1
                continue
            if mci_visits.iloc[0, :]['TIME_TO_PROGRESSION'] < 0:
                n_ad_before_mci += 1
                continue
            if self.__ad_before_mri(mci_visits.iloc[0, :]):
                n_ad_before_mri += 1
                continue
            mci_visits['PROGRESSION_CATEGORY'] = \
                    self.get_progression_category(
                        mci_visits.iloc[0, :][['TIME_TO_PROGRESSION',
                                          'TIME_TO_FINAL_DX']].to_numpy()
                    )
            final_visits.append(mci_visits.iloc[[0], :])
        logger.warning(f'\tN without MCI visit = {n_no_mci_visit}')
        logger.warning(f'\tN ad before MCI = {n_ad_before_mci}')
        logger.warning(f'\tN ad before MRI = {n_ad_before_mri}')
        final_tbl = pd.concat(final_visits, axis=0, ignore_index=True)
        self.retrieve_times(final_tbl)
        logger.warning(f'Final number: {len(final_tbl)}')
        return final_tbl

    def get_progression_data_time_to_progress_ad(self) -> pd.DataFrame:
        """
        Finds MCI visits. identifies first visit w/ corresponding MRI,
        assuming visit and MRI are both prior to AD conversion. Returns tbl.
        """
        self.tbl_merged.reset_index(inplace=True)
        final_visits = []
        for rid, rid_tbl, in self.tbl_merged.groupby('RID'):
            rid_tbl = rid_tbl.sort_values('EXAMDATE')
            mci_visits = rid_tbl.query('DX == \'MCI\'').copy()
            if len(mci_visits) == 0:
                continue
            if mci_visits.iloc[0, :]['TIME_TO_PROGRESSION'] < 0:
                continue
            if self.__ad_before_mri(mci_visits.iloc[0, :]):
                continue
            mci_visits['PROGRESSION_CATEGORY'] = \
                    self.get_progression_category(
                        mci_visits.iloc[0, :][['TIME_TO_PROGRESSION',
                                          'TIME_TO_FINAL_DX']].to_numpy()
                    )
            final_visits.append(mci_visits.iloc[[0], :])
        final_tbl = pd.concat(final_visits, axis=0, ignore_index=True)
        self.retrieve_times(final_tbl)
        return final_tbl

    def __assign_closest_visit(self, left_tbl, query_tbl):
        """
        Args:
            left_tbl:
            query_tbl:
        """
        new_tbl = left_tbl.copy()
        for idx, row in left_tbl.iterrows():
            exam_date_diffs = row['EXAMDATE'] - query_tbl['EXAMDATE_csf']
            exam_date_diffs = exam_date_diffs.apply(abs)
            if exam_date_diffs.min().days < self.threshold:
                new_tbl.loc[idx, query_tbl.columns] = query_tbl.loc[
                                                      exam_date_diffs.idxmin(), :]
                new_tbl.loc[idx, 'CSF_DATE_DIFF'] = exam_date_diffs.min().days
        return new_tbl

    @n_participants_wrapper
    def _merge_csf_values(self):
        # for each row in self.tables['reg'], get the date, find
        # the closest time at which a csf sample was taken. If taken
        # < threshold within date of visit, attach csf data to this visit
        # first, compute time
        self.tables['csf'].rename(columns={
                'EXAMDATE': 'EXAMDATE_csf',
                'RID': 'RID_csf'
        }, inplace=True)
        for col in self.tables['csf'].columns:
            self.tables['reg'].loc[:, col] = pd.NA
        self.tables['reg'].loc[:, 'CSF_DATE_DIFF'] = pd.NA
        for rid, sub_tbl in self.tables['reg'].groupby('RID'):
            csf_subtbl = self.tables['csf'].query('RID_csf == @rid').copy()
            if len(csf_subtbl) > 0:
                sub_tbl = self.__assign_closest_visit(sub_tbl, csf_subtbl)
                self.tables['reg'].loc[sub_tbl.index, :] = sub_tbl
        self.tables.pop('csf', None)