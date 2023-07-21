from typing import Type, Union
import warnings
import datetime
import unittest
import logging
import os
import pandas as pd
import numpy as np

log_file = os.path.join(os.path.abspath("./logs"),
                                        "cohort.log")
with open(log_file, 'w') as fi:
    fi.write(f'Logs {datetime.datetime.now()}')
hdlr = logging.FileHandler(log_file)
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)
logger.setLevel('INFO')

def gen_id(x: Union[list, dict, tuple]):
    if type(x) == list or type(x) == tuple or type(x) == np.array:
        args_copy = []
        for arg in x:
            if type(arg) == list:
                args_copy.append(list(map(id, arg)))
            elif type(arg) == dict:
                args_copy.append({x: id(y) for x, y in arg.items()})
            else:
                args_copy.append(id(arg))
        return args_copy
    elif type(x) == dict:
        kwargs_copy = {}
        for kw, val in x.items():
            if type(val) == list:
                kwargs_copy[kw] = list(map(id, val))
            elif type(val) == dict:
                kwargs_copy[kw] = {x: id(y) for x, y in val.items()}
            else:
                kwargs_copy[kw] = id(val)
        return kwargs_copy
    raise NotImplementedError


def mutable_argument_tester(fun):
    def logger_fun(*args, **kwargs):
        logger.warning(f'------------------------\nRunning: {fun.__name__}\n-----------------------------------\n')
        args_copy = gen_id(args)
        kwargs_copy = gen_id(kwargs)
        output = fun(*args, **kwargs)
        args = gen_id(args)
        kwargs = gen_id(kwargs)
        if args_copy != args or kwargs_copy != kwargs:
            warnings.warn(f'mutable arguments have changed!')
        return output
    return logger_fun

def test_mutable_argument_tester():
    @mutable_argument_tester
    def q(a,b,c):
        a += 5
        b += 4
        c[0] += 12
    g = [10,11]
    q(1,2,g)  # should print false

    @mutable_argument_tester
    def q2(a, b, c):
        a += 5
        b += 4
        c['a'] = 10
    g = {'a': 15}
    q2(1, 2, g)  # should print false

    @mutable_argument_tester
    def q3(a, b, c):
        a += 5
        b += 4
        d = c['a']
        d += 5
    g = {'a': 10}
    q3(1, 2, g)  # should print false

def _load_nii_adni():
    from preprocessing.move_nii_files import find_and_move_mri
    original_df = pd.read_csv('testing/merged_dataframe_cox_noqc_pruned_final.csv', dtype={'RID': str})
    new_df = find_and_move_mri(move=False)
    return new_df, original_df

def _load_unused_nii_adni():
    from preprocessing.move_nii_files import find_and_move_unused_mri
    original_df = pd.read_csv('testing/merged_dataframe_unused_cox_pruned.csv', dtype={'RID': str})
    new_df = find_and_move_unused_mri(move=False)
    return new_df, original_df

def _load_nii_nacc():
    from preprocessing.move_nacc_files import find_and_move_mri
    original_df = pd.read_csv('testing/merged_dataframe_cox_test_pruned_final.csv', dtype={'RID': str})
    new_df = find_and_move_mri(move=False)
    return new_df, original_df

def _compare_dataframes(original_df, new_df):
    print(f'columns in new not in old: \n\t{np.setdiff1d(new_df.columns, original_df.columns)}')
    print(f'columns in old not in new: \n\t{np.setdiff1d(original_df.columns, new_df.columns)}')
    cols = np.intersect1d(new_df.columns, original_df.columns)
    for col in cols:
        if not new_df[col].equals(original_df[col]):
            for i in range(len(new_df[col])):
                a = original_df[col].iloc[i]
                b = new_df[col].iloc[i]
                if type(a) == datetime.time or type(b) == datetime.time:
                    a = str(a)
                    b = str(b)
                if type(a) != type(b):
                    if a != b:
                        if not pd.isna(a) and not pd.isna(b):
                            print(f'Columns {col} are not equal!')
                            raise TypeError(f'Type a: {type(a)}, type b: {type(b)}, a={a}, b={b}')
                if type(a) == str:
                    if a != b:
                        if pd.isna(a) or pd.isna(b):
                            raise ValueError(f'a: {a}, b: {b}')
                        a,b = set(a.split(',')), set(b.split(','))
                        if a != b:
                            print(f'Columns {col} are not equal!')
                            raise ValueError(f'a: {a}, b: {b}')
                elif (type(a) == np.float64 or type(a) == float):
                    if pd.isna(a) and pd.isna(b):
                        continue
                    if not np.isclose(a,b,equal_nan=True):
                        print(f'Columns {col} are not equal!')
                        raise ValueError(f'a: {a}, b: {b}')
                else:
                    if a != b:
                        print(f'Columns {col} are not equal!')
                        raise ValueError(f'a: {a}, b: {b}')
        print(f'Columns {col} are equal...')
    return True

def _load_parcellation_files():
    from process_parcellations.make_imaging_sheet import main
    original_df = pd.read_csv('testing/mri3_cat12_vol_avg_combined_cox.csv', dtype={'RID': str})
    new_df, _ = main()
    return new_df.reset_index(), original_df

class TestNiiMovers(unittest.TestCase):
    def test_unused_nii_files(self):
        new_df, original_df = _load_unused_nii_adni()
        self.assertTrue(_compare_dataframes(original_df, new_df))

    def test_nii_files(self):
        new_df, original_df = _load_nii_adni()
        self.assertTrue(_compare_dataframes(original_df, new_df))

    def test_nacc_files(self):
        new_df, original_df = _load_nii_nacc()
        self.assertTrue(_compare_dataframes(original_df, new_df))

class TestParcellation(unittest.TestCase):
    def test_parcellations(self):
        new_df, original_df = _load_parcellation_files()
        self.assertTrue(_compare_dataframes(original_df, new_df))

class TestShapClusterMerge(unittest.TestCase):
    def test_merge(self):
        df1 = pd.DataFrame({
                'Region': np.tile(list(range(20)),5),
                'Dataset': np.repeat(['ADNI'], 100),
                'RID': np.repeat(['a','b','c','d','e'], 20)
        })
        df2 = pd.DataFrame({
                'RID': ['b','c','a','d','e'],
                'Cluster': ['0','2','1','3','4']
        })
        df_final = df1.merge(
                df2, left_on='RID',right_on='RID', validate='many_to_one')
        print(df_final.groupby('RID').apply(lambda x: pd.unique(x['Cluster'])))
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()