# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:04:13 2020

@author: mfromano
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
import datetime

fi = os.path.join(os.path.abspath("./logs"),
                                        "metadata_locations.log")
with open(fi,'w') as f:
    f.write(str(datetime.datetime.now()))
hdlr = logging.FileHandler(fi)
logger = logging.getLogger(__name__)
logger.addHandler(hdlr)

class DictionaryADNI():
    def __init__(self):
        curdir = os.getcwd()
        # hard-code csv"s w/ metadata
        self.base_dir = os.path.join(curdir, "metadata/data_raw/ADNI")
        # FDG scans
        fdg1 = MetadataCSV(
            loc=os.path.join(self.base_dir, "PETMETA_ADNI1.csv"),
            fields=[
                    "RID", "VISCODE", "PMCONDCT",
                    "EXAMDATE"
            ],
            dtype={
                    "RID": int,
                    "VISCODE": str,
                    "PMCONDCT": float,
                    "EXAMDATE": str,
            },
            date_fmt="%Y-%m-%d",
            colmap={"PMCONDCT": "DONE"},
            drop_nan_row={
                    "VISCODE": lambda x: pd.isna(x),
                    "DONE"   : lambda x: x == 0
            },
            duplicates=True
            )

        fdg2go = MetadataCSV(
                loc=os.path.join(self.base_dir, "PETMETA_ADNIGO2.csv"),
                fields=[
                        "Phase", "RID", "VISCODE",
                        "VISCODE2",
                        "PMCONDCT", "EXAMDATE"
                ],
                dtype={
                        "RID"     : int,
                        "VISCODE" : str,
                        "VISCODE2" : str,
                        "PMCONDCT": float,
                        "EXAMDATE": str,
                },
                date_fmt="%Y-%m-%d",
                colmap={"PMCONDCT": "DONE"},
                drop_nan_row={
                        "VISCODE" : lambda x: pd.isna(
                        x),
                        "VISCODE2": lambda x: pd.isna(
                        x),
                        "DONE"    : lambda x: x == 0
                },
                duplicates=True
                )

        fdg3 = MetadataCSV(
                loc=os.path.join(self.base_dir, "PETMETA3.csv"),
                fields=[
                        "Phase", "RID", "VISCODE", "VISCODE2",
                        "DONE",
                        "SCANDATE"
                ],
                dtype={
                        "RID": int,
                        "Phase"     : str,
                        "VISCODE" : str,
                        "VISCODE2": str,
                        "DONE": float,
                        "SCANDATE": str,
                },
                date_fmt="%Y-%m-%d",
                colmap={"SCANDATE": "EXAMDATE"},
                drop_nan_row={
                        "VISCODE" : lambda x: pd.isna(x),
                        "VISCODE2": lambda x: pd.isna(x),
                        "DONE"    : lambda x: x == 0
                },
                duplicates=True
                )

        medhist = MetadataCSV(
                loc=os.path.join(self.base_dir, "RECMHIST.csv"),
                fields=[
                        "Phase", "RID", "VISCODE","VISCODE2",
                        "MHNUM", "MHDESC","MHCUR"
                ],
                dtype={
                        "RID"     : int,
                        "Phase"   : str,
                        "VISCODE" : str,
                        "VISCODE2": str,
                        "MHNUM"    : float,
                        "MHDESC" : str,
                        "MHCUR": float,
                },
                date_fmt="",
                colmap={
                        "MHDESC": "MedhxDescription",
                        "MHCUR": "MedhxCurrent",
                },
                drop_nan_row={
                        "VISCODE" : lambda x: pd.isna(x),
                        "VISCODE2": lambda x: pd.isna(x),
                        "MedhxCurrent"   : lambda x: x == 0
                },
                duplicates=True
        ) # 1=1. Psychiatric; 2=2. Neurologic; 3=3. Head, Eyes, Ears, Nose, Throat; 4=4. Cardiovascular; 5=5. Respiratory; 6=6. Hepatic; 7=7. Dermatologic-Connective Tissue; 8=8. Musculoskeletal; 9=9. Endocrine-Metabolic; 10=10. Gastrointestinal; 11=11. Hematopoietic-Lymphatic; 12=12. Renal-Genitourinary; 13=13. Allergies or Drug Sensitivities; 14=14. Alcohol Abuse; 15=15. Drug Abuse; 16=16. Smoking; 17=17. Malignancy; 18=18. Major Surgical Procedures; 19=19. Other


        tau2 = MetadataCSV(
                loc=os.path.join(self.base_dir, "TAUMETA.csv"),
                fields=[
                        "Phase", "RID", "VISCODE", "VISCODE2",
                        "DONE",
                        "SCANDATE"
                ],
                dtype={
                        "RID"     : int,
                        "Phase"   : str,
                        "VISCODE" : str,
                        "VISCODE2": str,
                        "DONE"   : float,
                        "SCANDATE"   : str,
                },
                date_fmt="%Y-%m-%d",
                colmap={
                        "SCANDATE": "EXAMDATE"
                },
                drop_nan_row={
                        "VISCODE" : lambda x: pd.isna(x),
                        "VISCODE2": lambda x: pd.isna(x),
                        "DONE"    : lambda x: x == 0
                },
                duplicates=True
                )

        tau3 = MetadataCSV(
                loc=os.path.join(self.base_dir, "TAUMETA3.csv"),
                fields=[
                        "Phase", "RID", "VISCODE", "VISCODE2",
                        "DONE", "SCANDATE"
                ],
                colmap={
                        "DONE"    : "DONE",
                        "SCANDATE": "EXAMDATE"
                },
                dtype={
                        "RID"     : int,
                        "Phase"   : str,
                        "VISCODE" : str,
                        "VISCODE2": str,
                        "DONE"    : float,
                        "SCANDATE": str,
                },
                date_fmt="%Y-%m-%d",
                drop_nan_row={
                        "VISCODE" : lambda x: pd.isna(x),
                        "VISCODE2": lambda x: pd.isna(x),
                        "DONE"    : lambda x: x == 0
                },
                duplicates=True
                )

        mri3 = MetadataCSV(
                loc=os.path.join(self.base_dir, "MRI3META.csv"),
                fields=["PHASE", "RID", "VISCODE", "VISCODE2",
                        "MMCONDCT", "EXAMDATE", "MMRMPRAGE", ],
                dtype={
                        "RID"     : int,
                        "Phase"   : str,
                        "VISCODE" : str,
                        "VISCODE2": str,
                        "MMCONDCT"    : float,
                        "EXAMDATE": str,
                        "MMRMPRAGE": float,
                },
                date_fmt="%Y-%m-%d",
                colmap={
                        "PHASE" : "Phase",
                        "MMCONDCT": "DONE",
                },
                drop_nan_row={
                        "VISCODE"         : lambda x: pd.isna(x),
                        "VISCODE2"        : lambda x:
                        pd.isna(x), "DONE": lambda x: x == 0
                },
                duplicates=True
                )

        amyloid = MetadataCSV(
                loc=os.path.join(self.base_dir, "AV45META.csv"),
                fields=[
                        "Phase", "RID", "VISCODE",
                        "VISCODE2", "PMCONDCT", "EXAMDATE"
                ],
                dtype={
                        "RID"      : int,
                        "Phase"    : str,
                        "VISCODE"  : str,
                        "VISCODE2" : str,
                        "PMCONDCT" : float,
                        "EXAMDATE" : str,
                },
                date_fmt="%Y-%m-%d",
                colmap={"PMCONDCT": "DONE"},
                drop_nan_row={
                        "VISCODE2": lambda x: pd.isna(x),
                        "DONE"    : lambda x: x == 0
                },
                duplicates=True
                )

        amyloid2 = MetadataCSV(
                loc=os.path.join(self.base_dir, "AMYMETA.csv"),
                fields=[
                        "Phase", "RID", "VISCODE",
                        "VISCODE2", "DONE", "TRACERTYPE",
                        "SCANDATE"
                ],
                dtype={
                        "RID"     : int,
                        "Phase"   : str,
                        "VISCODE" : str,
                        "VISCODE2": str,
                        "DONE": float,
                        "SCANDATE": str,
                        "TRACERTYPE": float
                },
                date_fmt="%Y-%m-%d",
                colmap={"SCANDATE": "EXAMDATE"},
                drop_nan_row={
                        "VISCODE" : lambda x: pd.isna(x),
                        "VISCODE2": lambda x: pd.isna(x),
                        "DONE"    : lambda x: x == 0
                },
                duplicates=True
                )

        csf_12GO = MetadataCSV(
                loc=os.path.join(self.base_dir, "UPENNBIOMK9_04_19_17.csv"),
                fields=[
                        "RID", "PHASE",
                        "EXAMDATE",
                        "VISCODE", "VISCODE2",
                        "ABETA",
                        "TAU", "PTAU", "COMMENT"
                ],
                dtype={
                        "RID"       : int,
                        "Phase"     : str,
                        "VISCODE"   : str,
                        "VISCODE2"  : str,
                        "EXAMDATE"  : str,
                        "ABETA": str,
                        "TAU": str,
                        "PTAU": str,
                        "COMMENT": str
                },
                date_fmt="%Y-%m-%d",
                colmap={
                        "EXAMDATE": "EXAMDATE"
                },
                drop_nan_row={
                        "VISCODE" : lambda
                        x: pd.isna(x),
                        "VISCODE2": lambda
                        x: pd.isna(x)
                },
                duplicates=False
                )

        csf_3 = MetadataCSV(
                loc=os.path.join(self.base_dir, "UPENNBIOMK10_07_29_19.csv"),
                fields=[
                        "RID", "DRAWDATE",
                        "VISCODE",
                        "VISCODE2", "ABETA40",
                        "ABETA42",
                        "TAU", "PTAU", "NOTE"
                ],
                dtype={
                        "RID"     : int,
                        "Phase"   : str,
                        "VISCODE" : str,
                        "VISCODE2": str,
                        "DRAWDATE": str,
                        "ABETA40"   : str,
                        "ABETA42" : str,
                        "TAU"     : str,
                        "PTAU"    : str,
                        "NOTE" : str
                },
                date_fmt="%Y-%m-%d",
                colmap={"DRAWDATE": "EXAMDATE"},
                drop_nan_row={
                        "VISCODE" : lambda x: pd.isna(
                        x),
                        "VISCODE2": lambda x: pd.isna(
                        x)
                },
                duplicates=False
                )

        mmse = MetadataCSV(
                loc=os.path.join(self.base_dir, "MMSE.csv"),
                fields=[
                        "Phase", "RID", "VISCODE",
                        "VISCODE2",
                        "MMSCORE"
                ],
                dtype={
                        "RID"     : int,
                        "Phase"   : str,
                        "VISCODE" : str,
                        "VISCODE2": str,
                        "MMSCORE": float
                },
                date_fmt="",
                colmap={
                },
                drop_nan_row={
                        "VISCODE" : lambda x: pd.isna(x),
                        "VISCODE2": lambda x: pd.isna(x),
                        "MMSCORE": lambda x: x < 0
                },
                duplicates=False
                )

        reg = MetadataCSV(
            loc=os.path.join(self.base_dir, "REGISTRY.csv"),
            fields=["Phase", "RID", "VISCODE", "VISCODE2",
                    "EXAMDATE", "RGCONDCT","VISTYPE"],
            dtype={
                        "RID"     : int,
                        "Phase"   : str,
                        "VISCODE" : str,
                        "VISCODE2": str,
                        "EXAMDATE" : str,
                        "RGCONDCT" : float,
                        "VISTYPE" : float,
                },
            date_fmt="%Y-%m-%d",
            colmap={

            },
            drop_nan_row={
                    "VISCODE" : lambda x: pd.isna(x),
                    "VISCODE2": lambda x: pd.isna(x),
                    "RGCONDCT": lambda x: x == 0,
                    "VISTYPE": lambda x: x == 3,
            },
            duplicates=False
            )

        apoe = MetadataCSV(
            loc=os.path.join(self.base_dir, "APOERES.csv"),
            fields=["Phase", "RID", "VISCODE", "APGEN1",
                    "APGEN2"],
            dtype={
                    "RID"     : int,
                    "Phase"   : str,
                    "VISCODE" : str,
                    "APGEN1": float,
                    "APGEN2" : float
            },
            date_fmt="",
            colmap={},
            drop_nan_row={},
            duplicates=False
            )

        demo = MetadataCSV(
            loc=os.path.join(self.base_dir, "PTDEMOG.csv"),
            fields=[
                    "Phase", "RID", "VISCODE", "VISCODE2",
                    "PTGENDER", "PTDOBMM", "PTDOBYY",
                    "PTHAND", "PTEDUCAT", "PTRACCAT",
                    "PTETHCAT"
            ],
            dtype={
                    "RID": int,
                    "Phase": str,
                    "VISCODE": str,
                    "VISCODE2": str,
                    "PTGENDER": float,
                    "PTDOBMM": float,
                    "PTDOBYY": float,
                    "PTHAND": float,
                    "PTEDUCAT": float,
            },
            date_fmt="",
            colmap={
            },
            drop_nan_row={
                    "VISCODE" : lambda x: pd.isna(x),
                    "VISCODE2": lambda x: pd.isna(x)
            },
            duplicates=False,
            value_recode={
                    "PTGENDER": {
                        1: "M",
                        2: "F",
                        -4: np.nan,
                    },
                    "PTHAND": {
                        1 : "R",
                        2 : "L",
                        -4: np.nan,
                    },
                    "PTEDUCAT": {
                        -1: np.nan,
                        -4: np.nan
                    },
            }
            )

        dxsum = MetadataCSV(
            loc=os.path.join(self.base_dir, "DX_SUM_ALL.csv"),
            fields=[
                    "Phase", "RID", "VISCODE", "VISCODE2",
                    "EXAMDATE", "DXCHANGE", "DXCURREN",
                    "DIAGNOSIS", "DXDDUE", "DXAD"
            ],
            dtype={
                    "RID": int,
                    "Phase": str,
                    "VISCODE": str,
                    "VISCODE2": str,
                    "EXAMDATE": str,
                    "DXCHANGE": float,
                    "DXCURREN": float,
                    "DIAGNOSIS": float,
                    "DXDDUE": float,
                    "DXAD": float
            },
            date_fmt="",
            colmap={},
            drop_nan_row={
                    "VISCODE" : lambda x: pd.isna(x),
                    "VISCODE2": lambda x: pd.isna(x)
            },
            duplicates=False
            )# FDG scans
        self.file_list = {
                "tau2"   : tau2, "tau3": tau3, "mri3": mri3,
                "reg"    : reg, "dxsum": dxsum, "demo": demo, "mmse": mmse,
                "apoe"   : apoe,
                "amyloid": amyloid, "amyloid2": amyloid2,
                "csf_1"  : csf_12GO, "csf_2": csf_3, "fdg1": fdg1,
                "fdg2go" : fdg2go,
                "fdg3"   : fdg3,
                "medhx"  : medhist
        }

    def __getitem__(self, key):
        assert(key in self.file_list.keys())
        return self.file_list[key]

class MetadataCSV:
    def __init__(self, loc: str, fields: List,
                 colmap: Dict, drop_nan_row: Dict, duplicates:
            bool=False, dtype=None, date_fmt: str = "",
                 value_recode: Dict=None):
        """Summary

        Args:
            loc (str): file location
            fields (list): fields to select from the csv file
            colmap (dict): mapping of column names as read to new column names
            dateformat (str): format of the date in the stored file
            drop_nan_row (dict): handle to function used to decide which to drop
            duplicates (bool): whether or not we should expect duplicate entried
            in this file
        """

        if dtype is None:
            dtype = dict()
        self.props = {
            "loc": loc,
            "fields": fields,
            "colmap": colmap,
            "drop_nan_row": drop_nan_row,
            "duplicates": duplicates,
            "dtype": dtype,
            "date_fmt": date_fmt,
            "value_recode": value_recode
        }

    def __getitem__(self, item):
        assert(item in self.props.keys())
        return self.props[item]


"""
coding for "DIAGNOSIS" (ADNI3)
1=CN	2=MCI	3=Dementia
coding for DXCURREN (ADNI1)
1=NL	2=MCI	3=AD
coding for ADNI2GO:
1: "NL", 2: "MCI", 3: "AD", 4: "MCI", 5: "AD", 6: "AD",
7: "NL", 8: "MCI", 9: "NL"
"""

class DictionaryOasis():
    def __init__(self):
        curdir = os.getcwd()
        # hard-code csv"s w/ metadata
        self.base_dir = os.path.join(curdir, "metadata/data_raw/OASIS")
        # FDG scans
        clinical = MetadataCSV(
            loc=os.path.join(self.base_dir, "adrs_clinicaldata_oasis3.csv"),
            fields=["ADRC_ADRCCLINICALDATA ID","ageAtEntry",
                    "mmse", "apoe"],
            colmap={
                "ADRC_ADRCCLINICALDATA ID": "VISCODE",
                "mmse": "MMSE",
                "apoe": "APOE",
            },
            drop_nan_row={
            },
            )

        demo = MetadataCSV(
            loc=os.path.join(self.base_dir, "demo_oasis3.csv"),
            fields=["UDS_A1SUBDEMODATA ID",
                "SEX", "Education", "MARISTAT","HANDED"],
            colmap={
                "UDS_A1SUBDEMODATA ID": "VISCODE",
            },
            drop_nan_row={
                "HANDED": lambda x: float(x) not in [1, 2],
                "Education": lambda x: float(x)==99
            },
            )

        dx = MetadataCSV(
            loc=os.path.join(self.base_dir, "dx_oasis3.csv"),
            fields=["UDS_D1DXDATA ID","NORMCOG","DEMENTED","MCIAMEM","MCIAPLUS",
                "MCIAPLAN","MCIAPATT","MCIAPEX","MCIAPVIS",
                "MCINON1", "MCIN1LAN", "MCIN1ATT", "MCIN1EX", "MCIN1VIS",
                "MCINON2", "MCIN2LAN", "MCIN2ATT", "MCIN2EX", "MCIN2VIS", "MCIN2VIS",
                "PROBAD","PROBADIF","POSSAD","POSSADIF"],
            colmap={
                "UDS_D1DXDATA ID": "VISCODE",
            },
            drop_nan_row={
            },
            )

        pet = MetadataCSV(
            loc=os.path.join(self.base_dir, "pet_oasis3.csv"),
            fields=["XNAT_PETSESSIONDATA ID"],
            colmap={
                "XNAT_PETSESSIONDATA ID": "VISCODE",
            },
            drop_nan_row={            
            },
            )

        mri = MetadataCSV(
            loc=os.path.join(self.base_dir, "mri_oasis3.csv"),
            fields=["MR ID", 
                    "Scanner","Scans"],
            colmap={
                "Subject": "RID",
                "MR ID": "VISCODE",
            },
            drop_nan_row={
                "Scanner": lambda x: x != "3.0T"
            },
            )

        self.file_list = {
                "demo"   : demo, 
                "clinical" : clinical,
                "dx": dx,
                "mri": mri,
                "pet": pet,
        }


#
class DictionaryNacc():
    def __init__(self):
        curdir = os.getcwd()
        # hard-code csv"s w/ metadata
        self.base_dir = os.path.join(curdir, "metadata/data_raw/NACC")
        # FDG scans
        reg = MetadataCSV(
            loc=os.path.join(self.base_dir, "kolachalama12042020.csv"),
            fields=[
                    "NACCID", "FORMVER", "VISITMO", "VISITDAY",
                    "VISITYR", "NACCVNUM", "NACCAGE",
                    "SEX", "EDUC", "NACCAPOE",
                    "NACCUDSD", "NACCYOD", "NACCMOD",
                    "MMSECOMP", "NACCMMSE",
                    "NACCTMCI", "NACCALZD",
                    # "NPFORMVER", "NPSEX",
                    # "NACCBRNN",
                    "NPGRCCA", "NPGRLA", "NPGRHA",
                    # "NPTAN", "NPTANX", "NPABAN", "NPABANX", "NPASAN", "NPASANX",
                    # "NPTDPAN", "NPTDPANX",
                    "NPTHAL", "NACCBRAA", "NACCNEUR",
                    "NPADNC", "NACCDIFF",
                    # "NACCAMY",
                    "NPHIPSCL", "NPSCL", "NACCPROG",
                    # "NPTAU", "NPFTDTDP", "NPTDPA",
                    "NPTDPB", "NPTDPC",
                    "NPTDPD", "NPTDPE", "NACCDAGE", #"NPNIT"
            ],
            dtype={
                    "NACCID": str,
                    "NACCAGE": float,
                    "FORMVER": float,
                    "VISITMO": int,
                    "VISITDAY": int,
                    "VISITYR": int,
                    "NACCVNUM": float,
                    "SEX": float,
                    "EDUC": float, # 99 = unknown
                    "NACCAPOE": float,
                    # 1 = hx of family cog impairment
                    "NACCUDSD": float,
                    "NACCYOD": float,
                    "NACCMOD": float,
                    "MMSECOMP": float,
                    "NACCMMSE": float,
                    "NACCTMCI": float,
                    "NACCALZD": float,
                    # "NPFORMVER": float,
                    # "NPSEX": float,
                    # "NACCBRNN": float,
                    "NPGRCCA": float,
                    "NPGRLA": float,
                    "NPGRHA": float,
                    # "NPTAN": float,
                    # "NPTANX": str,
                    # "NPABAN": float,
                    # "NPABANX": str,
                    # "NPASAN": float,
                    # "NPASANX": str,
                    # "NPTDPAN": float,
                    # "NPTDPANX": str,
                    "NPTHAL": float,
                    "NACCBRAA": float,
                    "NACCNEUR": float,
                    "NPADNC": float,
                    "NACCDIFF": float,
                    # "NACCAMY": float,
                    "NPHIPSCL": float,
                    "NPSCL": float,
                    "NACCPROG": float,
                    # "NPTAU": float,
                    # "NPFTDTDP": float,
                    # "NPTDPA": float,
                    "NPTDPB": float,
                    "NPTDPC": float,
                    "NPTDPD": float,
                    "NPTDPE": float,
                    "NACCDAGE": float,
                    # "NPNIT": float,
            },
            date_fmt="",
            colmap={
                    "NACCID"  : "RID",
                    "NACCVNUM" : "Visit",
                    "NACCAGE": "AGE",
                    "NACCAPOE": "APOE",
                    "NACCMMSE": "MMSE",
                    "NPGRCCA": "CorticalAtrophy",
                    "NPGRLA": "LobarAtrophy",
                    "NPGRHA": "HippocampalAtrophy",
                    "NACCBRAA": "BRAAK",
                    "NACCNEUR": "CERAD",
                    "NPADNC": "NIA_ADNC",
                    "NACCDIFF": "CERAD_SemiQuant",
                    "NPHIPSCL": "HippocampalSclerosis",
                    "NPSCL": "MTLSclerosisWithHippocampus",
                    "NPTDPB": "TDP_Amygdala",
                    "NPTDPC": "TDP_Hippocampus",
                    "NPTDPD": "TDP_EntorhinalInfTemporal",
                    "NPTDPE": "TDP_Neocortex",
                    "NACCDAGE": "AgeAtDeath"
            },
            drop_nan_row={
            },
            value_recode={
                "MMSE": {
                        88: np.nan,
                        95: np.nan,
                        96: np.nan,
                        97: np.nan,
                        98: np.nan,
                        -4: np.nan
                },
                "SEX": {
                        1: "M",
                        2: "F"
                },

                "APOE": {
                        1: 0,
                        2: 1,
                        3: 0,
                        4: 2,
                        5: 1,
                        6: 0,
                        9: np.nan
                },
                "EDUC": {
                        99: np.nan
                },
                "BRAAK": {
                        -4: np.nan,
                        7: np.nan,
                        8: np.nan,
                        9: np.nan,
                        0: 'Stage 0',
                        1: 'Stage 1',
                        2: 'Stage 2',
                        3: 'Stage 3',
                        4: 'Stage 4',
                        5: 'Stage 5',
                        6: 'Stage 6'
                },
                "CERAD": {
                        -4: np.nan,
                        8: np.nan,
                        9: np.nan,
                        0: 'C0',
                        1: 'C1',
                        2: 'C2',
                        3: 'C3'
                },
                "CorticalAtrophy": {
                        8 : np.nan,
                        9 : np.nan,
                        -4: np.nan,
                        0: 'None',
                        1: 'Mild',
                        2: 'Moderate',
                        3: 'Severe'
                },
                "HippocampalAtrophy"  : {
                        8: np.nan,
                        9: np.nan,
                        -4: np.nan,
                        0: 'None',
                        1: 'Mild',
                        2: 'Moderate',
                        3: 'Severe'
                },

                "LobarAtrophy": {
                        8 : np.nan,
                        9 : np.nan,
                        -4: np.nan,
                        0: 'None',
                        1: 'Yes'
                },
                "NIA_ADNC": {
                        8 : np.nan,
                        9 : np.nan,
                        -4: np.nan,
                        0: 'Not AD',
                        1: 'Low',
                        2: 'Intermediate',
                        3: 'High'
                },
                "CERAD_SemiQuant": {
                        -4: np.nan,
                        8: np.nan,
                        9: np.nan,
                        0: 'None',
                        1: 'Sparse',
                        2: 'Moderate',
                        3: 'Frequent'
                },
                "HippocampalSclerosis": {
                        8: np.nan,
                        9: np.nan,
                        -4: np.nan,
                        0: 'None',
                        1: 'Unilateral',
                        2: 'Bilateral',
                        3: 'Present'
                },
                "MTLSclerosisWithHippocampus": {
                        -4: np.nan,
                        9: np.nan,
                        3: np.nan,
                        1: 'Yes',
                        2: 'No'
                },
                "TDP_Amygdala": {
                        8: np.nan,
                        9: np.nan,
                        -4: np.nan,
                        0: 'No',
                        1: 'Yes'
                },
                "TDP_Hippocampus": {
                        8: np.nan,
                        9: np.nan,
                        -4: np.nan,
                        0: 'No',
                        1: 'Yes'
                },
                "TDP_EntorhinalInfTemporal": {
                        8 : np.nan,
                        9 : np.nan,
                        -4: np.nan,
                        0: 'No',
                        1: 'Yes'
                },
                "TDP_Neocortex": {
                        8 : np.nan,
                        9 : np.nan,
                        -4: np.nan,
                        0: 'No',
                        1: 'Yes'
                },
                "AgeAtDeath": {
                        888: np.nan,
                        999: np.nan
                },
                "NACCYOD": {
                        8888: np.nan,
                        9999: np.nan
                },
                "NACCMOD": {
                        88: np.nan,
                        99: np.nan
                }
            }
        )

        apet = MetadataCSV(
            loc=os.path.join(self.base_dir, "kolachalama12042020apet.csv"),
            fields=[
                    "NACCID","APETMO","APETDY","APETYR","NACCAPTA",
                    "NACCAPTF","NACCAPTD","LIGANDN", "NACCAPNM"
            ],
            dtype={
                    "NACCID": str,
                    "APETMO": int,
                    "APETDY": int,
                    "APETYR": int,
                    "NACCAPTA": float,
                    "NACCAPTF": str,
                    "NACCAPTD": float,
                    "LIGANDN": float,
                    "NACCAPNM": float
            },
            date_fmt="",
            colmap={
                    "NACCID": "RID",
                    "NACCAPTA": "Age",
                    "NACCAPTF": "File_PET",
                    "NACCAPTD": "DaysToVisit",
                    "APETMO": "VISITMO",
                    "APETDY": "VISITDAY",
                    "APETYR": "VISITYR",
                    "NACCAPNM": "Visit"
            },
            drop_nan_row={
                "LIGANDN": lambda x: float(x) != 2,
            },
            )

        csf = MetadataCSV(
                loc=os.path.join(self.base_dir, "kolachalama12042020csf.csv"),
                fields=[
                        "NACCID","CSFABETA","CSFPTAU","CSFTTAU","CSFLPMO","CSFLPDY","CSFLPYR","CSFABMD","CSFPTMD","CSFTTMD"
                ],
                dtype={
                        "NACCID"  : str,
                        "CSFLPMO"  : int,
                        "CSFLPDY"  : int,
                        "CSFLPYR"  : int,
                        "CSFABETA": float,
                        "CSFPTAU": float,
                        "CSFTTAU": float,
                        "CSFABMD" : float,
                        "CSFPTMD": float,
                        "CSFTTMD": float
                },
                date_fmt="",
                colmap={
                        "NACCID"  : "RID",
                        "CSFLPMO"  : "VISITMO",
                        "CSFLPDY"  : "VISITDAY",
                        "CSFLPYR"  : "VISITYR",
                        "CSFABETA": "abeta",
                        "CSFPTAU": "ptau",
                        "CSFTTAU": "ttau",
                },
                drop_nan_row={
                        "abeta": lambda x: pd.isna(x),
                        "ptau": lambda x: pd.isna(x),
                        "ttau": lambda x: pd.isna(x)
                },
        )

        mri = MetadataCSV(
            loc=os.path.join(self.base_dir, "kolachalama12042020mri.csv"),
            fields=[
                    "NACCID", "MRIMO", "MRIDY", "MRIYR", "NACCMRIA", "NACCMRFI",
                    "NACCMRDY", "MRIT1", "MRIFIELD", "NACCMNUM"
            ],
            dtype={
                    "NACCID": str,
                    "MRIMO": int,
                    "MRIDY": int,
                    "MRIYR": int,
                    "NACCMRIA": float,
                    "NACCMRFI": str,
                    "NACCMRDY": float,
                    "MRIT1": float,
                    "MRIFIELD": float,
                    "NACCMNUM": float
            },
            date_fmt="",
            colmap={
                    "NACCID": "RID",
                    "NACCMRIA": "Age",
                    "NACCMRFI": "File_MRI",
                    "NACCMRDY": "DaysToVisit",
                    "MRIMO" : "VISITMO",
                    "MRIDY": "VISITDAY",
                    "MRIYR" : "VISITYR",
                    "NACCMNUM": "Visit"
            },
            drop_nan_row={
                    "MRIFIELD": lambda x: float(x) != 2,
                    "MRIT1": lambda x: x == 0,
            },
            )

        self.file_list = {
                "reg" : reg,
                "mri": mri,
                "csf": csf
        }

        def __getitem__(self, key):
            assert (key in self.file_list.keys())
            return self.file_list[key]
#
class _DxCodes(object):
    def __init__(self):
        """Diagnosis codes
        Creates a class with
        properties including:
        dx_trans: nested dictionary with diagnosis code to dx conversion
        dx_conversion_dict: a nested dictionary with the columns for the
                diagnosis in addition to the column for whether or not
                the diagnosis is due to AD dementia
        """
        ADNI3_TRANS = {1: "NL", 2: "MCI", 3: "AD"}
        ADNI1_TRANS = {1: "NL", 2: "MCI", 3: "AD"}
        ADNI2GO_TRANS = {
                1: "NL", 2: "MCI", 3: "AD", 4: "MCI", 5: "AD", 6: "AD",
                7: "NL", 8: "MCI", 9: "NL"
        }

        self.dx_trans = {
                "ADNI1" : ADNI1_TRANS, "ADNI2": ADNI2GO_TRANS,
                "ADNIGO": ADNI2GO_TRANS, "ADNI3": ADNI3_TRANS
        }
        self.dx_conversion_dict = {
                "ADNI1" : {
                        "dx" : "DXAD_dxsum",
                        "col": "DXCURREN_dxsum"
                },
                "ADNI2" : {
                        "dx" : "DXDDUE_dxsum",
                        "col": "DXCHANGE_dxsum"
                },
                "ADNIGO": {
                        "dx" : "DXDDUE_dxsum",
                        "col": "DXCHANGE_dxsum"
                },
                "ADNI3" : {
                        "dx" : "DXDDUE_dxsum",
                        "col": "DIAGNOSIS_dxsum"
                }
        }

FILE_LIST = DictionaryADNI().file_list
# OASIS_FILE_LIST = _OasisDictionary().file_list
NACC_FILE_LIST = DictionaryNacc().file_list

DXTRANS = _DxCodes().dx_trans
DX_CONVERSION_DICT = _DxCodes().dx_conversion_dict