import os
import numpy as np

RACE_MAP = {
    # "NACC": {
    #     1: "White",
    #     2: "BlackOrAfricanAmerican",
    #     3: "AmericanIndianOrAlaskanNative",
    #     4: "NativeHawaiianOrPacificIslander",
    #     5: "Asian",
    #     50: "Other",
    #     99: "Unknown",
    # },
    "NACC": {
        1: "White",
        2: "BlackOrAfricanAmerican",
        3: "AmericanIndianOrAlaskanNative",
        4: "NativeHawaiianOrPacificIslander",
        5: "Asian",
        6: "MoreThanOneRace",
        99: "Unknown"
    },
    "ADNI": {
        1: "AmericanIndianOrAlaskanNative",
        2: "Asian",
        3: "NativeHawaiianOrPacificIslander",
        4: "BlackOrAfricanAmerican",
        5: "White",
        6: "MoreThanOneRace",
        7: "Unknown",
        -4: np.nan,
    },
}

ETHN_MAP = {
    "NACC": {0: "NotHispanicOrLatino", 1: "HispanicOrLatino", 9: "Unknown"},
    "ADNI":
    {
        1: "HispanicOrLatino", 2: "NotHispanicOrLatino",
        3: "Unknown", -4: np.nan
    },
}

COL_NAMES = {
    "NACC": {
        "NACCID": str,
        "HISPANIC": float,
        "NACCNIHR": float,
    },
    "ADNI": {
        "PTRACCAT": float,
        "PTETHCAT": float,
        "RID": str,
    },
}

COL_NAMES_MAP = {
    "NACC": {"NACCNIHR": "RACE", "NACCID": "RID", "HISPANIC": "ETHNICITY"},
    "ADNI": {"PTRACCAT": "RACE", "PTETHCAT": "ETHNICITY"},
}

DIR = {
    "NACC": os.path.join(
        os.getcwd(), "metadata/data_raw/NACC", "kolachalama12042020.csv"
    ),
    "ADNI": os.path.join(os.getcwd(), "metadata/data_raw/ADNI", "PTDEMOG.csv"),
}

OUTPUT_FNAME = "metadata/data_processed/race_and_ethnicity.pkl"
