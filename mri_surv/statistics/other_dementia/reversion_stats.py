
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FI = {
    "ADNI": "./metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv",
    "NACC": "./metadata/data_processed/merged_dataframe_cox_test_pruned_final.csv"
}

COLS = {
    "ADNI": {
        'RID': str,
        'DX_DATES': str,
        'DX_VALUES': str,
        'PROGRESSES': float,
        'TIME_TO_PROGRESSION': float
    },
    "NACC": {
        'RID': str,
        'DX_DATES': str,
        'DX_VALUES': str,
        'PROGRESSES': float,
        'TIME_TO_PROGRESSION': float
    }
}

key_adni = {'NL': 0, 'MCI': 1, 'AD': 2}
key_nacc = {'NC': 0, 'IMP': 1, 'MCI': 2, 'AD': 3, 'OD': 4}

def load_csv(ds: str) -> pd.DataFrame:
    df = pd.read_csv(FI[ds], usecols=list(COLS[ds].keys()), dtype=COLS[ds])
    if ds == 'ADNI':
        df['RID'] = df['RID'].apply(lambda x: x.zfill(4))
    return df

def reverts(progresses: np.array, dx: np.ndarray, times: np.ndarray, rid: np.ndarray) -> dict:
    """
    reverts 
    
    Computes proportion of patients with AD who revert, MCI who revert, and
    the number with other types of dementia
    
    Computes this value utilizing the FINAL diagnosis given to a patient at a follow-up

    Parameters
    ----------
    progresses : np.array
        binary array, whether or not patient "progresses" to AD
    dx : np.ndarray
        array of string dx for patient
    times : np.ndarray
        array of times, in months, of dx for patient
    rid : np.ndarray
        array of RID corresponding to DX and TIMES

    Returns
    -------
    dict
        dict of proportion of AD who revert and proportion of MCI who revert
    """
    dx_last = np.zeros_like(dx)
    time_last = np.zeros_like(dx)
    has_od = 0
    for i in range(len(dx)):
        dx_last[i] = dx[i].split(',')[-1]  # take final diagnosis
        time_last[i] = times[i].split(',')[-1]
        if 'OD' in dx[i]:  # looks for other dementia
            has_od += 1
    
    # find indices where the patient "reverts", or, final diagnosis is not MCI or OD
    reversions = [(x == 0) and (y != 'MCI') and (y != 'OD') \
        for x,y in zip(progresses, dx_last)  
            ]
    rid = rid[reversions]
    times = time_last[reversions]
    pct = {
        'AD_reverted': np.sum(
            [(x == 1) and (y != 'AD') and (y != 'OD') \
                for x,y in zip(progresses, dx_last)
            ]) / np.sum(progresses),
        'MCI_reverted': np.sum(
            reversions) / np.sum(1-progresses),
        'OD diagnosis at least once:':
            has_od,
    }
    return pct, rid, times

def _dump_reversions(ds: str, pct: dict, fi):
    fi.write('\n' + '='*20 + '\n')
    fi.write(ds + '\n\n')
    for key, item in pct.items(): fi.write(f'{key}:\n\t{item}\n\n') 

def main():
    with open('results/reversion_counts','w') as fi:
        adni = load_csv('ADNI')

        pct, rid_adni, times_adni = reverts(adni['PROGRESSES'].to_numpy(), adni['DX_VALUES'].to_numpy(), adni['DX_DATES'].to_numpy(), adni['RID'].to_numpy())
        _dump_reversions('ADNI', pct, fi)

        nacc = load_csv('NACC')
        pct, rid_nacc, times_nacc = reverts(nacc['PROGRESSES'].to_numpy(), nacc['DX_VALUES'].to_numpy(), nacc['DX_DATES'].to_numpy(), nacc['RID'].to_numpy())
        _dump_reversions('NACC', pct, fi)

        df = pd.DataFrame(
            {'DS': np.concatenate([np.repeat('ADNI',len(rid_adni)), np.repeat('NACC', len(rid_nacc))]),
            'RID': np.concatenate([rid_adni, rid_nacc]),
            'T': np.concatenate([times_adni, times_nacc])}
            )
        
        df.to_csv('./metadata/data_processed/reverted_rids.csv', index=False)