from pandas._config.config import config_prefix
from statistics.clustered_mlp_output_wrappers import \
    load_clustered_mris
from statistics.utilities import CONFIG
from statistics.dataframe_validation import DataFrame, ClusteredMriSchema, pa
import pandas as pd
import numpy as np
import nibabel as nib
import sys, os


@pa.check_types
def _randomly_sample(cluster_dataframe: DataFrame[ClusteredMriSchema],
                     n_nonprogress: int,
                     n_progress: int) -> DataFrame[ClusteredMriSchema]:
    progress = cluster_dataframe['PROGRESSES'].to_numpy()
    progress = np.unique(np.round(progress))
    if progress == 0:
        return cluster_dataframe.sample(n=n_progress, random_state=1)
    elif progress == 1:
        return cluster_dataframe.sample(n=n_nonprogress, random_state=1)
    else:
        raise NotImplementedError

@pa.check_types
def _randomly_sample_from_progressors(
        cluster_dataframe: DataFrame[ClusteredMriSchema],
        n_nonprogress: int,
        n_progress: int) -> DataFrame[ClusteredMriSchema]:
    df = cluster_dataframe.groupby('PROGRESSES').apply(
            lambda x: _randomly_sample(x, n_nonprogress, n_progress)
    )
    df.reset_index(drop=True, inplace=True)
    assert(len(df) == n_nonprogress + n_progress)
    return df

@pa.check_types
def randomly_sample_from_cluster(
        cluster_dataframe: DataFrame[ClusteredMriSchema],
        n_nonprogress: int, n_progress: int) -> DataFrame[ClusteredMriSchema]:
    df = cluster_dataframe.groupby('Cluster Idx').apply(
            lambda x: _randomly_sample_from_progressors(
                    x, n_nonprogress, n_progress))
    df.reset_index(drop=True, inplace=True)
    assert(len(df) == (len(np.unique(df['Cluster Idx']))*(
            n_nonprogress+n_progress)))
    return df

def shuffle_df_entries(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.sample(frac=1).reset_index(drop=True)

def copy_mris(dataframe):
    os.makedirs(CONFIG['shuffled_mri_dir'], exist_ok=True)
    for idx, row in dataframe.iterrows():
        mri = nib.load(row['MRI_fname'])
        suffix = f'mri_mci_{idx}.nii'
        nib.save(mri, CONFIG['shuffled_mri_dir'] + suffix)

def convert_nii_to_dcm():
    shuffle_dir = CONFIG['shuffled_mri_dir'][:-1] + '_dcm'
    os.makedirs(shuffle_dir, exist_ok=True)
    for root, d, files in os.walk(CONFIG['shuffled_mri_dir']):
        for f in files:
            dcm_name = f[:-4] + ".dcm"
            os.system(f'medcon -c dicom -o {shuffle_dir}/{dcm_name} -f {root}/{f}')

def test_mri_ids():
    df = pd.read_csv(CONFIG['shuffled_mris'], dtype={'RID': str})
    clusters = load_clustered_mris()
    for idx, row in clusters.iterrows():
        pass

def compare_mris():
    d = '/data2/MRI_PET_DATA/raw_data/MRI_nii_cox_noqc/'
    for root, d, files in os.walk(d):
        for f in files:
            img = nib.load(f'{root}/{f}')
            rid = f[0:4]
            img_shuffled = _load_shuffled_mri_by_rid(rid)
            if img_shuffled is None:
                continue
            assert(np.allclose(img.get_fdata(), img_shuffled.get_fdata()))

def _load_shuffled_mri_by_rid(rid: str):
    df = pd.read_csv(CONFIG['shuffled_mris'], dtype={'RID': str}, index_col=0)
    idx = df.query('RID == @rid').index
    if len(idx) > 0:
        idx = idx[0]
    else:
        return None
    d = CONFIG['shuffled_mri_dir']
    img = nib.load(f'{d}mri_mci_{idx}.nii')
    return img

def main():
    clusters = load_clustered_mris()
    df = randomly_sample_from_cluster(clusters, 6, 6)
    df = shuffle_df_entries(df)
    df.to_csv(CONFIG['shuffled_mris'])
    # copy_mris(df)
    # convert_nii_to_dcm()
