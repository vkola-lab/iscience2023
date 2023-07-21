from statistics.mlp_output_wrappers import *
from statistics.dataframe_validation import *
from statistics.utilities import CONFIG, time_and_progress
from typing import Union
import pandas as pd

@pa.check_types
def load_metadata_pathology_mlp_pivot() -> DataFrame[MlpPathologySchema]:
    df_long : DataFrame[MlpSchema] = load_mlp()
    metadata : DataFrame[DemoSchemaNacc] = load_demographics()['nacc']
    df_pivot : DataFrame[MlpPivotSchema] = convert_to_pivot(df_long)
    pivot = df_pivot.reset_index().copy()
    metadata_nacc = metadata[CONFIG['path_config']['metadata_columns']].copy()
    nacc_mlp = pivot.merge(
        metadata_nacc, how='inner', on='RID',
        validate='one_to_one'
        )
    nacc_mlp = nacc_mlp.query(
            'AgeAtDeath > 0'
    ).reset_index(drop=True)
    clusters : DataFrame[ClusterSchema] = load_nacc_clusters()
    nacc_mlp['Cluster Idx'] = clusters.loc[nacc_mlp['RID'],
                                           'Cluster Idx'].to_numpy()
    return DataFrame[MlpPathologySchema](nacc_mlp)

def load_metadata_pathology_mlp_pivot() -> DataFrame[MlpPathologySchema]:
    df_long : DataFrame[MlpSchema] = load_mlp()
    metadata : DataFrame[DemoSchemaNacc] = load_demographics()['nacc']
    df_pivot : DataFrame[MlpPivotSchema] = convert_to_pivot(df_long)
    pivot = df_pivot.reset_index().copy()
    metadata_nacc = metadata[CONFIG['path_config']['metadata_columns']].copy()
    nacc_mlp = pivot.merge(
        metadata_nacc, how='inner', on='RID',
        validate='one_to_one'
        )
    nacc_mlp = nacc_mlp.query(
            'AgeAtDeath > 0'
    ).reset_index(drop=True)
    clusters : DataFrame[ClusterSchema] = load_nacc_clusters()
    nacc_mlp['Cluster Idx'] = clusters.loc[nacc_mlp['RID'],
                                           'Cluster Idx'].to_numpy()
    return DataFrame[MlpPathologySchema](nacc_mlp)

@pa.check_types
def load_shap_with_age_gender() -> DataFrame[ShapParcellationAgeSexSchema]:
    shap = load_shap_with_parcellations()
    demographics = load_demographics()
    demo_age_and_gender = retrieve_age_gender_from_demo()
    return shap.merge(demo_age_and_gender, left_on='RID',right_on='RID',validate='many_to_one')

@pa.check_types
def _add_gray_matter_vol_and_cluster(parcellation: DataFrame[
    ParcellationClusteredSchema], shap: DataFrame[ShapSchema]) -> \
        DataFrame[ShapParcellationClusterSchema]:
    """Adds two columns to shap property:
    'Gray Vol Bin Range': right-inclusive bin of gray matter volume
    'Gray Matter Vol Bin': bin from 0-(nbins-10) corresponding to ranked volume of gray matter
    ***Note: all standardized to ADNI gray matter volume
    Utilizes gray matter volumes from parcellation data

    Args:
        nbins (int, optional): number of bins to use for dividing up the gray matter volumes. Defaults to 10.
    """
    shap = shap.copy()
    shap.set_index(['RID','Dataset','Region'], inplace=True)
    parcellation_long = _make_gray_matter_vol_long(parcellation)
    parcellation_long.set_index(['RID','Dataset','Region'], inplace=True)
    shap.rename(columns={'Gray Matter Vol': 'Gray Matter Vol Raw'}, inplace=True)
    shap = \
        shap.merge(
                parcellation_long,
                left_index=True,
                right_index=True,
                validate='many_to_one'
        )
    shap.reset_index(inplace=True)
    return DataFrame[ShapParcellationClusterSchema](shap)

@pa.check_types
def _make_gray_matter_vol_long(
        parcellation: DataFrame[ParcellationClusteredSchema]) -> \
        DataFrame[ParcellationClusteredLongSchema]:
    """Melts gray matter dataframe:

    Args:
        nbins (int, optional): number of bins to use for dividing up the gray matter volumes. Defaults to 10.
    """
    parcellation = parcellation.copy()
    parcellation_long = parcellation.melt(
        id_vars=('RID','Dataset', 'Cluster Idx'),
        value_name='Gray Matter Vol',
        var_name='Region'
    )
    return DataFrame[ParcellationClusteredLongSchema](parcellation_long)

@pa.check_types
def load_shap_with_parcellations() -> DataFrame[ShapParcellationClusterSchema]:
    shap = load_shap('sur_ventricle')  # set self.shap
    parcellation = load_parcellations_clustered() # set self.parcellation
    return _add_gray_matter_vol_and_cluster(parcellation, shap)

@pa.check_types
def load_parcellations_clustered() -> DataFrame[ParcellationClusteredSchema]:
    parcellation = pd.read_csv(
            CONFIG['parcellation_csv_clustered'],
            dtype={'RID': str, 'Cluster Idx': str})
    return DataFrame[ParcellationClusteredSchema](parcellation)

@pa.check_types
def load_ad_parcellations_clustered() -> DataFrame[ParcellationClusteredSchema]:
    parcellation = pd.read_csv(CONFIG['parcellation_csv_ad_clustered'],
                               dtype={'RID': str, 'Cluster Idx': str})
    return DataFrame[ParcellationClusteredSchema](parcellation)


@pa.check_types
def load_parcellations_long() -> DataFrame[ParcellationClusteredLongSchema]:
    _parcellation = load_parcellations_clustered()
    return _make_gray_matter_vol_long(_parcellation)

@pa.check_types
def load_ad_parcellations_long() -> DataFrame[ParcellationClusteredLongSchema]:
    _parcellation = load_ad_parcellations_clustered()
    return _make_gray_matter_vol_long(_parcellation)

@pa.check_types
def load_any_long(dataset: str) -> Union[DataFrame[ParcellationClusteredLongSchema], None]:
    df = None
    if dataset not in ('ADNI','NACC','ADNI_AD'):
        raise KeyError(f'{dataset} not in (ADNI, NACC, ADNI_AD)!')
    if dataset in ('ADNI','NACC'):
        df = load_shap_with_parcellations().query('Dataset == @dataset').copy()
    elif dataset in ('ADNI_AD'):
        df = load_ad_parcellations_long()
    df.reset_index(drop=True, inplace=True)
    return df

@pa.check_types
def load_nacc_clusters() -> DataFrame[ClusterSchema]:
    df = load_parcellations_clustered()
    df = df.query('Dataset == \'NACC\'')[['RID','Cluster Idx']
            ].set_index('RID', drop=True)
    return DataFrame[ClusterSchema](df)

@pa.check_types
def load_all_clusters() -> DataFrame[ClusterSchema]:
    df = load_parcellations_clustered()
    df = df[['RID','Cluster Idx']].set_index('RID', drop=True)
    return ClusterSchema.validate(df)

@pa.check_types
def load_adni_ad_clusters() -> DataFrame[ClusterSchema]:
    cluster =  ClusterSchema.validate(load_ad_parcellations_clustered()[
        ['RID','Cluster Idx']].set_index('RID', drop=True))
    return cluster

@pa.check_types
def load_adni_clusters() -> DataFrame[ClusterSchema]:
    df = load_parcellations_clustered()
    df = df.query('Dataset == \'ADNI\'')[['RID','Cluster Idx']
            ].set_index('RID', drop=True)
    return DataFrame[ClusterSchema](df)

@pa.check_types
def load_clustered_mris() -> DataFrame[ClusteredMriSchema]:
    clusters = load_all_clusters()['Cluster Idx']
    demographics = load_demographics()['adni']
    demographics = \
        demographics[['RID','MRI_IID','MRI_fname','PROGRESSES']].copy()
    demographics['Cluster Idx'] = \
        demographics['RID'].replace(clusters.to_dict())
    return DataFrame[ClusteredMriSchema](demographics)

@pa.check_types
def load_metadata_mlp_pivot() -> DataFrame[MlpPivotClusterSchema]:
    df_long = load_mlp()
    metadata = load_demographics()
    metadata = time_and_progress(metadata)
    df_pivot = convert_to_pivot(df_long)
    pivot = df_pivot.reset_index().copy()
    mlp_pivot = pivot.merge(
        metadata, how='inner', on=['RID','Dataset'],
        validate='one_to_one'
        )
    clusters = load_all_clusters()
    mlp_pivot['Cluster Idx'] = clusters.loc[mlp_pivot['RID'], 'Cluster Idx'].to_numpy()
    return DataFrame[MlpPivotSchema](mlp_pivot)

@pa.check_types
def load_metadata_survival() -> DataFrame[MlpSurvivalSchema]:
    metadata = load_demographics()
    metadata = time_and_progress(metadata)
    clusters = load_all_clusters()
    metadata['Cluster Idx'] = clusters.loc[metadata['RID'], 'Cluster Idx'].to_numpy()
    return DataFrame[MlpSurvivalSchema](metadata)