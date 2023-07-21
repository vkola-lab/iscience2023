from statistics.clustered_mlp_output_wrappers import *
from statistics.mlp_output_wrappers import map_neuromorph_abbr_to_lobe
import numpy as np
from tqdm import tqdm

@pa.check_types
def aggregate_statistic_by_region(
        dataset: str='ADNI', value_set: str='Gray Matter Vol')\
        -> DataFrame[AggregatedOverRegionSchema]:
    assert dataset in ('ADNI','NACC', 'ADNI_AD')
    assert value_set in ('Gray Matter Vol', 'Shap Value')
    df : DataFrame[ParcellationClusteredLongSchema] = load_any_long(dataset)
    statistic_for_region = cluster_average(df=df, value_set=value_set)
    statistic_for_region['Dataset'] = dataset
    statistic_for_region.loc[:,'Cortex'] = statistic_for_region['Region'].apply(
        lambda x: map_neuromorph_abbr_to_lobe()[x]
    )
    return statistic_for_region

@pa.check_types
def aggregate_statistic_by_rid(
        dataset: str='ADNI', value_set: str='Gray Matter Vol') :
    assert dataset in ('ADNI','NACC', 'ADNI_AD')
    assert value_set in ('Gray Matter Vol', 'Shap Value')
    df = load_any_long(dataset)
    statistic_for_region = cluster_average_over_region(df=df, value_set=value_set)
    statistic_for_region['Dataset'] = dataset
    return statistic_for_region

def aggregate_statistic_by_cortex(
        dataset: str='ADNI', value_set: str='Gray Matter Vol'):
    assert dataset in ('ADNI', 'NACC', 'ADNI_AD')
    assert value_set in ('Gray Matter Vol', 'Shap Value')
    df = aggregate_statistic_by_region(dataset=dataset, value_set=value_set)
    df.drop(columns=['Dataset','Region'], inplace=True)
    df = df.groupby(['Cluster Idx', 'Cortex']).agg(np.mean).reset_index()
    return df

@pa.check_types
def aggregate_statistic_by_cortex_model(
        dataset: str='ADNI', value_set: str='Gray Matter Vol') -> \
    DataFrame[AggregatedOverLobeSchema]:
    assert dataset in ('ADNI','NACC', 'ADNI_AD')
    assert value_set in ('Gray Matter Vol', 'Shap Value')
    df = aggregate_statistic_by_rid(dataset=dataset, value_set=value_set)
    df.drop(columns=['Dataset'], inplace=True)
    return DataFrame[AggregatedOverLobeSchema](df)

@pa.check_types
def cluster_average(
        df: DataFrame[ParcellationClusteredLongSchema], value_set: str) -> \
        DataFrame[AggregatedOverRegionSchema]:
    assert value_set in ('Gray Matter Vol', 'Shap Value')
    df = df.copy()
    df.drop(columns=['RID'], inplace=True)
    region_average_over_cluster = df.groupby(
            ['Dataset', 'Cluster Idx', 'Region']).agg(
        np.mean
    )
    region_average_over_cluster = \
        region_average_over_cluster[[value_set]].reset_index()
    return region_average_over_cluster

@pa.check_types
def cluster_average_over_region(
        df: DataFrame[ParcellationClusteredLongSchema],
        value_set: str) -> DataFrame[AggregatedOverLobeSchema]:
    assert value_set in ('Gray Matter Vol', 'Shap Value')
    df = df.copy()
    _map = map_neuromorph_abbr_to_lobe()
    tqdm.pandas()
    df.loc[:,'Cortex'] = df['Region'].progress_apply(
        lambda x: _map[x]
    )
    df.drop(columns=['Region'], inplace=True)
    rid_average_over_cluster = df.groupby(['Cluster Idx', 'Cortex', 'RID']).agg(
        np.mean
    )
    rid_average_over_cluster = rid_average_over_cluster[[value_set]].reset_index()
    return DataFrame[AggregatedOverLobeSchema](rid_average_over_cluster)

def group_cortical_regions_by_subtype(dataset: str='ADNI'):
    tbl = aggregate_statistic_by_region(dataset=dataset, value_set='Gray Matter Vol')
    tbl = tbl.loc[tbl.Cortex != '', :]
    tbl = tbl.rename(columns={'Cortex': 'Cortical Region', 'Cluster Idx': 'Subtype', 'Gray Matter Vol': 'ZS Gray Matter Volume'})
    return tbl