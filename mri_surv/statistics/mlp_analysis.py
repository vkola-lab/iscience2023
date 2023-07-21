from tabulate import tabulate
import itertools
from scipy import  stats
from typing import Dict
import pandas as pd
import numpy as np
from statistics.utilities import *
from statistics.clustered_mlp_output_wrappers import *
from statistics.aggregate_utilities import aggregate_statistic_by_region, \
    aggregate_statistic_by_cortex_model


def _salient_regions(group: pd.DataFrame) -> Dict[str, dict]:
    df_schema = pa.DataFrameSchema({'Region': pa.Column(str), 'Centroid':
        pa.Column(float), 'Cortex': pa.Column(str)})
    group = df_schema.validate(group)
    group = group.sort_values('Centroid', ascending=False)
    head, tail = group.head(10), group.tail(10)
    salient_dict_head : dict = head.to_dict('records')
    salient_dict_tail : dict = tail.to_dict('records')
    salient_dict = {}
    salient_dict['Top'] = salient_dict_head
    salient_dict['Bottom'] = salient_dict_tail

    return salient_dict

@pa.check_types
def _transform_aggregate_to_centroid_schema(df: DataFrame[
    AggregatedOverRegionSchema], value_set: str) -> DataFrame[
    RegionCentroidSchema]:
    df = df.rename(columns={value_set: 'Centroid'}
        ).copy()
    df = df.drop(columns='Dataset')
    return DataFrame[RegionCentroidSchema](df)

def tabulate_centroids() -> None:
    with open(CONFIG['salient_centroids'],'w') as fi:
        for value_set in ('Shap Value', 'Gray Matter Vol'):
            fi.write(f'\n\n------------------\n{value_set}\n\n')
            for dataset in ('NACC', 'ADNI'):
                fi.write(f'\n\n----\n{dataset}\n\n')
                parcellation_averages_over_rid = \
                    aggregate_statistic_by_region(
                    dataset=dataset,
                    value_set=value_set
                )
                parcellation_averages_over_rid = \
                    _transform_aggregate_to_centroid_schema(
                            parcellation_averages_over_rid, value_set
                    )
                parcellation_averages_dict = \
                    parcellation_averages_over_rid.groupby(
                            'Cluster Idx').apply(
                            _salient_regions
                    )
                for _, value in parcellation_averages_dict.items():
                    fi.write('\n\n\nTop\n\n')
                    fi.write(tabulate(value['Top'], headers='keys'))
                    fi.write('\n\n\nBottom\n\n')
                    fi.write(tabulate(value['Bottom'], headers='keys'))

@pa.check_types
def _make_correlation_series(df: DataFrame[AggregatedOverRegionSchema]) -> \
        DataFrame[CentroidByRegionSchema]:
    df = df.copy().drop(
            columns=['Dataset', 'Cluster Idx']
    ).set_index('Region').sort_index()
    return DataFrame[CentroidByRegionSchema](df)

@pa.check_types
def _cluster_average_corr(
        statistics_by_region: Dict[str, DataFrame[AggregatedOverRegionSchema]],
        value_set: str) -> Dict[str, Dict[str,np.array]]:
    correlation_statistics = {
            'NACC'   :
                {
                        'p_value' : np.zeros((4, 4)),
                        'corrcoef': np.zeros((4, 4))
                },
            'ADNI_AD':
                {
                        'p_value' : np.zeros((4, 4)),
                        'corrcoef': np.zeros((4, 4))
                }
    }
    for cluster_idx, df in statistics_by_region['ADNI'].groupby(
            'Cluster Idx'):
        df = _make_correlation_series(df)
        for dataset in ('NACC', 'ADNI_AD'):
            if (dataset == 'ADNI_AD' and value_set == 'Shap Value'):
                continue
            for cluster_idx_inner, sub_tbl_inner in \
                    statistics_by_region[dataset].groupby('Cluster Idx'):
                sub_tbl_inner = _make_correlation_series(sub_tbl_inner)
                assert np.array_equal(df.columns, sub_tbl_inner.columns)
                correlation_statistics[dataset]['corrcoef'][
                    int(cluster_idx), int(cluster_idx_inner)
                ], correlation_statistics[dataset]['p_value'][
                    int(cluster_idx), int(cluster_idx_inner)
                ] = stats.spearmanr(df, sub_tbl_inner, nan_policy='raise')
                print(df)
    return correlation_statistics

def correlate_cluster_averages(value_set='Shap Value') -> \
        Dict[str, Dict[str,np.array]]:
    statistics_by_region = {}
    for dataset in ('NACC','ADNI','ADNI_AD'):
        if not (dataset == 'ADNI_AD' and value_set == 'Shap Value'):
            statistics_by_region[dataset] = aggregate_statistic_by_region(
                    value_set=value_set, dataset=dataset
            )
            statistics_by_region[dataset].drop(columns='Cortex', inplace=True)
    correlation_statistics = _cluster_average_corr(
            statistics_by_region, value_set
    )
    with open(
            CONFIG['correlate_cluster_averages'] +
            value_set.lower().replace(' ','') + '.txt', 'w') as fi:
        fi.write(f'{value_set}\n')
        for key, table in correlation_statistics.items():
            fi.write(f'\nCorrelation coefficients between clusters, '
                     f'ADNI vs {key}\n')
            fi.write(
                tabulate(
                    table['corrcoef'],
                        headers=np.arange(0,4).astype('str'), showindex='always'
                )
            )
            fi.write(f'\nP-values between clusters, ADNI vs {key}\n')
            fi.write(
                tabulate(
                    table['p_value'], headers=np.arange(0,4).astype('str'),
                        showindex='always'
                    )
            )
    return correlation_statistics

def correlate_cluster_averages_by_cortex(value_set='Gray Matter Vol') -> dict:
    statistics_by_region = {}
    for dataset in ('NACC','ADNI','ADNI_AD'):
        if not (dataset == 'ADNI_AD' and value_set == 'Shap Value'):
            statistics_by_region[dataset] = aggregate_statistic_by_region(
                    value_set=value_set, dataset=dataset
            )
            statistics_by_region[dataset].drop(columns='Region', inplace=True)
            statistics_by_region[dataset] = \
                statistics_by_region[dataset].groupby(
                        ['Dataset','Cluster Idx','Cortex']
                ).agg(np.mean)
            statistics_by_region[dataset].reset_index(inplace=True)
    correlation_statistics = {
            'NACC':
                                  {
                                          'p_value': np.zeros((4,4)),
                                          'corrcoef': np.zeros((4,4))},
                              'ADNI_AD':
                                    {
                                            'p_value': np.zeros((4,4)),
                                            'corrcoef': np.zeros((4,4))
                                    }}
    for cluster_idx, sub_tbl in \
            statistics_by_region['ADNI'].groupby('Cluster Idx'):
        sub_tbl = sub_tbl.copy().drop(
                columns=['Dataset','Cluster Idx']
        ).set_index('Cortex').sort_index()
        for dataset in ('NACC','ADNI_AD'):
            if (dataset == 'ADNI_AD' and value_set == 'Shap Value'):
                continue
            for cluster_idx_inner, sub_tbl_inner \
                    in statistics_by_region[dataset].groupby('Cluster Idx'):
                sub_tbl_inner = sub_tbl_inner.copy().drop(
                        columns=['Dataset','Cluster Idx']
                ).set_index('Cortex').sort_index()
                assert np.array_equal(sub_tbl.columns, sub_tbl_inner.columns)
                correlation_statistics[dataset]['corrcoef'][
                    int(cluster_idx), int(cluster_idx_inner)
                ], correlation_statistics[dataset]['p_value'][
                    int(cluster_idx), int(cluster_idx_inner)
                ] = stats.spearmanr(sub_tbl, sub_tbl_inner, nan_policy='raise')
    with open(
            CONFIG['correlate_cluster_averages'] +
            value_set.lower().replace(' ','') +
            '_by_cortex.txt', 'w') as fi:
        fi.write(f'{value_set}\n')
        for key, table in correlation_statistics.items():
            fi.write(f'\nCorrelation coefficients between cortical regions,'
                     f'ADNI vs {key}\n')
            fi.write(
                tabulate(
                    table['corrcoef'], headers=np.arange(0,4).astype('str'),
                        showindex='always'
                )
            )
            fi.write(f'\nP-values between cortical regions, ADNI vs {key}\n')
            fi.write(
                tabulate(
                    table['p_value'], headers=np.arange(0,4).astype('str'),
                        showindex='always'
                    )
            )
    return correlation_statistics

def transition_state_model() -> pd.DataFrame:
    # subtype_t2 ~ subtype_t1 + (1|subject)
    statistics_by_region = {}
    for dataset in ('ADNI','ADNI_AD'):
        statistics_by_region[dataset] = load_any_long(dataset=dataset)
        statistics_by_region[dataset] = _combine_cluster_idx(
                statistics_by_region[dataset]
        )
        statistics_by_region[dataset] = \
            statistics_by_region[dataset].rename(
                    columns={'Cluster Idx': f'ClusterIdx_{dataset}'}
            )
    df = pd.merge(
            left=statistics_by_region['ADNI_AD'],
            right=statistics_by_region['ADNI'],
            left_on='RID',
            right_on='RID',
            how='inner'
    )
    cross_tabbed_data = pd.crosstab(
                                df['ClusterIdx_ADNI'],
                                df['ClusterIdx_ADNI_AD']
                        )
    with open(CONFIG['transition_state_model'],'w') as fi:
        fi.write(
                tabulate(
                        cross_tabbed_data,
                        headers=np.arange(0, 4).astype('str'),
                        showindex='always'
                )
        )
    return cross_tabbed_data

def _combine_cluster_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['RID','Cluster Idx']].copy()
    df = df.groupby(['RID']).agg(pd.unique).reset_index()
    return df

def write_cortical_regions_by_subtype():
    tbl = aggregate_statistic_by_cortex_model(
        dataset='ADNI', value_set='Gray Matter Vol'
    )
    tbl['Visit'] = 'MCI'
    tbl_ad = aggregate_statistic_by_cortex_model(
        dataset='ADNI_AD', value_set='Gray Matter Vol'
    )
    tbl_ad['Visit'] = 'AD'
    tbl = pd.concat([tbl, tbl_ad], axis=0, ignore_index=True)
    tbl = tbl.loc[tbl.Cortex != '', :]
    tbl = tbl.rename(
            columns={
                    'Cortex': 'Cortical Region',
                    'Cluster Idx': 'Subtype',
                    'Gray Matter Vol': 'ZS Gray Matter Volume'
            }
    )
    tbl['Subtype'] = tbl['Subtype'].replace(
            {'0': 'H', '1': 'IH', '2': 'IL', '3': 'L'}
    )
    age_gender_ad = retrieve_age_gender_from_demo_ad()
    age_gender_ad['Visit'] = 'AD'
    age_gender = retrieve_age_gender_from_demo()
    age_gender['Visit'] = 'MCI'
    age_gender = pd.concat([age_gender, age_gender_ad], axis=0)
    tbl = tbl.merge(age_gender, left_on=['RID','Visit'], right_on=['RID','Visit'], validate='many_to_one')
    tbl.to_csv(
            './metadata/data_processed/cortical_regions_by_subtype.csv',
            index=False
    )
    return tbl

def main():
    s = load_shap_with_age_gender()
    s.to_csv(
            './metadata/data_processed/shap_with_parcellations_long.csv',
            index=True
    )
    write_cortical_regions_by_subtype()

    tabulate_centroids()
    ####debugged through here

    correlate_cluster_averages('Shap Value')
    correlate_cluster_averages('Gray Matter Vol')

    transition_state_model()

    # generate_mri_brains_by_cluster()

    correlate_cluster_averages_by_cortex()
    print('Warning: not dropping missing cortical areas; also, results may '
          'not align with write_cortical_regions_by_subtype')

if __name__ == '__main__':
    main()
