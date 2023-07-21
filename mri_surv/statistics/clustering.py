from typing import Tuple
from statistics.mlp_output_wrappers import *
from statistics.dataframe_validation import *
from statistics.utilities import CONFIG
from statistics.dataframe_validation import *
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = [
        'cluster_parcellations',
        'cluster_ad_parcellations',
        'cluster_parcellations_generic',
        'cluster_ad_parcellations_generic',
        'load_zscored_parcellations',
        'parcellation_summary_stats',
        'zscore_parcellations']



@pa.check_types
def _cluster_csf_data(csf_centroids: DataFrame[ParcellationCentroidSchema],
                      parcellation_df: DataFrame[ParcellationFullSchema]) -> \
        pd.DataFrame:
    """Clusters parcellation data based on centroids obtained for each
    parcellated region from csf quartiles

    Args:
        csf_centroids (pd.DataFrame): dataframe with a row corresponding to
        each quantile of biomarker, and column for each parcellated region
        parcellation_df (pd.DataFrame): dataframe with row for each
        participant and column for each quantile of biomarker

    Raises:
        NotImplementedError: csf_centroids dataframe must be an array with
        indices Biomarker Quartile and abeta
        TypeError: parcellation columns must be the same as the csf centroid
        columns

    Returns:
        corr_df (pd.DataFrame): dataframe w/ index ('RID','Dataset') and
        columns each corresponding to spearman correlation w/ centroids
    """
    if not np.array_equal(parcellation_df.columns,
                          csf_centroids.columns):
        print(parcellation_df.columns)
        print(csf_centroids.columns)
        raise TypeError
    cols = parcellation_df.columns
    csf_centroids = csf_centroids[
        cols]  # by accessing this way, ensure correct ordering
    corr_df = parcellation_df.reset_index()[
        ['RID', 'Dataset']].copy()  # get empty dataframe for storage
    for idx, row in csf_centroids.iterrows():  # for each centroid
        cors = parcellation_df.corrwith(
                row, axis=1, method='spearman'
        )  # get spearman correlation
        # coefficient between each centroid and each patient
        cors = cors.rename(str(idx[0]))  # rename the index -> set to
        # corresponding to cluster #
        corr_df = corr_df.merge(
                cors, left_on=['RID', 'Dataset'],
                right_on=['RID', 'Dataset'] # corresponding to cluster #
        )  # merge on RID and dataset
    return corr_df  # return df

def retrieve_cluster_ids_csf(df: pd.DataFrame, biomarker: str = 'abeta',
                             n_cuts: int = 4, reverse_quartiles=False):
    '''
    Takes raw array (dataframe with shap values from ADNI csf model), divides
    into quantiles using qcut, returns values returned by qcut appended to
    dataframe
     
     Then, for the function that I templated yesterday, load the ADNI metadata 
     and apply qcut to divide into 4 groups, and return the metadata sheet with 
     an additional column called Biomarker Quartile
    '''
    biomarker_col = df[biomarker].copy()
    # Apply qcut
    df['Biomarker Quartile'] = pd.qcut(biomarker_col, q=n_cuts, labels=False)

    if reverse_quartiles is True:
        df['Biomarker Quartile'] = 3-df['Biomarker Quartile']

    return df

def _load_abeta_centroids() -> pd.DataFrame:
    return pd.read_csv(CONFIG['parcellation_csv_centroids'],
                       index_col=['Biomarker Quartile', 'abeta'])

def _load_centroids(biomarker: str) -> pd.DataFrame:
    return pd.read_csv(CONFIG['parcellation_' + biomarker + '_csv_centroids'],
                       index_col=['Biomarker Quartile', biomarker])

@pa.check_types
def load_zscored_parcellations(parcellation: DataFrame[ParcellationSchema],
                               parcellation_for_zs: DataFrame[
                                   ParcellationSchema] =
                               None) -> DataFrame[ParcellationFullSchema]:
    """assigns parcellation (pd.DataFrame) property, parcellation_means (
    pd.Series), parcellation_std (pd.Series).
    Final columns include RID, Dataset, and z-scored regions.

    Raises:
        ValueError: if data are not properly z-scored
    """
    if parcellation_for_zs is None:
        parcellation_for_zs = parcellation.copy()
    parcellation.drop(columns=CONFIG['ventricles'] + \
                              ['PROGRESSION_CATEGORY'], inplace=True)
    parcellation_for_zs.drop(columns=CONFIG['ventricles'] + \
                                     ['PROGRESSION_CATEGORY'], inplace=True)

    neuromorph_abbr_dict = load_roiabbr_to_roiname_map()

    parcellation.rename(columns=neuromorph_abbr_dict, inplace=True)

    parcellation = DataFrame[ParcellationFullSchema](parcellation)

    parcellation_for_zs.rename(columns=neuromorph_abbr_dict, inplace=True)

    parcellation_for_zs = DataFrame[ParcellationFullSchema](
            parcellation_for_zs)

    parcellation_means, parcellation_std = parcellation_summary_stats(
        parcellation)
    parcellation_for_zs = zscore_parcellations(parcellation_for_zs,
                                               parcellation_means,
                                               parcellation_std)
    parcellation = zscore_parcellations(parcellation, parcellation_means,
                                        parcellation_std)
    if not _check_parcellation_zscore(parcellation):
        raise ValueError
    return parcellation_for_zs

@pa.check_types
def _check_parcellation_zscore(_parcellation: DataFrame[
    ParcellationFullSchema]) -> bool:
    sub_df = _parcellation.query('Dataset == \'ADNI\'').drop(
        columns=['RID', 'Dataset'])
    mn_check = np.allclose(sub_df.mean(), 0)
    std_check = np.allclose(sub_df.std(), 1)
    return mn_check and std_check

@pa.check_types
def zscore_parcellations(parcellation_dataframe: DataFrame[
                        ParcellationFullSchema],
                         parcellation_means: pd.Series,
                         parcellation_std: pd.Series
                         ) -> DataFrame[ParcellationFullSchema]:
    """Takes as argument wide-form parcellation data and z-scores according
    to class properties

    Args:
        parcellation_dataframe (pd.DataFrame): wide-form data with columns
        RID, Dataset, and the rest brain regions

    Returns:
        pd.DataFrame: dataframe z-scored according to self.parcellation_means
        and self.parcellation_std
    """
    df = parcellation_dataframe.set_index(['RID', 'Dataset']).subtract(
            parcellation_means, axis=1).divide(
            parcellation_std, axis=1).reset_index()
    return DataFrame[ParcellationFullSchema](df)

@pa.check_types
def parcellation_summary_stats(parcellation_df:
DataFrame[ParcellationFullSchema]) -> Tuple[
    pd.Series, pd.Series]:
    """Takes wide-form parcellation data. computes mean and std for all of
    ADNI dataset.

    Args:
        parcellation_df (pd.DataFrame): dataframe consisting of brain regions
        and RID and Dataset columns

    Returns:
        Tuple[pd.Series, pd.Series]: means and std for ADNI dataset for each
        parcellated region
    """
    adni_parcellation = parcellation_df.query('Dataset == \'ADNI\'').copy()
    parcellation_means = adni_parcellation.drop(columns=[
            'RID', 'Dataset']).mean()
    parcellation_std = adni_parcellation.drop(columns=[
            'RID', 'Dataset']).std()
    return parcellation_means, parcellation_std

@pa.check_types
def _generate_classifiers(adni_metadata: DataFrame[DemoSchemaAdni],
                          parcellation_data: DataFrame[ParcellationFullSchema],
                          biomarker='abeta') -> DataFrame[
    ParcellationCentroidSchema]:
    """takes a biomarker (currently implemented for abeta, ptau, and ttau)
    and assigns mean values
        for each parcellated region corresponding to each quartile.

    Args:
        biomarker (str, optional): CSF biomarker to use for classifying CSF.
        Defaults to 'abeta'.

    Raises:
        NotImplementedError: must be a CSF biomarker
    """
    if biomarker not in ['abeta', 'ptau', 'tau']:
        raise NotImplementedError

    adni_metadata = retrieve_cluster_ids_csf(
            adni_metadata, biomarker=biomarker, n_cuts=4, reverse_quartiles=(
                    biomarker in ('ptau','tau',))
    )
    parcellation_adni = \
        parcellation_data.query('Dataset == \'ADNI\'').copy()
    parcellation_adni = parcellation_adni.merge(
            adni_metadata[['RID', 'Biomarker Quartile', biomarker]],
            left_on=['RID'], right_on=['RID'],
            validate='one_to_one'
    )
    parcellation_no_labels = parcellation_adni.drop(
            columns=['RID', 'Dataset']
    )
    parcellation_centroids = parcellation_no_labels.groupby(
            'Biomarker Quartile').agg(np.mean).reset_index()
    parcellation_centroids.set_index(
            ['Biomarker Quartile', biomarker], inplace=True
    )
    return DataFrame[ParcellationCentroidSchema](parcellation_centroids)

@pa.check_types
def _retrieve_cluster_ids(
        parcellation_centroids: DataFrame[ParcellationCentroidSchema],
        parcellation: DataFrame[ParcellationFullSchema]) -> pd.Series:
    """Assigns property self.cluster_df, clusters ids based on CSF biomarker
    data to each patient using spearman correlation. Takes argmax as cluster.
    """
    parcellation_centroids = parcellation_centroids.copy()
    parcellations = parcellation.copy()
    parcellations.set_index(['RID', 'Dataset'], inplace=True)
    cluster_df = _cluster_csf_data(
            parcellation_centroids, parcellations
    ).set_index(['RID', 'Dataset'])
    cluster_df = cluster_df.idxmax(axis=1)
    cluster_df.rename('Cluster Idx', inplace=True)
    return cluster_df

@pa.check_types
def generate_parcellation_zscored() -> Tuple[
    DataFrame[ParcellationSchema], DataFrame[ParcellationFullSchema]]:
    parcellation = load_parcellations()
    parcellation_zscored = load_zscored_parcellations(parcellation)
    return parcellation, parcellation_zscored

@pa.check_types
def generate_ad_parcellation_zscored() -> Tuple[
    DataFrame[ParcellationSchema], DataFrame[ParcellationFullSchema]]:
    parcellation_ad = load_ad_parcellations()
    parcellation = load_parcellations()
    parcellation_zscored = load_zscored_parcellations(parcellation,
                                                      parcellation_ad)
    return parcellation_ad, parcellation_zscored

@pa.check_types
def cluster_parcellations() -> DataFrame[
    ParcellationClusteredSchema]:
    adni_metadata = DataFrame[DemoSchemaAdni](DemoDataFrameDict()['adni'])
    parcellation, parcellation_zscored = generate_parcellation_zscored()
    parcellation_centroids = _generate_classifiers(
            adni_metadata, parcellation_zscored,
            biomarker='abeta'
    )
    parcellation_centroids.to_csv(CONFIG['parcellation_csv_centroids'])
    idx = _retrieve_cluster_ids(parcellation_centroids, parcellation_zscored)
    parcellation = parcellation_zscored
    parcellation.set_index(['RID', 'Dataset'], inplace=True)
    parcellation = parcellation.merge(idx, how='inner', left_index=True,
                                      right_index=True, validate='one_to_one')
    parcellation = DataFrame[ParcellationClusteredSchema](parcellation)
    parcellation.to_csv(CONFIG['parcellation_csv_clustered'])
    return parcellation.reset_index()

@pa.check_types
def cluster_ad_parcellations() -> DataFrame[
    ParcellationClusteredSchema]:
    parcellation_ad, parcellation_ad_zscored = \
        generate_ad_parcellation_zscored()
    parcellation_centroids = _load_abeta_centroids()
    idx = _retrieve_cluster_ids(parcellation_centroids, parcellation_ad_zscored)
    parcellation_ad = parcellation_ad_zscored
    parcellation_ad.set_index(['RID', 'Dataset'], inplace=True)
    parcellation = parcellation_ad.merge(idx, how='inner',
                                               left_index=True,
                                         right_index=True,
                                         validate='one_to_one')
    parcellation = DataFrame[ParcellationClusteredSchema](parcellation)
    parcellation.to_csv(CONFIG['parcellation_csv_ad_clustered'])
    return parcellation.reset_index()

@pa.check_types
def cluster_ad_parcellations_generic(biomarker: str) -> DataFrame[
    ParcellationClusteredSchema]:
    parcellation_ad, parcellation_ad_zscored = \
        generate_ad_parcellation_zscored()
    parcellation_centroids = _load_centroids(biomarker)
    idx = _retrieve_cluster_ids(parcellation_centroids, parcellation_ad_zscored)
    parcellation_ad = parcellation_ad_zscored
    parcellation_ad.set_index(['RID', 'Dataset'], inplace=True)
    parcellation = parcellation_ad.merge(idx, how='inner',
                                               left_index=True,
                                         right_index=True,
                                         validate='one_to_one')
    parcellation = DataFrame[ParcellationClusteredSchema](parcellation)
    parcellation.to_csv(CONFIG['parcellation_' + biomarker +
                               '_csv_ad_clustered'])
    return parcellation.reset_index()

@pa.check_types
def cluster_parcellations_generic(biomarker: str) -> DataFrame[
    ParcellationClusteredSchema]:
    adni_metadata = DataFrame[DemoSchemaAdni](DemoDataFrameDict()['adni'])
    parcellation, parcellation_zscored = generate_parcellation_zscored()
    parcellation_centroids = _generate_classifiers(
            adni_metadata, parcellation_zscored,
            biomarker=biomarker
    )
    parcellation_centroids.to_csv(
            CONFIG['parcellation_' + biomarker + '_csv_centroids'])
    idx = _retrieve_cluster_ids(parcellation_centroids, parcellation_zscored)
    parcellation = parcellation_zscored
    parcellation.set_index(['RID', 'Dataset'], inplace=True)
    parcellation = parcellation.merge(idx, how='inner', left_index=True,
                                      right_index=True, validate='one_to_one')
    parcellation = DataFrame[ParcellationClusteredSchema](parcellation)
    parcellation.to_csv(
            CONFIG['parcellation_' + biomarker + '_csv_clustered'])
    return parcellation.reset_index()

@pa.check_types
def cluster_similarity(
        biomarker1: DataFrame[ParcellationClusteredSchema],
        biomarker2: DataFrame[ParcellationClusteredSchema],
        dataset: str,
        metric=np.mean
):
    set1 = biomarker1.query('Dataset == @dataset').copy()
    set1 = set1[['RID','Cluster Idx']].sort_values(by='RID')
    set2 = biomarker2.query('Dataset == @dataset').copy()
    set2 = set2[['RID','Cluster Idx']].sort_values(by='RID')
    clusters1 = set1[['Cluster Idx']].to_numpy()
    clusters2 = set2[['Cluster Idx']].to_numpy()
    assert(np.array_equal(set1['RID'].to_numpy(), set2['RID'].to_numpy()))
    return metric(clusters1,clusters2)

def _mae(x,y):
    x = x.astype('float')
    y = y.astype('float')
    assert(len(x) == len(y))
    return np.sum(np.abs(x-y))/len(x)

def _similarity(x,y):
    return np.mean(x == y)

def plot_confusion_matrix(conf_mat: np.ndarray, tit_le: str):
    ax = sns.heatmap(conf_mat)
    ax.invert_yaxis()
    ax.set_xticklabels(['H','IH','IL','L'])
    ax.set_yticklabels(['H','IH','IL','L'])
    plt.savefig(f'figures/confusion_matrix_{tit_le}.png')
    plt.close()

def main():
    abeta = cluster_parcellations()
    tau = cluster_parcellations_generic('tau')
    ptau = cluster_parcellations_generic('ptau')
    print(cluster_similarity(abeta, tau, 'ADNI', _mae))
    print(cluster_similarity(abeta, tau, 'ADNI', _similarity))
    cmat = cluster_similarity(abeta, tau, 'ADNI', confusion_matrix)
    plot_confusion_matrix(cmat, 'abeta_vs_tau')
    print(cluster_similarity(abeta, ptau, 'ADNI', _mae))
    print(cluster_similarity(abeta, ptau, 'ADNI', _similarity))
    cmat = (cluster_similarity(abeta, ptau, 'ADNI', confusion_matrix))
    plot_confusion_matrix(cmat, 'abeta_vs_ptau')
    print(cluster_similarity(tau, ptau, 'ADNI', _mae))
    print(cluster_similarity(tau, ptau, 'ADNI', _similarity))
    cmat = (cluster_similarity(tau, ptau, 'ADNI', confusion_matrix))
    plot_confusion_matrix(cmat, 'tau_vs_ptau')