from statistics.utilities import CONFIG
from typing import Dict
from statistics.dataframe_validation import *
from pandera.typing import DataFrame
import pandera as pa
import numpy as np
import pandas as pd
import re

__all__ = [
    'DemoDataFrameDict',
    'retrieve_age_gender_from_demo',
    'retrieve_age_gender_from_demo_ad',
    'MetadataSurvival',
    'convert_to_pivot',
    'load_parcellations',
    'load_ad_parcellations',
    'load_demographics',
    'load_demographics_ad',
    'load_mlp',
    'load_shap',
    'map_neuromorph_abbr_to_lobe',
    'load_roiname_to_roiid_map', 
    'load_roiname_to_lobe_map',
    'load_roiabbr_to_roiname_map',
    'load_neuromorph_roi_map',
]

class MetadataSurvival(pd.DataFrame):
    @property
    def _constructor(self):
        return MetadataSurvival

class SummaryStats(pd.DataFrame):
    @property
    def _constructor(self):
        return SummaryStats

class DemoDataFrameDict:
    def __init__(self):
        self.metadata = load_demographics()

    def __getitem__(self, key):
        return self.metadata[key]

@pa.check_types
def convert_to_pivot(df_long: DataFrame[MlpSchema]) -> DataFrame[MlpPivotSchema]:
    """Generates pivot table from the long-form mlp data imported previously. Averages over all experiments for each RID and bin
    """
    df_long['Bins'] = df_long['Bins'].astype(str)
    df_pivot = pd.pivot_table(
        df_long, values='Predictions', index=['RID','Dataset'], columns='Bins', aggfunc=np.mean
        )
    return DataFrame[MlpPivotSchema](df_pivot)

def convert_to_pivot_preserve_exp(df_long: DataFrame[MlpSchema]) -> DataFrame:
    """Generates pivot table from the long-form mlp data imported previously. Averages over all experiments for each RID and bin
    """
    df_long['Bins'] = df_long['Bins'].astype(str)
    df_pivot = pd.pivot_table(
        df_long, values='Predictions', index=['RID','Dataset','Experiment'], columns='Bins', aggfunc=np.mean
        )
    df_pivot.rename(columns={'Experiment': 'Exp'}, inplace=True)
    return DataFrame[MlpPivotSchema](df_pivot)

@pa.check_types
def load_parcellations() -> DataFrame[ParcellationSchema]:
    return DataFrame[ParcellationSchema](
            pd.read_csv(CONFIG['parcellation_csv'], dtype={'RID': str}))

@pa.check_types
def load_ad_parcellations() -> DataFrame[ParcellationSchema]:
    return pd.read_csv(CONFIG['parcellation_csv_ad'], dtype={'RID': str})

@pa.check_types
def load_demographics() -> Dict[str, Union[DataFrame[DemoSchemaAdni],
                                           DataFrame[DemoSchemaNacc]]]:
    metadata = {}
    metadata['nacc'] = DemoSchemaNacc.validate(pd.read_csv(
            CONFIG['NACC_config']['metadata_csv'], dtype={'RID': str}
    ))
    metadata['adni'] = DemoSchemaAdni.validate(pd.read_csv(
            CONFIG['ADNI_config']['metadata_csv'], dtype={'RID': str}
    ))
    return metadata

@pa.check_types
def load_demographics_ad() -> DataFrame[DemoSchemaAdni]:
    df = pd.read_csv(
            CONFIG['ADNI_AD_config']['metadata_csv'], dtype={'RID': str}
        )
    return DataFrame[DemoSchemaAdni](df)

@pa.check_types
def load_mlp(mlp_case: str="sur_ventricle") -> DataFrame[MlpSchema]:
    if mlp_case != "sur_ventricle":
        raise NotImplementedError
    mlp = DataFrame[MlpSchema](pd.read_csv(
        CONFIG['path_config']['results'][mlp_case], dtype={'RID': str}
    ))
    return mlp

@pa.check_types
def load_shap(mlp_case: str="sur_ventricle") -> DataFrame[ShapSchema]:
    """Assigns shap (pd.DataFrame) as property. Averages shap values over all Bins and Experiments.

    Args:
        mlp_case (str, optional): type of mlp to load shap info from. Defaults to "sur_ventricle".

    Raises:
        NotImplementedError: not implemented for mlp models besides sur_ventricle
    """
    if mlp_case != "sur_ventricle":
        raise NotImplementedError
    shap = pd.read_csv(
            CONFIG['path_config']['shap_results'][mlp_case], dtype={'RID': str}
    )
    shap.drop(columns=['Bin','ExpNo'], inplace=True)
    shap = shap.groupby(
            ['RID','variable','Dataset']
        ).agg(np.nanmean).reset_index()
    shap.rename(
        columns={
            'variable': 'Region',
            'value': 'Shap Value',
            'value_gmvol': 'Gray Matter Vol'
            }, inplace=True
        )
    shap = DataFrame[ShapSchema](shap)
    return shap

def retrieve_age_gender_from_demo():
    demographics = load_demographics()
    age_and_gender = demographics['nacc'][['RID','AGE','SEX']]
    age_and_gender_adni = demographics['adni'][['RID','AGE','PTGENDER_demo']]
    age_and_gender_adni = age_and_gender_adni.rename(columns={'PTGENDER_demo': 'SEX'})
    return pd.concat([age_and_gender, age_and_gender_adni], axis=0)

def retrieve_age_gender_from_demo_ad():
    demographics = load_demographics_ad()
    age_and_gender_adni = demographics[['RID','AGE','PTGENDER_demo']]
    age_and_gender_adni = age_and_gender_adni.rename(columns={'PTGENDER_demo': 'SEX'})
    return age_and_gender_adni


def map_neuromorph_abbr_to_lobe() -> Dict:
    neuromorph_dict_lobe = load_roiname_to_lobe_map()
    neuromorph_roi_map = load_neuromorph_roi_map()
    return {
        x: (neuromorph_roi_map[y] if y in neuromorph_roi_map.keys() else '')
        for x,y in neuromorph_dict_lobe.items()
    }

def map_neuromorph_nm_to_lobe() -> Dict[str,str]:
    neuromorph_dict_lobe = map_neuromorph_abbr_to_lobe()
    abbr_to_name_map = load_roiabbr_to_roiname_map()
    return {
        x: neuromorph_dict_lobe[y] for x,y in abbr_to_name_map.items()
    }

def load_roiname_to_roiid_map() -> dict:
    neuromorph_df = _load_neuromorph_region_map()
    neuromorph_id_df = neuromorph_df[['ROIname','ROIid']].groupby('ROIname').agg(list)
    return neuromorph_id_df.to_dict()['ROIid']

def load_roiabbr_to_roiname_map() -> Dict[str, str]:
    neuromorph_df = _load_neuromorph_region_map()
    neuromorph_abbr_df = neuromorph_df[['ROIname','ROIabbr']].groupby('ROIabbr').agg(pd.unique)
    return neuromorph_abbr_df.to_dict()['ROIname']

def load_roiname_to_lobe_map() -> dict:
    neuromorph_df = _load_neuromorph_region_map()
    neuromorph_lobe_df = neuromorph_df[['ROIname','ROIbasename']].groupby('ROIname').agg(pd.unique)
    return neuromorph_lobe_df.to_dict()['ROIbasename']

def load_neuromorph_roi_map() -> Dict[int, str]:
    neuromorph_df = pd.read_csv(CONFIG['neuromorphometrics_roi'],
                                        usecols=['ROI','Cortex'],
                                        sep=',')
    neuromorph_df['Cortex'] = neuromorph_df['Cortex'].apply(_consolidate_cortex_values)
    neuromorph_df.set_index('ROI', drop=True, inplace=True)
    return neuromorph_df.to_dict()['Cortex']

def name_to_lobe_map():
    roi_to_abbr = load_roiname_to_lobe_map()
    abbr_to_lobe = pd.read_csv(CONFIG['neuromorphometrics_roi'],
                                        usecols=['ROI','Cortex'],
                                        sep=',')
    abbr_to_lobe.set_index('ROI', drop=True, inplace=True)
    abbr_to_lobe = abbr_to_lobe.to_dict()['Cortex']
    return {x: 
        abbr_to_lobe[y] for x,y in roi_to_abbr.items() if y in abbr_to_lobe.keys()}

def _consolidate_cortex_values(cortex_val: str) -> str:
    if cortex_val == 'TL-M':
        return 'TL-M'
    elif cortex_val[0:2] == 'TL':
        return 'TL-O'
    elif cortex_val[0:2] == 'OL':
        return 'OL'
    elif cortex_val[0:2] == 'PL':
        return 'PL'
    elif cortex_val[0:2] == 'FL':
        return 'FL'
    elif cortex_val == 'Limbic-Cing':
        return 'Cing'
    elif cortex_val == 'Insula':
        return 'Ins'
    elif cortex_val == 'Subcortical':
        return 'SC'
    elif cortex_val == 'Basal-Ganglia':
        return 'BG'
    else:
        raise ValueError(f'{cortex_val} is not a valid cortex region!')

def _load_neuromorph_region_map() -> pd.DataFrame:
    neuromorph_df = pd.read_csv(CONFIG['neuromorphometrics'],
                                        usecols=['ROIabbr','ROIname','ROIid','ROIbasename'],
                                        sep=';')
    neuromorph_df['ROIname'] = neuromorph_df['ROIname'].apply(
        lambda x: x.replace('Left ', '').replace('Right ', ''))
    neuromorph_df['ROIabbr'] = neuromorph_df['ROIabbr'].apply(
        lambda x: re.sub('^[rl]?', '', x))
    neuromorph_df['ROIbasename'] = neuromorph_df['ROIbasename'].apply(
        lambda x: x.split(':')[-1])

    return neuromorph_df
