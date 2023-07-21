import pandera as pa
from pandera.typing import Index, DataFrame, Series, Object
from typing import Union, Optional
"""
'ShaplessParcellationDataFrame',
'MlpPivotDataFrame',
'MlpPivotPathologyDataFrame',
'ShapParcellationDataFrame',
"""

demo_schema = {}

class DemoSchemaAdni(pa.SchemaModel):
    RID: Series[str]
    PROGRESSION_CATEGORY: Series[float] = pa.Field(nullable=True)
    PROGRESSION_CATEGORY_2YR: Series[float] = pa.Field(nullable=True)
    PROGRESSES: Series[float]
    APOE: Series[float]
    TIME_TO_PROGRESSION: Series[float] = pa.Field(nullable=True)
    TIME_TO_FINAL_DX: Series[float]
    TIMES: Series[float]
    TIMES_ROUNDED: Series[float]
    VISCODE2: Series[str]
    AGE: Series[float]
    MHNUM_medhx: Series[str] = pa.Field(nullable=True)
    MMSCORE_mmse: Series[float]
    MedhxCurrent_medhx: Series[float] = pa.Field(nullable=True)
    MedhxDescription_medhx: Series[str] = pa.Field(nullable=True)
    PTEDUCAT_demo: Series[float]
    PTGENDER_demo: Series[str]
    PTHAND_demo: Series[str]
    abeta: Series[float]
    ptau: Series[float]
    tau: Series[float]
    DX_VALUES: Series[str]
    DX_DATES: Series[str]
    DX_STEPS: Series[str]
    STEPS_UNDER_1_YR: Series[bool]
    MRI_IID: Series[str]
    MRI_fname: Series[str]

    class Config:
        name = 'baseconfig'
        strict = True

class DemoSchemaNacc(pa.SchemaModel):
    RID: Series[str]
    json: Series[str]
    nii: Series[str]
    unnamed: Series[int] = pa.Field(alias='Unnamed: 0')
    Description: Series[str]
    AcquisitionTime: Series[str] = pa.Field(nullable=True)
    ImageType: Series[str]
    Is_Original: Series[bool]
    Visit: Series[float]
    FORMVER: Series[float]
    SEX: Series[str]
    EDUC: Series[float]
    MMSECOMP: Series[float]
    MMSE: Series[float] = pa.Field(nullable=True)
    AGE: Series[float]
    NACCMOD: Series[float] = pa.Field(nullable=True)
    NACCYOD: Series[float] = pa.Field(nullable=True)
    APOE: Series[float] = pa.Field(nullable=True)
    CorticalAtrophy: Series[str] = pa.Field(nullable=True)
    LobarAtrophy: Series[str] = pa.Field(nullable=True)
    HippocampalAtrophy: Series[str] = pa.Field(nullable=True)
    NPTHAL: Series[float] = pa.Field(nullable=True)
    BRAAK: Series[str] = pa.Field(nullable=True)
    CERAD: Series[str] = pa.Field(nullable=True)
    NIA_ADNC: Series[str] = pa.Field(nullable=True)
    CERAD_SemiQuant: Series[str] = pa.Field(nullable=True)
    HippocampalSclerosis: Series[str] = pa.Field(nullable=True)
    MTLSclerosisWithHippocampus: Series[str] = pa.Field(nullable=True)
    NACCPROG: Series[float] = pa.Field(nullable=True)
    TDP_Amygdala: Series[str] = pa.Field(nullable=True)
    TDP_Hippocampus: Series[str] = pa.Field(nullable=True)
    TDP_EntorhinalInfTemporal: Series[str] = pa.Field(nullable=True)
    TDP_Neocortex: Series[str] = pa.Field(nullable=True)
    AgeAtDeath: Series[float] = pa.Field(nullable=True)
    EXAMDATE: Series[str]
    DX: Series[str]
    RID_csf: Series[str] = pa.Field(nullable=True)
    abeta: Series[float] = pa.Field(nullable=True)
    ptau: Series[float] = pa.Field(nullable=True)
    ttau: Series[float] = pa.Field(nullable=True)
    CSFABMD: Series[float] = pa.Field(nullable=True)
    CSFPTMD: Series[float] = pa.Field(nullable=True)
    CSFTTMD: Series[float] = pa.Field(nullable=True)
    EXAMDATE_csf: Series[str] = pa.Field(nullable=True)
    CSF_DATE_DIFF: Series[float] = pa.Field(nullable=True)
    FIRST_MCI: Series[str] = pa.Field(nullable=True)
    FIRST_AD_VISIT: Series[str] = pa.Field(nullable=True)
    FIRST_AD_VISIT_NO: Series[float] = pa.Field(nullable=True)
    PROGRESSES: Series[int]
    TIME_TO_PROGRESSION: Series[float] = pa.Field(nullable=True)
    FINAL_OBS_DATE: Series[str]
    AGE_AT_FINAL_DX: Series[float]
    TIME_TO_FINAL_DX: Series[float]
    DATE_OF_DEATH: Series[str] = pa.Field(nullable=True)
    File_MRI: Series[str]
    DATE_MRI: Series[str]
    Time_between_visits: Series[float]
    MCI_MRI_VISIT: Series[bool]
    File_MRI_AD: Series[str] = pa.Field(nullable=True)
    DATE_MRI_AD: Series[str] = pa.Field(nullable=True)
    Time_between_visits_AD: Series[float] = pa.Field(nullable=True)
    DX_DATES: Series[str]
    DX_VALUES: Series[str]
    PROGRESSION_CATEGORY: Series[float] = pa.Field(nullable=True)
    TIMES: Series[float]
    TIMES_ROUNDED: Series[float]
    STEPS_UNDER_1_YR: Series[bool]
    NEW_MRI: Series[str]

    class Config:
        name = 'baseconfig'
        strict = True

class CentroidByRegionSchema(pa.SchemaModel):
    ShapValue: Optional[Series[float]] = pa.Field(alias='Shap Value')
    GrayMatterVolume: Optional[Series[float]] = pa.Field(alias='Gray Matter Vol')
    Region: Index[str]

    class Config:
        name = 'baseconfig'
        strict = True

class ParcellationLongSchema(pa.SchemaModel):
    RID: Series[str]
    Dataset: Series[str]
    Region: Series[str]
    GrayMatterVol: Series[float] = pa.Field(alias='Gray Matter Vol')

class ParcellationClusteredLongSchema(ParcellationLongSchema):
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')
    ShapValue: Optional[Series[float]] = pa.Field(alias='Shap Value')

class AggregatedOverLobeSchema(pa.SchemaModel):
    RID: Series[str]
    GrayMatterVol: Optional[Series[float]] = pa.Field(alias='Gray Matter Vol')
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')
    ShapValue: Optional[Series[float]] = pa.Field(alias='Shap Value')
    GrayMatterVolRaw: Optional[Series[float]] = \
        pa.Field(alias='Gray Matter Vol Raw')
    Cortex: Series[str]

class AggregatedOverRegionSchema(pa.SchemaModel):
    GrayMatterVol: Optional[Series[float]] = pa.Field(alias='Gray Matter Vol')
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')
    ShapValue: Optional[Series[float]] = pa.Field(alias='Shap Value')
    GrayMatterVolRaw: Optional[Series[float]] = \
        pa.Field(alias='Gray Matter Vol Raw')
    Region: Series[str]
    Cortex: Optional[Series[str]]
    Dataset: Optional[Series[str]]

class RegionCentroidSchema(pa.SchemaModel):
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')
    Centroid: Series[float]
    Region: Series[str]
    Cortex: Series[str]

class ShapSchema(ParcellationLongSchema):
    ShapValue: Series[float] = pa.Field(alias='Shap Value')

class ShapParcellationClusterSchema(
        ShapSchema, ParcellationClusteredLongSchema):
    GrayMatterVolRaw: Series[float] = pa.Field(alias='Gray Matter Vol Raw')

class ShapParcellationAgeSexSchema(ShapParcellationClusterSchema):
    AGE: Series[float]
    SEX: Series[str]

class ClusterSchema(pa.SchemaModel):
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')
    RID: Index[str]

class BaseParcellationSchema(pa.SchemaModel):
    RID: Series[str]
    Acc: Series[float]
    Amy: Series[float]
    AngGy: Series[float]
    AntCinGy: Series[float]
    AntIns: Series[float]
    AntOrbGy: Series[float]
    BasCbr_FobBr: Series[float] = pa.Field(alias='BasCbr+FobBr')
    Bst: Series[float]
    Cal_Cbr: Series[float] = pa.Field(alias='Cal+Cbr')
    Cau: Series[float]
    CbeLoCbe1_5: Series[float] = pa.Field(alias='CbeLoCbe1-5')
    CbeLoCbe6_7: Series[float] = pa.Field(alias='CbeLoCbe6-7')
    CbeLoCbe8_10: Series[float] = pa.Field(alias='CbeLoCbe8-10')
    CbeWM: Series[float]
    Cbr_Mot: Series[float] = pa.Field(alias='Cbr+Mot')
    CbrWM: Series[float]
    CenOpe: Series[float]
    Cun: Series[float]
    Ent: Series[float]
    ExtCbe: Series[float]
    FroOpe: Series[float]
    FroPo: Series[float]
    FusGy: Series[float]
    Hip: Series[float]
    InfFroAngGy: Series[float]
    InfFroGy: Series[float]
    InfFroOrbGy: Series[float]
    InfOccGy: Series[float]
    InfTemGy: Series[float]
    LatOrbGy: Series[float]
    LinGy: Series[float]
    MedFroCbr: Series[float]
    MedOrbGy: Series[float]
    MedPoCGy: Series[float]
    MedPrcGy: Series[float]
    MidCinGy: Series[float]
    MidFroGy: Series[float]
    MidOccGy: Series[float]
    MidTemGy: Series[float]
    OC: Series[float]
    OccFusGy: Series[float]
    OccPo: Series[float]
    PCu: Series[float]
    Pal: Series[float]
    ParHipGy: Series[float]
    ParOpe: Series[float]
    Pla: Series[float]
    PoCGy: Series[float]
    PosCinGy: Series[float]
    PosIns: Series[float]
    PosOrbGy: Series[float]
    PrcGy: Series[float]
    Put: Series[float]
    RecGy: Series[float]
    SCA: Series[float]
    SupFroGy: Series[float]
    SupMarGy: Series[float]
    SupMedFroGy: Series[float]
    SupOccGy: Series[float]
    SupParLo: Series[float]
    SupTemGy: Series[float]
    Tem: Series[float]
    TemPo: Series[float]
    TemTraGy: Series[float]
    ThaPro: Series[float]
    VenVen: Series[float]
    PROGRESSION_CATEGORY: Series[str]
    Dataset: Series[str]

    class Config:
        name = 'baseconfig'
        strict = True

class ParcellationSchema(BaseParcellationSchema):
    ThirdVen: Series[float] = pa.Field(alias='3thVen')
    FourthVen: Series[float] = pa.Field(alias='4thVen')
    CSF: Series[float]
    InfLatVen: Series[float]
    LatVen: Series[float]

class ParcellationFullSchema(pa.SchemaModel):
    RID: Optional[Series[str]] = pa.Field(alias='RID')
    Dataset: Optional[Series[str]] = pa.Field(alias='Dataset')
    Accumbens: Series[float] = pa.Field(alias='Accumbens')
    Amygdala: Series[float] = pa.Field(alias='Amygdala')
    AngularGyrus: Series[float] = pa.Field(alias='Angular Gyrus')
    AnteriorCingulateGyrus: Series[float] = pa.Field(
        alias='Anterior Cingulate Gyrus')
    AnteriorInsula: Series[float] = pa.Field(alias='Anterior Insula')
    AnteriorOrbitalGyrus: Series[float] = pa.Field(
        alias='Anterior Orbital Gyrus')
    BasalForebrain: Series[float] = pa.Field(alias='Basal Forebrain')
    Brainstem: Series[float] = pa.Field(alias='Brainstem')
    CalcarineCortex: Series[float] = pa.Field(alias='Calcarine Cortex')
    Caudate: Series[float] = pa.Field(alias='Caudate')
    CerebellarVermalLobulesIV: Series[float] = pa.Field(
        alias='Cerebellar Vermal Lobules I-V')
    CerebellarVermalLobulesVIVII: Series[float] = pa.Field(
        alias='Cerebellar Vermal Lobules VI-VII')
    CerebellarVermalLobulesVIIIX: Series[float] = pa.Field(
        alias='Cerebellar Vermal Lobules VIII-X')
    CerebellumWhiteMatter: Series[float] = pa.Field(
        alias='Cerebellum White Matter')
    SupplementaryMotorCortex: Series[float] = pa.Field(
        alias='Supplementary Motor Cortex')
    CerebralWhiteMatter: Series[float] = pa.Field(
        alias='Cerebral White Matter')
    CentralOperculum: Series[float] = pa.Field(alias='Central Operculum')
    Cuneus: Series[float] = pa.Field(alias='Cuneus')
    EntorhinalArea: Series[float] = pa.Field(alias='Entorhinal Area')
    CerebellumExterior: Series[float] = pa.Field(alias='Cerebellum Exterior')
    FrontalOperculum: Series[float] = pa.Field(alias='Frontal Operculum')
    FrontalPole: Series[float] = pa.Field(alias='Frontal Pole')
    FusiformGyrus: Series[float] = pa.Field(alias='Fusiform Gyrus')
    Hippocampus: Series[float] = pa.Field(alias='Hippocampus')
    TriangularPartoftheInferiorFrontalGyrus: Series[float] = pa.Field(
        alias='Triangular Part of the Inferior Frontal Gyrus')
    OpercularPartoftheInferiorFrontalGyrus: Series[float] = pa.Field(
        alias='Opercular Part of the Inferior Frontal Gyrus')
    OrbitalPartoftheInferiorFrontalGyrus: Series[float] = pa.Field(
        alias='Orbital Part of the Inferior Frontal Gyrus')
    InferiorOccipitalGyrus: Series[float] = pa.Field(
        alias='Inferior Occipital Gyrus')
    InferiorTemporalGyrus: Series[float] = pa.Field(
        alias='Inferior Temporal Gyrus')
    LateralOrbitalGyrus: Series[float] = pa.Field(
        alias='Lateral Orbital Gyrus')
    LingualGyrus: Series[float] = pa.Field(alias='Lingual Gyrus')
    MedialFrontalCortex: Series[float] = pa.Field(
        alias='Medial Frontal Cortex')
    MedialOrbitalGyrus: Series[float] = pa.Field(alias='Medial Orbital Gyrus')
    PostcentralGyrusMedialSegment: Series[float] = pa.Field(
        alias='Postcentral Gyrus Medial Segment')
    PrecentralGyrusMedialSegment: Series[float] = pa.Field(
        alias='Precentral Gyrus Medial Segment')
    MiddleCingulateGyrus: Series[float] = pa.Field(
        alias='Middle Cingulate Gyrus')
    MiddleFrontalGyrus: Series[float] = pa.Field(alias='Middle Frontal Gyrus')
    MiddleOccipitalGyrus: Series[float] = pa.Field(
        alias='Middle Occipital Gyrus')
    MiddleTemporalGyrus: Series[float] = pa.Field(
        alias='Middle Temporal Gyrus')
    OpticChiasm: Series[float] = pa.Field(alias='Optic Chiasm')
    OccipitalFusiformGyrus: Series[float] = pa.Field(
        alias='Occipital Fusiform Gyrus')
    OccipitalPole: Series[float] = pa.Field(alias='Occipital Pole')
    Precuneus: Series[float] = pa.Field(alias='Precuneus')
    Pallidum: Series[float] = pa.Field(alias='Pallidum')
    ParahippocampusGyrus: Series[float] = pa.Field(
        alias='Parahippocampus Gyrus')
    ParietalOperculum: Series[float] = pa.Field(alias='Parietal Operculum')
    PlanumPolare: Series[float] = pa.Field(alias='Planum Polare')
    PostcentralGyrus: Series[float] = pa.Field(alias='Postcentral Gyrus')
    PosteriorCingulateGyrus: Series[float] = pa.Field(
        alias='Posterior Cingulate Gyrus')
    PosteriorInsula: Series[float] = pa.Field(alias='Posterior Insula')
    PosteriorOrbitalGyrus: Series[float] = pa.Field(
        alias='Posterior Orbital Gyrus')
    PrecentralGyrus: Series[float] = pa.Field(alias='Precentral Gyrus')
    Putamen: Series[float] = pa.Field(alias='Putamen')
    GyrusRectus: Series[float] = pa.Field(alias='Gyrus Rectus')
    SubcallosalArea: Series[float] = pa.Field(alias='Subcallosal Area')
    SuperiorFrontalGyrus: Series[float] = pa.Field(
        alias='Superior Frontal Gyrus')
    SupramarginalGyrus: Series[float] = pa.Field(alias='Supramarginal Gyrus')
    SuperiorFrontalGyrusMedialSegment: Series[float] = pa.Field(
        alias='Superior Frontal Gyrus Medial Segment')
    SuperiorOccipitalGyrus: Series[float] = pa.Field(
        alias='Superior Occipital Gyrus')
    SuperiorParietalLobule: Series[float] = pa.Field(
        alias='Superior Parietal Lobule')
    SuperiorTemporalGyrus: Series[float] = pa.Field(
        alias='Superior Temporal Gyrus')
    PlanumTemporale: Series[float] = pa.Field(alias='Planum Temporale')
    TemporalPole: Series[float] = pa.Field(alias='Temporal Pole')
    TransverseTemporalGyrus: Series[float] = pa.Field(
        alias='Transverse Temporal Gyrus')
    ThalamusProper: Series[float] = pa.Field(alias='Thalamus Proper')
    VentralDC: Series[float] = pa.Field(alias='Ventral DC')

    class Config:
        name = 'baseconfig'
        strict = True

class ParcellationCentroidSchema(pa.SchemaModel):
    Accumbens: Series[float] = pa.Field(alias='Accumbens')
    Amygdala: Series[float] = pa.Field(alias='Amygdala')
    AngularGyrus: Series[float] = pa.Field(alias='Angular Gyrus')
    AnteriorCingulateGyrus: Series[float] = pa.Field(
        alias='Anterior Cingulate Gyrus')
    AnteriorInsula: Series[float] = pa.Field(alias='Anterior Insula')
    AnteriorOrbitalGyrus: Series[float] = pa.Field(
        alias='Anterior Orbital Gyrus')
    BasalForebrain: Series[float] = pa.Field(alias='Basal Forebrain')
    Brainstem: Series[float] = pa.Field(alias='Brainstem')
    CalcarineCortex: Series[float] = pa.Field(alias='Calcarine Cortex')
    Caudate: Series[float] = pa.Field(alias='Caudate')
    CerebellarVermalLobulesIV: Series[float] = pa.Field(
        alias='Cerebellar Vermal Lobules I-V')
    CerebellarVermalLobulesVIVII: Series[float] = pa.Field(
        alias='Cerebellar Vermal Lobules VI-VII')
    CerebellarVermalLobulesVIIIX: Series[float] = pa.Field(
        alias='Cerebellar Vermal Lobules VIII-X')
    CerebellumWhiteMatter: Series[float] = pa.Field(
        alias='Cerebellum White Matter')
    SupplementaryMotorCortex: Series[float] = pa.Field(
        alias='Supplementary Motor Cortex')
    CerebralWhiteMatter: Series[float] = pa.Field(
        alias='Cerebral White Matter')
    CentralOperculum: Series[float] = pa.Field(alias='Central Operculum')
    Cuneus: Series[float] = pa.Field(alias='Cuneus')
    EntorhinalArea: Series[float] = pa.Field(alias='Entorhinal Area')
    CerebellumExterior: Series[float] = pa.Field(alias='Cerebellum Exterior')
    FrontalOperculum: Series[float] = pa.Field(alias='Frontal Operculum')
    FrontalPole: Series[float] = pa.Field(alias='Frontal Pole')
    FusiformGyrus: Series[float] = pa.Field(alias='Fusiform Gyrus')
    Hippocampus: Series[float] = pa.Field(alias='Hippocampus')
    TriangularPartoftheInferiorFrontalGyrus: Series[float] = pa.Field(
        alias='Triangular Part of the Inferior Frontal Gyrus')
    OpercularPartoftheInferiorFrontalGyrus: Series[float] = pa.Field(
        alias='Opercular Part of the Inferior Frontal Gyrus')
    OrbitalPartoftheInferiorFrontalGyrus: Series[float] = pa.Field(
        alias='Orbital Part of the Inferior Frontal Gyrus')
    InferiorOccipitalGyrus: Series[float] = pa.Field(
        alias='Inferior Occipital Gyrus')
    InferiorTemporalGyrus: Series[float] = pa.Field(
        alias='Inferior Temporal Gyrus')
    LateralOrbitalGyrus: Series[float] = pa.Field(
        alias='Lateral Orbital Gyrus')
    LingualGyrus: Series[float] = pa.Field(alias='Lingual Gyrus')
    MedialFrontalCortex: Series[float] = pa.Field(
        alias='Medial Frontal Cortex')
    MedialOrbitalGyrus: Series[float] = pa.Field(alias='Medial Orbital Gyrus')
    PostcentralGyrusMedialSegment: Series[float] = pa.Field(
        alias='Postcentral Gyrus Medial Segment')
    PrecentralGyrusMedialSegment: Series[float] = pa.Field(
        alias='Precentral Gyrus Medial Segment')
    MiddleCingulateGyrus: Series[float] = pa.Field(
        alias='Middle Cingulate Gyrus')
    MiddleFrontalGyrus: Series[float] = pa.Field(alias='Middle Frontal Gyrus')
    MiddleOccipitalGyrus: Series[float] = pa.Field(
        alias='Middle Occipital Gyrus')
    MiddleTemporalGyrus: Series[float] = pa.Field(
        alias='Middle Temporal Gyrus')
    OpticChiasm: Series[float] = pa.Field(alias='Optic Chiasm')
    OccipitalFusiformGyrus: Series[float] = pa.Field(
        alias='Occipital Fusiform Gyrus')
    OccipitalPole: Series[float] = pa.Field(alias='Occipital Pole')
    Precuneus: Series[float] = pa.Field(alias='Precuneus')
    Pallidum: Series[float] = pa.Field(alias='Pallidum')
    ParahippocampusGyrus: Series[float] = pa.Field(
        alias='Parahippocampus Gyrus')
    ParietalOperculum: Series[float] = pa.Field(alias='Parietal Operculum')
    PlanumPolare: Series[float] = pa.Field(alias='Planum Polare')
    PostcentralGyrus: Series[float] = pa.Field(alias='Postcentral Gyrus')
    PosteriorCingulateGyrus: Series[float] = pa.Field(
        alias='Posterior Cingulate Gyrus')
    PosteriorInsula: Series[float] = pa.Field(alias='Posterior Insula')
    PosteriorOrbitalGyrus: Series[float] = pa.Field(
        alias='Posterior Orbital Gyrus')
    PrecentralGyrus: Series[float] = pa.Field(alias='Precentral Gyrus')
    Putamen: Series[float] = pa.Field(alias='Putamen')
    GyrusRectus: Series[float] = pa.Field(alias='Gyrus Rectus')
    SubcallosalArea: Series[float] = pa.Field(alias='Subcallosal Area')
    SuperiorFrontalGyrus: Series[float] = pa.Field(
        alias='Superior Frontal Gyrus')
    SupramarginalGyrus: Series[float] = pa.Field(alias='Supramarginal Gyrus')
    SuperiorFrontalGyrusMedialSegment: Series[float] = pa.Field(
        alias='Superior Frontal Gyrus Medial Segment')
    SuperiorOccipitalGyrus: Series[float] = pa.Field(
        alias='Superior Occipital Gyrus')
    SuperiorParietalLobule: Series[float] = pa.Field(
        alias='Superior Parietal Lobule')
    SuperiorTemporalGyrus: Series[float] = pa.Field(
        alias='Superior Temporal Gyrus')
    PlanumTemporale: Series[float] = pa.Field(alias='Planum Temporale')
    TemporalPole: Series[float] = pa.Field(alias='Temporal Pole')
    TransverseTemporalGyrus: Series[float] = pa.Field(
        alias='Transverse Temporal Gyrus')
    ThalamusProper: Series[float] = pa.Field(alias='Thalamus Proper')
    VentralDC: Series[float] = pa.Field(alias='Ventral DC')
    # Quartile: Index[int] = pa.Field(alias='Biomarker Quartile')
    # abeta: Index[float]
    # tau: Index[float]
    # ptau: Index[float]

    class Config:
        name = 'baseconfig'
        strict = True

class ParcellationClusteredSchema(ParcellationFullSchema):
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')

class ClusteredMriSchema(pa.SchemaModel):
    RID: Series[str]
    MRI_IID: Series[str]
    MRI_fname: Series[str]
    PROGRESSES: Series[float]
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')

    class Config:
        name = 'baseconfig'
        strict = True

class MlpSchema(pa.SchemaModel):
    RID: Series[str]
    Predictions: Series[float]
    Experiment: Series[float]
    Bins: Series[int]
    Dataset: Series[str]

    class Config:
        name = 'baseconfig'
        strict = True

class MlpPivotSchema(pa.SchemaModel):
    RID: Index[str]
    Dataset: Index[str]
    Zero: Series[float] = pa.Field(alias='0')
    Two: Series[float] = pa.Field(alias='24')
    Four: Series[float] = pa.Field(alias='48')
    Nine: Series[float] = pa.Field(alias='108')

    class Config:
        name = 'baseconfig'
        strict = True

class MlpSurvivalSchema(pa.SchemaModel):
    RID: Series[str]
    Dataset: Series[str]
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')
    TIMES: Series[float]
    PROGRESSES: Series[float]

class MlpPivotClusterSchema(MlpPivotSchema):
    RID: Series[str]
    Dataset: Series[str]
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')
    TIMES: Series[float]
    PROGRESSES: Series[float]

class MlpPivotSurvSchema(MlpSurvivalSchema):
    Zero: Series[float] = pa.Field(alias='0')
    Two: Series[float] = pa.Field(alias='24')
    Four: Series[float] = pa.Field(alias='48')
    Nine: Series[float] = pa.Field(alias='108')
    TIMES: Series[float]

class MlpPivotSurvSchemaRaw(MlpPivotSurvSchema):
    Exp: Series[float]

class MlpPathologySchema(MlpPivotSchema):
    RID: Series[str]
    Dataset: Series[str]
    AGE: Series[float]
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')
    CorticalAtrophy: Series[str] = pa.Field(nullable=True)
    LobarAtrophy: Series[str] = pa.Field(nullable=True)
    HippocampalAtrophy: Series[str] = pa.Field(nullable=True)
    BRAAK: Series[str] = pa.Field(nullable=True)
    CERAD: Series[str] = pa.Field(nullable=True)
    NIA_ADNC: Series[str] = pa.Field(nullable=True)
    CERAD_SemiQuant: Series[str] = pa.Field(nullable=True)
    HippocampalSclerosis: Series[str] = pa.Field(nullable=True)
    MTLSclerosisWithHippocampus: Series[str] = pa.Field(nullable=True)
    TDP_Amygdala: Series[str] = pa.Field(nullable=True)
    TDP_Hippocampus: Series[str] = pa.Field(nullable=True)
    TDP_EntorhinalInfTemporal: Series[str] = pa.Field(nullable=True)
    TDP_Neocortex: Series[str] = pa.Field(nullable=True)
    AgeAtDeath: Series[float] = pa.Field(nullable=True)
    EXAMDATE: Series[str]
    DATE_OF_DEATH: Series[str] = pa.Field(nullable=True)
    PROGRESSES: Series[int]
    TIME_TO_PROGRESSION: Series[float] = pa.Field(nullable=True)
    FINAL_OBS_DATE: Series[str]
    AGE_AT_FINAL_DX: Series[float]
    TIME_TO_FINAL_DX: Series[float]
    TIMES: Series[float]

class SwarmPlotSchema(pa.SchemaModel):
    RID: Series[str]
    Dataset: Series[str]
    Region: Series[str]
    ShapValue: Series[float] = pa.Field(alias='Shap Value')
    GrayMatterVolRaw: Series[float] = pa.Field(alias='Gray Matter Vol Raw')
    ClusterIdx: Series[str] = pa.Field(alias='Cluster Idx')
    GrayMatterVol: Series[float] = pa.Field(alias='Gray Matter Vol')
    GrayMatterVolBin: Series[int] = pa.Field(alias='Gray Matter Vol Bin')