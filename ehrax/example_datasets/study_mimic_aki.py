from typing import Final, Optional

import sqlalchemy

from .mimic import MIMICDatasetAuxiliaryResources, MIMICDatasetSchemeSuffixes, ScopedSchemeNames, load_mimic
from .mimic_sql import SQLMIMICTablesResources
from ..coding_scheme import CodingSchemesManager
from ..dataset import AbstractDatasetPipeline, Dataset, DatasetColumns, DatasetConfig, DatasetSchemeConfig
from ..transformations import CastTimestamps, FilterClampTimestampsToAdmissionInterval, FilterInvalidInputRatesSubjects, \
    FilterShortAdmissions, FilterSubjectsNegativeAdmissionLengths, FilterUnsupportedCodes, ICUInputRateUnitConversion, \
    MergeOverlappingAdmissions, SelectSubjectsWithObservation, SetAdmissionRelativeTimes, SetIndex
from ..tvx_ehr import AbstractTVxPipeline, DatasetNumericalProcessorsConfig, DemographicVectorConfig, \
    LeadingObservableExtractorConfig, TVxEHRConfig, TVxEHRSampleConfig, TVxEHRSchemeConfig, TVxEHRSplitsConfig
from ..tvx_transformations import InputScaler, InterventionSegmentation, \
    LeadingObservableExtraction, ObsAdaptiveScaler, ObsIQROutlierRemover, ObsTimeBinning, RandomSplits, SampleSubjects, \
    TVxConcepts

OBSERVABLE_AKI_TARGET_CODE: Final[str] = 'renal_aki.aki_binary'


def default_suffixes() -> MIMICDatasetSchemeSuffixes:
    return MIMICDatasetSchemeSuffixes(ethnicity='ethnicity',
                                      gender='gender',
                                      dx_discharge='dx_discharge',
                                      hosp_procedures='hosp_procedures',
                                      icu_procedures='icu_procedures',
                                      icu_inputs='icu_inputs')


def default_auxiliary_resources() -> MIMICDatasetAuxiliaryResources:
    return MIMICDatasetAuxiliaryResources.make_resources(
        name_prefix='mimiciv_aki', resources_root='aki_study',
        suffixes=default_suffixes(),
        icu_inputs_uom_normalization=("uom_normalization", "icu_inputs.csv"),
        icu_inputs_aggregation_column="aggregation")


def dataset_schemes_config(scoped_names: ScopedSchemeNames) -> DatasetSchemeConfig:
    return DatasetSchemeConfig(ethnicity=scoped_names.ethnicity,
                               gender=scoped_names.gender,
                               dx_discharge=scoped_names.dx_discharge,
                               obs=scoped_names.obs,
                               icu_procedures=scoped_names.icu_procedures,
                               hosp_procedures=scoped_names.hosp_procedures,
                               icu_inputs=scoped_names.icu_inputs)


def dataset_config(scoped_names: ScopedSchemeNames) -> DatasetConfig:
    return DatasetConfig(
        scheme=dataset_schemes_config(scoped_names),
        columns=DatasetColumns(),
        select_subjects_with_observation=OBSERVABLE_AKI_TARGET_CODE,
        admission_minimum_los=12.0 / 24.0  # 12 hours.
    )


def dataset_pipeline() -> AbstractDatasetPipeline:
    pipeline = [
        SetIndex(),
        SelectSubjectsWithObservation(),
        CastTimestamps(),
        MergeOverlappingAdmissions(),
        FilterSubjectsNegativeAdmissionLengths(),
        FilterShortAdmissions(),
        FilterClampTimestampsToAdmissionInterval(),
        FilterUnsupportedCodes(),
        ICUInputRateUnitConversion(),
        FilterInvalidInputRatesSubjects(),
        SetAdmissionRelativeTimes()
    ]
    return AbstractDatasetPipeline(transformations=pipeline)


def tvx_schemes_config(config: DatasetSchemeConfig, scoped_names: ScopedSchemeNames) -> TVxEHRSchemeConfig:
    names = scoped_names.mapped
    return TVxEHRSchemeConfig(
        gender=config.gender,
        ethnicity=names.ethnicity,
        dx_discharge='icd9cm',
        obs=config.obs,
        icu_inputs=names.icu_inputs,
        icu_procedures=names.icu_procedures,
        hosp_procedures=names.hosp_procedures,
        outcome='icd9cm_v1')


def tvx_ehr_config(scoped_names: ScopedSchemeNames) -> TVxEHRConfig:
    scheme = tvx_schemes_config(dataset_schemes_config(scoped_names), scoped_names)
    return TVxEHRConfig(
        scheme=scheme,
        demographic=DemographicVectorConfig(age=True,
                                            gender=True,
                                            ethnicity=True),
        leading_observable=LeadingObservableExtractorConfig(
            observable_code=OBSERVABLE_AKI_TARGET_CODE,
            scheme=scheme.obs,
            leading_hours=[6., 12., 24., 48., 72.],  # hours
            entry_neglect_window=6.,  # hours
            minimum_acquisitions=2,  # number of observable acquisitions.
            recovery_window=12.),  # hours
        sample=TVxEHRSampleConfig(n_subjects=6000, seed=0, offset=0),  # no subsetting now
        splits=TVxEHRSplitsConfig(split_quantiles=[0.6, 0.7, 0.8], seed=0,
                                  discount_first_admission=False,
                                  balance='admissions'),
        numerical_processors=DatasetNumericalProcessorsConfig(),
        interventions=True,
        observables=True,
        time_binning=None,
        interventions_segmentation=True,
    )


def tvx_ehr_pipeline() -> AbstractTVxPipeline:
    pipeline = [
        SampleSubjects(),
        RandomSplits(),
        ObsIQROutlierRemover(),
        ObsAdaptiveScaler(),
        InputScaler(),
        TVxConcepts(),
        ObsTimeBinning(),
        LeadingObservableExtraction(),
        InterventionSegmentation(),
    ]
    return AbstractTVxPipeline(transformations=pipeline)


def mimiciv_from_env_sql(dataset_tables_resources: SQLMIMICTablesResources = SQLMIMICTablesResources(),
                         schemes_config: Optional[DatasetSchemeConfig] = None,
                         aux_resources: MIMICDatasetAuxiliaryResources = default_auxiliary_resources()) -> tuple[
    Dataset, CodingSchemesManager]:
    if schemes_config is None:
        schemes_config = dataset_schemes_config(aux_resources.scoped_names)
    engine = sqlalchemy.create_engine(dataset_tables_resources.url())
    return load_mimic(config=DatasetConfig(scheme=schemes_config),
                      tables=dataset_tables_resources,
                      aux=aux_resources,
                      data_connection=engine)
