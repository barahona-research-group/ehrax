from typing import Final

import equinox as eqx
import sqlalchemy

from ..coding_scheme import CodingSchemesManager
from ..dataset import AbstractDatasetPipeline, Dataset, DatasetColumns, DatasetConfig, DatasetSchemeConfig
from ..transformations import (
    CastTimestamps,
    FilterAdmissionsWithNoObservables, FilterClampTimestampsToAdmissionInterval,
    FilterInvalidInputRatesSubjects,
    FilterShortAdmissions,
    FilterSubjectsNegativeAdmissionLengths,
    FilterSubjectsWithInvalidInputInterval,
    FilterUnsupportedCodes,
    ICUInputRateUnitConversion,
    MergeOverlappingAdmissions,
    SelectSubjectsWithObservation,
    SetAdmissionRelativeTimes,
    SetIndex,
)
from ..tvx_ehr import (
    AbstractTVxPipeline,
    DatasetNumericalProcessorsConfig,
    DemographicVectorConfig,
    LeadingObservableExtractorConfig,
    TVxEHRConfig,
    TVxEHRSampleConfig,
    TVxEHRSchemeConfig,
    TVxEHRSplitsConfig,
)
from ..tvx_transformations import (
    InputScaler,
    InterventionSegmentation,
    LeadingObservableExtraction,
    ObsAdaptiveScaler,
    ObsIQROutlierRemover,
    ObsTimeBinning,
    RandomSplits,
    SampleSubjects,
    TVxConcepts,
)
from .mimic import load_mimic, MIMICDatasetAuxiliaryResources, MIMICDatasetSchemeSuffixes, ScopedSchemeNames
from .mimic_sql import SQLMIMICTablesResources


OBSERVABLE_AKI_TARGET_CODE: Final[str] = "renal_aki.aki_binary"


def default_suffixes() -> MIMICDatasetSchemeSuffixes:
    return MIMICDatasetSchemeSuffixes(
        ethnicity="ethnicity",
        gender="gender",
        obs="obs",
        dx_discharge="dx_discharge",
        hosp_procedures="hosp_procedures",
        icu_procedures="icu_procedures",
        icu_inputs="icu_inputs",
    )


def default_auxiliary_resources() -> MIMICDatasetAuxiliaryResources:
    return MIMICDatasetAuxiliaryResources.make_resources(
        name_prefix="mimiciv_aki",
        resources_root="aki_study",
        suffixes=default_suffixes(),
        icu_inputs_uom_normalization=("uom_normalization", "icu_inputs.csv"),
        icu_inputs_aggregation_column="aggregation",
    )


def default_scoped_names() -> ScopedSchemeNames:
    return default_auxiliary_resources().scoped_names


def default_dataset_schemes_config(scoped_names: ScopedSchemeNames) -> DatasetSchemeConfig:
    return DatasetSchemeConfig(
        ethnicity=scoped_names.ethnicity,
        gender=scoped_names.gender,
        dx_discharge=scoped_names.dx_discharge,
        obs=scoped_names.obs,
        icu_procedures=scoped_names.icu_procedures,
        hosp_procedures=scoped_names.hosp_procedures,
        icu_inputs=scoped_names.icu_inputs,
    )


def default_dataset_config(scoped_names: ScopedSchemeNames) -> DatasetConfig:
    return DatasetConfig(
        scheme=default_dataset_schemes_config(scoped_names),
        columns=DatasetColumns(),
        select_subjects_with_observation=OBSERVABLE_AKI_TARGET_CODE,
        admission_minimum_los=12.0 / 24.0,  # 12 hours.
    )


def default_dataset_pipeline() -> AbstractDatasetPipeline:
    pipeline = [
        SetIndex(),
        SelectSubjectsWithObservation(),
        CastTimestamps(),
        MergeOverlappingAdmissions(),
        FilterSubjectsNegativeAdmissionLengths(),
        FilterAdmissionsWithNoObservables(),
        FilterShortAdmissions(),
        FilterClampTimestampsToAdmissionInterval(),
        FilterUnsupportedCodes(),
        ICUInputRateUnitConversion(),
        FilterInvalidInputRatesSubjects(),
        FilterSubjectsWithInvalidInputInterval(),
        SetAdmissionRelativeTimes(),
    ]
    return AbstractDatasetPipeline(transformations=pipeline)


def default_tvx_schemes_config(config: DatasetSchemeConfig, scoped_names: ScopedSchemeNames) -> TVxEHRSchemeConfig:
    names = scoped_names.mapped
    return TVxEHRSchemeConfig(
        gender=config.gender,
        ethnicity=names.ethnicity,
        dx_discharge="dx_ccs",
        obs=config.obs,
        icu_inputs=names.icu_inputs,
        icu_procedures=names.icu_procedures,
        hosp_procedures=names.hosp_procedures,
        outcome="dx_flat_ccs_v1",
    )


def default_tvx_ehr_config(scoped_names: ScopedSchemeNames = default_scoped_names()) -> TVxEHRConfig:
    scheme = default_tvx_schemes_config(default_dataset_schemes_config(scoped_names), scoped_names)
    return TVxEHRConfig(
        scheme=scheme,
        demographic=DemographicVectorConfig(age=True, gender=True, ethnicity=True),
        leading_observable=LeadingObservableExtractorConfig(
            observable_code=OBSERVABLE_AKI_TARGET_CODE,
            scheme=scheme.obs,
            leading_hours=[6.0, 12.0, 24.0, 48.0, 72.0],  # hours
            entry_neglect_window=6.0,  # hours
            minimum_acquisitions=2,  # number of observable acquisitions.
            recovery_window=12.0,
        ),  # hours
        sample=TVxEHRSampleConfig(n_subjects=6000, seed=0, offset=0),  # no subsetting now
        splits=TVxEHRSplitsConfig(
            split_quantiles=[0.6, 0.7, 0.8], seed=0, discount_first_admission=False, balance="admissions"
        ),
        numerical_processors=DatasetNumericalProcessorsConfig(),
        interventions=True,
        observables=True,
        time_binning=None,
        interventions_segmentation=True,
    )


def default_tb_tvx_ehr_config(scoped_names: ScopedSchemeNames = default_scoped_names()) -> TVxEHRConfig:
    config = default_tvx_ehr_config(scoped_names)
    return eqx.tree_at(lambda x: x.time_binning, config, 6.0, is_leaf=lambda x: x is None)


def default_tvx_ehr_pipeline() -> AbstractTVxPipeline:
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


def mimiciv_from_env_sql(
    dataset_tables_resources: SQLMIMICTablesResources = SQLMIMICTablesResources(),
    aux_resources: MIMICDatasetAuxiliaryResources = default_auxiliary_resources(),
) -> tuple[Dataset, CodingSchemesManager]:
    engine = sqlalchemy.create_engine(dataset_tables_resources.url())
    return load_mimic(
        config=default_dataset_config(aux_resources.scoped_names),
        tables=dataset_tables_resources,
        aux=aux_resources,
        data_connection=engine,
    )
