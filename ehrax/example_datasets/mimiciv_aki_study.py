from typing import Final

from .mimic_resources import ScopedSchemeNames
from ..dataset import AbstractDatasetPipeline, DatasetSchemeConfig, DatasetConfig, DatasetColumns
from ..transformations import SetIndex, CastTimestamps, \
    SelectSubjectsWithObservation, ProcessOverlappingAdmissions, FilterSubjectsNegativeAdmissionLengths, \
    FilterClampTimestampsToAdmissionInterval, FilterUnsupportedCodes, ICUInputRateUnitConversion, \
    FilterInvalidInputRatesSubjects, SetAdmissionRelativeTimes
from ..tvx_ehr import TVxEHRSchemeConfig, TVxEHRSplitsConfig, DatasetNumericalProcessorsConfig, AbstractTVxPipeline, \
    TVxEHRConfig, DemographicVectorConfig, LeadingObservableExtractorConfig
from ..tvx_transformations import SampleSubjects, ObsIQROutlierRemover, RandomSplits, ObsAdaptiveScaler, \
    InputScaler, ObsTimeBinning, TVxConcepts, InterventionSegmentation, ExcludeShortAdmissions, \
    LeadingObservableExtraction

OBSERVABLE_AKI_TARGET_CODE: Final[str] = 'renal_aki.aki_binary'
AKI_STUDY_PREFIX: str = 'mimiciv.aki_study'
AKI_STUDY_RESOURCES_ROOT: str = 'mimiciv/aki_study'


def dataset_schemes_config():
    names = ScopedSchemeNames(name_separator='.', name_prefix=AKI_STUDY_PREFIX)
    return DatasetSchemeConfig(ethnicity=names.ethnicity,
                               gender=names.gender,
                               dx_discharge=names.dx_discharge,
                               obs=names.obs,
                               icu_procedures=names.icu_procedures,
                               hosp_procedures=names.hosp_procedures,
                               icu_inputs=names.icu_inputs)


def dataset_config():
    return DatasetConfig(
        scheme=dataset_schemes_config(),
        columns=DatasetColumns(),
        overlapping_admissions="merge",
        select_subjects_with_observation=OBSERVABLE_AKI_TARGET_CODE)


def dataset_pipeline() -> AbstractDatasetPipeline:
    pipeline = [
        SetIndex(),
        SelectSubjectsWithObservation(),
        CastTimestamps(),
        ProcessOverlappingAdmissions(),
        FilterSubjectsNegativeAdmissionLengths(),
        FilterClampTimestampsToAdmissionInterval(),
        FilterUnsupportedCodes(),
        ICUInputRateUnitConversion(),
        FilterInvalidInputRatesSubjects(),
        SetAdmissionRelativeTimes()
    ]
    return AbstractDatasetPipeline(transformations=pipeline)


def tvx_schemes_config(config: DatasetSchemeConfig) -> TVxEHRSchemeConfig:
    names = ScopedSchemeNames(name_separator='.', name_prefix='aki_study').target
    return TVxEHRSchemeConfig(
        gender=config.gender,
        ethnicity=names.ethnicity,
        dx_discharge='dx_icd9',
        obs=config.obs,
        icu_inputs=names.icu_inputs,
        icu_procedures=names.icu_procedures,
        hosp_procedures=names.hosp_procedures,
        outcome='dx_icd9_v1')


def tvx_ehr_config() -> TVxEHRConfig:
    scheme = tvx_schemes_config()
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
            minimum_acquisitions=2,  # number of observables acquisitions.
            recovery_window=12.),  # hours
        sample=None,  # no subsetting now
        splits=TVxEHRSplitsConfig(split_quantiles=[0.6, 0.7, 0.8], seed=0,
                                  discount_first_admission=False,
                                  balance='admissions'),
        numerical_processors=DatasetNumericalProcessorsConfig(),
        interventions=True,
        observables=True,
        time_binning=None,
        interventions_segmentation=True

    )


def tvx_ehr_pipeline() -> AbstractTVxPipeline:
    pipeline = [
        SampleSubjects(),
        RandomSplits(),
        ObsIQROutlierRemover(),
        ObsAdaptiveScaler(),
        InputScaler(),
        TVxConcepts(),
        ExcludeShortAdmissions(),
        ObsTimeBinning(),
        LeadingObservableExtraction(),
        InterventionSegmentation(),
    ]
    return AbstractTVxPipeline(transformations=pipeline)
