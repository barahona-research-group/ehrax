from typing import Literal

from ..coding_scheme import CodingSchemesManager
from ..dataset import AbstractDatasetPipeline, Dataset, DatasetColumns, DatasetConfig, DatasetSchemeConfig
from ..transformations import (
    CastTimestamps,
    FilterAdmissionsWithNoDiagnoses,
    FilterSubjectsNegativeAdmissionLengths,
    FilterSubjectsWithLongAdmission,
    FilterSubjectsWithSingleOrNoAdmission,
    FilterUnsupportedCodes,
    MergeOverlappingAdmissions,
    SetIndex,
    SqueezeToStandardColumns,
)
from ..tvx_concepts import DemographicVectorConfig
from ..tvx_ehr import AbstractTVxPipeline, TVxEHRConfig, TVxEHRSchemeConfig, TVxEHRSplitsConfig
from ..tvx_transformations import RandomSplits, SampleSubjects, TVxConcepts
from .mimic import load_mimic, MIMICDatasetAuxiliaryResources, MIMICDatasetSchemeSuffixes, ScopedSchemeNames
from .mimic_in_memory import (
    InMemoryMIMICTableFiles,
    MIMICIII_ADMISSIONS_COLMAP,
    MIMICIII_D_ICD_DIAGNOSES_COLMAP,
    MIMICIII_DIAGNOSES_ICD_COLMAP,
    MIMICIII_STATIC_COLMAP,
    MIMICIII_TABLES_RESOURCES,
    MIMICIV_ADMISSIONS_COLMAP,
    MIMICIV_D_ICD_DIAGNOSES_COLMAP,
    MIMICIV_DIAGNOSES_ICD_COLMAP,
    MIMICIV_STATIC_COLMAP,
    MIMICIV_TABLES_RESOURCES,
    MIMICTablesResources,
)


DatasetName = Literal["mimiciii", "mimiciv"]


def default_suffixes() -> MIMICDatasetSchemeSuffixes:
    return MIMICDatasetSchemeSuffixes(ethnicity="ethnicity", gender="gender", dx_discharge="dx_discharge")


def default_auxiliary_resources(dataset_name: DatasetName) -> MIMICTablesResources:
    return MIMICDatasetAuxiliaryResources.make_resources(
        name_prefix=f"{dataset_name}.dx_summary", resources_root="dx_summary_study", suffixes=default_suffixes()
    )


def default_dataset_schemes_config(scoped_names: ScopedSchemeNames) -> DatasetSchemeConfig:
    return DatasetSchemeConfig(
        ethnicity=scoped_names.ethnicity, gender=scoped_names.gender, dx_discharge=scoped_names.dx_discharge
    )


def default_dataset_config(scoped_names: ScopedSchemeNames) -> DatasetConfig:
    return DatasetConfig(
        scheme=default_dataset_schemes_config(scoped_names),
        columns=DatasetColumns(),
        select_subjects_with_observation=None,
        select_subjects_with_short_admissions=14.0,
    )  # two weeks.


def default_tvx_schemes_config(config: DatasetSchemeConfig) -> TVxEHRSchemeConfig:
    return TVxEHRSchemeConfig(gender=config.gender, ethnicity=None, dx_discharge="dx_ccs", outcome="dx_flat_ccs_v1")


def default_dataset_pipeline() -> AbstractDatasetPipeline:
    pipeline = [
        SqueezeToStandardColumns(),
        SetIndex(),
        CastTimestamps(),
        FilterSubjectsNegativeAdmissionLengths(),
        MergeOverlappingAdmissions(),
        FilterUnsupportedCodes(),
        FilterAdmissionsWithNoDiagnoses(),
        FilterSubjectsWithSingleOrNoAdmission(),
        FilterSubjectsWithLongAdmission(),
    ]
    return AbstractDatasetPipeline(transformations=pipeline)


def _mimic_from_memory(
    dataset_tables_resources: MIMICTablesResources,
    aux: MIMICDatasetAuxiliaryResources,
    schemes_config: DatasetSchemeConfig | None,
    in_memory_tables: InMemoryMIMICTableFiles,
) -> tuple[Dataset, CodingSchemesManager]:
    if schemes_config is None:
        schemes_config = default_dataset_schemes_config(aux.scoped_names)
    return load_mimic(
        config=DatasetConfig(scheme=schemes_config),
        tables=dataset_tables_resources,
        aux=aux,
        data_connection=in_memory_tables,
    )


def mimiciii_from_paths(
    patients: str,
    admissions: str,
    diagnoses_icd: str,
    d_icd_diagnoses: str,
    aux_resources: MIMICDatasetAuxiliaryResources = default_auxiliary_resources("mimiciii"),
    schemes_config: DatasetSchemeConfig | None = None,
    dataset_tables_resources: MIMICTablesResources = MIMICIII_TABLES_RESOURCES,
) -> tuple[Dataset, CodingSchemesManager]:
    # Optional argument. For faster loading time and lower memory footprint, only load the relevant raw columns.
    USECOLS = {
        "patients": tuple(MIMICIII_STATIC_COLMAP.keys()),
        "admissions": tuple(MIMICIII_ADMISSIONS_COLMAP.keys()),
        "diagnoses_icd": tuple(MIMICIII_DIAGNOSES_ICD_COLMAP.keys()),
        "d_icd_diagnoses": tuple(MIMICIII_D_ICD_DIAGNOSES_COLMAP.keys()),
    }
    in_memory_tables = InMemoryMIMICTableFiles.from_path(
        patients=patients,
        admissions=admissions,
        diagnoses_icd=diagnoses_icd,
        d_icd_diagnoses=d_icd_diagnoses,
        usecols=USECOLS,
    )
    return _mimic_from_memory(
        schemes_config=schemes_config,
        dataset_tables_resources=dataset_tables_resources,
        in_memory_tables=in_memory_tables,
        aux=aux_resources,
    )


def mimiciv_from_paths(
    patients: str,
    admissions: str,
    diagnoses_icd: str,
    d_icd_diagnoses: str,
    aux_resources: MIMICDatasetAuxiliaryResources = default_auxiliary_resources("mimiciv"),
    schemes_config: DatasetSchemeConfig | None = None,
    dataset_tables_resources: MIMICTablesResources = MIMICIV_TABLES_RESOURCES,
) -> tuple[Dataset, CodingSchemesManager]:
    # Optional argument. For faster loading time and lower memory footprint, only load the relevant raw columns.
    USECOLS = {
        "patients": tuple(MIMICIV_STATIC_COLMAP.keys()),
        "admissions": tuple(MIMICIV_ADMISSIONS_COLMAP.keys()),
        "diagnoses_icd": tuple(MIMICIV_DIAGNOSES_ICD_COLMAP.keys()),
        "d_icd_diagnoses": tuple(MIMICIV_D_ICD_DIAGNOSES_COLMAP.keys()),
    }
    in_memory_tables = InMemoryMIMICTableFiles.from_path(
        patients=patients,
        admissions=admissions,
        diagnoses_icd=diagnoses_icd,
        d_icd_diagnoses=d_icd_diagnoses,
        usecols=USECOLS,
    )
    return _mimic_from_memory(
        schemes_config=schemes_config,
        dataset_tables_resources=dataset_tables_resources,
        in_memory_tables=in_memory_tables,
        aux=aux_resources,
    )


def match_gender_schemes(
    scheme_config_a: DatasetSchemeConfig, scheme_config_b: DatasetSchemeConfig, schemes: CodingSchemesManager
) -> CodingSchemesManager:
    return schemes.add_match_map(scheme_config_a.gender, scheme_config_b.gender)


def default_tvx_ehr_config(config: DatasetSchemeConfig) -> TVxEHRConfig:
    scheme = default_tvx_schemes_config(config)
    return TVxEHRConfig(
        scheme=scheme,
        demographic=DemographicVectorConfig(age=True, gender=True, ethnicity=False),
        sample=None,  # no subsetting now
        splits=TVxEHRSplitsConfig(
            split_quantiles=[0.7, 0.85],
            seed=0,  # 0.7:0.15:0.15
            discount_first_admission=False,
            balance="admissions",
        ),
    )


def default_tvx_ehr_pipeline() -> AbstractTVxPipeline:
    pipeline = [SampleSubjects(), RandomSplits(), TVxConcepts()]
    return AbstractTVxPipeline(transformations=pipeline)
