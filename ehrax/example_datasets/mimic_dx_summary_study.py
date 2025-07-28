from typing import Literal

from .mimic_in_memory import MIMICTablesResources, InMemoryMIMICTableFiles
from .mimic_resources import ScopedSchemeNames, MIMICDataset, MIMICDatasetAuxiliaryResources
from .mimiciii_in_memory import MIMICIII_TABLES_RESOURCES
from .mimiciv_in_memory import MIMICIV_TABLES_RESOURCES
from ..coding_scheme import CodingSchemesManager
from ..dataset import DatasetSchemeConfig, DatasetConfig, DatasetColumns, AbstractDatasetPipeline
from ..transformations import SetIndex, CastTimestamps, ProcessOverlappingAdmissions, \
    FilterSubjectsNegativeAdmissionLengths, \
    FilterUnsupportedCodes
from ..tvx_concepts import DemographicVectorConfig
from ..tvx_ehr import TVxEHRSplitsConfig, AbstractTVxPipeline, TVxEHRSchemeConfig, TVxEHRConfig
from ..tvx_transformations import RandomSplits, TVxConcepts, ExcludeShortAdmissions, SampleSubjects

STUDY_PREFIX: str = 'mimiciv.aki_study'
STUDY_RESOURCES_ROOT: str = 'mimiciv/aki_study'

DatasetName = Literal['mimiciii', 'mimiciv']


def scoped_names(dataset_name: DatasetName) -> ScopedSchemeNames:
    return ScopedSchemeNames(name_separator='.', name_prefix=f'{dataset_name}.dx_summary')


def dataset_schemes_config(dataset_name: DatasetName) -> DatasetSchemeConfig:
    names = scoped_names(dataset_name)
    return DatasetSchemeConfig(ethnicity=names.ethnicity,
                               gender=names.gender,
                               dx_discharge=names.dx_discharge,
                               obs=names.obs,
                               icu_procedures=names.icu_procedures,
                               hosp_procedures=names.hosp_procedures,
                               icu_inputs=names.icu_inputs)


def dataset_config(dataset_name: DatasetName) -> DatasetConfig:
    return DatasetConfig(
        scheme=dataset_schemes_config(dataset_name),
        columns=DatasetColumns(),
        overlapping_admissions="merge",
        select_subjects_with_observation=None)


def dataset_pipeline() -> AbstractDatasetPipeline:
    pipeline = [
        SetIndex(),
        CastTimestamps(),
        ProcessOverlappingAdmissions(),
        FilterSubjectsNegativeAdmissionLengths(),
        FilterUnsupportedCodes(),
    ]
    return AbstractDatasetPipeline(transformations=pipeline)


def tvx_schemes_config(config: DatasetSchemeConfig, dataset_name: DatasetName) -> TVxEHRSchemeConfig:
    names = scoped_names(dataset_name).target
    return TVxEHRSchemeConfig(
        gender=config.gender,
        ethnicity=names.ethnicity,
        dx_discharge='dx_icd9',
        outcome='dx_icd9_v1')


def tvx_ehr_config(dataset_name: DatasetName) -> TVxEHRConfig:
    scheme = tvx_schemes_config(dataset_schemes_config(dataset_name), dataset_name)
    return TVxEHRConfig(
        scheme=scheme,
        demographic=DemographicVectorConfig(age=True,
                                            gender=True,
                                            ethnicity=True),
        sample=None,  # no subsetting now
        splits=TVxEHRSplitsConfig(split_quantiles=[0.6, 0.7, 0.8], seed=0,
                                  discount_first_admission=False,
                                  balance='admissions')
    )


def tvx_ehr_pipeline() -> AbstractTVxPipeline:
    pipeline = [
        SampleSubjects(),
        RandomSplits(),
        TVxConcepts(),
        ExcludeShortAdmissions()
    ]
    return AbstractTVxPipeline(transformations=pipeline)


def mimic_from_memory(dataset_scheme_config: DatasetSchemeConfig,
                      dataset_tables_resources: MIMICTablesResources,
                      in_memory_tables: InMemoryMIMICTableFiles) -> tuple[MIMICDataset, CodingSchemesManager]:
    return MIMICDataset.compile(config=DatasetConfig(scheme=dataset_scheme_config),
                                tables=dataset_tables_resources,
                                aux=MIMICDatasetAuxiliaryResources.make_resources(),
                                data_connection=in_memory_tables)


def mimiciii_from_paths(patients: str, admissions: str, diagnoses_icd: str, d_icd_diagnoses: str) -> tuple[
    MIMICDataset, CodingSchemesManager]:
    in_memory_tables = InMemoryMIMICTableFiles.from_path(patients=patients, admissions=admissions,
                                                         diagnoses_icd=diagnoses_icd, d_icd_diagnoses=d_icd_diagnoses)
    return mimic_from_memory(dataset_scheme_config=dataset_schemes_config('mimiciii'),
                             dataset_tables_resources=MIMICIII_TABLES_RESOURCES,
                             in_memory_tables=in_memory_tables)


def mimiciv_from_paths(patients: str, admissions: str, diagnoses_icd: str, d_icd_diagnoses: str) -> tuple[
    MIMICDataset, CodingSchemesManager]:
    in_memory_tables = InMemoryMIMICTableFiles.from_path(patients=patients, admissions=admissions,
                                                         diagnoses_icd=diagnoses_icd, d_icd_diagnoses=d_icd_diagnoses)
    return mimic_from_memory(dataset_scheme_config=dataset_schemes_config('mimiciv'),
                             dataset_tables_resources=MIMICIV_TABLES_RESOURCES,
                             in_memory_tables=in_memory_tables)
