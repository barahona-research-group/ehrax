from typing import Literal

from .mimic import ScopedSchemeNames, MIMICDataset, MIMICDatasetAuxiliaryResources
from .mimic_in_memory import MIMICTablesResources, InMemoryMIMICTableFiles, MIMICIII_TABLES_RESOURCES, \
    MIMICIV_TABLES_RESOURCES
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


def default_dataset_schemes_config(dataset_name: DatasetName) -> DatasetSchemeConfig:
    names = scoped_names(dataset_name)
    return DatasetSchemeConfig(ethnicity=names.ethnicity,
                               gender=names.gender,
                               dx_discharge=names.dx_discharge,
                               obs=names.obs,
                               icu_procedures=names.icu_procedures,
                               hosp_procedures=names.hosp_procedures,
                               icu_inputs=names.icu_inputs)


def default_dataset_config(dataset_name: DatasetName) -> DatasetConfig:
    return DatasetConfig(
        scheme=default_dataset_schemes_config(dataset_name),
        columns=DatasetColumns(),
        overlapping_admissions="merge",
        select_subjects_with_observation=None)


def default_dataset_pipeline() -> AbstractDatasetPipeline:
    pipeline = [
        SetIndex(),
        CastTimestamps(),
        ProcessOverlappingAdmissions(),
        FilterSubjectsNegativeAdmissionLengths(),
        FilterUnsupportedCodes(),
    ]
    return AbstractDatasetPipeline(transformations=pipeline)


def default_tvx_schemes_config(config: DatasetSchemeConfig, dataset_name: DatasetName) -> TVxEHRSchemeConfig:
    names = scoped_names(dataset_name).target
    return TVxEHRSchemeConfig(
        gender=config.gender,
        ethnicity=names.ethnicity,
        dx_discharge='dx_icd9',
        outcome='dx_icd9_v1')


def default_tvx_ehr_config(dataset_name: DatasetName) -> TVxEHRConfig:
    scheme = default_tvx_schemes_config(default_dataset_schemes_config(dataset_name), dataset_name)
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


def default_tvx_ehr_pipeline() -> AbstractTVxPipeline:
    pipeline = [
        SampleSubjects(),
        RandomSplits(),
        TVxConcepts(),
        ExcludeShortAdmissions()
    ]
    return AbstractTVxPipeline(transformations=pipeline)


def _mimic_from_memory(dataset_scheme_config: DatasetSchemeConfig,
                       dataset_tables_resources: MIMICTablesResources,
                       aux: MIMICDatasetAuxiliaryResources,
                       in_memory_tables: InMemoryMIMICTableFiles) -> tuple[MIMICDataset, CodingSchemesManager]:
    return MIMICDataset.compile(config=DatasetConfig(scheme=dataset_scheme_config),
                                tables=dataset_tables_resources,
                                aux=aux,
                                data_connection=in_memory_tables)


def mimiciii_from_paths(patients: str, admissions: str, diagnoses_icd: str, d_icd_diagnoses: str,
                        schemes_config: DatasetSchemeConfig = default_dataset_schemes_config('mimiciii'),
                        aux_resources: MIMICDatasetAuxiliaryResources = MIMICDatasetAuxiliaryResources.make_resources(),
                        dataset_tables_resources: MIMICTablesResources = MIMICIII_TABLES_RESOURCES,
                        ) -> tuple[
    MIMICDataset, CodingSchemesManager]:
    in_memory_tables = InMemoryMIMICTableFiles.from_path(patients=patients, admissions=admissions,
                                                         diagnoses_icd=diagnoses_icd, d_icd_diagnoses=d_icd_diagnoses)
    return _mimic_from_memory(dataset_scheme_config=schemes_config,
                              dataset_tables_resources=dataset_tables_resources,
                              in_memory_tables=in_memory_tables, aux=aux_resources)


def mimiciv_from_paths(patients: str, admissions: str, diagnoses_icd: str, d_icd_diagnoses: str,
                       schemes_config: DatasetSchemeConfig = default_dataset_schemes_config('mimiciv'),
                       aux_resources: MIMICDatasetAuxiliaryResources = MIMICDatasetAuxiliaryResources.make_resources(),
                       dataset_tables_resources: MIMICTablesResources = MIMICIV_TABLES_RESOURCES) -> tuple[
    MIMICDataset, CodingSchemesManager]:
    in_memory_tables = InMemoryMIMICTableFiles.from_path(patients=patients, admissions=admissions,
                                                         diagnoses_icd=diagnoses_icd, d_icd_diagnoses=d_icd_diagnoses)
    return _mimic_from_memory(dataset_scheme_config=schemes_config,
                              dataset_tables_resources=dataset_tables_resources,
                              in_memory_tables=in_memory_tables,
                              aux=aux_resources)
