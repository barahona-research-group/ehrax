from typing import Iterator, Generator
from unittest.mock import patch, PropertyMock

import equinox as eqx
import numpy as np
import pandas as pd
import pytest
import tables as tb

from ehrax.coding_scheme import CodesVector
from ehrax.dataset import DatasetTables, Dataset, \
    DatasetConfig
from ehrax.transformations import SetIndex, CastTimestamps
from ehrax.tvx_concepts import SegmentedAdmission, InpatientInterventions, Admission, \
    SegmentedInpatientInterventions, Patient, SegmentedPatient, StaticInfo
from ehrax.tvx_ehr import TVxEHR, InpatientObservables
from test.common_setup import ALIAS, DATASET_SCHEME_CONF, TVXEHR_CONF, NaiveEHR, _dataset_tables, SCHEMES, \
    _dx_codes, OUTCOME_EXTRACTOR, \
    DATASET_SCHEME_MANAGER, \
    _singular_codevec, _static_info, _dx_codes_history, _inpatient_observables, _icu_inputs, _proc, _outcome, \
    DATASET_CONFIG, _segmented_inpatient_interventions, _inpatient_interventions, leading_observables_extractor, \
    _admission, _admissions, DATASET_TABLES_CONF, NaiveDataset, MockMIMICIVDatasetSchemeConfig, MockMIMICIVDataset, \
    TARGET_SCHEMES


@pytest.fixture(params=[(1, 0, 0), (1, 2, 0), (1, 2, 10), (500, 5, 25)],
                ids=lambda x: f"_{x[0]}_subjects_{x[0] * x[1]}_admissions_{x[0] * x[1] * x[2]}_records",
                scope='session')
def dataset_tables(request) -> DatasetTables:
    return _dataset_tables(DATASET_TABLES_CONF, DATASET_SCHEME_CONF, DATASET_SCHEME_MANAGER, request.param)


@pytest.fixture(scope='session')
def large_dataset_tables():
    return _dataset_tables(DATASET_TABLES_CONF, DATASET_SCHEME_CONF, DATASET_SCHEME_MANAGER, (1000, 5, 1))


@pytest.fixture(scope='session')
def dataset(dataset_tables: DatasetTables):
    return NaiveDataset(tables=dataset_tables, config=DATASET_CONFIG)


@pytest.fixture(scope='session')
def large_dataset(large_dataset_tables: DatasetTables):
    return NaiveDataset(tables=large_dataset_tables, config=DATASET_CONFIG)


@pytest.fixture(scope='session')
def indexed_dataset(dataset) -> NaiveDataset:
    return dataset.execute_pipeline(dataset.make_default_pipeline(), DATASET_SCHEME_MANAGER)


@pytest.fixture(scope='session')
def indexed_large_dataset(large_dataset) -> NaiveDataset:
    return large_dataset.execute_pipeline(large_dataset.make_default_pipeline(), DATASET_SCHEME_MANAGER)


@pytest.fixture(scope='session')
def has_admissions_dataset(indexed_dataset: NaiveDataset) -> Generator[NaiveDataset, None, None]:
    if len(indexed_dataset.tables.admissions) == 0:
        raise pytest.skip("No admissions data found in dataset.")
    yield indexed_dataset


@pytest.fixture(scope='session')
def has_codes_dataset(has_admissions_dataset: Dataset):
    if all(len(getattr(has_admissions_dataset.tables, k)) == 0 for k in
           has_admissions_dataset.config.tables.code_column.keys()):
        raise pytest.skip("No coded tables or they are all empty.")
    yield has_admissions_dataset


@pytest.fixture(scope='session')
def has_obs_dataset(has_admissions_dataset: Dataset) -> Generator[NaiveDataset, None, None]:
    if len(has_admissions_dataset.tables.obs) == 0:
        raise pytest.skip("No obs data found in dataset.")
    yield has_admissions_dataset


@pytest.fixture(scope='session')
def subject_id_column(indexed_dataset: Dataset) -> str:
    return indexed_dataset.config.tables.subject_id_alias


@pytest.fixture(scope='session')
def admission_id_column(indexed_dataset: Dataset) -> str:
    return indexed_dataset.config.tables.admission_id_alias


@pytest.fixture(scope='session')
def sample_subject_id(has_admissions_dataset: Dataset) -> str:
    return has_admissions_dataset.tables.admissions[ALIAS['subject_id']].iloc[0]


@pytest.fixture(scope='session')
def sample_admission_id(has_admissions_dataset: Dataset) -> str:
    return has_admissions_dataset.tables.admissions.index[0]


@pytest.fixture(scope='session')
def unit_converter_table(dataset_tables: DatasetTables) -> Generator[pd.DataFrame, None, None]:
    if 'icu_inputs' not in dataset_tables.tables_dict or len(dataset_tables.icu_inputs) == 0:
        raise pytest.skip("No ICU inputs in dataset. Required for the unit conversion table generation.")
    assert DATASET_CONFIG.tables.icu_inputs is not None, \
        "Dataset configuration must have ICU inputs table defined."
    c_code = DATASET_CONFIG.tables.icu_inputs.code_alias
    c_amount_unit = DATASET_CONFIG.tables.icu_inputs.amount_unit_alias
    c_norm_factor = DATASET_CONFIG.tables.icu_inputs.derived_unit_normalization_factor
    c_universal_unit = DATASET_CONFIG.tables.icu_inputs.derived_universal_unit
    icu_inputs = dataset_tables.icu_inputs

    table = pd.DataFrame(columns=[c_code, c_amount_unit],
                         data=[(code, unit) for code, unit in
                               icu_inputs.groupby([c_code, c_amount_unit]).groups.keys()])

    for code, df in table.groupby(c_code):
        units = df[c_amount_unit].unique()
        universal_unit = np.random.choice(units, size=1)[0]
        norm_factor = 1
        if len(units) > 1:
            norm_factor = np.random.choice([1e-3, 100, 10, 1e3], size=len(units))
            norm_factor = np.where(units == universal_unit, 1, norm_factor)
        table.loc[df.index, c_norm_factor] = norm_factor
        table.loc[df.index, c_universal_unit] = universal_unit
    yield table


@pytest.fixture(scope='session')
def mimiciv_dataset_scheme_config() -> MockMIMICIVDatasetSchemeConfig:
    return MockMIMICIVDatasetSchemeConfig(
        ethnicity=SCHEMES['ethnicity'].name,
        gender=SCHEMES['gender'].name,
        dx_discharge=SCHEMES['dx_discharge'].name,
        icu_procedures=SCHEMES['icu_procedures'].name,
        icu_inputs=SCHEMES['icu_inputs'].name,
        obs=SCHEMES['obs'].name,
        hosp_procedures=SCHEMES['hosp_procedures'].name)


@pytest.fixture(scope='session')
def mimiciv_dataset_config(mimiciv_dataset_scheme_config: MockMIMICIVDatasetSchemeConfig):
    return DatasetConfig(scheme=mimiciv_dataset_scheme_config, tables=DATASET_TABLES_CONF)


@pytest.fixture(scope='session')
def mimiciv_dataset_no_conv(mimiciv_dataset_config, dataset_tables,
                            unit_converter_table) -> Iterator[MockMIMICIVDataset]:
    ds = MockMIMICIVDataset(tables=dataset_tables, config=mimiciv_dataset_config)
    with patch(__name__ + '.MockMIMICIVDatasetSchemeConfig.icu_inputs_uom_normalization_table',
               return_value=unit_converter_table,
               new_callable=PropertyMock):
        yield eqx.tree_at(lambda x: x.tables, ds, dataset_tables,
                          is_leaf=lambda x: x is None)._execute_pipeline([SetIndex(), CastTimestamps()],
                                                                         DATASET_SCHEME_MANAGER)


@pytest.fixture(scope='session')
def mimiciv_dataset(dataset_tables: DatasetTables,
                    unit_converter_table: pd.DataFrame) -> Iterator[MockMIMICIVDataset]:
    config = eqx.tree_at(lambda x: x.scheme, DATASET_CONFIG,
                         MockMIMICIVDatasetSchemeConfig(**DATASET_CONFIG.scheme.scheme_fields()))
    ds = MockMIMICIVDataset(tables=dataset_tables, config=config)
    with patch('test.common_setup.MockMIMICIVDatasetSchemeConfig.icu_inputs_uom_normalization_table',
               return_value=unit_converter_table,
               new_callable=PropertyMock):
        yield ds.execute_pipeline(MockMIMICIVDataset.make_default_pipeline(DATASET_CONFIG), DATASET_SCHEME_MANAGER)


@pytest.fixture(scope='session')
def tvx_ehr(mimiciv_dataset: MockMIMICIVDataset) -> TVxEHR:
    return NaiveEHR(dataset=mimiciv_dataset, config=TVXEHR_CONF)


@pytest.fixture
def hf5_writer_file(tmpdir) -> Iterator[tb.File]:
    # No compression for faster tests.
    with tb.open_file(tmpdir.join('test.h5'), 'w', filters=tb.Filters(complevel=0)) as h5f:
        yield h5f


@pytest.fixture
def hf5_group_writer(hf5_writer_file: tb.File) -> tb.Group:
    return hf5_writer_file.create_group('/', 'test')


@pytest.fixture(scope='session')
def gender() -> CodesVector:
    return _singular_codevec(SCHEMES['gender'])


@pytest.fixture(scope='session')
def ethnicity() -> CodesVector:
    return _singular_codevec(SCHEMES['ethnicity'])


@pytest.fixture(scope='session')
def static_info(ethnicity: CodesVector, gender: CodesVector) -> StaticInfo:
    return _static_info(ethnicity, gender)


@pytest.fixture(scope='session')
def dx_codes():
    return _dx_codes(TARGET_SCHEMES['dx_discharge'])


@pytest.fixture(scope='session')
def dx_codes_history(dx_codes: CodesVector):
    return _dx_codes_history(dx_codes)


@pytest.fixture(scope='session')
def outcome(dx_codes: CodesVector):
    return _outcome(OUTCOME_EXTRACTOR, DATASET_SCHEME_MANAGER, dx_codes)


@pytest.fixture(params=[0, 1, 301], scope='session', ids=['0-obs', '1-obs', '301-obs'])
def inpatient_observables(request):
    n_timestamps = request.param
    return _inpatient_observables(TARGET_SCHEMES['obs'], n_timestamps)


@pytest.fixture(params=[0, 1, 5], scope='session', ids=['0icuin', '1icuin', '5icuin'])
def icu_inputs(request):
    return _icu_inputs(SCHEMES['icu_inputs'], request.param)


@pytest.fixture(params=[0, 5], scope='session', ids=['0icuproc', '5icuproc'])
def icu_proc(request):
    return _proc(SCHEMES['icu_procedures'], request.param)


@pytest.fixture(params=[0, 5], scope='session', ids=['0hosproc', '5hosproc'])
def hosp_proc(request):
    return _proc(SCHEMES['hosp_procedures'], n_timestamps=request.param)


@pytest.fixture(params=[0, 1, 2, -1], scope='session')
def inpatient_interventions_with_a_none(hosp_proc, icu_proc, icu_inputs, request):
    whoisnull = request.param
    return _inpatient_interventions(None if whoisnull == 0 else hosp_proc,
                                    None if whoisnull == 1 else icu_proc,
                                    None if whoisnull == 2 else icu_inputs)


@pytest.fixture(scope='session')
def inpatient_interventions(hosp_proc, icu_proc, icu_inputs):
    return _inpatient_interventions(hosp_proc, icu_proc, icu_inputs)


@pytest.fixture(scope='session')
def segmented_inpatient_interventions(
        inpatient_interventions_with_a_none: InpatientInterventions) -> SegmentedInpatientInterventions:
    return _segmented_inpatient_interventions(inpatient_interventions_with_a_none,
                                              hosp_proc_scheme=SCHEMES['hosp_procedures'],
                                              icu_proc_scheme=SCHEMES['icu_procedures'],
                                              icu_inputs_scheme=SCHEMES['icu_inputs'],
                                              maximum_padding=1)


@pytest.fixture(scope='session')
def leading_observable(inpatient_observables: InpatientObservables) -> InpatientObservables:
    return leading_observables_extractor(observation_scheme=SCHEMES['obs'])(inpatient_observables)


@pytest.fixture(scope='session')
def admission(dx_codes: CodesVector, dx_codes_history: CodesVector,
              outcome: CodesVector, inpatient_observables: InpatientObservables,
              inpatient_interventions: InpatientInterventions,
              leading_observable: InpatientObservables) -> Admission:
    admission_id = 'test'
    return _admission(admission_id=admission_id, admission_date=pd.to_datetime('now'),
                      dx_codes=dx_codes, dx_codes_history=dx_codes_history, outcome=outcome,
                      observables=inpatient_observables, interventions=inpatient_interventions,
                      leading_observable=leading_observable)


@pytest.fixture(scope='session')
def segmented_admission(admission: Admission) -> SegmentedAdmission:
    return SegmentedAdmission.from_admission(admission=admission, maximum_padding=1,
                                             icu_inputs_size=len(SCHEMES['icu_inputs']),
                                             icu_procedures_size=len(SCHEMES['icu_procedures']),
                                             hosp_procedures_size=len(SCHEMES['hosp_procedures']))


@pytest.fixture(scope='session')
def segmented_patient(patient: Patient) -> SegmentedPatient:
    return SegmentedPatient.from_patient(patient=patient, maximum_padding=1,
                                         icu_inputs_size=len(SCHEMES['icu_inputs']),
                                         icu_procedures_size=len(SCHEMES['icu_procedures']),
                                         hosp_procedures_size=len(SCHEMES['hosp_procedures']))


@pytest.fixture(params=[0, 10], scope='session', ids=['0adms', '10adms'])
def patient(request, static_info: StaticInfo) -> Patient:
    admissions = _admissions(n_admissions=request.param, dx_scheme=SCHEMES['dx_discharge'],
                             outcome_extractor_=OUTCOME_EXTRACTOR, observation_scheme=SCHEMES['obs'],
                             icu_inputs_scheme=SCHEMES['icu_inputs'], icu_proc_scheme=SCHEMES['icu_procedures'],
                             hosp_proc_scheme=SCHEMES['hosp_procedures'],
                             dataset_scheme_manager=DATASET_SCHEME_MANAGER)
    return Patient(subject_id='test', admissions=admissions, static_info=static_info)
