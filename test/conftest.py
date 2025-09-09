from collections.abc import Iterator

import ehrax as rx
import equinox as eqx
import numpy as np
import pandas as pd
import pytest
import tables as tb
from ehrax.testing.common_setup import (
    _admission,
    _admissions,
    _dataset_tables,
    _dx_codes,
    _dx_codes_history,
    _icu_inputs,
    _inpatient_interventions,
    _inpatient_observables,
    _outcome,
    _proc,
    _segmented_inpatient_interventions,
    _singular_codevec,
    _static_info,
    DATASET_CONFIG,
    DATASET_SCHEME_CONF,
    DATASET_SCHEME_MANAGER,
    leading_observables_extractor,
    OUTCOME_DATA,
    SCHEMES,
    TARGET_SCHEMES,
    TVXEHR_CONF,
)

from .common_setup import (
    INTERVENTIONS_MAX_N_ITEMS,
    OBS_MAX_N_TIMESTAMPS,
    ADMISSION_CONCEPT_MAX_STAY_DAYS,
    ADMISSION_CONCEPT_MAX_STAY_HOURS,
    ADMISSION_TABLES_MAX_STAY_DAYS,
)


class Dataset(rx.Dataset):
    @classmethod
    def load_tables(cls, config: rx.DatasetConfig, scheme: rx.DatasetSchemeProxy) -> rx.DatasetTables:
        raise NotImplementedError("Still not implemented.")

    @classmethod
    def make_default_pipeline(cls) -> rx.AbstractDatasetPipeline:
        raise NotImplementedError("No.")


@pytest.fixture(
    params=[(1, 0, 0), (1, 2, 0)],
    ids=lambda x: f"_{x[0]}_subjects_{x[0] * x[1]}_admissions_{x[0] * x[1] * x[2]}_records",
    scope="session",
)
def dataset_tables_without_records(request) -> rx.DatasetTables:
    return _dataset_tables(
        DATASET_SCHEME_CONF, DATASET_SCHEME_MANAGER, request.param, max_stay_days=ADMISSION_TABLES_MAX_STAY_DAYS
    )


@pytest.fixture(scope="session")
def dataset_without_records(dataset_tables_without_records: rx.DatasetTables) -> Dataset:
    return Dataset(tables=dataset_tables_without_records, config=DATASET_CONFIG)


@pytest.fixture(
    params=[(1, 2, 10), (300, 3, 25)],
    ids=lambda x: f"_{x[0]}_subjects_{x[0] * x[1]}_admissions_{x[0] * x[1] * x[2]}_records",
    scope="session",
)
def dataset_tables_with_records(request) -> rx.DatasetTables:
    return _dataset_tables(
        DATASET_SCHEME_CONF, DATASET_SCHEME_MANAGER, request.param, max_stay_days=ADMISSION_TABLES_MAX_STAY_DAYS
    )


@pytest.fixture(scope="session")
def dataset_with_records(dataset_tables_with_records: rx.DatasetTables) -> Dataset:
    return Dataset(tables=dataset_tables_with_records, config=DATASET_CONFIG)


@pytest.fixture(scope="session")
def large_dataset_tables():
    return _dataset_tables(
        DATASET_SCHEME_CONF, DATASET_SCHEME_MANAGER, (1000, 5, 1), max_stay_days=ADMISSION_TABLES_MAX_STAY_DAYS
    )


@pytest.fixture(scope="session")
def large_dataset(large_dataset_tables: rx.DatasetTables) -> Dataset:
    return Dataset(tables=large_dataset_tables, config=DATASET_CONFIG)


@pytest.fixture(scope="session")
def unit_converter_table(dataset_tables_with_records: rx.DatasetTables) -> pd.DataFrame | None:
    assert "icu_inputs" in dataset_tables_with_records.tables_dict or len(dataset_tables_with_records.icu_inputs) == 0
    c_code = str(rx.COLUMN.code)
    c_amount_unit = str(rx.COLUMN.amount_unit)
    c_norm_factor = str(rx.COLUMN.derived_unit_normalization_factor)
    c_universal_unit = str(rx.COLUMN.derived_universal_unit)
    icu_inputs = dataset_tables_with_records.icu_inputs

    table = pd.DataFrame(
        columns=[c_code, c_amount_unit],
        data=[(code, unit) for code, unit in icu_inputs.groupby([c_code, c_amount_unit]).groups.keys()],
    )
    table[c_norm_factor] = 1.0
    table[c_universal_unit] = table[c_amount_unit].map(
        {c: np.random.choice(df[c_amount_unit].unique(), size=1)[0] for c, df in table.groupby(c_code)}
    )
    table[c_norm_factor] = table[c_norm_factor].where(
        table[c_universal_unit] == table[c_amount_unit], np.random.choice([1e-3, 100, 10, 1e3])
    )
    return table


@pytest.fixture(scope="session")
def mimiciv_dataset_scheme_config() -> rx.DatasetSchemeConfig:
    return rx.DatasetSchemeConfig(
        ethnicity=SCHEMES["ethnicity"].name,
        gender=SCHEMES["gender"].name,
        dx_discharge=SCHEMES["dx_discharge"].name,
        icu_procedures=SCHEMES["icu_procedures"].name,
        icu_inputs=SCHEMES["icu_inputs"].name,
        obs=SCHEMES["obs"].name,
        hosp_procedures=SCHEMES["hosp_procedures"].name,
    )


@pytest.fixture(scope="session")
def mimiciv_dataset_config(mimiciv_dataset_scheme_config: rx.DatasetSchemeConfig) -> rx.DatasetConfig:
    return rx.DatasetConfig(scheme=mimiciv_dataset_scheme_config)


@pytest.fixture(scope="session")
def mimiciv_dataset_without_records(mimiciv_dataset_config, dataset_tables_without_records) -> Dataset:
    ds = Dataset(tables=dataset_tables_without_records, config=mimiciv_dataset_config)
    return eqx.tree_at(
        lambda x: x.columns, ds, dataset_tables_without_records, is_leaf=lambda x: x is None
    )._execute_pipeline(
        [rx.SetIndex(), rx.SynchronizeSubjects(), rx.CastTimestamps(), rx.SetAdmissionRelativeTimes()],
        DATASET_SCHEME_MANAGER,
    )


@pytest.fixture(scope="session")
def mimiciv_dataset(dataset_tables_with_records: rx.DatasetTables, unit_converter_table: pd.DataFrame) -> Dataset:
    config = eqx.tree_at(
        lambda x: x.scheme, DATASET_CONFIG, rx.DatasetSchemeConfig(**DATASET_CONFIG.scheme.scheme_fields())
    )
    ds = Dataset(tables=dataset_tables_with_records, config=config)
    return ds._execute_pipeline(
        [
            rx.SetIndex(),
            rx.SynchronizeSubjects(),
            rx.CastTimestamps(),
            rx.ICUInputRateUnitConversion(),
            rx.SetAdmissionRelativeTimes(),
        ],
        DATASET_SCHEME_MANAGER,
    )


@pytest.fixture(scope="session")
def tvx_ehr(mimiciv_dataset: Dataset) -> rx.TVxEHR:
    return rx.TVxEHR(dataset=mimiciv_dataset, config=TVXEHR_CONF)


@pytest.fixture(scope="session")
def tvx_ehr_without_records(mimiciv_dataset_without_records: Dataset) -> rx.TVxEHR:
    return rx.TVxEHR(dataset=mimiciv_dataset_without_records, config=TVXEHR_CONF)


@pytest.fixture
def hf5_writer_file(tmpdir) -> Iterator[tb.File]:
    # No compression for faster tests.
    with tb.open_file(tmpdir.join("test.h5"), "w", filters=tb.Filters(complevel=0)) as h5f:
        yield h5f


@pytest.fixture
def hf5_group_writer(hf5_writer_file: tb.File) -> tb.Group:
    return hf5_writer_file.create_group("/", "test")


@pytest.fixture(scope="session")
def gender() -> rx.CodesVector:
    return _singular_codevec(SCHEMES["gender"])


@pytest.fixture(scope="session")
def ethnicity() -> rx.CodesVector:
    return _singular_codevec(SCHEMES["ethnicity"])


@pytest.fixture(scope="session")
def static_info(ethnicity: rx.CodesVector, gender: rx.CodesVector) -> rx.StaticInfo:
    return _static_info(ethnicity, gender)


@pytest.fixture(scope="session")
def dx_codes():
    return _dx_codes(TARGET_SCHEMES["dx_discharge"])


@pytest.fixture(scope="session")
def dx_codes_history(dx_codes: rx.CodesVector):
    return _dx_codes_history(dx_codes)


@pytest.fixture(scope="session")
def outcome(dx_codes: rx.CodesVector):
    return _outcome(OUTCOME_DATA, DATASET_SCHEME_MANAGER, dx_codes)


@pytest.fixture(params=[0, 1, 301], scope="session", ids=["0-obs", "1-obs", "301-obs"])
def inpatient_observables(request):
    n_timestamps = request.param
    return _inpatient_observables(TARGET_SCHEMES["obs"], n_timestamps, los_hours=ADMISSION_CONCEPT_MAX_STAY_HOURS)


@pytest.fixture(params=[0, 1, 5], scope="session", ids=["0icuin", "1icuin", "5icuin"])
def icu_inputs(request) -> rx.InpatientInput:
    return _icu_inputs(SCHEMES["icu_inputs"], request.param, los_hours=ADMISSION_CONCEPT_MAX_STAY_HOURS)


@pytest.fixture(params=[0, 5], scope="session", ids=["0icuproc", "5icuproc"])
def icu_proc(request):
    return _proc(SCHEMES["icu_procedures"], request.param, los_hours=ADMISSION_CONCEPT_MAX_STAY_HOURS)


@pytest.fixture(params=[0, 5], scope="session", ids=["0hosproc", "5hosproc"])
def hosp_proc(request):
    return _proc(SCHEMES["hosp_procedures"], n_timestamps=request.param, los_hours=ADMISSION_CONCEPT_MAX_STAY_HOURS)


@pytest.fixture(params=[0, 1, 2, -1], scope="session")
def inpatient_interventions_with_a_none(hosp_proc, icu_proc, icu_inputs, request):
    whoisnull = request.param
    return _inpatient_interventions(
        None if whoisnull == 0 else hosp_proc,
        None if whoisnull == 1 else icu_proc,
        None if whoisnull == 2 else icu_inputs,
    )


@pytest.fixture(scope="session")
def inpatient_interventions(hosp_proc, icu_proc, icu_inputs):
    return _inpatient_interventions(hosp_proc, icu_proc, icu_inputs)


@pytest.fixture(scope="session")
def segmented_inpatient_interventions(
    inpatient_interventions_with_a_none: rx.InpatientInterventions,
) -> rx.SegmentedInpatientInterventions:
    return _segmented_inpatient_interventions(
        inpatient_interventions_with_a_none,
        hosp_proc_scheme=SCHEMES["hosp_procedures"],
        icu_proc_scheme=SCHEMES["icu_procedures"],
        icu_inputs_scheme=SCHEMES["icu_inputs"],
        maximum_padding=1,
        max_los_hours=ADMISSION_CONCEPT_MAX_STAY_HOURS,
    )


@pytest.fixture(scope="session")
def leading_observable(inpatient_observables: rx.InpatientObservables) -> rx.InpatientObservables:
    return leading_observables_extractor(observation_scheme=SCHEMES["obs"])(inpatient_observables)


@pytest.fixture(scope="session")
def admission(
    dx_codes: rx.CodesVector,
    dx_codes_history: rx.CodesVector,
    outcome: rx.CodesVector,
    inpatient_observables: rx.InpatientObservables,
    inpatient_interventions: rx.InpatientInterventions,
    leading_observable: rx.InpatientObservables,
) -> rx.Admission:
    admission_id = "test"
    return _admission(
        admission_id=admission_id,
        admission_date=pd.to_datetime("now"),
        dx_codes=dx_codes,
        dx_codes_history=dx_codes_history,
        outcome=outcome,
        observables=inpatient_observables,
        interventions=inpatient_interventions,
        leading_observable=leading_observable,
        los_days=ADMISSION_CONCEPT_MAX_STAY_DAYS,
    )


@pytest.fixture(scope="session")
def segmented_admission(admission: rx.Admission) -> rx.SegmentedAdmission:
    return rx.SegmentedAdmission.from_admission(
        admission=admission,
        maximum_padding=1,
        icu_inputs_size=len(SCHEMES["icu_inputs"]),
        icu_procedures_size=len(SCHEMES["icu_procedures"]),
        hosp_procedures_size=len(SCHEMES["hosp_procedures"]),
    )


@pytest.fixture(scope="session")
def segmented_patient(patient: rx.Patient) -> rx.SegmentedPatient:
    return rx.SegmentedPatient.from_patient(
        patient=patient,
        maximum_padding=1,
        icu_inputs_size=len(SCHEMES["icu_inputs"]),
        icu_procedures_size=len(SCHEMES["icu_procedures"]),
        hosp_procedures_size=len(SCHEMES["hosp_procedures"]),
    )


@pytest.fixture(params=[0, 10], scope="session", ids=["0adms", "10adms"])
def patient(request, static_info: rx.StaticInfo) -> rx.Patient:
    admissions = _admissions(
        n_admissions=request.param,
        dx_scheme=SCHEMES["dx_discharge"],
        outcome_extractor_=OUTCOME_DATA,
        observation_scheme=SCHEMES["obs"],
        icu_inputs_scheme=SCHEMES["icu_inputs"],
        icu_proc_scheme=SCHEMES["icu_procedures"],
        hosp_proc_scheme=SCHEMES["hosp_procedures"],
        dataset_scheme_manager=DATASET_SCHEME_MANAGER,
        max_los_days=ADMISSION_CONCEPT_MAX_STAY_DAYS,
        max_n_timestamps_obs=OBS_MAX_N_TIMESTAMPS,
        max_n_inputs=INTERVENTIONS_MAX_N_ITEMS,
    )
    return rx.Patient(subject_id="test", admissions=admissions, static_info=static_info)
