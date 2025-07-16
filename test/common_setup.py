import random

import numpy as np
import numpy.random as nr
import pandas as pd

from ehrax.coding_scheme import FrozenDict11, ReducedCodeMapN1, NumericScheme, CodingScheme, CodingSchemesManager, \
    FrozenDict1N, OutcomeExtractor, ExcludingOutcomeExtractor, CodesVector, CodeMap
from ehrax.dataset import DatasetTablesConfig, RatedInputTableConfig, AdmissionTimestampedCodedValueTableConfig, \
    AdmissionLinkedCodedValueTableConfig, AdmissionIntervalBasedCodedTableConfig, StaticTableConfig, \
    AdmissionTableConfig, DatasetTables, DatasetConfig, DatasetSchemeConfig
from ehrax.tvx_concepts import AdmissionDates, InpatientInterventions, LeadingObservableExtractorConfig, \
    LeadingObservableExtractor, Admission, SegmentedInpatientInterventions, InpatientObservables, InpatientInput, \
    DemographicVectorConfig, StaticInfo
from ehrax.tvx_ehr import TVxEHRConfig, TVxEHRSchemeConfig

MAX_STAY_DAYS = 356
LENGTH_OF_STAY = 5.0


def scheme(name: str, codes: list[str]) -> CodingScheme:
    return CodingScheme(name=name, codes=tuple(sorted(codes)),
                        desc=FrozenDict11(dict(zip(codes, codes))))


def outcome_extractor(dx_scheme: CodingScheme) -> OutcomeExtractor:
    name = f'{dx_scheme.name}_outcome'
    k = max(3, len(dx_scheme) - 1)
    random.seed(0)
    excluded = tuple(random.sample(dx_scheme.codes, k=k))
    return ExcludingOutcomeExtractor(name=name, base_name=dx_scheme.name, exclude_codes=excluded)


def sample_codes(scheme: CodingScheme, n: int) -> list[str]:
    codes = scheme.codes
    return random.choices(codes, k=n)


def _dx_codes(dx_scheme: CodingScheme):
    v = nr.binomial(1, 0.5, size=len(dx_scheme)).astype(bool)
    return CodesVector(vec=v, scheme=dx_scheme.name)


def inpatient_binary_input(n: int, p: int):
    starttime = np.array(
        sorted(nr.choice(np.linspace(0, LENGTH_OF_STAY, max(1000, n)), replace=False, size=n)))
    endtime = starttime + nr.uniform(0, LENGTH_OF_STAY - starttime, size=(n,))
    code_index = nr.choice(p, size=n, replace=True)
    return InpatientInput(starttime=starttime, endtime=endtime, code_index=code_index)


def inpatient_rated_input(n: int, p: int):
    bin_input = inpatient_binary_input(n, p)
    return InpatientInput(starttime=bin_input.starttime, endtime=bin_input.endtime, code_index=bin_input.code_index,
                          rate=nr.uniform(0, 1, size=(n,)))


def _singular_codevec(scheme: CodingScheme) -> CodesVector:
    return scheme.codeset2vec({random.choice(scheme.codes)})


def _icu_inputs(icu_inputs_scheme: CodingScheme, n_timestamps: int):
    return inpatient_rated_input(n_timestamps, len(icu_inputs_scheme))


def _proc(scheme: CodingScheme, n_timestamps: int):
    return inpatient_binary_input(n_timestamps, len(scheme))


def demographic_vector_config() -> DemographicVectorConfig:
    flags = random.choices([True, False], k=3)
    return DemographicVectorConfig(*flags)


def date_of_birth() -> pd.Timestamp:
    return pd.to_datetime(pd.Timestamp('now') - pd.to_timedelta(nr.randint(0, 100 * 365), unit='D'))


def _static_info(ethnicity: CodesVector, gender: CodesVector) -> StaticInfo:
    return StaticInfo(ethnicity=ethnicity, gender=gender,
                      date_of_birth=date_of_birth())


def _dx_codes_history(dx_codes: CodesVector):
    v = nr.binomial(1, 0.5, size=len(dx_codes.vec)).astype(bool)
    return CodesVector(vec=v + dx_codes.vec, scheme=dx_codes.scheme)


def _outcome(outcome_extractor_: OutcomeExtractor, dataset_scheme_manager: CodingSchemesManager,
             dx_codes: CodesVector):
    source_scheme = dataset_scheme_manager.scheme[dx_codes.scheme]
    outcome_base_scheme = dataset_scheme_manager.scheme[outcome_extractor_.base_name]
    code_map = dataset_scheme_manager.map[(dx_codes.scheme, outcome_extractor_.base_name)]
    extractor = outcome_extractor_.codeset2vec_extractor(outcome_base_scheme, code_map)
    return extractor(source_scheme.vec2codeset(dx_codes.vec))


def sample_subjects_dataframe(n: int, static_table_config: StaticTableConfig, ethnicity_scheme: CodingScheme,
                              gender_scheme: CodingScheme) -> pd.DataFrame:
    return pd.DataFrame({
        static_table_config.subject_id_alias: list(str(i) for i in range(n)),
        static_table_config.race_alias: random.choices(ethnicity_scheme.codes, k=n),
        static_table_config.gender_alias: random.choices(gender_scheme.codes, k=n),
        static_table_config.date_of_birth_alias: pd.to_datetime(
            random.choices(pd.date_range(start='1/1/1900', end='1/1/2000', freq='D'), k=n))
    })


def sample_admissions_dataframe(subjects_df: pd.DataFrame,
                                n: int, static_table_config: StaticTableConfig,
                                admission_table_config: AdmissionTableConfig) -> pd.DataFrame:
    c_subject = static_table_config.subject_id_alias
    c_admission = admission_table_config.admission_id_alias
    c_admission_time = admission_table_config.admission_time_alias
    c_discharge_time = admission_table_config.discharge_time_alias
    admit_dates = pd.to_datetime(random.choices(pd.date_range(start='1/1/2000', end='1/1/2020', freq='D'), k=n))
    disch_dates = admit_dates + pd.to_timedelta(random.choices(range(1, MAX_STAY_DAYS), k=n), unit='D')

    return pd.DataFrame({
        c_subject: random.choices(subjects_df[c_subject], k=n),
        c_admission: list(str(i) for i in range(n)),
        c_admission_time: admit_dates,
        c_discharge_time: disch_dates
    })


def sample_dx_dataframe(admissions_df: pd.DataFrame,
                        admission_table_config: AdmissionTableConfig,
                        dx_discharge_table_config: AdmissionLinkedCodedValueTableConfig,
                        dx_scheme: CodingScheme, n: int) -> pd.DataFrame:
    c_admission = admission_table_config.admission_id_alias
    c_dx = dx_discharge_table_config.code_alias
    dx_codes = sample_codes(dx_scheme, n)
    return pd.DataFrame({
        c_admission: random.choices(admissions_df[c_admission], k=n),
        c_dx: dx_codes
    })


def _sample_proc_dataframe(admissions_df: pd.DataFrame,
                           admission_table_config: AdmissionTableConfig,
                           table_config: AdmissionIntervalBasedCodedTableConfig,
                           scheme: CodingScheme,
                           n: int) -> pd.DataFrame:
    c_admission = admission_table_config.admission_id_alias
    c_admittime = admission_table_config.admission_time_alias
    c_dischtime = admission_table_config.discharge_time_alias
    c_code = table_config.code_alias
    c_start = table_config.start_time_alias
    c_end = table_config.end_time_alias
    codes = sample_codes(scheme, n)
    df = pd.DataFrame({
        c_admission: random.choices(admissions_df[c_admission], k=n),
        c_code: codes
    })
    df = pd.merge(df, admissions_df[[c_admission, c_admittime, c_dischtime]],
                  on=c_admission,
                  suffixes=(None, '_admission'))
    df['los'] = (df[c_dischtime] - df[c_admittime]).dt.total_seconds() / 3600

    relative_start = nr.uniform(0, df['los'].values.tolist(), size=n)
    df[c_start] = df[c_admittime] + pd.to_timedelta(relative_start, unit='hours')

    relative_end = nr.uniform(low=relative_start, high=df['los'].values.tolist(), size=n)

    df[c_end] = df[c_admittime] + pd.to_timedelta(relative_end, unit='hours')
    return df[[c_admission, c_code, c_start, c_end]]


def sample_icu_inputs_dataframe(admissions_df: pd.DataFrame,
                                admission_table_config: AdmissionTableConfig,
                                table_config: RatedInputTableConfig,
                                icu_input_scheme: CodingScheme,
                                n: int) -> pd.DataFrame:
    df = _sample_proc_dataframe(admissions_df, admission_table_config, table_config, icu_input_scheme, n)
    c_amount = table_config.amount_alias
    c_unit = table_config.amount_unit_alias
    df[c_amount] = np.random.uniform(low=0, high=1000, size=n)
    df[c_unit] = random.choices(['mg', 'g', 'kg', 'cm', 'dose', 'ml'], k=n)
    return df


def sample_obs_dataframe(admissions_df: pd.DataFrame,
                         admission_table_config: AdmissionTableConfig,
                         obs_table_config: AdmissionTimestampedCodedValueTableConfig,
                         obs_scheme: CodingScheme,
                         n: int) -> pd.DataFrame:
    c_admission = admission_table_config.admission_id_alias
    c_admittime = admission_table_config.admission_time_alias
    c_dischtime = admission_table_config.discharge_time_alias
    c_obs = obs_table_config.code_alias
    c_time = obs_table_config.time_alias
    c_value = obs_table_config.value_alias

    codes = sample_codes(obs_scheme, n)
    df = pd.DataFrame({
        c_admission: random.choices(admissions_df[c_admission], k=n),
        c_obs: codes
    })
    df = pd.merge(df, admissions_df[[c_admission, c_admittime, c_dischtime]], on=c_admission,
                  suffixes=(None, '_y'))
    df['los'] = (df[c_dischtime] - df[c_admittime]).dt.total_seconds() / 3600
    relative_time = nr.uniform(0, df['los'].values.tolist(), size=n)
    df[c_time] = df[c_admittime] + pd.to_timedelta(relative_time, unit='hours')

    assert isinstance(obs_scheme, NumericScheme), 'Only numeric schemes are supported'
    df['obs_type'] = df[c_obs].map(obs_scheme.type_hint.data)
    df.loc[df.obs_type == 'N', c_value] = np.random.uniform(low=0, high=1000, size=(df.obs_type == 'N').sum())
    df.loc[df.obs_type.isin(('C', 'O')), c_value] = random.choices([0, 1, 2],
                                                                   k=df.obs_type.isin(('C', 'O')).sum())
    df.loc[df.obs_type == 'B', c_value] = random.choices([0, 1], k=(df.obs_type == 'B').sum())

    return df[[c_admission, c_obs, c_time, c_value]]


def _dataset_tables(dataset_tables_config: DatasetTablesConfig,
                    dataset_scheme_config: DatasetSchemeConfig,
                    dataset_scheme_manager: CodingSchemesManager,
                    freqs: tuple[int, ...]) -> DatasetTables:
    n_subjects, n_admission_per_subject, n_per_admission = freqs
    assert dataset_scheme_config.ethnicity is not None
    assert dataset_scheme_config.gender is not None
    assert dataset_scheme_config.dx_discharge is not None
    assert dataset_scheme_config.icu_procedures is not None
    assert dataset_scheme_config.icu_inputs is not None
    assert dataset_scheme_config.obs is not None
    assert dataset_scheme_config.hosp_procedures is not None
    assert dataset_tables_config.dx_discharge is not None
    assert dataset_tables_config.obs is not None
    assert dataset_tables_config.icu_procedures is not None
    assert dataset_tables_config.hosp_procedures is not None
    assert dataset_tables_config.icu_inputs is not None
    subjects_df = sample_subjects_dataframe(n_subjects, dataset_tables_config.static,
                                            dataset_scheme_manager.scheme[dataset_scheme_config.ethnicity],
                                            dataset_scheme_manager.scheme[dataset_scheme_config.gender])
    admissions_df = sample_admissions_dataframe(subjects_df, n_admission_per_subject * n_subjects,
                                                dataset_tables_config.static,
                                                dataset_tables_config.admissions)
    dx_df = sample_dx_dataframe(admissions_df, dataset_tables_config.admissions,
                                dataset_tables_config.dx_discharge,
                                dataset_scheme_manager.scheme[dataset_scheme_config.dx_discharge],
                                n_per_admission * n_subjects * n_admission_per_subject)

    obs_df = sample_obs_dataframe(admissions_df, dataset_tables_config.admissions,
                                  dataset_tables_config.obs,
                                  dataset_scheme_manager.scheme[dataset_scheme_config.obs],
                                  n_per_admission * n_subjects * n_admission_per_subject)

    icu_proc_df = _sample_proc_dataframe(admissions_df, dataset_tables_config.admissions,
                                         dataset_tables_config.icu_procedures,
                                         dataset_scheme_manager.scheme[dataset_scheme_config.icu_procedures],
                                         n_per_admission * n_subjects * n_admission_per_subject)

    hosp_proc_df = _sample_proc_dataframe(admissions_df, dataset_tables_config.admissions,
                                          dataset_tables_config.hosp_procedures,
                                          dataset_scheme_manager.scheme[dataset_scheme_config.hosp_procedures],
                                          n_per_admission * n_subjects * n_admission_per_subject)

    icu_inputs_df = sample_icu_inputs_dataframe(admissions_df, dataset_tables_config.admissions,
                                                dataset_tables_config.icu_inputs,
                                                dataset_scheme_manager.scheme[dataset_scheme_config.icu_inputs],
                                                n_per_admission * n_subjects * n_admission_per_subject)

    return DatasetTables(static=subjects_df,
                         admissions=admissions_df,
                         dx_discharge=dx_df,
                         obs=obs_df,
                         icu_procedures=icu_proc_df,
                         hosp_procedures=hosp_proc_df,
                         icu_inputs=icu_inputs_df)


def make_targets_schemes_with_maps(n_scheme_targets: dict[str, int], source_schemes: dict[str, CodingScheme]) -> tuple[
    dict[str, CodingScheme], dict[str, CodeMap]]:
    def make_target_scheme_with_map(size: int, space: str, source_scheme: CodingScheme) -> tuple[
        str, CodingScheme, CodeMap]:
        assert size <= len(source_scheme)
        target_name = f'{source_scheme.name}_target'
        target_codes = tuple(f'{source_scheme}_target_{i}' for i in range(size))
        target_desc = FrozenDict11(dict(zip(target_codes, target_codes)))
        target_scheme = CodingScheme(name=target_name, codes=target_codes, desc=target_desc)
        mapp = {c: {t} for c, t in zip(source_scheme.codes, target_codes)}
        mapp |= {c: {random.choice(target_codes)} for c in source_scheme.codes[len(target_codes):]}
        map_data = FrozenDict1N({c: {random.choice(target_codes)} for c in source_scheme.codes})
        code_map = CodeMap(source_name=source_scheme.name, target_name=target_name, data=map_data)
        return space, target_scheme, code_map

    space, schemes, maps = zip(*list(make_target_scheme_with_map(size, space, source_schemes[space])
                                     for space, size in n_scheme_targets.items()))
    return dict(zip(space, schemes)), dict(zip(space, maps))


ALIAS = {
    'subject_id': 'SUBJECT_IDXYZ',
    'race': 'race',
    'gender': 'gender',
    'date_of_birth': 'date_of_birth',
    'admission_id': 'ADMISSION_IDX',
    'admission_time': 'admission_time',
    'discharge_time': 'discharge_time',
    'obs_time': 'time_bin',
    'obs_code': 'measurement',
    'obs_code_desc': 'measurement description',
    'obs_value': 'obs_val',
    'dx_code': 'dx_code',
    'dx_code_desc': 'dx_code description',
    'hosp_proc_code': 'hosp_proc_code',
    'hosp_proc_code_desc': 'hosp_proc_code description',
    'hosp_proc_start_time': 'hosp_proc_start_time',
    'hosp_proc_end_time': 'hosp_proc_end_time',
    'icu_proc_code': 'icu_proc_code',
    'icu_proc_code_desc': 'icu_proc_code description',
    'icu_proc_start_time': 'icu_proc_start_time',
    'icu_proc_end_time': 'icu_proc_end_time',
    'icu_input_code': 'icu_input_code',
    'icu_input_code_desc': 'icu_input_code description',
    'icu_input_start_time': 'icu_input_start_time',
    'icu_input_end_time': 'icu_input_end_time',
    'icu_input_amount_alias': 'icu_input_amount_alias',
    'icu_input_amount_unit_alias': 'icu_input_amount_unit_alias',
    'icu_input_derived_normalized_amount': 'icu_input_derived_normalized_amount',
    'icu_input_derived_normalized_amount_per_hour': 'icu_input_derived_normalized_amount_per_hour',
    'icu_input_derived_unit_normalization_factor': 'icu_input_derived_unit_normalization_factor',
    'icu_input_derived_universal_unit': 'icu_input_derived_universal_unit',
}
TABLE_CONF = dict(
    static=StaticTableConfig(subject_id_alias=ALIAS['subject_id'],
                             gender_alias=ALIAS['gender'], race_alias=ALIAS['race'],
                             date_of_birth_alias=ALIAS['date_of_birth']),
    admissions=AdmissionTableConfig(subject_id_alias=ALIAS['subject_id'],
                                    admission_id_alias=ALIAS['admission_id'],
                                    admission_time_alias=ALIAS['admission_time'],
                                    discharge_time_alias=ALIAS['discharge_time']),

    obs=AdmissionTimestampedCodedValueTableConfig(admission_id_alias=ALIAS['admission_id'],
                                                  time_alias=ALIAS['obs_time'],
                                                  code_alias=ALIAS['obs_code'],
                                                  description_alias=ALIAS['obs_code_desc'],
                                                  value_alias=ALIAS['obs_value']),
    dx_discharge=AdmissionLinkedCodedValueTableConfig(admission_id_alias=ALIAS['admission_id'],
                                                      code_alias=ALIAS['dx_code'],
                                                      description_alias=ALIAS['dx_code_desc']),
    hosp_procedures=AdmissionIntervalBasedCodedTableConfig(admission_id_alias=ALIAS['admission_id'],
                                                           code_alias=ALIAS['hosp_proc_code'],
                                                           description_alias=ALIAS['hosp_proc_code_desc'],
                                                           start_time_alias=ALIAS['hosp_proc_start_time'],
                                                           end_time_alias=ALIAS['hosp_proc_end_time']),
    icu_procedures=AdmissionIntervalBasedCodedTableConfig(admission_id_alias=ALIAS['admission_id'],
                                                          code_alias=ALIAS['icu_proc_code'],
                                                          description_alias=ALIAS['icu_proc_code_desc'],
                                                          start_time_alias=ALIAS['icu_proc_start_time'],
                                                          end_time_alias=ALIAS['icu_proc_end_time']),
    icu_inputs=RatedInputTableConfig(admission_id_alias=ALIAS['admission_id'],
                                     code_alias=ALIAS['icu_input_code'],
                                     description_alias=ALIAS['icu_input_code_desc'],
                                     start_time_alias=ALIAS['icu_input_start_time'],
                                     end_time_alias=ALIAS['icu_input_end_time'],
                                     amount_alias=ALIAS['icu_input_amount_alias'],
                                     amount_unit_alias=ALIAS['icu_input_amount_unit_alias'],
                                     derived_normalized_amount=ALIAS['icu_input_derived_normalized_amount'],
                                     derived_normalized_amount_per_hour=ALIAS[
                                         'icu_input_derived_normalized_amount_per_hour'],
                                     derived_unit_normalization_factor=ALIAS[
                                         'icu_input_derived_unit_normalization_factor'],
                                     derived_universal_unit=ALIAS['icu_input_derived_universal_unit'])

)
DATASET_TABLES_CONF = DatasetTablesConfig(**TABLE_CONF)  # type: ignore
SCHEMES: dict[str, CodingScheme] = dict(
    ethnicity=scheme('ethnicity', ['E1', 'E2', 'E3']),
    gender=scheme('genderrrr', ['M', 'F']),
    dx_discharge=scheme('dx1', ['Dx1', 'Dx2', 'Dx3', 'Dx4', 'Dx5', 'Dx6', 'Dx7', 'Dx8', 'Dx9', 'Dx10']),
    hosp_procedures=scheme('hosp_proc1', ['HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'HP6']),
    icu_procedures=scheme('icu_proc2', ['ICU1', 'ICU2', 'ICU3', 'ICU4', 'ICU5', 'ICU6']),
    icu_inputs=scheme('icu_inputs', ['ICUI1', 'ICUI2', 'ICUI3', 'ICUI4', 'ICUI5', 'ICUI6']),
    obs=NumericScheme(name='observation11',
                      codes=tuple(sorted(('O1', 'O2', 'O3', 'O4', 'O5'))),
                      type_hint=FrozenDict11(  # type: ignore
                          dict(zip(('O1', 'O2', 'O3', 'O4', 'O5'), ('B', 'C', 'O', 'N', 'N')))))
)

TARGET_SCHEMES, TARGET_SCHEMES_MAPS = make_targets_schemes_with_maps(
    n_scheme_targets={
        'dx_discharge': 5,
        'obs': 5,
        'icu_procedures': 3,
        'icu_inputs': 3,
        'hosp_procedures': 3
    },
    source_schemes=SCHEMES
)

TARGET_SCHEMES_MAPS['icu_inputs'] = ReducedCodeMapN1.from_data(SCHEMES['icu_inputs'].name,
                                                               TARGET_SCHEMES['icu_inputs'].name,
                                                               TARGET_SCHEMES_MAPS['icu_inputs'].data,
                                                               FrozenDict11({c: 'w_sum' for c in
                                                                             TARGET_SCHEMES[
                                                                                 'icu_inputs'].codes}))
OUTCOME_EXTRACTOR = outcome_extractor(TARGET_SCHEMES['dx_discharge'])
DATASET_SCHEME_MANAGER = CodingSchemesManager(
    outcomes=(OUTCOME_EXTRACTOR,),
    schemes=tuple(SCHEMES.values()) + tuple(TARGET_SCHEMES.values()),
    maps=tuple(TARGET_SCHEMES_MAPS.values())
)

BINARY_OBSERVATION_CODE_INDEX = 0
CATEGORICAL_OBSERVATION_CODE_INDEX = 1
ORDINAL_OBSERVATION_CODE_INDEX = 2
NUMERIC_OBSERVATION_CODE_INDEX = 3

DATASET_SCHEME_CONF = DatasetSchemeConfig(ethnicity=SCHEMES['ethnicity'].name,
                                          gender=SCHEMES['gender'].name,
                                          dx_discharge=SCHEMES['dx_discharge'].name,
                                          icu_procedures=SCHEMES['icu_procedures'].name,
                                          icu_inputs=SCHEMES['icu_inputs'].name,
                                          obs=SCHEMES['obs'].name,
                                          hosp_procedures=SCHEMES['hosp_procedures'].name)
DATASET_CONFIG = DatasetConfig(scheme=DATASET_SCHEME_CONF, tables=DATASET_TABLES_CONF)
TVXEHR_SCHEME_CONF = TVxEHRSchemeConfig(ethnicity=SCHEMES['ethnicity'].name,
                                        gender=SCHEMES['gender'].name,
                                        dx_discharge=TARGET_SCHEMES['dx_discharge'].name,
                                        outcome=OUTCOME_EXTRACTOR.name,
                                        icu_procedures=SCHEMES['icu_procedures'].name,
                                        icu_inputs=TARGET_SCHEMES['icu_inputs'].name,
                                        obs=SCHEMES['obs'].name,
                                        hosp_procedures=SCHEMES['hosp_procedures'].name)

TVXEHR_CONF = TVxEHRConfig(scheme=TVXEHR_SCHEME_CONF, demographic=DemographicVectorConfig())


def leading_observables_extractor(observation_scheme: NumericScheme,
                                  leading_hours: tuple[float, ...] | list[float] = (1.0,),
                                  entry_neglect_window: float = 0.0,
                                  recovery_window: float = 0.0,
                                  minimum_acquisitions: int = 0,
                                  code_index: int = BINARY_OBSERVATION_CODE_INDEX) -> LeadingObservableExtractor:
    config = LeadingObservableExtractorConfig(observable_code=observation_scheme.codes[code_index],
                                              scheme=observation_scheme.name,
                                              entry_neglect_window=entry_neglect_window,
                                              recovery_window=recovery_window,
                                              minimum_acquisitions=minimum_acquisitions,
                                              leading_hours=leading_hours)
    return LeadingObservableExtractor(config=config, observable_scheme=observation_scheme)


def _inpatient_observables(observation_scheme: CodingScheme, n_timestamps: int):
    d = len(observation_scheme)
    timestamps_grid = np.linspace(0, LENGTH_OF_STAY, 1000, dtype=np.float64)
    t = np.array(sorted(nr.choice(timestamps_grid, replace=False, size=n_timestamps)))
    v = nr.randn(n_timestamps, d)
    mask = nr.binomial(1, 0.5, size=(n_timestamps, d)).astype(bool)
    return InpatientObservables(t, v, mask)


def _inpatient_interventions(hosp_proc, icu_proc, icu_inputs):
    return InpatientInterventions(hosp_proc, icu_proc, icu_inputs)


def _segmented_inpatient_interventions(inpatient_interventions: InpatientInterventions, hosp_proc_scheme,
                                       icu_proc_scheme,
                                       icu_inputs_scheme,
                                       maximum_padding: int = 1) -> SegmentedInpatientInterventions:
    assert all(isinstance(s, CodingScheme) for s in [hosp_proc_scheme, icu_proc_scheme, icu_inputs_scheme])
    return SegmentedInpatientInterventions.from_interventions(inpatient_interventions, LENGTH_OF_STAY,
                                                              hosp_procedures_size=len(hosp_proc_scheme),
                                                              icu_procedures_size=len(icu_proc_scheme),
                                                              icu_inputs_size=len(SCHEMES['icu_inputs']),
                                                              maximum_padding=maximum_padding)


def _admission(admission_id: str, admission_date: pd.Timestamp,
               dx_codes: CodesVector,
               dx_codes_history: CodesVector, outcome: CodesVector, observables: InpatientObservables,
               interventions: InpatientInterventions, leading_observable: InpatientObservables) -> Admission:
    discharge_date = pd.to_datetime(admission_date + pd.to_timedelta(LENGTH_OF_STAY, unit='hours'))

    return Admission(admission_id=admission_id, admission_dates=AdmissionDates(admission_date, discharge_date),
                     dx_codes=dx_codes,
                     dx_codes_history=dx_codes_history, outcome=outcome, observables=observables,
                     interventions=interventions, leading_observable=leading_observable)


def _admissions(n_admissions, dx_scheme: CodingScheme,
                outcome_extractor_: OutcomeExtractor, observation_scheme: NumericScheme,
                icu_inputs_scheme: CodingScheme, icu_proc_scheme: CodingScheme,
                hosp_proc_scheme: CodingScheme,
                dataset_scheme_manager: CodingSchemesManager) -> list[Admission]:
    admissions = []
    for i in range(n_admissions):
        dx_codes = _dx_codes(dx_scheme)
        obs = _inpatient_observables(observation_scheme, n_timestamps=nr.randint(0, 100))
        lead = leading_observables_extractor(observation_scheme=observation_scheme)(obs)
        icu_proc = _proc(icu_proc_scheme, n_timestamps=nr.randint(0, 50))
        hosp_proc = _proc(hosp_proc_scheme, n_timestamps=nr.randint(0, 50))
        icu_inputs = _icu_inputs(icu_inputs_scheme, n_timestamps=nr.randint(0, 50))

        admissions.append(_admission(admission_id=f'test_{i}', admission_date=pd.to_datetime('now'),
                                     dx_codes=dx_codes,
                                     dx_codes_history=_dx_codes_history(dx_codes),
                                     outcome=_outcome(outcome_extractor_, dataset_scheme_manager, dx_codes),
                                     observables=obs,
                                     interventions=_inpatient_interventions(hosp_proc=hosp_proc, icu_proc=icu_proc,
                                                                            icu_inputs=icu_inputs),
                                     leading_observable=lead))
    return admissions
