import random
from typing import cast

import numpy as np
import numpy.random as nr
import pandas as pd

import ehrax as rx
from ehrax import CodingSchemeWithUOM


UOM = ["m", "s", "g", "mg", "KG", "ml"]


def scheme(name: str, codes: list[str]) -> rx.CodingScheme:
    return rx.CodingScheme(name=name, codes=tuple(sorted(codes)), desc=rx.FrozenDict11(dict(zip(codes, codes))))


def scheme_with_uom(name: str, codes: list[str]) -> rx.CodingScheme:
    universal_unit = rx.FrozenDict11({c: random.choice(UOM) for c in codes})
    uom_normalization_factor = rx.FrozenDict1NM({c: {u.lower(): 1.0 for u in random.sample(UOM, k=3)} for c in codes})
    return rx.CodingSchemeWithUOM(
        name=name,
        codes=tuple(sorted(codes)),
        desc=rx.FrozenDict11(dict(zip(codes, codes))),
        uom_normalization_factor=uom_normalization_factor,
        universal_unit=universal_unit,
    )


def outcome_extractor(dx_scheme: rx.CodingScheme) -> rx.FilterOutcomeMapData:
    name = f"{dx_scheme.name}_outcome"
    k = max(3, len(dx_scheme) - 1)
    random.seed(0)
    excluded = tuple(random.sample(dx_scheme.codes, k=k))
    return rx.FilterOutcomeMapData(name=name, base_name=dx_scheme.name, exclude_codes=excluded)


def sample_codes(scheme: rx.CodingScheme, n: int) -> list[str]:
    codes = scheme.codes
    return random.choices(codes, k=n)


def _dx_codes(dx_scheme: rx.CodingScheme):
    v = nr.binomial(1, 0.5, size=len(dx_scheme)).astype(bool)
    return rx.CodesVector(vec=v, scheme=dx_scheme.name)


def inpatient_binary_input(n: int, p: int, los_hours: float):
    starttime = np.array(sorted(nr.choice(np.linspace(0, los_hours, max(n + 1, 1000)), replace=False, size=n)))
    endtime = starttime + nr.uniform(0, los_hours - starttime, size=(n,))
    code_index = nr.choice(p, size=n, replace=True)
    return rx.InpatientInput(starttime=starttime, endtime=endtime, code_index=code_index)


def inpatient_rated_input(n: int, p: int, los_hours: float):
    bin_input = inpatient_binary_input(n, p, los_hours)
    return rx.InpatientInput(
        starttime=bin_input.starttime,
        endtime=bin_input.endtime,
        code_index=bin_input.code_index,
        rate=nr.uniform(0, 1, size=(n,)),
    )


def _singular_codevec(scheme: rx.CodingScheme) -> rx.CodesVector:
    return scheme.codeset2vec({random.choice(scheme.codes)})


def _icu_inputs(icu_inputs_scheme: rx.CodingScheme, n_timestamps: int, los_hours: float):
    return inpatient_rated_input(n_timestamps, len(icu_inputs_scheme), los_hours=los_hours)


def _proc(scheme: rx.CodingScheme, n_timestamps: int, los_hours: float):
    return inpatient_binary_input(n_timestamps, len(scheme), los_hours=los_hours)


def demographic_vector_config() -> rx.DemographicVectorConfig:
    flags = random.choices([True, False], k=3)
    return rx.DemographicVectorConfig(*flags)


def date_of_birth() -> pd.Timestamp:
    return pd.to_datetime(pd.Timestamp("now") - pd.to_timedelta(nr.randint(0, 100 * 365), unit="D"))


def _static_info(ethnicity: rx.CodesVector, gender: rx.CodesVector) -> rx.StaticInfo:
    return rx.StaticInfo(ethnicity=ethnicity, gender=gender, date_of_birth=date_of_birth())


def _dx_codes_history(dx_codes: rx.CodesVector):
    v = nr.binomial(1, 0.5, size=len(dx_codes.vec)).astype(bool)
    return rx.CodesVector(vec=v + dx_codes.vec, scheme=dx_codes.scheme)


def _outcome(
    outcome_extractor_: rx.FilterOutcomeMapData,
    dataset_scheme_manager: rx.CodingSchemesManager,
    dx_codes: rx.CodesVector,
):
    source_codes = dataset_scheme_manager.scheme[dx_codes.scheme].vec2codeset(dx_codes.vec)
    extractor = dataset_scheme_manager.outcome[dx_codes.scheme, outcome_extractor_.name]
    return extractor(source_codes)


def sample_subjects_dataframe(
    n: int, ethnicity_scheme: rx.CodingScheme, gender_scheme: rx.CodingScheme
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            str(rx.COLUMN.subject_id): list(str(i) for i in range(n)),
            str(rx.COLUMN.race): random.choices(ethnicity_scheme.codes, k=n),
            str(rx.COLUMN.gender): random.choices(gender_scheme.codes, k=n),
            str(rx.COLUMN.date_of_birth): pd.to_datetime(
                random.choices(pd.date_range(start="1/1/1900", end="1/1/2000", freq="D"), k=n)
            ),
        }
    )


def sample_admissions_dataframe(subjects_df: pd.DataFrame, n: int, max_stay_days: int) -> pd.DataFrame:
    c_subject = str(rx.COLUMN.subject_id)
    c_admission = str(rx.COLUMN.admission_id)
    c_admission_time = str(rx.COLUMN.start_time)
    c_discharge_time = str(rx.COLUMN.end_time)
    admit_dates = pd.to_datetime(random.choices(pd.date_range(start="1/1/2000", end="1/1/2020", freq="D"), k=n))
    los = [random.uniform(0.5, max_stay_days * 0.99) for _ in range(n)]
    disch_dates = admit_dates + pd.to_timedelta(los, unit="D")

    return pd.DataFrame(
        {
            c_subject: random.choices(subjects_df[c_subject], k=n),
            c_admission: list(str(i) for i in range(n)),
            c_admission_time: admit_dates,
            c_discharge_time: disch_dates,
        }
    )


def sample_dx_dataframe(admissions_df: pd.DataFrame, dx_scheme: rx.CodingScheme, n: int) -> pd.DataFrame:
    dx_codes = sample_codes(dx_scheme, n)
    return pd.DataFrame(
        {
            str(rx.COLUMN.admission_id): random.choices(admissions_df[str(rx.COLUMN.admission_id)], k=n),
            str(rx.COLUMN.code): dx_codes,
        }
    )


def _sample_proc_dataframe(admissions_df: pd.DataFrame, scheme: rx.CodingScheme, n: int) -> pd.DataFrame:
    c_admission = str(rx.COLUMN.admission_id)
    c_admittime = str(rx.COLUMN.start_time)
    c_dischtime = str(rx.COLUMN.end_time)
    c_code = str(rx.COLUMN.code)
    c_start = str(rx.COLUMN.start_time)
    c_end = str(rx.COLUMN.end_time)
    codes = sample_codes(scheme, n)
    df_in = pd.DataFrame(
        {
            c_admission: random.choices(admissions_df[c_admission], k=n),
            c_code: codes,
            c_start: pd.Timestamp(0),
            c_end: pd.Timestamp(0),
        }
    )
    df = pd.merge(df_in, admissions_df[[c_admission, c_admittime, c_dischtime]], on=c_admission, suffixes=(None, "_y"))
    c_admittime = f"{c_admittime}_y" if c_admittime in df_in.columns else c_admittime
    c_dischtime = f"{c_dischtime}_y" if c_dischtime in df_in.columns else c_dischtime

    df["los"] = (df[c_dischtime] - df[c_admittime]).dt.total_seconds() / 3600
    relative_start = nr.uniform(0, df["los"].values.tolist(), size=n)
    df[c_start] = df[c_admittime] + pd.to_timedelta(relative_start, unit="hours")
    relative_end = nr.uniform(low=relative_start, high=df["los"].values.tolist(), size=n)
    df[c_end] = df[c_admittime] + pd.to_timedelta(relative_end, unit="hours")
    assert df[c_start].between(df[c_admittime], df[c_dischtime]).all()
    assert df[c_end].between(df[c_admittime], df[c_dischtime]).all()
    return df[[c_admission, c_code, c_start, c_end]]


def sample_icu_inputs_dataframe(admissions_df: pd.DataFrame, icu_input_scheme: rx.CodingScheme, n: int) -> pd.DataFrame:
    df = _sample_proc_dataframe(admissions_df, icu_input_scheme, n)
    c_amount = str(rx.COLUMN.amount)
    c_unit = str(rx.COLUMN.amount_unit)
    c_code = str(rx.COLUMN.code)
    df[c_amount] = np.random.uniform(low=0, high=1000, size=n)
    scheme = cast(CodingSchemeWithUOM, SCHEMES["icu_inputs"])
    normalizer = scheme.uom_normalization_factor
    units_map = {c: list(d.keys()) for c, d in normalizer.items()}
    df[c_unit] = df[c_code].map(lambda c: random.choice(units_map[c]))
    return df


def sample_obs_dataframe(admissions_df: pd.DataFrame, obs_scheme: rx.CodingScheme, n: int) -> pd.DataFrame:
    c_admission = str(rx.COLUMN.admission_id)
    c_admittime = str(rx.COLUMN.start_time)
    c_dischtime = str(rx.COLUMN.end_time)
    c_obs = str(rx.COLUMN.code)
    c_time = str(rx.COLUMN.time)
    c_value = str(rx.COLUMN.measurement)

    codes = sample_codes(obs_scheme, n)
    df_in = pd.DataFrame(
        {c_admission: random.choices(admissions_df[c_admission], k=n), c_obs: codes, c_time: pd.Timestamp(0)}
    )
    df = pd.merge(df_in, admissions_df[[c_admission, c_admittime, c_dischtime]], on=c_admission, suffixes=(None, "_y"))
    c_admittime = f"{c_admittime}_y" if c_admittime in df_in.columns else c_admittime
    c_dischtime = f"{c_dischtime}_y" if c_dischtime in df_in.columns else c_dischtime

    df["los"] = (df[c_dischtime] - df[c_admittime]).dt.total_seconds() / 3600
    relative_time = nr.uniform(0, df["los"].values.tolist(), size=n)
    df[c_time] = df[c_admittime] + pd.to_timedelta(relative_time, unit="hours")

    assert isinstance(obs_scheme, rx.NumericScheme), "Only numeric schemes are supported"
    df["obs_type"] = df[c_obs].map(obs_scheme.type_hint.data)
    df.loc[df.obs_type == "N", c_value] = np.random.uniform(low=0, high=1000, size=(df.obs_type == "N").sum())
    df.loc[df.obs_type.isin(("C", "O")), c_value] = random.choices([0, 1, 2], k=df.obs_type.isin(("C", "O")).sum())
    df.loc[df.obs_type == "B", c_value] = random.choices([0, 1], k=(df.obs_type == "B").sum())

    return df[[c_admission, c_obs, c_time, c_value]]


def _dataset_tables(
    dataset_scheme_config: rx.DatasetSchemeConfig,
    dataset_scheme_manager: rx.CodingSchemesManager,
    freqs: tuple[int, ...],
    max_stay_days: int,
) -> rx.DatasetTables:
    n_subjects, n_admission_per_subject, n_per_admission = freqs
    assert dataset_scheme_config.ethnicity is not None
    assert dataset_scheme_config.gender is not None
    assert dataset_scheme_config.dx_discharge is not None
    assert dataset_scheme_config.icu_procedures is not None
    assert dataset_scheme_config.icu_inputs is not None
    assert dataset_scheme_config.obs is not None
    assert dataset_scheme_config.hosp_procedures is not None
    subjects_df = sample_subjects_dataframe(
        n_subjects,
        dataset_scheme_manager.scheme[dataset_scheme_config.ethnicity],
        dataset_scheme_manager.scheme[dataset_scheme_config.gender],
    )
    admissions_df = sample_admissions_dataframe(subjects_df, n_admission_per_subject * n_subjects, max_stay_days)
    dx_df = sample_dx_dataframe(
        admissions_df,
        dataset_scheme_manager.scheme[dataset_scheme_config.dx_discharge],
        n_per_admission * n_subjects * n_admission_per_subject,
    )

    obs_df = sample_obs_dataframe(
        admissions_df,
        dataset_scheme_manager.scheme[dataset_scheme_config.obs],
        n_per_admission * n_subjects * n_admission_per_subject,
    )

    icu_proc_df = _sample_proc_dataframe(
        admissions_df,
        dataset_scheme_manager.scheme[dataset_scheme_config.icu_procedures],
        n_per_admission * n_subjects * n_admission_per_subject,
    )

    hosp_proc_df = _sample_proc_dataframe(
        admissions_df,
        dataset_scheme_manager.scheme[dataset_scheme_config.hosp_procedures],
        n_per_admission * n_subjects * n_admission_per_subject,
    )

    icu_inputs_df = sample_icu_inputs_dataframe(
        admissions_df,
        dataset_scheme_manager.scheme[dataset_scheme_config.icu_inputs],
        n_per_admission * n_subjects * n_admission_per_subject,
    )

    return rx.DatasetTables(
        static=subjects_df,
        admissions=admissions_df,
        dx_discharge=dx_df,
        obs=obs_df,
        icu_procedures=icu_proc_df,
        hosp_procedures=hosp_proc_df,
        icu_inputs=icu_inputs_df,
    )


def make_targets_schemes_with_maps(
    n_scheme_targets: dict[str, int], source_schemes: dict[str, rx.CodingScheme]
) -> tuple[dict[str, rx.CodingScheme], dict[str, rx.CodeMap]]:
    def make_target_scheme_with_map(
        size: int, space: str, source_scheme: rx.CodingScheme
    ) -> tuple[str, rx.CodingScheme, rx.CodeMap]:
        assert size <= len(source_scheme)
        target_name = f"{source_scheme.name}_target"
        target_codes = tuple(f"{source_scheme}_target_{i}" for i in range(size))
        target_desc = rx.FrozenDict11(dict(zip(target_codes, target_codes)))
        target_scheme = rx.CodingScheme(name=target_name, codes=target_codes, desc=target_desc)
        mapp = {c: {t} for c, t in zip(source_scheme.codes, target_codes)}
        mapp |= {c: {random.choice(target_codes)} for c in source_scheme.codes[len(target_codes) :]}
        map_data = rx.FrozenDict1N({c: {random.choice(target_codes)} for c in source_scheme.codes})
        code_map = rx.CodeMap(source_name=source_scheme.name, target_name=target_name, data=map_data)
        return space, target_scheme, code_map

    space, schemes, maps = zip(
        *list(
            make_target_scheme_with_map(size, space, source_schemes[space]) for space, size in n_scheme_targets.items()
        )
    )
    return dict(zip(space, schemes)), dict(zip(space, maps))


SCHEMES: dict[str, rx.CodingScheme] = dict(
    ethnicity=scheme("ethnicity", ["E1", "E2", "E3"]),
    gender=scheme("genderrrr", ["M", "F"]),
    dx_discharge=scheme("dx1", ["Dx1", "Dx2", "Dx3", "Dx4", "Dx5", "Dx6", "Dx7", "Dx8", "Dx9", "Dx10"]),
    hosp_procedures=scheme("hosp_proc1", ["HP1", "HP2", "HP3", "HP4", "HP5", "HP6"]),
    icu_procedures=scheme("icu_proc2", ["ICU1", "ICU2", "ICU3", "ICU4", "ICU5", "ICU6"]),
    icu_inputs=scheme_with_uom("icu_inputs", ["ICUI1", "ICUI2", "ICUI3", "ICUI4", "ICUI5", "ICUI6"]),
    obs=rx.NumericScheme(
        name="observation11",
        codes=tuple(sorted(("O1", "O2", "O3", "O4", "O5"))),
        type_hint=rx.FrozenDict11(  # type: ignore
            dict(zip(("O1", "O2", "O3", "O4", "O5"), ("B", "C", "O", "N", "N")))
        ),
    ),
)

TARGET_SCHEMES, TARGET_SCHEMES_MAPS = make_targets_schemes_with_maps(
    n_scheme_targets={"dx_discharge": 5, "obs": 5, "icu_procedures": 3, "icu_inputs": 3, "hosp_procedures": 3},
    source_schemes=SCHEMES,
)

TARGET_SCHEMES_MAPS["icu_inputs"] = rx.ReducedCodeMapN1.from_data(
    SCHEMES["icu_inputs"].name,
    TARGET_SCHEMES["icu_inputs"].name,
    TARGET_SCHEMES_MAPS["icu_inputs"].data,
    rx.FrozenDict11({c: "w_sum" for c in TARGET_SCHEMES["icu_inputs"].codes}),
)
OUTCOME_DATA = outcome_extractor(TARGET_SCHEMES["dx_discharge"])
DATASET_SCHEME_MANAGER = rx.CodingSchemesManager(
    outcomes=(OUTCOME_DATA,),
    schemes=tuple(SCHEMES.values()) + tuple(TARGET_SCHEMES.values()),
    maps=tuple(TARGET_SCHEMES_MAPS.values()),
)

BINARY_OBSERVATION_CODE_INDEX = 0
CATEGORICAL_OBSERVATION_CODE_INDEX = 1
ORDINAL_OBSERVATION_CODE_INDEX = 2
NUMERIC_OBSERVATION_CODE_INDEX = 3

DATASET_SCHEME_CONF = rx.DatasetSchemeConfig(
    ethnicity=SCHEMES["ethnicity"].name,
    gender=SCHEMES["gender"].name,
    dx_discharge=SCHEMES["dx_discharge"].name,
    icu_procedures=SCHEMES["icu_procedures"].name,
    icu_inputs=SCHEMES["icu_inputs"].name,
    obs=SCHEMES["obs"].name,
    hosp_procedures=SCHEMES["hosp_procedures"].name,
)
DATASET_CONFIG = rx.DatasetConfig(scheme=DATASET_SCHEME_CONF)
TVXEHR_SCHEME_CONF = rx.TVxEHRSchemeConfig(
    ethnicity=SCHEMES["ethnicity"].name,
    gender=SCHEMES["gender"].name,
    dx_discharge=TARGET_SCHEMES["dx_discharge"].name,
    outcome=OUTCOME_DATA.name,
    icu_procedures=SCHEMES["icu_procedures"].name,
    icu_inputs=TARGET_SCHEMES["icu_inputs"].name,
    obs=SCHEMES["obs"].name,
    hosp_procedures=SCHEMES["hosp_procedures"].name,
)

TVXEHR_CONF = rx.TVxEHRConfig(scheme=TVXEHR_SCHEME_CONF, demographic=rx.DemographicVectorConfig())


def leading_observables_extractor(
    observation_scheme: rx.NumericScheme,
    leading_hours: tuple[float, ...] | list[float] = (1.0,),
    entry_neglect_window: float = 0.0,
    recovery_window: float = 0.0,
    minimum_acquisitions: int = 0,
    code_index: int = BINARY_OBSERVATION_CODE_INDEX,
) -> rx.LeadingObservableExtractor:
    config = rx.LeadingObservableExtractorConfig(
        observable_code=observation_scheme.codes[code_index],
        scheme=observation_scheme.name,
        entry_neglect_window=entry_neglect_window,
        recovery_window=recovery_window,
        minimum_acquisitions=minimum_acquisitions,
        leading_hours=leading_hours,
    )
    return rx.LeadingObservableExtractor(config=config, observable_scheme=observation_scheme)


def _inpatient_observables(observation_scheme: rx.CodingScheme, n_timestamps: int, los_hours: float):
    d = len(observation_scheme)
    timestamps_grid = np.linspace(0, los_hours, max(n_timestamps + 1, 1000), dtype=np.float64)
    t = np.array(sorted(nr.choice(timestamps_grid, replace=False, size=n_timestamps)))
    v = nr.randn(n_timestamps, d)
    mask = nr.binomial(1, 0.5, size=(n_timestamps, d)).astype(bool)
    return rx.InpatientObservables(t, v, mask)


def _inpatient_interventions(hosp_proc, icu_proc, icu_inputs):
    return rx.InpatientInterventions(hosp_proc, icu_proc, icu_inputs)


def _segmented_inpatient_interventions(
    inpatient_interventions: rx.InpatientInterventions,
    hosp_proc_scheme,
    icu_proc_scheme,
    icu_inputs_scheme,
    max_los_hours: float,
    maximum_padding: int = 1,
) -> rx.SegmentedInpatientInterventions:
    assert all(isinstance(s, rx.CodingScheme) for s in [hosp_proc_scheme, icu_proc_scheme, icu_inputs_scheme])
    return rx.SegmentedInpatientInterventions.from_interventions(
        inpatient_interventions,
        max_los_hours,
        hosp_procedures_size=len(hosp_proc_scheme),
        icu_procedures_size=len(icu_proc_scheme),
        icu_inputs_size=len(SCHEMES["icu_inputs"]),
        maximum_padding=maximum_padding,
    )


def _admission(
    admission_id: str,
    admission_date: pd.Timestamp,
    dx_codes: rx.CodesVector,
    dx_codes_history: rx.CodesVector,
    outcome: rx.CodesVector,
    observables: rx.InpatientObservables,
    interventions: rx.InpatientInterventions,
    leading_observable: rx.InpatientObservables,
    los_days: float,
) -> rx.Admission:
    discharge_date = pd.to_datetime(admission_date + pd.to_timedelta(los_days, unit="D") + pd.to_timedelta(1, unit="S"))

    return rx.Admission(
        admission_id=admission_id,
        admission_dates=rx.AdmissionDates(admission_date, discharge_date),
        dx_codes=dx_codes,
        dx_codes_history=dx_codes_history,
        outcome=outcome,
        observables=observables,
        interventions=interventions,
        leading_observable=leading_observable,
    )


def _admissions(
    n_admissions,
    dx_scheme: rx.CodingScheme,
    outcome_extractor_: rx.FilterOutcomeMap,
    observation_scheme: rx.NumericScheme,
    icu_inputs_scheme: rx.CodingScheme,
    icu_proc_scheme: rx.CodingScheme,
    hosp_proc_scheme: rx.CodingScheme,
    dataset_scheme_manager: rx.CodingSchemesManager,
    max_los_days: int,
    max_n_timestamps_obs: int,
    max_n_inputs: int,
) -> list[rx.Admission]:
    admissions = []
    multiplier = list(sorted(random.sample(range(10 * n_admissions), k=n_admissions)))
    admission_dates = [pd.to_datetime("now") + pd.to_timedelta(max_los_days * m, unit="D") for m in multiplier]
    for i, admission_date in enumerate(admission_dates):
        los_days = random.uniform(0, max_los_days)
        los_h = los_days * 24.0
        dx_codes = _dx_codes(dx_scheme)
        obs = _inpatient_observables(
            observation_scheme, n_timestamps=nr.randint(0, max_n_timestamps_obs), los_hours=los_h
        )
        lead = leading_observables_extractor(observation_scheme=observation_scheme)(obs)
        icu_proc = _proc(icu_proc_scheme, n_timestamps=nr.randint(0, max_n_inputs), los_hours=los_h)
        hosp_proc = _proc(hosp_proc_scheme, n_timestamps=nr.randint(0, max_n_inputs), los_hours=los_h)
        icu_inputs = _icu_inputs(icu_inputs_scheme, n_timestamps=nr.randint(0, max_n_inputs), los_hours=los_h)
        admissions.append(
            _admission(
                admission_id=f"test_{i}",
                admission_date=admission_date,
                dx_codes=dx_codes,
                dx_codes_history=_dx_codes_history(dx_codes),
                outcome=_outcome(outcome_extractor_, dataset_scheme_manager, dx_codes),
                observables=obs,
                los_days=los_days,
                interventions=_inpatient_interventions(hosp_proc=hosp_proc, icu_proc=icu_proc, icu_inputs=icu_inputs),
                leading_observable=lead,
            )
        )
    return admissions
