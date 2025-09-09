"""."""

import dataclasses
import enum
import logging
import random
from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import field
from datetime import datetime
from functools import cached_property
from typing import ClassVar, Final, Self

import equinox as eqx
import numpy as np
import pandas as pd

from ._literals import NumericalTypeHint, SplitLiteral
from ._stats.dataset import DatasetStatsInterface, MultiDatasetsStatsInterface, TwoDatasetsStatsInterface
from .base import AbstractConfig, AbstractVxData, HDFVirtualNode
from .coding_scheme import CodingScheme, CodingSchemesManager, NumericScheme
from .utils import tqdm_constructor

SECONDS_TO_HOURS_SCALER: Final[float] = 1 / 3600.0  # convert seconds to hours
SECONDS_TO_DAYS_SCALER: Final[float] = SECONDS_TO_HOURS_SCALER * 1 / 24.0  # convert seconds to days


# This Enum will be used as a reference to ensure consistent column names
# across the multiple (relational) columns representing a single dataset.
class COLUMN(enum.StrEnum):
    subject_id = enum.auto()
    admission_id = enum.auto()
    gender = enum.auto()
    race = enum.auto()
    date_of_birth = enum.auto()
    anchor_year = enum.auto()
    anchor_age = enum.auto()
    code = enum.auto()
    version = enum.auto()
    description = enum.auto()
    time = enum.auto()  # for singly timestamped events.
    measurement = enum.auto()
    start_time = enum.auto()  # for events defined by intervals rather than a single timestamp
    end_time = enum.auto()  # ..
    amount = enum.auto()
    amount_unit = enum.auto()
    derived_unit_normalization_factor = enum.auto()
    derived_universal_unit = enum.auto()
    derived_normalized_amount = enum.auto()
    derived_normalized_amount_per_hour = enum.auto()
    mapped_code = enum.auto()
    mapped_description = enum.auto()

    @property
    def is_time(self):
        return self in (
            COLUMN.time,
            COLUMN.start_time,
            COLUMN.end_time,
            COLUMN.date_of_birth,
            COLUMN.anchor_year,
        )

    @property
    def is_code(self):
        return self in (COLUMN.code, COLUMN.mapped_code)

    @property
    def is_id(self):
        return self in (COLUMN.subject_id, COLUMN.admission_id)

    @staticmethod
    def as_dict() -> dict[str, str]:
        """
        Returns a dictionary representation of the enum member.
        """
        return {str(c): c.value for c in COLUMN}


class TableColumns(AbstractConfig):
    def __check_init__(self):
        self.validate()

    def validate(self):
        assert all(k in COLUMN and k == v for k, v in self.as_dict().items()), f"Fields must be one of {COLUMN}."

    def as_tuple(self) -> tuple[str, ...]:
        return tuple(self.as_dict().values())

    @property
    def id_cols(self) -> tuple[str, ...]:
        return tuple(v for k, v in self.as_one_level_dict().items() if COLUMN[k].is_id)

    def __contains__(self, item: str | COLUMN):
        if isinstance(item, str):
            return item in self.as_one_level_dict()
        elif isinstance(item, COLUMN):
            return str(item) in self.as_one_level_dict()
        else:
            raise ValueError(f"Unsupported type {type(item)}")

    @property
    def index(self) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(self) if f.metadata.get("index", False))

    @property
    def time_cols(self) -> tuple[str, ...]:
        return tuple(v for k, v in self.as_dict().items() if COLUMN[k].is_time)

    @property
    def code_cols(self) -> tuple[str, ...]:
        return tuple(v for k, v in self.as_dict().items() if COLUMN[k].is_code)


class StaticTableColumns(TableColumns):
    subject_id: str = field(default=str(COLUMN.subject_id), metadata={"index": True})
    race: str = str(COLUMN.race)
    gender: str = str(COLUMN.gender)
    date_of_birth: str = str(COLUMN.date_of_birth)


class AdmissionsTableColumns(TableColumns):
    subject_id: str = str(COLUMN.subject_id)
    admission_id: str = field(default=str(COLUMN.admission_id), metadata={"index": True})
    start_time: str = str(COLUMN.start_time)
    end_time: str = str(COLUMN.end_time)


class AdmissionSummaryTableColumns(TableColumns):
    admission_id: str = str(COLUMN.admission_id)
    code: str = str(COLUMN.code)
    description: str = str(COLUMN.description)


class AdmissionTimeSeriesTableColumns(TableColumns):
    admission_id: str = str(COLUMN.admission_id)
    code: str = str(COLUMN.code)
    time: str = str(COLUMN.time)
    measurement: str = str(COLUMN.measurement)
    description: str = str(COLUMN.description)


class AdmissionIntervalEventsTableColumns(TableColumns):
    admission_id: str = str(COLUMN.admission_id)
    code: str = str(COLUMN.code)
    description: str = str(COLUMN.description)
    start_time: str = str(COLUMN.start_time)
    end_time: str = str(COLUMN.end_time)


class AdmissionIntervalRatesTableColumns(TableColumns):
    admission_id: str = str(COLUMN.admission_id)
    code: str = str(COLUMN.code)
    description: str = str(COLUMN.description)
    start_time: str = str(COLUMN.start_time)
    end_time: str = str(COLUMN.end_time)
    amount: str = str(COLUMN.amount)
    amount_unit: str = str(COLUMN.amount_unit)
    derived_unit_normalization_factor: str = str(COLUMN.derived_unit_normalization_factor)
    derived_universal_unit: str = str(COLUMN.derived_universal_unit)
    derived_normalized_amount: str = str(COLUMN.derived_normalized_amount)
    derived_normalized_amount_per_hour: str = str(COLUMN.derived_normalized_amount_per_hour)


class MultivariateTimeSeriesTableMeta(AbstractConfig):
    name: str
    attributes: tuple[str, ...]
    type_hint: tuple[NumericalTypeHint, ...]
    default_type_hint: NumericalTypeHint

    def __init__(
        self,
        name: str,
        attributes: tuple[str, ...],
        type_hint: tuple[NumericalTypeHint, ...] | None = None,
        default_type_hint: NumericalTypeHint = "N",
    ):
        self.name = name
        self.attributes = attributes
        self.default_type_hint = default_type_hint
        self.type_hint = type_hint or ((default_type_hint,) * len(attributes))

    def __check_init__(self):
        assert len(self.attributes) == len(self.type_hint), (
            f"Length of attributes and type_hint must be the same. Got {len(self.attributes)} and "
            f"{len(self.type_hint)}."
        )
        assert all(t in ("N", "C", "B", "O") for t in self.type_hint), (
            f"type hint must be one of 'N', 'C', 'B', 'O'. Got {self.type_hint}."
        )


class DatasetColumns(AbstractConfig):
    static: StaticTableColumns
    admissions: AdmissionsTableColumns
    dx_discharge: AdmissionSummaryTableColumns
    obs: AdmissionTimeSeriesTableColumns
    icu_procedures: AdmissionIntervalEventsTableColumns
    icu_inputs: AdmissionIntervalRatesTableColumns
    hosp_procedures: AdmissionIntervalEventsTableColumns

    def __init__(
        self,
        static: StaticTableColumns = StaticTableColumns(),
        admissions: AdmissionsTableColumns = AdmissionsTableColumns(),
        dx_discharge: AdmissionSummaryTableColumns = AdmissionSummaryTableColumns(),
        obs: AdmissionTimeSeriesTableColumns = AdmissionTimeSeriesTableColumns(),
        icu_procedures: AdmissionIntervalEventsTableColumns = AdmissionIntervalEventsTableColumns(),
        icu_inputs: AdmissionIntervalRatesTableColumns = AdmissionIntervalRatesTableColumns(),
        hosp_procedures: AdmissionIntervalEventsTableColumns = AdmissionIntervalEventsTableColumns(),
    ):
        self.static = static
        self.admissions = admissions
        self.dx_discharge = dx_discharge
        self.obs = obs
        self.icu_procedures = icu_procedures
        self.icu_inputs = icu_inputs
        self.hosp_procedures = hosp_procedures

    def __check_init__(self):
        self.validate()

    def validate(self):
        assert all(isinstance(v, TableColumns) for v in self.as_one_level_dict().values())
        for v in self.as_one_level_dict().values():
            v.validate()
        column_names = defaultdict(set)
        for v in self.as_dict().values():
            for k, v in v.items():
                assert k == v
                column_names[k].add(v)
        for k, v in column_names.items():
            if len(v) > 1:
                raise ValueError(f"Column {k} is present with different names: {v}")

    def columns_dict(self) -> dict[str, tuple[str, ...]]:
        return {k: v.as_dict() for k, v in self.as_one_level_dict().items()}

    @property
    def admission_id(self) -> str:
        return self.admissions.admission_id

    @property
    def subject_id(self) -> str:
        return self.static.subject_id

    @property
    def timestamped_tables_config_dict(self):
        return {k: v for k, v in self.as_one_level_dict().items() if COLUMN.time in v.as_dict().keys()}

    @property
    def interval_based_table_config_dict(self):
        return {
            k: v
            for k, v in self.as_one_level_dict().items()
            if {COLUMN.start_time, str(COLUMN.end_time)}.issubset(set(v.as_dict().keys()))
        }

    @property
    def indices(self) -> dict[str, str]:
        return {k: v.index for k, v in self.as_one_level_dict().items() if len(v.index) > 0}

    @property
    def time_cols(self) -> dict[str, tuple[str, ...]]:
        return {k: v.time_cols for k, v in self.as_one_level_dict().items() if len(v.time_cols) > 0}

    @property
    def code_column(self) -> dict[str, str]:
        return {k: v.code_cols for k, v in self.as_one_level_dict().items() if len(v.code_cols) > 0}

    def temporal_admission_linked_table(self, table_name: str) -> bool:
        conf = getattr(self, table_name)
        return len(conf.time_cols) > 0 and COLUMN.admission_id in conf and str(COLUMN.admission_id) not in conf.index


class DatasetTables(AbstractVxData):
    static: pd.DataFrame
    admissions: pd.DataFrame
    dx_discharge: pd.DataFrame | None
    obs: pd.DataFrame | None
    icu_procedures: pd.DataFrame | None
    icu_inputs: pd.DataFrame | None
    hosp_procedures: pd.DataFrame | None

    def __init__(
        self,
        static: pd.DataFrame,
        admissions: pd.DataFrame,
        dx_discharge: pd.DataFrame | None = None,
        obs: pd.DataFrame | None = None,
        icu_procedures: pd.DataFrame | None = None,
        icu_inputs: pd.DataFrame | None = None,
        hosp_procedures: pd.DataFrame | None = None,
    ):
        self.static = static
        self.admissions = admissions
        self.dx_discharge = dx_discharge
        self.obs = obs
        self.icu_procedures = icu_procedures
        self.icu_inputs = icu_inputs
        self.hosp_procedures = hosp_procedures

    def __check_init__(self):
        if isinstance(self.admissions, HDFVirtualNode):
            return
        if COLUMN.admission_id in self.admissions.columns:
            admission_id = self.admissions[COLUMN.admission_id]
        elif self.admissions.index.name == str(COLUMN.admission_id):
            admission_id = self.admissions.index
        else:
            raise ValueError(
                f"Where is the admission_id? columns: {self.admissions.columns}. Index: {self.admissions.index.name}"
            )

        assert admission_id.nunique() == len(admission_id), (
            "Admission IDs in MIMIC-III and MIMIC-IV were found to be globally unique, i.e. two patients cannot share "
            "an admission ID. This allowed simpler dataset representation. Since we have the admissions table listing "
            "both subject_id and admission_id (unique 1:1 relation), no need to include the subject_id in other tables."
            "However, in the future there might be a need for an extension/adaptation to deal with non-unique "
            "admission IDs, where both (subject_id, admission_id) are needed for identification. "
            "In case you are getting this error message, you can either rewrite all admission_ids of your dataset "
            "tables to be globally unique, or, if not an urgent request, you can post an Issue "
            "at the repository of this code or email at: (asem.a.abdelaziz@proton.me). "
            "TODO: fix this potential limitation. Refer to this issue in the comments with ISSUE_ADM_UNIQ"
        )

    @property
    def tables_dict(self) -> dict[str, pd.DataFrame]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, pd.DataFrame)}


class DatasetSchemeConfig(AbstractConfig):
    ethnicity: str | None
    gender: str | None
    dx_discharge: str | None
    obs: str | None
    icu_procedures: str | None
    hosp_procedures: str | None
    icu_inputs: str | None

    def __init__(
        self,
        ethnicity: str | None = None,
        gender: str | None = None,
        dx_discharge: str | None = None,
        obs: str | None = None,
        icu_procedures: str | None = None,
        hosp_procedures: str | None = None,
        icu_inputs: str | None = None,
    ):
        self.ethnicity = ethnicity
        self.gender = gender
        self.dx_discharge = dx_discharge
        self.obs = obs
        self.icu_procedures = icu_procedures
        self.hosp_procedures = hosp_procedures
        self.icu_inputs = icu_inputs

    def scheme_fields(self) -> dict[str, str | None]:
        return {
            "gender": self.gender,
            "ethnicity": self.ethnicity,
            "dx_discharge": self.dx_discharge,
            "obs": self.obs,
            "icu_inputs": self.icu_inputs,
            "icu_procedures": self.icu_procedures,
            "hosp_procedures": self.hosp_procedures,
        }


@dataclasses.dataclass
class DatasetSchemeProxy:
    """
    Represents a dataset scheme that defines the coding schemes and outcome extractor for a dataset.

    Attributes:
        config (DatasetSchemeConfig): the configuration for the dataset scheme.

    Methods:
        __init__(self, config: DatasetSchemeConfig, **kwargs): initializes a new instance of the DatasetScheme class.
        scheme_dict(self): returns a dictionary of the coding schemes in the dataset scheme.
        make_target_scheme_config(self, **kwargs): creates a new target scheme configuration based on the
            current scheme.
        make_target_scheme(self, config=None, **kwargs): creates a new target scheme based on the current scheme.
        demographic_vector_size(self, demographic_vector_config: DemographicVectorConfig): calculates the size of
            the demographic vector.
        dx_mapper(self, target_scheme: DatasetScheme): returns the mapper for the diagnosis coding scheme to the
            corresponding target scheme.
        ethnicity_mapper(self, target_scheme: DatasetScheme): returns the mapper for the ethnicity coding scheme to
            the corresponding target scheme.
        supported_target_scheme_options(self): returns the supported target scheme options for each coding scheme.
    """

    config: DatasetSchemeConfig
    schemes_context: CodingSchemesManager

    def __init__(self, config: DatasetSchemeConfig, schemes_context: CodingSchemesManager):
        self.config = config
        self.schemes_context = schemes_context

    def _scheme(self, name: str | None) -> CodingScheme | None:
        if name is None:
            return None
        try:
            return self.schemes_context.scheme[name]
        except KeyError:
            return None

    @property
    def ethnicity(self) -> CodingScheme | None:
        return self._scheme(self.config.ethnicity)

    @property
    def gender(self) -> CodingScheme | None:
        return self._scheme(self.config.gender)

    @property
    def dx_discharge(self) -> CodingScheme | None:
        return self._scheme(self.config.dx_discharge)

    @property
    def obs(self) -> NumericScheme | None:
        s = self._scheme(self.config.obs)
        assert s is None or isinstance(s, NumericScheme), f"Observation scheme must be numeric. Got {type(s)}."
        return s

    @property
    def icu_procedures(self) -> CodingScheme | None:
        return self._scheme(self.config.icu_procedures)

    @property
    def hosp_procedures(self) -> CodingScheme | None:
        return self._scheme(self.config.hosp_procedures)

    @property
    def icu_inputs(self) -> CodingScheme | None:
        s = self._scheme(self.config.icu_inputs)
        return s

    @property
    def scheme_dict(self):
        return {k: self._scheme(v) for k, v in self.config.scheme_fields().items() if self._scheme(v) is not None}


class ReportAttributes(AbstractConfig):
    transformation: str | None = None
    operation: str | None = None
    table: str | None = None
    column: str | None = None
    value_type: str | None = None
    before: str | int | float | bool | None = None
    after: str | int | float | bool | None = None
    timestamp: str | None = None

    def __init__(
        self,
        transformation: str | type | None = None,
        operation: str | None = None,
        table: str | None = None,
        column: str | None = None,
        value_type: str | type | None = None,
        before: str | int | float | bool | type | None = None,
        after: str | int | float | bool | type | None = None,
        timestamp: str | None = None,
    ):
        def strtype(x):
            if x is None:
                return x
            elif isinstance(x, type):
                return x.__name__
            elif isinstance(x, np.dtype):
                return x.name
            else:
                return x

        self.transformation = strtype(transformation)
        self.operation = operation
        self.table = table
        self.column = column
        self.value_type = strtype(value_type)
        self.before = strtype(before)
        self.after = strtype(after)
        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp = datetime.now().isoformat()


class PipelineReportTable(pd.DataFrame):
    # We need to exclude the timestamps of the steps from the equality tests.
    def equals(self, other: object) -> bool:  # type: ignore[override]
        assert isinstance(other, pd.DataFrame), f"Can only compare with pd.DataFrame. Got {type(other)}."
        # Exclude timestamps from comparison.
        report = self
        if all("timestamp" in r for r in (self.columns, other.columns)):
            report = report.drop(columns=["timestamp"])
            other = other.drop(columns=["timestamp"])
        return pd.DataFrame.equals(report, other)


class Report(AbstractConfig):
    incidents: tuple[ReportAttributes, ...]
    incident_class: ClassVar[type[ReportAttributes]] = ReportAttributes

    def __init__(self, incidents: tuple[ReportAttributes, ...] = ()):
        self.incidents = incidents

    def __add__(self, other: Self) -> Self:
        return type(self)(incidents=self.incidents + other.incidents)

    def add(self, *args, **kwargs) -> Self:
        return type(self)(incidents=self.incidents + (self.incident_class(*args, **kwargs),))

    def __getitem__(self, item) -> ReportAttributes:
        return self.incidents[item]

    def __iter__(self) -> Iterator[ReportAttributes]:
        return iter(self.incidents)

    def __len__(self) -> int:
        return len(self.incidents)

    def compile(self, previous_report: pd.DataFrame | None = None) -> PipelineReportTable:
        report = self.incidents
        if len(report) == 0:
            report = (ReportAttributes(transformation="identity"),)

        df = pd.DataFrame([x.as_dict() for x in report]).astype(str)
        object_columns = [c for c in df.columns if df[c].dtype == "object"]
        type_rows = df["value_type"] == "dtype"
        type_cols = ["after", "before"]
        nan_mask = df.loc[:, object_columns].isnull() | df.loc[:, object_columns].isin((None, "nan", "NaN", "None"))
        df.loc[:, object_columns] = df.loc[:, object_columns].where(~nan_mask, "-")
        df.loc[type_rows, type_cols] = df.loc[type_rows, type_cols].map(lambda x: f"{x}_type")
        if previous_report is None:
            return PipelineReportTable(df)
        else:
            return PipelineReportTable(pd.concat([previous_report, df], ignore_index=True, axis=0, sort=False))


class AbstractDataset(AbstractVxData, ABC):
    config: AbstractConfig
    report_class: ClassVar[type[Report]] = Report

    @abstractmethod
    def scheme_proxy(self, schemes_context: CodingSchemesManager) -> DatasetSchemeProxy: ...


class AbstractTransformation[DType: AbstractDataset, RType: Report](eqx.Module):
    @classmethod
    @abstractmethod
    def apply(cls, dataset: DType, schemes_context: CodingSchemesManager, report: RType, /) -> tuple[DType, RType]:
        raise NotImplementedError

    @classmethod
    def skip(cls, dataset: DType, report: RType, reason: str = "") -> tuple[DType, RType]:
        return dataset, report.add(transformation=cls, operation=": ".join(("skip", reason)))


class AbstractDatasetPipelineConfig(AbstractConfig):
    pass


class AbstractDatasetPipeline(AbstractVxData, metaclass=ABCMeta):
    config: AbstractDatasetPipelineConfig
    transformations: list[AbstractTransformation]
    report_class: ClassVar[type[Report]] = Report

    def __init__(
        self,
        config: AbstractDatasetPipelineConfig = AbstractDatasetPipelineConfig(),
        *,
        transformations: list[AbstractTransformation],
    ):
        self.config = config
        self.transformations = transformations


class AbstractProcessedDataset(AbstractDataset):
    pipeline_report: PipelineReportTable

    @property
    def pipeline_executed(self) -> bool:
        return len(self.pipeline_report) != 0

    def execute_pipeline(self, pipeline: AbstractDatasetPipeline, schemes_context: CodingSchemesManager) -> Self:
        if self.pipeline_executed:
            logging.warning("A pipeline has already been executed. Doing nothing.")
            return self
        return self._execute_pipeline(pipeline.transformations, schemes_context)

    def _execute_pipeline(
        self, transformations: list[AbstractTransformation], schemes_context: CodingSchemesManager
    ) -> Self:
        if self.pipeline_executed:
            logging.warning("A pipeline has already been executed. This will replace the pipeline execution history.")
        report = self.report_class()
        dataset = self
        with tqdm_constructor(desc="Transforming Dataset", unit="transformations", total=len(transformations)) as pbar:
            for t in transformations:
                pbar.set_description(f"Transforming Dataset: {type(t).__name__}")
                report = report.add(transformation=type(t), operation="start")
                dataset, report = t.apply(dataset, schemes_context, report)
                report = report.add(transformation=type(t), operation="end")
                pbar.update(1)

        return eqx.tree_at(lambda x: x.pipeline_report, dataset, report.compile(dataset.pipeline_report))


class DatasetConfig(AbstractConfig):
    scheme: DatasetSchemeConfig
    columns: DatasetColumns
    select_subjects_with_observation: str | None
    select_subjects_with_short_admissions: float | None  # number of days.
    admission_minimum_los: float | None

    def __init__(
        self,
        scheme: DatasetSchemeConfig,
        columns: DatasetColumns = DatasetColumns(),
        select_subjects_with_observation: str | None = None,
        select_subjects_with_short_admissions: float | None = None,
        admission_minimum_los: float | None = None,
    ):
        self.scheme = scheme
        self.columns = columns
        self.select_subjects_with_observation = select_subjects_with_observation
        self.select_subjects_with_short_admissions = select_subjects_with_short_admissions
        self.admission_minimum_los = admission_minimum_los


class Dataset(AbstractProcessedDataset):
    """
    A class representing a dataset.

    Attributes:
        config (DatasetConfig): the configuration object for the dataset.

    Methods:
        __init__(self, config: DatasetConfig = None, config_path: str = None, **kwargs): initializes the Dataset object.
        supported_target_scheme_options(self): returns the supported target scheme options.
        to_subjects(self, **kwargs): converts the dataset to subject objects.
        save(self, path: Union[str, Path], overwrite: bool = False): saves the dataset to disk.
        load(cls, path: Union[str, Path]): loads the dataset from disk.
    """

    config: DatasetConfig
    tables: DatasetTables

    def __init__(
        self, config: DatasetConfig, tables: DatasetTables, pipeline_report: PipelineReportTable = PipelineReportTable()
    ):
        self.config = config
        self.tables = tables
        self.pipeline_report = PipelineReportTable(pipeline_report)

    def scheme_proxy(self, schemes_context: CodingSchemesManager) -> DatasetSchemeProxy:
        return DatasetSchemeProxy(self.config.scheme, schemes_context)

    def stats(self, coding_schemes_manager: CodingSchemesManager) -> DatasetStatsInterface:
        return DatasetStatsInterface(self, coding_schemes_manager)

    @classmethod
    def multi_stats(cls, *datasets: Self, coding_schemes_manager: CodingSchemesManager) -> MultiDatasetsStatsInterface:
        return MultiDatasetsStatsInterface(*datasets, schemes_manager=coding_schemes_manager)

    @classmethod
    def two_stats(
        cls, dataset1: Self, dataset2: Self, coding_schemes_manager: CodingSchemesManager
    ) -> TwoDatasetsStatsInterface:
        return TwoDatasetsStatsInterface(dataset1, dataset2, schemes_manager=coding_schemes_manager)

    @cached_property
    def subject_ids(self) -> pd.Index:
        assert self.tables.static.index.name == self.config.columns.static.subject_id, (
            f"Index name of static table must be {self.config.columns.static.subject_id}."
        )
        return self.tables.static.index.unique()

    @cached_property
    def subjects_intervals_sum(self) -> pd.Series:
        c_admittime = self.config.columns.admissions.start_time
        c_dischtime = self.config.columns.admissions.end_time
        c_subject_id = self.config.columns.admissions.subject_id
        admissions = self.tables.admissions
        interval = (admissions[c_dischtime] - admissions[c_admittime]).dt.total_seconds()
        admissions = admissions.assign(interval=interval)
        missed_subjects = set(self.subject_ids).difference(set(admissions[c_subject_id]))
        intervals_sum = admissions.groupby(c_subject_id)["interval"].sum()
        assert isinstance(intervals_sum, pd.Series)
        missing_subjects = pd.Series([0] * len(missed_subjects), index=pd.Series(list(missed_subjects)))
        return pd.concat([intervals_sum, missing_subjects])

    @cached_property
    def subjects_n_admissions(self) -> pd.Series:
        c_subject_id = self.config.columns.admissions.subject_id
        admissions = self.tables.admissions
        missed_subjects = set(self.subject_ids).difference(set(admissions[c_subject_id]))
        n_admissions = [admissions.groupby(c_subject_id).size()]
        if len(missed_subjects) > 0:
            n_admissions.append(pd.Series([0] * len(missed_subjects), index=pd.Series(list(missed_subjects))))
        n_admissions_s = pd.concat(n_admissions)
        assert isinstance(n_admissions_s, pd.Series)
        return n_admissions_s

    def random_splits(
        self,
        splits: list[float],
        subject_ids: list[str] | None = None,
        random_seed: int = 42,
        balance: SplitLiteral = "subjects",
        discount_first_admission: bool = False,
    ) -> tuple[list[str], ...]:
        assert len(splits) > 0, "Split quantiles must be non-empty."
        assert list(splits) == sorted(splits), "Splits must be sorted."
        assert balance in ("subjects", "admissions", "admissions_intervals"), (
            "Balanced must be'subjects', 'admissions', or 'admissions_intervals'."
        )
        if subject_ids is None:
            subject_ids = list(self.subject_ids)
        assert len(subject_ids) > 0, "No subjects in the dataset."

        subject_ids = sorted(subject_ids)

        random.Random(random_seed).shuffle(subject_ids)
        subject_ids_arr = np.array(subject_ids)

        c_subject_id = self.config.columns.static.subject_id

        admissions = self.tables.admissions.loc[self.tables.admissions.loc[:, c_subject_id].isin(subject_ids_arr), :]

        if balance == "subjects":
            probs = (np.ones(len(subject_ids_arr)) / len(subject_ids_arr)).cumsum()

        elif balance == "admissions":
            assert len(admissions) > 0, "No admissions in the dataset."
            n_admissions = self.subjects_n_admissions.loc[subject_ids_arr]
            if discount_first_admission:
                n_admissions = n_admissions - 1
            p_admissions = n_admissions / n_admissions.sum()
            probs = p_admissions.values.cumsum()

        elif balance == "admissions_intervals":
            assert len(admissions) > 0, "No admissions in the dataset."
            subjects_intervals_sum = self.subjects_intervals_sum.loc[subject_ids_arr]
            p_subject_intervals = subjects_intervals_sum / subjects_intervals_sum.sum()
            probs = p_subject_intervals.values.cumsum()
        else:
            raise ValueError(f"Unknown balanced option: {balance}")

        # Deal with edge cases where the splits are exactly the same as the probabilities.
        for i in range(len(splits)):
            if any(abs(probs - splits[i]) < 1e-6):
                splits[i] = splits[i] + 1e-6

        splits_array = np.searchsorted(probs, splits)
        return tuple(a.tolist() for a in np.split(subject_ids_arr, splits_array))
