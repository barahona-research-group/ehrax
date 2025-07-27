"""."""

import dataclasses
import enum
import logging
import random
from abc import abstractmethod, ABCMeta, ABC
from collections import defaultdict
from dataclasses import field
from datetime import datetime
from functools import cached_property
from typing import Optional, ClassVar, Final, Self, Iterator

import equinox as eqx
import numpy as np
import pandas as pd

from .base import AbstractConfig, AbstractVxData
from .coding_scheme import (CodingScheme, CodingSchemesManager, NumericScheme, CodingSchemeWithUOM)
from .literals import OverlappingAction, NumericalTypeHint, SplitLiteral
from .utils import tqdm_constructor

SECONDS_TO_HOURS_SCALER: Final[float] = 1 / 3600.0  # convert seconds to hours


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
        return self in (COLUMN.time,
                        COLUMN.start_time,
                        COLUMN.end_time,
                        COLUMN.date_of_birth,
                        COLUMN.anchor_year,)

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
        assert all(k in COLUMN and k == v for k, v in self.as_dict().items()), f"Fields must be one of {COLUMN}."

    @property
    def id_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.as_dict() if COLUMN[k].is_id}

    def __contains__(self, item: str | COLUMN):
        if isinstance(item, str):
            return item in self.as_dict()
        elif isinstance(item, COLUMN):
            return str(item) in self.as_dict()
        else:
            raise ValueError(f"Unsupported type {type(item)}")

    def _index(self) -> Optional[str]:
        return None

    @property
    def index(self) -> Optional[str]:
        return self._index()

    @property
    def time_cols(self) -> tuple[str, ...]:
        return tuple(v for k, v in self.as_dict().items() if COLUMN[k].is_time)

    @property
    def code_cols(self) -> tuple[str, ...]:
        return tuple(v for k, v in self.as_dict().items() if COLUMN[k].is_code)


def include_cols(*cols: COLUMN, index: Optional[COLUMN] = None):
    """Add the columns as attributes to the subclass and augment the annotations as appropriate."""
    assert all(isinstance(c, COLUMN) for c in cols), f"Columns must be a set of {COLUMN}."
    assert len(cols) == len(set(cols)), f"Columns must be unique."
    assert index is None or isinstance(index, COLUMN), f"Index must be None or a {COLUMN}."
    annotations = {str(c): str for c in cols}
    defaults = {str(c): str(c) for c in cols}

    def init(self, **kwargs):
        assert kwargs == defaults, f"Expected {defaults}, got {kwargs}."
        for c in cols:
            setattr(self, str(c), c.value)

    def decorator(subclass: type[TableColumns]) -> type[TableColumns]:
        assert issubclass(subclass, TableColumns), f"Class must be a subclass of {TableColumns}."
        # Update the subclass with the new annotations and defaults.
        subclass.__annotations__.update(annotations)
        init(subclass, **{str(c): c.value for c in cols})
        subclass.__defaults__ = tuple(defaults[k] for k in subclass.__annotations__)
        subclass._index = lambda self: str(index) if index is not None else None
        subclass.__repr__ = lambda \
                self: f"{subclass.__name__}({', '.join(f'{k}={v}' for k, v in self.as_dict().items())})"
        return subclass

    return decorator


@include_cols(COLUMN.subject_id, COLUMN.race, COLUMN.gender, COLUMN.date_of_birth, index=COLUMN.subject_id)
class StaticTableColumns(TableColumns):
    pass


@include_cols(COLUMN.subject_id, COLUMN.admission_id, COLUMN.start_time, COLUMN.end_time, index=COLUMN.admission_id)
class AdmissionTableColumns(TableColumns):
    pass


@include_cols(COLUMN.admission_id, COLUMN.code, COLUMN.description)
class AdmissionSummaryTableColumns(TableColumns):
    pass


@include_cols(COLUMN.admission_id, COLUMN.code, COLUMN.time, COLUMN.measurement, COLUMN.description)
class AdmissionTimeSeriesTableColumns(TableColumns):
    pass


@include_cols(COLUMN.admission_id, COLUMN.code, COLUMN.start_time, COLUMN.end_time, COLUMN.description)
class AdmissionIntervalEventsTableColumns(TableColumns):
    pass


@include_cols(COLUMN.admission_id, COLUMN.code, COLUMN.description,
              COLUMN.start_time, COLUMN.end_time,
              COLUMN.amount,
              COLUMN.amount_unit,
              COLUMN.derived_unit_normalization_factor,
              COLUMN.derived_universal_unit,
              COLUMN.derived_normalized_amount,
              COLUMN.derived_normalized_amount_per_hour)
class AdmissionIntervalRatesTableColumns(TableColumns):
    pass


class MultivariateTimeSeriesTableMeta(AbstractConfig):
    name: str
    attributes: tuple[str, ...]
    type_hint: tuple[NumericalTypeHint, ...]
    default_type_hint: NumericalTypeHint

    def __init__(self, name: str, attributes: tuple[str, ...],
                 type_hint: Optional[tuple[NumericalTypeHint, ...]] = None,
                 default_type_hint: NumericalTypeHint = 'N'):
        self.name = name
        self.attributes = attributes
        self.default_type_hint = default_type_hint
        self.type_hint = type_hint or ((default_type_hint,) * len(attributes))

    def __check_init__(self):
        assert len(self.attributes) == len(self.type_hint), \
            f"Length of attributes and type_hint must be the same. Got {len(self.attributes)} and {len(self.type_hint)}."
        assert all(t in ('N', 'C', 'B', 'O') for t in self.type_hint), \
            f"type hint must be one of 'N', 'C', 'B', 'O'. Got {self.type_hint}."


class DatasetColumns(AbstractConfig):
    static: StaticTableColumns
    admissions: AdmissionTableColumns
    dx_discharge: AdmissionSummaryTableColumns
    obs: AdmissionTimeSeriesTableColumns
    icu_procedures: AdmissionIntervalEventsTableColumns
    icu_inputs: AdmissionIntervalRatesTableColumns
    hosp_procedures: AdmissionIntervalEventsTableColumns

    def __init__(self, static: StaticTableColumns = StaticTableColumns(),
                 admissions: AdmissionTableColumns = AdmissionTableColumns(),
                 dx_discharge: Optional[AdmissionSummaryTableColumns] = AdmissionSummaryTableColumns(),
                 obs: Optional[AdmissionTimeSeriesTableColumns] = AdmissionTimeSeriesTableColumns(),
                 icu_procedures: Optional[AdmissionIntervalEventsTableColumns] = AdmissionTimeSeriesTableColumns(),
                 icu_inputs: Optional[AdmissionIntervalRatesTableColumns] = AdmissionTimeSeriesTableColumns(),
                 hosp_procedures: Optional[AdmissionIntervalEventsTableColumns] = AdmissionTimeSeriesTableColumns()):
        self.static = static
        self.admissions = admissions
        self.dx_discharge = dx_discharge
        self.obs = obs
        self.icu_procedures = icu_procedures
        self.icu_inputs = icu_inputs
        self.hosp_procedures = hosp_procedures

    def __check_init__(self):
        assert all(isinstance(v, TableColumns) for v in self.as_dict().values())
        column_names = defaultdict(set)
        for v in self.as_dict().values():
            for k, v in v.items():
                assert k == v
                column_names[k].add(v)
        for k, v in column_names.items():
            if len(v) > 1:
                raise ValueError(f"Column {k} is present with different names: {v}")

    @property
    def admission_id_alias(self) -> str:
        return self.admissions.admission_id

    @property
    def subject_id_alias(self) -> str:
        return self.static.subject_id

    @property
    def timestamped_tables_config_dict(self):
        return {k: v for k, v in self.as_dict().items()
                if str(COLUMN.time) in v.as_dict().keys()}

    @property
    def interval_based_table_config_dict(self):
        return {k: v for k, v in self.as_dict().items()
                if {str(COLUMN.start_time), str(COLUMN.end_time)}.issubset(set(v.as_dict().keys))}

    @property
    def indices(self) -> dict[str, str]:
        return {
            k: v.index
            for k, v in self.as_dict().items() if v.index is not None
        }

    @property
    def time_cols(self) -> dict[str, tuple[str, ...]]:
        return {k: v.time_cols for k, v in self.as_dict().items() if len(v.time_cols) > 0}

    @property
    def code_column(self) -> dict[str, str]:
        return {k: v.code_cols for k, v in self.as_dict().items() if len(v.code_cols) > 0}

    def temporal_admission_linked_table(self, table_name: str) -> bool:
        conf = getattr(self, table_name)
        return len(conf.time_cols) > 0 and COLUMN.admission_id in conf


class DatasetTables(AbstractVxData):
    static: pd.DataFrame
    admissions: pd.DataFrame
    dx_discharge: Optional[pd.DataFrame]
    obs: Optional[pd.DataFrame]
    icu_procedures: Optional[pd.DataFrame]
    icu_inputs: Optional[pd.DataFrame]
    hosp_procedures: Optional[pd.DataFrame]

    def __init__(self, static: pd.DataFrame, admissions: pd.DataFrame, dx_discharge: Optional[pd.DataFrame] = None,
                 obs: Optional[pd.DataFrame] = None, icu_procedures: Optional[pd.DataFrame] = None,
                 icu_inputs: Optional[pd.DataFrame] = None, hosp_procedures: Optional[pd.DataFrame] = None):
        self.static = static
        self.admissions = admissions
        self.dx_discharge = dx_discharge
        self.obs = obs
        self.icu_procedures = icu_procedures
        self.icu_inputs = icu_inputs
        self.hosp_procedures = hosp_procedures

    @property
    def tables_dict(self) -> dict[str, pd.DataFrame]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if isinstance(v, pd.DataFrame)
        }


class DatasetSchemeConfig(AbstractConfig):
    ethnicity: Optional[str]
    gender: Optional[str]
    dx_discharge: Optional[str]
    obs: Optional[str]
    icu_procedures: Optional[str]
    hosp_procedures: Optional[str]
    icu_inputs: Optional[str]

    def __init__(self, ethnicity: Optional[str] = None, gender: Optional[str] = None,
                 dx_discharge: Optional[str] = None, obs: Optional[str] = None,
                 icu_procedures: Optional[str] = None, hosp_procedures: Optional[str] = None,
                 icu_inputs: Optional[str] = None):
        self.ethnicity = ethnicity
        self.gender = gender
        self.dx_discharge = dx_discharge
        self.obs = obs
        self.icu_procedures = icu_procedures
        self.hosp_procedures = hosp_procedures
        self.icu_inputs = icu_inputs

    def scheme_fields(self) -> dict[str, str]:
        return {'gender': self.gender,
                'ethnicity': self.ethnicity,
                'dx_discharge': self.dx_discharge,
                'obs': self.obs,
                'icu_inputs': self.icu_inputs,
                'icu_procedures': self.icu_procedures,
                'hosp_procedures': self.hosp_procedures}


@dataclasses.dataclass
class DatasetSchemeProxy:
    """
    Represents a dataset scheme that defines the coding schemes and outcome extractor for a dataset.

    Attributes:
        config (DatasetSchemeConfig): the configuration for the dataset scheme.

    Methods:
        __init__(self, config: DatasetSchemeConfig, **kwargs): initializes a new instance of the DatasetScheme class.
        scheme_dict(self): returns a dictionary of the coding schemes in the dataset scheme.
        make_target_scheme_config(self, **kwargs): creates a new target scheme configuration based on the current scheme.
        make_target_scheme(self, config=None, **kwargs): creates a new target scheme based on the current scheme.
        demographic_vector_size(self, demographic_vector_config: DemographicVectorConfig): calculates the size of the demographic vector.
        dx_mapper(self, target_scheme: DatasetScheme): returns the mapper for the diagnosis coding scheme to the corresponding target scheme.
        ethnicity_mapper(self, target_scheme: DatasetScheme): returns the mapper for the ethnicity coding scheme to the corresponding target scheme.
        supported_target_scheme_options(self): returns the supported target scheme options for each coding scheme.
    """
    config: DatasetSchemeConfig
    schemes_context: CodingSchemesManager

    def __init__(self, config: DatasetSchemeConfig, schemes_context: CodingSchemesManager):
        self.config = config
        self.schemes_context = schemes_context

    def _scheme(self, name: str) -> Optional[CodingScheme]:
        try:
            return self.schemes_context.scheme[name]
        except KeyError as e:
            return None

    @property
    def ethnicity(self) -> CodingScheme:
        return self._scheme(self.config.ethnicity)

    @property
    def gender(self) -> CodingScheme:
        return self._scheme(self.config.gender)

    @property
    def dx_discharge(self) -> CodingScheme:
        return self._scheme(self.config.dx_discharge)

    @property
    def obs(self) -> Optional[NumericScheme]:
        return self._scheme(self.config.obs)

    @property
    def icu_procedures(self) -> Optional[CodingScheme]:
        return self._scheme(self.config.icu_procedures)

    @property
    def hosp_procedures(self) -> Optional[CodingScheme]:
        return self._scheme(self.config.hosp_procedures)

    @property
    def icu_inputs(self) -> Optional[CodingSchemeWithUOM]:
        return self._scheme(self.config.icu_inputs)

    @property
    def scheme_dict(self):
        return {
            k: self._scheme(v)
            for k, v in self.config.scheme_fields().items() if self._scheme(v) is not None}


class ReportAttributes(AbstractConfig):
    transformation: str = None
    operation: str = None
    table: str = None
    column: str = None
    value_type: str = None
    before: Optional[str | int | float | bool] = None
    after: Optional[str | int | float | bool] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(), init=True,
                           compare=False, repr=False, hash=False)

    def __post_init__(self):

        for k, v in self.__dict__.items():
            if not k.startswith('_') and v is not None:
                if isinstance(v, type):
                    setattr(self, k, v.__name__)
                elif isinstance(v, np.dtype):
                    setattr(self, k, v.name)


class PipelineReportTable(pd.DataFrame):
    # We need to exclude the timestamps of the steps from the equality tests.
    def equals(self, other: Self):
        # Exclude timestamps from comparison.
        report = self
        if all('timestamp' in r for r in (self.columns, other.columns)):
            report = report.drop(columns=['timestamp'])
            other = other.drop(columns=['timestamp'])
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

    def compile(self, previous_report: Optional[pd.DataFrame] = None) -> PipelineReportTable:
        report = self.incidents
        if len(report) == 0:
            report = (ReportAttributes(transformation='identity'),)

        df = pd.DataFrame([x.as_dict() for x in report]).astype(str)
        object_columns = [c for c in df.columns if df[c].dtype == 'object']
        type_rows = df['value_type'] == 'dtype'
        type_cols = ['after', 'before']
        nan_mask = df.loc[:, object_columns].isnull() | df.loc[:, object_columns].isin((None, 'nan', 'NaN', 'None'))
        df.loc[:, object_columns] = df.loc[:, object_columns].where(~nan_mask, '-')
        df.loc[type_rows, type_cols] = df.loc[type_rows, type_cols].map(lambda x: f'{x}_type')
        if previous_report is None:
            return PipelineReportTable(df)
        else:
            return PipelineReportTable(pd.concat([previous_report, df], ignore_index=True,
                                                 axis=0, sort=False))


class AbstractDataset(AbstractVxData, ABC):
    config: AbstractConfig
    report_class: ClassVar[type[Report]] = Report

    @abstractmethod
    def scheme_proxy(self, schemes_context: CodingSchemesManager):
        ...


class AbstractTransformation(eqx.Module):

    @classmethod
    @abstractmethod
    def apply(cls, dataset: AbstractDataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[
        AbstractDataset, Report]:
        raise NotImplementedError

    @classmethod
    def skip(cls, dataset: AbstractDataset, report: Report) -> tuple[AbstractDataset, Report]:
        return dataset, report.add(transformation=cls, operation='skip')


#
# class TransformationSequenceException(TypeError):
#     pass
#
#
# class DuplicateTransformationException(TransformationSequenceException):
#     pass
#
#
# class MissingDependencyException(TransformationSequenceException):
#     pass
#
#
# class BlockedTransformationException(TransformationSequenceException):
#     pass
#
#
# class TransformationsDependency(AbstractConfig):
#     depends: dict[type[AbstractTransformation], set[type[AbstractTransformation]]]
#     blocked_by: dict[type[AbstractTransformation], set[type[AbstractTransformation]]]
#
#     def __init__(self, depends: dict[type[AbstractTransformation], set[type[AbstractTransformation]]],
#                  blocked_by: dict[type[AbstractTransformation], set[type[AbstractTransformation]]]):
#         self.depends = depends
#         self.blocked_by = blocked_by
#
#     @staticmethod
#     def empty():
#         return TransformationsDependency(depends={}, blocked_by={})
#
#     @staticmethod
#     def inherit_features(transformation_type: type[AbstractTransformation],
#                          inheritable_map: dict[type[AbstractTransformation], set[type[AbstractTransformation]]]) -> set[
#         type[AbstractTransformation]]:
#         inherited_features = inheritable_map.get(transformation_type, set())
#         for d in inheritable_map:
#             if issubclass(transformation_type, d):
#                 inherited_features |= inheritable_map[d]
#         return inherited_features
#
#     def merge(self, other: 'TransformationsDependency') -> 'TransformationsDependency':
#         common_depend_keys = set(self.depends.keys()) & set(other.depends.keys())
#         common_blocked_keys = set(self.blocked_by.keys()) & set(other.blocked_by.keys())
#         common_depends = {k: self.depends[k] | other.depends[k] for k in common_depend_keys}
#         common_blocked = {k: self.blocked_by[k] | other.blocked_by[k] for k in common_blocked_keys}
#         return TransformationsDependency(depends={**self.depends, **other.depends, **common_depends},
#                                          blocked_by={**self.blocked_by, **other.blocked_by, **common_blocked})
#
#     def get_dependencies(self, transformation_type: type[AbstractTransformation]) -> set[
#         type[AbstractTransformation]]:
#         return self.inherit_features(transformation_type, self.depends)
#
#     def get_blocked_by(self, transformation_type: type[AbstractTransformation]) -> set[
#         type[AbstractTransformation]]:
#         return self.inherit_features(transformation_type, self.blocked_by)
#
#     def validate_sequence(self, transformations: list[AbstractTransformation]):
#         transformations_type: list[type[AbstractTransformation]] = list(map(type, transformations))
#         if len(set(transformations_type)) != len(transformations_type):
#             raise DuplicateTransformationException("Transformation sequence contains duplicate transformations. "
#                                                    "Each transformation must appear only once. "
#                                                    f"Got {transformations}.")
#         applied_set: set[type[AbstractTransformation]] = set()
#         for t in transformations_type:
#             applied_set.add(t)
#             dependency_gap = self.get_dependencies(t) - applied_set
#             block_incidents = self.get_blocked_by(t) & applied_set
#             if len(dependency_gap) > 0:
#                 raise MissingDependencyException(f"Transformation {t} depends on "
#                                                  f"{dependency_gap} which "
#                                                  "was not applied before."
#                                                  f"Got {transformations_type}.")
#             if len(block_incidents) > 0:
#                 raise BlockedTransformationException(f"Transformation {t} is blocked by "
#                                                      f"{block_incidents} "
#                                                      f"which was applied before. Got {transformations_type}.")


class AbstractDatasetPipelineConfig(AbstractConfig):
    pass


class AbstractDatasetPipeline(AbstractVxData, metaclass=ABCMeta):
    config: AbstractDatasetPipelineConfig
    transformations: list[AbstractTransformation]
    # validator: ClassVar[TransformationsDependency] = TransformationsDependency.empty()
    report_class: ClassVar[type[Report]] = Report

    def __init__(self, config: AbstractDatasetPipelineConfig = AbstractDatasetPipelineConfig(), *,
                 transformations: list[AbstractTransformation]):
        self.config = config
        self.transformations = transformations

    # def __check_init__(self):
    # self.validator.validate_sequence(self.transformations)


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

    def _execute_pipeline(self, transformations: list[AbstractTransformation],
                          schemes_context: CodingSchemesManager) -> Self:
        if self.pipeline_executed:
            logging.warning("A pipeline has already been executed. This will replace the pipeline execution history.")
        report = self.report_class()
        dataset = self
        with tqdm_constructor(desc='Transforming Dataset', unit='transformations',
                              total=len(transformations)) as pbar:
            for t in transformations:
                pbar.set_description(f"Transforming Dataset: {type(t).__name__}")
                report = report.add(transformation=type(t), operation='start')
                dataset, report = t.apply(dataset, schemes_context, report)
                report = report.add(transformation=type(t), operation='end')
                pbar.update(0)

        return eqx.tree_at(lambda x: x.pipeline_report, dataset, report.compile(dataset.pipeline_report))


class DatasetConfig(AbstractConfig):
    scheme: DatasetSchemeConfig
    columns: DatasetColumns
    overlapping_admissions: OverlappingAction
    select_subjects_with_observation: Optional[str]

    def __init__(self, scheme: DatasetSchemeConfig, columns: DatasetColumns = DatasetColumns(),
                 overlapping_admissions: OverlappingAction = "merge",
                 select_subjects_with_observation: Optional[str] = None):
        self.scheme = scheme
        self.columns = columns
        self.overlapping_admissions = overlapping_admissions
        self.select_subjects_with_observation = select_subjects_with_observation


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

    def __init__(self, config: DatasetConfig, tables: DatasetTables,
                 pipeline_report: PipelineReportTable = PipelineReportTable()):
        self.config = config
        self.tables = tables
        self.pipeline_report = PipelineReportTable(pipeline_report)

    @classmethod
    @abstractmethod
    def make_default_pipeline(cls) -> AbstractDatasetPipeline:
        ...

    def scheme_proxy(self, coding_schemes_manger: CodingSchemesManager) -> DatasetSchemeProxy:  # type: ignore[override]
        return DatasetSchemeProxy(self.config.scheme, coding_schemes_manger)

    @cached_property
    def subject_ids(self):
        assert self.tables.static.index.name == self.config.columns.static.subject_id, \
            f"Index name of static table must be {self.config.columns.static.subject_id}."
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
        return pd.concat([admissions.groupby(c_subject_id)['interval'].sum(),
                          pd.Series([0] * len(missed_subjects), index=list(missed_subjects))])

    @cached_property
    def subjects_n_admissions(self) -> pd.Series:
        c_subject_id = self.config.columns.admissions.subject_id
        admissions = self.tables.admissions
        missed_subjects = set(self.subject_ids).difference(set(admissions[c_subject_id]))
        return pd.concat([admissions.groupby(c_subject_id).size(),
                          pd.Series([0] * len(missed_subjects), index=list(missed_subjects))])

    def random_splits(self,
                      splits: list[float],
                      subject_ids: Optional[list[str]] = None,
                      random_seed: int = 42,
                      balance: SplitLiteral = 'subjects',
                      discount_first_admission: bool = False) -> tuple[list[str], ...]:
        assert len(splits) > 0, "Split quantiles must be non-empty."
        assert list(splits) == sorted(splits), "Splits must be sorted."
        assert balance in ('subjects', 'admissions',
                           'admissions_intervals'), "Balanced must be'subjects', 'admissions', or 'admissions_intervals'."
        if subject_ids is None:
            subject_ids = self.subject_ids
        assert len(subject_ids) > 0, "No subjects in the dataset."

        subject_ids = sorted(subject_ids)

        random.Random(random_seed).shuffle(subject_ids)
        subject_ids = np.array(subject_ids)

        c_subject_id = self.config.columns.static.subject_id

        admissions = self.tables.admissions[self.tables.admissions[c_subject_id].isin(subject_ids)]

        if balance == 'subjects':
            probs = (np.ones(len(subject_ids)) / len(subject_ids)).cumsum()

        elif balance == 'admissions':
            assert len(admissions) > 0, "No admissions in the dataset."
            n_admissions = self.subjects_n_admissions.loc[subject_ids]
            if discount_first_admission:
                n_admissions = n_admissions - 1
            p_admissions = n_admissions / n_admissions.sum()
            probs = p_admissions.values.cumsum()

        elif balance == 'admissions_intervals':
            assert len(admissions) > 0, "No admissions in the dataset."
            subjects_intervals_sum = self.subjects_intervals_sum.loc[subject_ids]
            p_subject_intervals = subjects_intervals_sum / subjects_intervals_sum.sum()
            probs = p_subject_intervals.values.cumsum()
        else:
            raise ValueError(f'Unknown balanced option: {balance}')

        # Deal with edge cases where the splits are exactly the same as the probabilities.
        for i in range(len(splits)):
            if any(abs(probs - splits[i]) < 1e-6):
                splits[i] = splits[i] + 1e-6

        splits_array = np.searchsorted(probs, splits)
        return tuple(a.tolist() for a in np.split(subject_ids, splits_array))
