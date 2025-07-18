"""."""

import dataclasses
import logging
import random
from abc import abstractmethod, ABCMeta, ABC
from dataclasses import field
from datetime import datetime
from functools import cached_property
from typing import Optional, ClassVar, Literal, Final, Self, Iterator

import equinox as eqx
import numpy as np
import pandas as pd

from .base import AbstractConfig, AbstractVxData
from .coding_scheme import (CodingScheme, NumericalTypeHint, CodingSchemesManager, NumericScheme, CodingSchemeWithUOM)
from .utils import tqdm_constructor

SECONDS_TO_HOURS_SCALER: Final[float] = 1 / 3600.0  # convert seconds to hours


class TableConfig(AbstractConfig):

    @staticmethod
    def _alias_dict(data) -> dict[str, str]:
        return {k: v for k, v in data.items() if k.endswith('_alias')}

    @property
    def alias_dict(self) -> dict[str, str]:
        return self._alias_dict(self.as_dict())

    @staticmethod
    def _alias_id_dict(data) -> dict[str, str]:
        return {k: v for k, v in data.items() if '_id_' in k}

    @property
    def alias_id_dict(self) -> dict[str, str]:
        return self._alias_id_dict(self.as_dict())

    @property
    def index(self) -> Optional[str]:
        return None

    @staticmethod
    def _time_cols(data) -> tuple[str, ...]:
        return tuple(v for k, v in data.items() if 'time' in k or 'date' in k)

    @property
    def time_cols(self) -> tuple[str, ...]:
        return self._time_cols(self.alias_dict)

    @staticmethod
    def _coded_cols(data) -> tuple[str, ...]:
        return tuple(v for k, v in data.items() if 'code' in k)

    @property
    def coded_cols(self) -> tuple[str, ...]:
        return self._coded_cols(self.alias_dict)


class AdmissionLinkedTableConfig(TableConfig):
    admission_id_alias: str

    def __init__(self, admission_id_alias: str):
        self.admission_id_alias = admission_id_alias


class SubjectLinkedTableConfig(TableConfig):
    subject_id_alias: str

    def __init__(self, subject_id_alias: str):
        self.subject_id_alias = subject_id_alias


class TimestampedTableConfig(TableConfig):
    time_alias: str

    def __init__(self, time_alias: str):
        self.time_alias = time_alias


class TimestampedMultiColumnTableConfig(TimestampedTableConfig):
    attributes: tuple[str, ...]
    type_hint: tuple[NumericalTypeHint, ...]
    default_type_hint: NumericalTypeHint

    def __init__(self, time_alias: str, attributes: tuple[str, ...],
                 type_hint: Optional[tuple[NumericalTypeHint, ...]] = None,
                 default_type_hint: NumericalTypeHint = 'N'):
        super().__init__(time_alias)
        self.attributes = attributes
        self.default_type_hint = default_type_hint
        self.type_hint = type_hint or ((default_type_hint,) * len(attributes))

    def __check_init__(self):
        assert len(self.attributes) == len(self.type_hint), \
            f"Length of attributes and type_hint must be the same. Got {len(self.attributes)} and {len(self.type_hint)}."
        assert all(t in ('N', 'C', 'B', 'O') for t in self.type_hint), \
            f"type hint must be one of 'N', 'C', 'B', 'O'. Got {self.type_hint}."


class CodedTableConfig(TableConfig):
    code_alias: str
    description_alias: str

    def __init__(self, code_alias: str, description_alias: str):
        self.code_alias = code_alias
        self.description_alias = description_alias


class TimestampedCodedTableConfig(CodedTableConfig, TimestampedTableConfig):
    def __init__(self, code_alias: str, description_alias: str, time_alias: str):
        CodedTableConfig.__init__(self, code_alias=code_alias, description_alias=description_alias)
        TimestampedTableConfig.__init__(self, time_alias=time_alias)


class TimestampedCodedValueTableConfig(TimestampedCodedTableConfig):
    value_alias: str

    def __init__(self, value_alias: str, code_alias: str, description_alias: str, time_alias: str):
        super().__init__(code_alias=code_alias, description_alias=description_alias, time_alias=time_alias)
        self.value_alias = value_alias


class AdmissionLinkedCodedValueTableConfig(CodedTableConfig, AdmissionLinkedTableConfig):
    def __init__(self, code_alias: str, description_alias: str, admission_id_alias: str):
        CodedTableConfig.__init__(self, code_alias=code_alias, description_alias=description_alias)
        AdmissionLinkedTableConfig.__init__(self, admission_id_alias=admission_id_alias)


class IntervalBasedTableConfig(TableConfig):
    start_time_alias: str
    end_time_alias: str

    def __init__(self, start_time_alias: str, end_time_alias: str):
        self.start_time_alias = start_time_alias
        self.end_time_alias = end_time_alias


class AdmissionTableConfig(AdmissionLinkedTableConfig, SubjectLinkedTableConfig):
    admission_time_alias: str
    discharge_time_alias: str

    def __init__(self, admission_id_alias: str, subject_id_alias: str, admission_time_alias: str,
                 discharge_time_alias: str):
        AdmissionLinkedTableConfig.__init__(self, admission_id_alias=admission_id_alias)
        SubjectLinkedTableConfig.__init__(self, subject_id_alias=subject_id_alias)
        self.admission_time_alias = admission_time_alias
        self.discharge_time_alias = discharge_time_alias

    @property
    def index(self):
        return self.admission_id_alias


class StaticTableConfig(SubjectLinkedTableConfig):
    gender_alias: str
    race_alias: str
    date_of_birth_alias: str

    def __init__(self, subject_id_alias: str, gender_alias: str, race_alias: str, date_of_birth_alias: str):
        super().__init__(subject_id_alias)
        self.gender_alias = gender_alias
        self.race_alias = race_alias
        self.date_of_birth_alias = date_of_birth_alias

    @property
    def index(self):
        return self.subject_id_alias


class AdmissionTimestampedMultiColumnTableConfig(TimestampedMultiColumnTableConfig, AdmissionLinkedTableConfig):
    name: str

    def __init__(self, name: str, admission_id_alias: str, time_alias: str, attributes: tuple[str, ...],
                 type_hint: Optional[tuple[NumericalTypeHint, ...]] = None,
                 default_type_hint: NumericalTypeHint = 'N'):
        TimestampedMultiColumnTableConfig.__init__(self, time_alias=time_alias, attributes=attributes,
                                                   type_hint=type_hint, default_type_hint=default_type_hint)
        AdmissionLinkedTableConfig.__init__(self, admission_id_alias=admission_id_alias)
        self.name = name


class AdmissionTimestampedCodedValueTableConfig(TimestampedCodedValueTableConfig, AdmissionLinkedTableConfig):

    def __init__(self, admission_id_alias: str, value_alias: str, code_alias: str, description_alias: str,
                 time_alias: str):
        TimestampedCodedValueTableConfig.__init__(self, code_alias=code_alias, value_alias=value_alias,
                                                  description_alias=description_alias,
                                                  time_alias=time_alias)
        AdmissionLinkedTableConfig.__init__(self, admission_id_alias=admission_id_alias)


class AdmissionIntervalBasedCodedTableConfig(IntervalBasedTableConfig, CodedTableConfig,
                                             AdmissionLinkedTableConfig):
    def __init__(self, admission_id_alias: str, start_time_alias: str, end_time_alias: str,
                 code_alias: str, description_alias: str, ):
        IntervalBasedTableConfig.__init__(self, start_time_alias=start_time_alias, end_time_alias=end_time_alias)
        CodedTableConfig.__init__(self, code_alias=code_alias, description_alias=description_alias)
        AdmissionLinkedTableConfig.__init__(self, admission_id_alias=admission_id_alias)


class RatedInputTableConfig(AdmissionIntervalBasedCodedTableConfig):
    amount_alias: str
    amount_unit_alias: str
    derived_unit_normalization_factor: str
    derived_universal_unit: str
    derived_normalized_amount: str
    derived_normalized_amount_per_hour: str

    def __init__(self, admission_id_alias: str, start_time_alias: str, end_time_alias: str,
                 code_alias: str, description_alias: str, amount_alias: str, amount_unit_alias: str,
                 derived_unit_normalization_factor: str, derived_universal_unit: str,
                 derived_normalized_amount: str,
                 derived_normalized_amount_per_hour: str):
        AdmissionIntervalBasedCodedTableConfig.__init__(self, admission_id_alias=admission_id_alias,
                                                        start_time_alias=start_time_alias,
                                                        end_time_alias=end_time_alias,
                                                        code_alias=code_alias,
                                                        description_alias=description_alias)
        self.amount_alias = amount_alias
        self.amount_unit_alias = amount_unit_alias
        self.derived_unit_normalization_factor = derived_unit_normalization_factor
        self.derived_universal_unit = derived_universal_unit
        self.derived_normalized_amount = derived_normalized_amount
        self.derived_normalized_amount_per_hour = derived_normalized_amount_per_hour


class DatasetTablesConfig(AbstractConfig):
    static: StaticTableConfig
    admissions: AdmissionTableConfig
    dx_discharge: Optional[AdmissionLinkedCodedValueTableConfig]
    obs: Optional[AdmissionTimestampedCodedValueTableConfig]
    icu_procedures: Optional[AdmissionIntervalBasedCodedTableConfig]
    icu_inputs: Optional[RatedInputTableConfig]
    hosp_procedures: Optional[AdmissionIntervalBasedCodedTableConfig]

    def __init__(self, static: StaticTableConfig, admissions: AdmissionTableConfig,
                 dx_discharge: Optional[AdmissionLinkedCodedValueTableConfig],
                 obs: Optional[AdmissionTimestampedCodedValueTableConfig],
                 icu_procedures: Optional[AdmissionIntervalBasedCodedTableConfig],
                 icu_inputs: Optional[RatedInputTableConfig],
                 hosp_procedures: Optional[AdmissionIntervalBasedCodedTableConfig]
                 ):
        self.static = static
        self.admissions = admissions
        self.dx_discharge = dx_discharge
        self.obs = obs
        self.icu_procedures = icu_procedures
        self.icu_inputs = icu_inputs
        self.hosp_procedures = hosp_procedures

    def __check_init__(self):
        self._assert_consistent_aliases()

    def _assert_consistent_aliases(self):
        config_dict = self.table_config_dict

        for k, v in config_dict.items():
            if k == 'static' or not isinstance(v, SubjectLinkedTableConfig):
                continue
            assert v.subject_id_alias == self.static.subject_id_alias, \
                f"Subject id alias for {k} must be the same as the one in static table. Got {v.subject_id_alias}." \
                f"Expected {self.static.subject_id_alias}."

        for k, v in config_dict.items():
            if k == 'admissions' or not isinstance(v, AdmissionLinkedTableConfig):
                continue
            assert v.admission_id_alias == self.admissions.admission_id_alias, \
                f"Admission id alias for {k} must be the same as the one in admissions table. Got {v.admission_id_alias}." \
                f"Expected {self.admissions.admission_id_alias}."

    @property
    def admission_id_alias(self):
        return self.admissions.admission_id_alias

    @property
    def subject_id_alias(self):
        return self.static.subject_id_alias

    @property
    def table_config_dict(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, TableConfig)}

    @property
    def timestamped_table_config_dict(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, TimestampedTableConfig)}

    @property
    def interval_based_table_config_dict(self):
        return {k: v for k, v in self.__dict__.items() if
                isinstance(v, IntervalBasedTableConfig)}

    @property
    def indices(self) -> dict[str, str]:
        return {
            k: v.index
            for k, v in self.__dict__.items()
            if isinstance(v, TableConfig) and v.index is not None
        }

    @property
    def time_cols(self) -> dict[str, tuple[str, ...]]:
        return {
            k: v.time_cols
            for k, v in self.__dict__.items()
            if isinstance(v, TableConfig) and len(v.time_cols) > 0
        }

    @property
    def code_column(self) -> dict[str, str]:
        return {k: v.code_alias for k, v in self.__dict__.items() if isinstance(v, CodedTableConfig)}

    def temporal_admission_linked_table(self, table_name: str) -> bool:
        conf = getattr(self, table_name)
        temporal = isinstance(conf, TimestampedTableConfig) or isinstance(conf, IntervalBasedTableConfig)
        admission_linked = isinstance(conf, AdmissionLinkedTableConfig)
        return temporal and admission_linked


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
                 icu_inputs: Optional[str] = None,
                 icu_inputs_uom_normalizer: Optional[str] = None):
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
    tables: DatasetTablesConfig
    overlapping_admissions: Literal["merge", "remove"]
    filter_subjects_with_observation: Optional[str]

    def __init__(self, scheme: DatasetSchemeConfig, tables: DatasetTablesConfig,
                 overlapping_admissions: Literal["merge", "remove"] = "merge",
                 filter_subjects_with_observation: Optional[str] = None):
        self.scheme = scheme
        self.tables = tables
        self.overlapping_admissions = overlapping_admissions
        self.filter_subjects_with_observation = filter_subjects_with_observation


SplitLiteral = Literal['subjects', 'admissions', 'admissions_intervals']


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
    def load_tables(cls, config: DatasetConfig, scheme: DatasetSchemeProxy) -> DatasetTables:
        pass

    @classmethod
    @abstractmethod
    def make_default_pipeline(cls) -> AbstractDatasetPipeline:
        ...

    def scheme_proxy(self, coding_schemes_manger: CodingSchemesManager) -> DatasetSchemeProxy:  # type: ignore[override]
        return DatasetSchemeProxy(self.config.scheme, coding_schemes_manger)

    @classmethod
    def load_scheme_manager(cls, config: DatasetConfig) -> CodingSchemesManager:
        raise NotImplementedError

    @cached_property
    def subject_ids(self):
        assert self.tables.static.index.name == self.config.tables.static.subject_id_alias, \
            f"Index name of static table must be {self.config.tables.static.subject_id_alias}."
        return self.tables.static.index.unique()

    @cached_property
    def subjects_intervals_sum(self) -> pd.Series:
        c_admittime = self.config.tables.admissions.admission_time_alias
        c_dischtime = self.config.tables.admissions.discharge_time_alias
        c_subject_id = self.config.tables.admissions.subject_id_alias
        admissions = self.tables.admissions
        interval = (admissions[c_dischtime] - admissions[c_admittime]).dt.total_seconds()
        admissions = admissions.assign(interval=interval)
        missed_subjects = set(self.subject_ids).difference(set(admissions[c_subject_id]))
        return pd.concat([admissions.groupby(c_subject_id)['interval'].sum(),
                          pd.Series([0] * len(missed_subjects), index=list(missed_subjects))])

    @cached_property
    def subjects_n_admissions(self) -> pd.Series:
        c_subject_id = self.config.tables.admissions.subject_id_alias
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

        c_subject_id = self.config.tables.static.subject_id_alias

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
