from abc import ABC, ABCMeta, abstractmethod
from functools import cached_property
from types import MappingProxyType
from typing import ClassVar, Iterable, Optional, Self

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pandas as pd

from .base import AbstractConfig, AbstractVxData, HDFVirtualNode, fetch_at
from .coding_scheme import CodeMap, CodesVector, CodingSchemesManager, GroupingData, OutcomeExtractor, ReducedCodeMapN1
from .dataset import AbstractDatasetPipeline, AbstractProcessedDataset, AbstractTransformation, COLUMN, Dataset, \
    DatasetSchemeConfig, DatasetSchemeProxy, PipelineReportTable, Report, ReportAttributes
from .literals import SplitLiteral
from .tvx_concepts import (Admission, AdmissionDates, DemographicVectorConfig, InpatientInput, InpatientInterventions,
                           InpatientObservables, LeadingObservableExtractorConfig, Patient, SegmentedPatient,
                           StaticInfo)
from .utils import tqdm_constructor


class ScalerConfig(AbstractConfig):
    use_float16: bool

    def __init__(self, use_float16: bool):
        self.use_float16 = use_float16


class TVxEHRSplitsConfig(AbstractConfig):
    split_quantiles: list[float]
    seed: int
    balance: SplitLiteral
    discount_first_admission: bool

    def __init__(self, split_quantiles: list[float], seed: int = 0, balance: SplitLiteral = 'subjects',
                 discount_first_admission: bool = False):
        self.split_quantiles = split_quantiles
        self.seed = seed
        self.balance = balance
        self.discount_first_admission = discount_first_admission


class TVxEHRSampleConfig(AbstractConfig):
    n_subjects: int
    seed: int
    offset: int

    def __init__(self, n_subjects: int, seed: int = 0, offset: int = 0):
        self.n_subjects = n_subjects
        self.seed = seed
        self.offset = offset


class IQROutlierRemoverConfig(AbstractConfig):
    outlier_q1: float
    outlier_q2: float
    outlier_iqr_scale: float
    outlier_z1: float
    outlier_z2: float

    def __init__(self, outlier_q1: float = 0.25,
                 outlier_q2: float = 0.75,
                 outlier_iqr_scale: float = 1.5,
                 outlier_z1: float = -2.5,
                 outlier_z2: float = 2.5):
        self.outlier_q1 = outlier_q1
        self.outlier_q2 = outlier_q2
        self.outlier_iqr_scale = outlier_iqr_scale
        self.outlier_z1 = outlier_z1
        self.outlier_z2 = outlier_z2


class OutlierRemoversConfig(AbstractConfig):
    obs: Optional[IQROutlierRemoverConfig]

    def __init__(self, obs: IQROutlierRemoverConfig = IQROutlierRemoverConfig()):
        self.obs = obs


class ScalersConfig(AbstractConfig):
    obs: Optional[ScalerConfig]
    icu_inputs: Optional[ScalerConfig]

    def __init__(self, obs: ScalerConfig = ScalerConfig(use_float16=True),
                 icu_inputs: ScalerConfig = ScalerConfig(use_float16=True)):
        self.obs = obs
        self.icu_inputs = icu_inputs


class DatasetNumericalProcessorsConfig(AbstractConfig):
    scalers: Optional[ScalersConfig]
    outlier_removers: Optional[OutlierRemoversConfig]

    def __init__(self, scalers: ScalersConfig = ScalersConfig(),
                 outlier_removers: OutlierRemoversConfig = OutlierRemoversConfig()):
        self.scalers = scalers
        self.outlier_removers = outlier_removers


class CodedValueProcessor(AbstractVxData, ABC):
    config: ScalerConfig | IQROutlierRemoverConfig
    table_name: Optional[str]
    code_column: Optional[str]
    value_column: Optional[str]

    def __init__(self, config: ScalerConfig | IQROutlierRemoverConfig, table_name: Optional[str] = None,
                 code_column: Optional[str] = None, value_column: Optional[str] = None):
        self.config = config
        self.table_name = table_name
        self.code_column = code_column
        self.value_column = value_column

    def table_getter(self, dataset: Dataset) -> pd.DataFrame:
        return getattr(dataset.tables, self.table_name)

    def fit(self, dataset: Dataset, admission_ids: list[str],
            table_name: str, code_column: str, value_column: str) -> Self:
        df = getattr(dataset.tables, table_name)
        c_adm_id = getattr(dataset.config.columns, table_name).admission_id
        df = df[[code_column, value_column, c_adm_id]]
        df = df[df[c_adm_id].isin(admission_ids)]

        fitted = self
        for k, v in self._extract_stats(df, code_column, value_column).items():
            fitted = eqx.tree_at(lambda x: getattr(x, k), fitted, v)

        for k, v in {'table_name': table_name, 'code_column': code_column, 'value_column': value_column}.items():
            fitted = eqx.tree_at(lambda x: getattr(x, k), fitted, v,
                                 is_leaf=lambda x: x is None)
        return fitted

    @abstractmethod
    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> dict[str, pd.Series]:
        pass

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        pass

    @property
    def series_dict(self) -> dict[str, pd.Series]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, pd.Series) and len(v) > 0}

    @property
    def processing_target(self) -> dict[str, str]:
        return {'table_name': self.table_name, 'code_column': self.code_column, 'value_column': self.value_column}


class CodedValueScaler(CodedValueProcessor, ABC):
    config: ScalerConfig

    @property
    @abstractmethod
    def original_dtype(self) -> np.dtype:
        pass

    @abstractmethod
    def unscale(self, array: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def unscale_code(self, array: np.ndarray, code_index: int) -> np.ndarray:
        pass


def outcome_first_occurrence(sorted_admissions: list[Admission]):
    """
    Find the first occurrence admission index of each outcome in a list of sorted admissions.

    Args:
        sorted_admissions (list[Admission]): a list of sorted Admission objects.

    Returns:
        np.ndarray: an array containing the admission index of the first occurrence of each outcome for each admission. -1 means no occurrence.
    """
    first_occurrence = np.empty_like(
        sorted_admissions[0].outcome.vec,
        dtype=type[sorted_admissions[0].admission_id])
    first_occurrence[:] = -1
    for adm in sorted_admissions:
        update_mask = (first_occurrence < 0) & adm.outcome.vec
        first_occurrence[update_mask] = adm.admission_id
    return first_occurrence


class OutlierRemovers(AbstractVxData):
    obs: Optional[CodedValueProcessor]

    def __init__(self, obs: CodedValueProcessor = None):
        self.obs = obs


class Scalers(AbstractVxData):
    obs: Optional[CodedValueScaler]
    icu_inputs: Optional[CodedValueScaler]

    def __init__(self, obs: CodedValueScaler = None, icu_inputs: CodedValueScaler = None):
        self.obs = obs
        self.icu_inputs = icu_inputs


class DatasetNumericalProcessors(AbstractVxData):
    outlier_removers: OutlierRemovers
    scalers: Scalers

    def __init__(self, outlier_removers: OutlierRemovers = OutlierRemovers(), scalers: Scalers = Scalers()):
        self.outlier_removers = outlier_removers
        self.scalers = scalers


class TVxEHRSchemeConfig(DatasetSchemeConfig):
    outcome: Optional[str] = None

    def __init__(self, ethnicity: Optional[str] = None, gender: Optional[str] = None,
                 dx_discharge: Optional[str] = None, obs: Optional[str] = None,
                 icu_procedures: Optional[str] = None, hosp_procedures: Optional[str] = None,
                 icu_inputs: Optional[str] = None, outcome: Optional[str] = None):
        super().__init__(ethnicity=ethnicity, gender=gender, dx_discharge=dx_discharge, obs=obs,
                         icu_procedures=icu_procedures, hosp_procedures=hosp_procedures, icu_inputs=icu_inputs)
        self.outcome = outcome


class TVxEHRSchemeProxy(DatasetSchemeProxy):
    config: TVxEHRSchemeConfig

    def __init__(self, config: TVxEHRSchemeConfig, schemes_context: CodingSchemesManager):
        super().__init__(config=config, schemes_context=schemes_context)

    @property
    def outcome(self) -> Optional[OutcomeExtractor]:
        return self.schemes_context.outcome[self.config.outcome] if self.config.outcome else None

    @cached_property
    def outcome_size(self) -> int | None:
        return len(self.outcome.codes(self.schemes_context.scheme[self.outcome.base_name])) if self.outcome else None

    @cached_property
    def outcome_base_mapper(self) -> Optional[CodeMap]:
        if self.config.dx_discharge == self.outcome.base_name:
            return None
        return self.schemes_context.map[(self.config.dx_discharge, self.outcome.base_name)]

    @staticmethod
    def validate_mapping(coding_scheme_manager: CodingSchemesManager, source: DatasetSchemeConfig,
                         target: TVxEHRSchemeConfig):
        target_schemes = target.scheme_fields()
        for key, source_scheme in source.scheme_fields().items():
            target_scheme = target_schemes[key]
            assert target_scheme is None or coding_scheme_manager.map[(source_scheme, target_scheme)] is not None, \
                f"Cannot map {key} from {source_scheme} to {target_scheme}"

        if target.outcome is not None:
            assert (coding_scheme_manager.supported_outcome(target.outcome, target.dx_discharge)
                    ), f"Outcome {target.outcome} not supported for {target.dx_discharge}"

    def demographic_vector_size(self, demographic_vector_config: DemographicVectorConfig):
        size = 0
        if demographic_vector_config.gender:
            size += len(self.gender)
        if demographic_vector_config.age:
            size += 1
        if demographic_vector_config.ethnicity:
            size += len(self.ethnicity)
        return size

    def mapper(self, source_scheme_config: DatasetSchemeConfig, key: str) -> Optional[CodeMap]:
        pair = (getattr(source_scheme_config, key), getattr(self, key).name)
        return self.schemes_context.map[pair] if pair in self.schemes_context.map else None

    def dx_mapper(self, source_scheme_config: DatasetSchemeConfig) -> Optional[CodeMap]:
        return self.mapper(source_scheme_config, 'dx_discharge')

    def ethnicity_mapper(self, source_scheme_config: DatasetSchemeConfig) -> Optional[CodeMap]:
        return self.mapper(source_scheme_config, 'ethnicity')

    def gender_mapper(self, source_scheme_config: DatasetSchemeConfig) -> Optional[CodeMap]:
        return self.mapper(source_scheme_config, 'gender')

    def icu_procedures_mapper(self, source_scheme_config: DatasetSchemeConfig) -> Optional[CodeMap]:
        return self.mapper(source_scheme_config, 'icu_procedures')

    def hosp_procedures_mapper(self, source_scheme_config: DatasetSchemeConfig) -> Optional[CodeMap]:
        return self.mapper(source_scheme_config, 'hosp_procedures')

    def icu_inputs_mapper(self, source_scheme_config: DatasetSchemeConfig) -> Optional[ReducedCodeMapN1]:
        return self.mapper(source_scheme_config, 'icu_inputs')

    def icu_inputs_grouping(self, source_scheme_config: DatasetSchemeConfig) -> Optional[GroupingData]:
        mapper = self.icu_inputs_mapper(source_scheme_config)
        return mapper.grouping_data(self.schemes_context.scheme[source_scheme_config.icu_inputs].index)

    @staticmethod
    def check_target_scheme_support(scheme_manager: CodingSchemesManager, dataset_scheme: DatasetSchemeProxy) -> dict[
        str, tuple[str, ...]]:
        supported_attr_targets = {
            k: scheme_manager.scheme_supported_targets(v)
            for k, v in dataset_scheme.scheme_dict.items()
        }
        supported_outcomes = scheme_manager.supported_outcomes(dataset_scheme.dx_discharge.name)
        return supported_attr_targets | {'outcome': supported_outcomes}


class TVxEHRConfig(AbstractConfig):
    """
    Configuration class for the interface.

    Attributes:
        demographic (DemographicVectorConfig): configuration for the demographic vector.
        leading_observable (Optional[LeadingObservableExtractorConfig]): configuration for the leading observable (optional).
        scheme (dict[str, str]): dictionary representing the scheme.
        time_binning (Optional[int]): time binning configuration to aggregate observables/measurements over intervals (optional).
    """
    scheme: TVxEHRSchemeConfig
    demographic: DemographicVectorConfig
    sample: Optional[TVxEHRSampleConfig]
    splits: Optional[TVxEHRSplitsConfig]
    numerical_processors: DatasetNumericalProcessorsConfig
    interventions: bool
    observables: bool
    time_binning: Optional[float]
    leading_observable: Optional[LeadingObservableExtractorConfig]
    interventions_segmentation: bool
    admission_minimum_los: Optional[float]

    def __init__(self, scheme: TVxEHRSchemeConfig, demographic: DemographicVectorConfig,
                 sample: Optional[TVxEHRSampleConfig] = None, splits: Optional[TVxEHRSplitsConfig] = None,
                 numerical_processors: DatasetNumericalProcessorsConfig = DatasetNumericalProcessorsConfig(),
                 interventions: bool = False, observables: bool = False,
                 time_binning: Optional[float] = None,
                 leading_observable: Optional[LeadingObservableExtractorConfig] = None,
                 interventions_segmentation: bool = False, admission_minimum_los: Optional[float] = None):
        self.scheme = scheme
        self.demographic = demographic
        self.sample = sample
        self.splits = splits
        self.numerical_processors = numerical_processors
        self.interventions = interventions
        self.observables = observables
        self.time_binning = time_binning
        self.leading_observable = leading_observable
        self.interventions_segmentation = interventions_segmentation
        self.admission_minimum_los = admission_minimum_los


class TVxReportAttributes(ReportAttributes):
    tvx_concept: str = None

    @staticmethod
    def _t(object_or_type: type | AbstractVxData) -> str:
        return object_or_type.__name__ if isinstance(object_or_type, type) else object_or_type.__class__.__name__

    @classmethod
    def ehr_prefix(cls) -> str:
        return cls._t(TVxEHR)

    @classmethod
    def subjects_prefix(cls) -> str:
        return f'{cls.ehr_prefix()}.dict[str, {cls._t(Patient)}]'

    @classmethod
    def subject_prefix(cls) -> str:
        return f'{cls.subjects_prefix()}[i]'

    @classmethod
    def static_info_prefix(cls) -> str:
        return f'{cls.subject_prefix()}.{cls._t(StaticInfo)}'

    @classmethod
    def admissions_prefix(cls) -> str:
        return f'{cls.subject_prefix()}.list[{cls._t(Admission)}]'

    @classmethod
    def inpatient_input_prefix(cls, attribute) -> str:
        return f'{cls.admissions_prefix()}[j].{cls._t(InpatientInterventions)}.{attribute}({cls._t(InpatientInput)})'

    @classmethod
    def admission_attribute_prefix(cls, attribute: str, attribute_type: str | type) -> str:
        if isinstance(attribute_type, type):
            attribute_type = cls._t(attribute_type)

        return f'{cls.admissions_prefix()}[j].{attribute}({attribute_type})'

    @classmethod
    def admission_codes_attribute(cls, attribute) -> str:
        return f'{cls.admission_attribute_prefix(attribute, CodesVector)}'


class TVxReport(Report):
    incident_class: ClassVar[type[TVxReportAttributes]] = TVxReportAttributes


class AbstractTVxPipeline(AbstractDatasetPipeline, metaclass=ABCMeta):
    transformations: list[AbstractTransformation]
    report_class: ClassVar[type[TVxReport]] = TVxReport


_SplitsType = tuple[tuple[str, ...], ...]


class TVxEHR(AbstractProcessedDataset):
    """
    A class representing a collection of patients in the EHR system, in ML-compliant format.

    Attributes:
        config (TVxEHRConfig): the configuration for the interface.
        dataset (Dataset): the dataset containing the patient data.
        subjects (dict[str, Patient]): a dictionary of patient objects, where the keys are subject IDs.

    Methods:
        subject_ids: returns a sorted list of subject IDs.
        scheme: returns the target scheme for the dataset.
        schemes: returns a tuple containing the dataset scheme and the target scheme.
        __len__: returns the number of patients in the dataset.
        equal_config: checks if the given configuration is equal to the cached configuration.
        save: saves the Patients object to disk.
        load: loads a Patients object from disk.
        try_load_cached: tries to load a cached Patients object, or creates a new one if the cache does not match the configuration.
        load_subjects: loads the subjects from the dataset.
        device_batch: loads the subjects and moves them to the device (e.g. GPU).
        epoch_splits: generates epoch splits for the subjects.
        batch_gen: generates batches of subjects.
        n_admissions: returns the total number of admissions for the given subjects.
        n_segments: returns the total number of segments for the given subjects.
        n_obs_times: returns the total number of observation timestamps for the given subjects.
        d2d_interval_days: returns the total number of days between the first and last discharge.
        interval_days: returns the total number of days between admissions for the given subjects.
        interval_hours: returns the total number of hours between admissions for the given subjects.
        p_obs: returns the proportion of present observations in the timestamped vectorized representation in the dataset.
        obs_coocurrence_matrix: returns the co-occurrence (or co-presence) matrix of the observables.
        size_in_bytes: returns the size of the Patients object in bytes.
        _unscaled_observation: unscales the observation values, undos the preprocessing scaling.
        _unscaled_leading_observable: unscales the leading observable values, undos the preprocessing scaling.
    """

    config: TVxEHRConfig
    dataset: Dataset
    numerical_processors: DatasetNumericalProcessors
    splits: Optional[_SplitsType]
    subjects: Optional[MappingProxyType[str, Patient]]
    patient_class: ClassVar[type[Patient]] = Patient
    report_class: ClassVar[type[TVxReport]] = TVxReport

    def __init__(self, config: TVxEHRConfig, dataset: Dataset,
                 numerical_processors: DatasetNumericalProcessors = DatasetNumericalProcessors(),
                 splits: _SplitsType = None, subjects: dict[str, Patient] = None,
                 pipeline_report: PipelineReportTable = PipelineReportTable()):
        self.config = config
        self.dataset = dataset
        self.numerical_processors = numerical_processors
        self.splits = splits
        self.subjects = subjects
        self.pipeline_report = PipelineReportTable(pipeline_report)

    @property
    def subject_ids(self) -> tuple[str, ...]:
        """Get the list of subject IDs."""
        return tuple(sorted(self.subjects.keys())) if self.subjects is not None else ()

    def fetch_subjects(self, subject_ids: Optional[tuple[str, ...]] = None) -> Self:
        if subject_ids is None:
            subject_ids = self.subject_ids
        # generating lambdas inside generators can lead to unexpected behaviour, e.g. all lambdas can be bounded
        # to one value of subject_id (the last one of the collection).
        # https://stackoverflow.com/a/452660
        return fetch_at(tuple(map(lambda k: lambda x: x.subjects[k], subject_ids)), self)

    def try_fetch_subjects(self, subject_ids: Optional[tuple[str, ...]] = None) -> Self:
        if subject_ids is None:
            subject_ids = self.subject_ids

        # only fetch those not loaded already.
        subject_ids = tuple(i for i in subject_ids if not isinstance(self.subjects[i], HDFVirtualNode))
        return fetch_at(tuple(map(lambda k: lambda x: x.subjects[k], subject_ids)), self)

    def scheme_proxy(self, schemes_context: CodingSchemesManager) -> TVxEHRSchemeProxy:
        """Get the scheme."""
        scheme = TVxEHRSchemeProxy(self.config.scheme, schemes_context)
        TVxEHRSchemeProxy.validate_mapping(schemes_context, self.dataset.config.scheme, self.config.scheme)
        return scheme

    def gender_mapper(self, schemes_proxy: TVxEHRSchemeProxy) -> Optional[CodeMap]:
        return schemes_proxy.gender_mapper(self.dataset.config.scheme)

    def ethnicity_mapper(self, schemes_proxy: TVxEHRSchemeProxy) -> Optional[CodeMap]:
        return schemes_proxy.ethnicity_mapper(self.dataset.config.scheme)

    def dx_mapper(self, schemes_proxy: TVxEHRSchemeProxy) -> Optional[CodeMap]:
        return schemes_proxy.dx_mapper(self.dataset.config.scheme)

    def icu_procedures_mapper(self, schemes_proxy: TVxEHRSchemeProxy) -> Optional[CodeMap]:
        return schemes_proxy.icu_procedures_mapper(self.dataset.config.scheme)

    def hosp_procedures_mapper(self, schemes_proxy: TVxEHRSchemeProxy) -> Optional[CodeMap]:
        return schemes_proxy.hosp_procedures_mapper(self.dataset.config.scheme)

    def icu_inputs_grouping(self, schemes_proxy: TVxEHRSchemeProxy) -> Optional[GroupingData]:
        return schemes_proxy.icu_inputs_grouping(self.dataset.config.scheme)

    def demographic_vector_size(self, schemes_proxy: TVxEHRSchemeProxy) -> int:
        return schemes_proxy.demographic_vector_size(self.config.demographic)

    @cached_property
    def subjects_sorted_admission_ids(self) -> dict[str, list[str]]:
        admissions = self.dataset.tables.admissions
        # For each subject, get the list of admission id sorted by admission date.
        sorted_index = lambda x: x.sort_values().index.to_list()
        return admissions.groupby(COLUMN.subject_id)[COLUMN.start_time].apply(sorted_index).to_dict()

    @cached_property
    def admission_ids(self) -> list[str]:

        admission_ids = sum(self.subjects_sorted_admission_ids.values(), [])
        # Check unique admission IDs across all subjects.
        if len(admission_ids) != len(set(admission_ids)):
            raise ValueError("Duplicate admission IDs found.")

        return admission_ids

    def subject_admission_demographics(self, subject_id: str) -> dict[str, jnp.ndarray]:
        return self.subjects[subject_id].admission_demographics(self.config.demographic)

    @cached_property
    def admission_demographics(self) -> dict[str, jnp.ndarray]:
        return {admission_id: admission_demo for subject_id in self.subjects for admission_id, admission_demo in
                self.subject_admission_demographics(subject_id).items()}

    @cached_property
    def admission_dates(self) -> dict[str, AdmissionDates]:
        admissions = self.dataset.tables.admissions
        c_admittime = self.dataset.config.columns.admissions.start_time
        c_dischtime = self.dataset.config.columns.admissions.end_time
        return admissions.apply(lambda x: AdmissionDates(x[c_admittime], x[c_dischtime]), axis=1).to_dict()

    def fetch_device_batch(self, subject_ids: Optional[tuple[str, ...]] = None) -> tuple[Self, Self]:
        # 1. Fetch from disk if the subjects are lazy-loaded in a new tree `ehr`.
        # 2. Load the subjects to the device in a new tree `device_ehr`.
        # 3. return as a tuple (`ehr`, `device_ehr`) as `device_ehr` can be re-used avoiding disk reading.
        if subject_ids is None:
            subject_ids = self.subject_ids
        ehr = self.try_fetch_subjects(subject_ids)
        device_subjects = {
            i: ehr.subjects[i].to_jax_arrays()
            for i in tqdm_constructor(subject_ids,
                                      desc="Loading to device",
                                      unit='subject',
                                      leave=False)
        }
        return ehr, eqx.tree_at(lambda x: x.subjects, self, device_subjects)

    def epoch_splits(self,
                     subject_ids: Optional[list[str]],
                     batch_n_admissions: int,
                     discount_first_admission: bool = False):
        """Generate epoch splits for training.

        Args:
            subject_ids (Optional[list[str]]): list of subject IDs to split.
            batch_n_admissions (int): number of admissions per batch.
            discount_first_admission (bool, optional): whether to ignore the first admission from the counts. Defaults to False.

        Returns:
            list[list[str]]: list of lists containing the split subject IDs.
        """
        if subject_ids is None:
            subject_ids = list(self.subjects.keys())

        n_splits = self.n_admissions(
            subject_ids, discount_first_admission) // batch_n_admissions
        if n_splits == 0:
            n_splits = 1
        p_splits = np.linspace(0, 1, n_splits + 1)[1:-1]

        subject_ids = np.array(subject_ids,
                               dtype=type(list(self.subjects.keys())[0]))

        n_adms = self.dataset.subjects_n_admissions
        if discount_first_admission:
            n_adms = n_adms - 1

        w_adms = n_adms.loc[subject_ids] / n_adms.loc[subject_ids].sum()
        weights = w_adms.values.cumsum()
        splits = np.searchsorted(weights, p_splits)
        splits = [a.tolist() for a in np.split(subject_ids, splits)]
        splits = [s for s in splits if len(s) > 0]
        return splits

    def batch_gen(self,
                  subject_ids,
                  batch_n_admissions: int,
                  ignore_first_admission: bool = False):
        """Generate batches of subjects.

        Args:
            subject_ids: list of subject IDs.
            batch_n_admissions (int): number of admissions per batch.
            ignore_first_admission (bool, optional): whether to ignore the first admission from the counts. Defaults to False.

        Yields:
            Patients: Patients object with a batch of subjects.
        """
        splits = self.epoch_splits(subject_ids, batch_n_admissions,
                                   ignore_first_admission)
        for split in splits:
            yield self.fetch_device_batch(split)

    def n_admissions(self,
                     subject_ids=None,
                     ignore_first_admission: bool = False):
        """Get the total number of admissions.

        Args:
            subject_ids: list of subject IDs.
            ignore_first_admission (bool, optional): Whether to ignore the first admission from the counts. Defaults to False.

        Returns:
            int: Total number of admissions.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        if ignore_first_admission:
            return sum(
                len(self.subjects[s].admissions) - 1 for s in subject_ids)
        return sum(len(self.subjects[s].admissions) for s in subject_ids)

    def iter_obs(self, subject_ids=None) -> Iterable[InpatientObservables]:
        """Iterate over the observables for the given subject IDs.

        Args:
            subject_ids: list of subject IDs.

        Yields:
            InpatientObservables: InpatientObservables object.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        for s in subject_ids:
            for adm in self.subjects[s].admissions:
                yield adm.observables

    def iter_lead_obs(self, subject_ids=None) -> Iterable[InpatientObservables]:
        """Iterate over the leading observables for the given subject IDs.

        Args:
            subject_ids: list of subject IDs.

        Yields:
            InpatientObservables: InpatientObservables object.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        for s in subject_ids:
            for adm in self.subjects[s].admissions:
                yield adm.leading_observable

    def n_obs_times(self, subject_ids=None):
        """Get the total number of observation times.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            int: total number of observation times.
        """
        return sum(len(obs) for obs in self.iter_obs(subject_ids))

    def d2d_interval_days(self, subject_ids=None):
        """Get the total number of days between first discharge and last discharge.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            int: total number of days between first discharge and last discharge.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(self.subjects[s].d2d_interval_days for s in subject_ids)

    def interval_days(self, subject_ids=None):
        """Get the total number of days in-hospital.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            int: total number of days in-hospital.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(a.interval_days for s in subject_ids for a in self.subjects[s].admissions)

    def interval_hours(self, subject_ids=None):
        """Get the total number of hours in-hospital.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            int: Total number of hours in-hospital.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(a.interval_hours for s in subject_ids for a in self.subjects[s].admissions)

    def p_obs(self, subject_ids: list[str] = None) -> float:
        """For a colelction of subjects, compute a measure that is proportional to rate of presence per observation timestamp.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            float: proportion of observables presence per unique timestamp.
        """
        return sum(obs.mask.sum() for obs in self.iter_obs(subject_ids)) / self.n_obs_times()

    def obs_coocurrence_matrix(self, subject_ids=None):
        """Compute the co-occurrence (or co-presence) matrix of observables.

        Returns:
            jnp.ndarray: co-occurrence (or co-presence) matrix of observables.
        """
        obs = [obs.mask for obs in self.iter_obs(subject_ids) if len(obs) > 0]
        obs = jnp.vstack(obs, dtype=int)
        return obs.T @ obs

    def size_in_bytes(self):
        """Get the size of the Patients object in bytes.

        Returns:
            int: size of the Patients object in bytes.
        """
        is_arr = eqx.filter(self.subjects, eqx.is_array)
        arr_size = jtu.tree_map(
            lambda a, m: a.size * a.itemsize
            if m is not None else 0, self.subjects, is_arr)
        return sum(jtu.tree_leaves(arr_size))

    def _unscaled_observation(self, obs: InpatientObservables) -> InpatientObservables:
        """Unscale the observation values, undo the preprocessing scaling.

        Args:
            obs (InpatientObservables): observation to unscale.

        Returns:
            InpatientObservables: unscaled observation.
        """
        obs_scaler = self.numerical_processors.scalers.obs
        value = obs_scaler.unscale(obs.value)
        return InpatientObservables(time=obs.time, value=value, mask=obs.mask)

    def _unscaled_leading_observable(self, lead: InpatientObservables, code_index: int):
        """Unscale the leading observable values, undo the preprocessing scaling.

        Args:
            lead (InpatientObservables): leading observable to unscale.

        Returns:
            InpatientObservables: unscaled leading observable.
        """
        lead_scaler = self.numerical_processors.scalers.obs
        value = lead_scaler.unscale_code(lead.value, code_index)
        return InpatientObservables(time=lead.time, value=value, mask=lead.mask)

    def subject_size_in_bytes(self, subject_id):
        """Get the size of the subject object in bytes.

        Args:
            subject_id (str): subject ID.

        Returns:
            int: size of the subject object in bytes.
        """
        is_arr = eqx.filter(self.subjects[subject_id], eqx.is_array)
        arr_size = jtu.tree_map(
            lambda a, m: a.size * a.itemsize
            if m is not None else 0, self.subjects[subject_id], is_arr)
        return sum(jtu.tree_leaves(arr_size))

    def outcome_frequency_vec(self, subjects: list[str]):
        """Get the outcome frequency vector for a list of subjects.

        Args:
            subjects (list[str]): list of subject IDs.

        Returns:
            jnp.ndarray: outcome frequency vector.
        """
        return sum(self.subjects[i].outcome_frequency_vec() for i in subjects)

    def outcome_frequency_partitions(self, n_partitions, subjects: list[str]):
        """
        Get the outcome codes partitioned by their frequency of occurrence into `n_partitions` partitions. The codes in each partition contributes to 1 / n_partitions of the all outcome occurrences.
        
        Args:
            n_partitions (int): number of partitions.
            subjects (list[str]): list of subject IDs.

        Returns:
            list[list[int]]: list of outcome codes partitioned by frequency into `n_partitions` partitions.
        
        """
        frequency_vec = self.outcome_frequency_vec(subjects)
        frequency_vec = frequency_vec / frequency_vec.sum()
        sorted_codes = np.argsort(frequency_vec)
        frequency_vec = frequency_vec[sorted_codes]
        cumsum = np.cumsum(frequency_vec)
        partitions = np.linspace(0, 1, n_partitions + 1)[1:-1]
        splitters = np.searchsorted(cumsum, partitions)
        return np.hsplit(sorted_codes, splitters)

    def outcome_first_occurrence(self, subject_id):
        """Get the first occurrence admission index of each outcome for a subject. If an outcome does not occur, the index is set to -1.

        Args:
            subject_id (str): subject ID.

        Returns:
            int: first occurrence admission index of each outcome for a subject.
        """
        return outcome_first_occurrence(self.subjects[subject_id].admissions)

    def outcome_first_occurrence_masks(self, subject_id):
        """Get a list of masks indicating whether an outcome occurs for a subject for the first time.

        Args:
            subject_id (str): subject ID.

        Returns:
            list[bool]: list of masks indicating whether an outcome occurs for a subject for the first time.

        """
        adms = self.subjects[subject_id].admissions
        first_occ_adm_id = outcome_first_occurrence(adms)
        return [first_occ_adm_id == a.admission_id for a in adms]

    def outcome_all_masks(self, subject_id):
        """Get a list of full-masks with the same shape as the outcome vector."""
        adms = self.subjects[subject_id].admissions
        if isinstance(adms[0].outcome.vec, jnp.ndarray):
            _np = jnp
        else:
            _np = np
        return [_np.ones_like(a.outcome.vec, dtype=bool) for a in adms]


class SegmentedTVxEHR(TVxEHR):
    subjects: dict[str, SegmentedPatient]
    patient_class: ClassVar[type[SegmentedPatient]] = SegmentedPatient

    def execute_pipeline(self, pipeline: AbstractDatasetPipeline,
                         schemes_context: CodingSchemesManager) -> AbstractProcessedDataset:
        raise NotImplementedError("SegmentedPatient is a final representation. It cannot have a pipeline.")

    def iter_obs(self, subject_ids=None) -> Iterable[InpatientObservables]:
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        for s in subject_ids:
            for adm in self.subjects[s].admissions:
                for obs_segment in adm.observables:
                    yield obs_segment

    def iter_lead_obs(self, subject_ids=None) -> Iterable[InpatientObservables]:
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        for s in subject_ids:
            for adm in self.subjects[s].admissions:
                yield adm.leading_observable

    @classmethod
    def from_tvx_ehr(cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, maximum_padding: int = 100) -> Self:
        dataset_scheme_proxy = tvx_ehr.dataset.scheme_proxy(schemes_context)
        tvx_scheme_proxy = tvx_ehr.scheme_proxy(schemes_context)
        hosp_procedures_size = len(tvx_scheme_proxy.hosp_procedures) if tvx_scheme_proxy.hosp_procedures else None
        icu_procedures_size = len(tvx_scheme_proxy.icu_procedures) if tvx_scheme_proxy.icu_procedures else None
        icu_inputs_size = len(dataset_scheme_proxy.icu_inputs) if dataset_scheme_proxy.icu_inputs else None
        subjects = {k: SegmentedPatient.from_patient(v, hosp_procedures_size=hosp_procedures_size,
                                                     icu_procedures_size=icu_procedures_size,
                                                     icu_inputs_size=icu_inputs_size,
                                                     maximum_padding=maximum_padding) for k, v in
                    tvx_ehr.subjects.items()}
        return SegmentedTVxEHR(config=tvx_ehr.config, dataset=tvx_ehr.dataset,
                               numerical_processors=tvx_ehr.numerical_processors,
                               subjects=subjects, splits=tvx_ehr.splits, pipeline_report=tvx_ehr.pipeline_report)
