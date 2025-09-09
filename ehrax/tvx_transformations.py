import random
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Hashable, Generator
from typing import Any, cast

import equinox as eqx
import numpy as np
import pandas as pd

from ._literals import SplitLiteral
from .coding_scheme import CodeMap, CodingSchemesManager, NumericScheme
from .dataset import (
    AbstractTransformation,
    AdmissionIntervalEventsTableColumns,
    AdmissionIntervalRatesTableColumns,
    Dataset,
    Report,
)
from .transformations import DatasetTransformation
from .tvx_concepts import (
    Admission,
    CodesVector,
    InpatientInput,
    InpatientInterventions,
    InpatientObservables,
    LeadingObservableExtractor,
    Patient,
    StaticInfo,
)
from .tvx_ehr import (
    CodedValueProcessor,
    CodedValueScaler,
    IQROutlierRemoverConfig,
    ScalerConfig,
    SegmentedTVxEHR,
    TVxEHR,
    TVxReport,
    TVxReportAttributes,
)
from .utils import dataframe_log


def dataset_surgery(getter: Callable[[TVxEHR], Any], dataset: TVxEHR, replacement: Any) -> TVxEHR:
    return eqx.tree_at(getter, dataset, replacement)


class TrainableTransformation(AbstractTransformation, metaclass=ABCMeta):
    # dependencies: ClassVar[tuple[type[DatasetTransformation], ...]] = (RandomSplits, setIndex, SetCodeIntegerIndices)

    @staticmethod
    def get_admission_ids(tvx_ehr: TVxEHR) -> list[str]:
        c_subject_id = tvx_ehr.dataset.config.columns.static.subject_id
        c_admission_id = tvx_ehr.dataset.config.columns.admissions.admission_id
        admissions = tvx_ehr.dataset.tables.admissions[[c_subject_id]]
        assert c_admission_id in admissions.index.names, f"Column {c_admission_id} not found in admissions table index."
        assert tvx_ehr.splits is not None, "TVxEHR.splits is None"
        training_subject_ids = tvx_ehr.splits[0]
        return admissions.loc[admissions.loc[:, c_subject_id].isin(training_subject_ids)].index.unique().tolist()


class TVxTransformation(AbstractTransformation[TVxEHR, TVxReport]):
    @classmethod
    @abstractmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[TVxEHR, TVxReport]: ...


class SampleSubjects(TVxTransformation):
    @classmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[TVxEHR, TVxReport]:
        static = tvx_ehr.dataset.tables.static
        # assert index name is subject_id
        c_subject_id = tvx_ehr.dataset.config.columns.static.subject_id
        assert c_subject_id in static.index.names, f"Index name must be {c_subject_id}"
        config = tvx_ehr.config.sample
        if config is None:
            return cls.skip(tvx_ehr, report, reason="config.sample is not configured.")

        rng = random.Random(config.seed)
        subjects = static.index.unique().tolist()
        rng.shuffle(subjects)
        subjects = subjects[config.offset : config.offset + config.n_subjects]
        n1 = len(static)
        static = static.loc[subjects]
        n2 = len(static)
        report = report.add(
            table="static",
            column="index",
            before=n1,
            after=n2,
            value_type="count",
            transformation=cls,
            operation="sample",
        )
        dataset = eqx.tree_at(lambda x: x.tables.static, tvx_ehr.dataset, static)
        r = cast(Report, report)
        dataset, r = DatasetTransformation.synchronize_subjects(dataset, r)
        tvx_ehr = eqx.tree_at(lambda x: x.dataset, tvx_ehr, dataset)
        return tvx_ehr, cast(TVxReport, r)


class RandomSplits(TVxTransformation):
    @classmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[TVxEHR, TVxReport]:
        config = tvx_ehr.config.splits

        if config is None:
            return cls.skip(tvx_ehr, report, "config.split is not configured.")

        splits = tvx_ehr.dataset.random_splits(
            splits=config.split_quantiles,
            random_seed=config.seed,
            balance=config.balance,
            discount_first_admission=config.discount_first_admission,
        )

        report = report.add(
            table="static",
            column=None,
            value_type="splits",
            transformation=cls,
            operation="TVxEHR.splits<-TVxEHR.dataset.random_splits(TVxEHR.config.splits)",
            before=(len(tvx_ehr.dataset.tables.static),),
            after=tuple(len(x) for x in splits),
        )
        tvx_ehr = eqx.tree_at(lambda x: x.splits, tvx_ehr, splits, is_leaf=lambda x: x is None)
        return tvx_ehr, report


class TrainingSplitGroups(TVxTransformation):
    @classmethod
    def sync_dataset(cls, dataset: Dataset, subject_ids: tuple[str, ...]) -> Dataset:
        static = dataset.tables.static
        c_subject_id = dataset.config.columns.static.subject_id
        assert c_subject_id in static.index.names, f"Index name must be {c_subject_id}"
        static = static.loc[list(subject_ids)]
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return DatasetTransformation.synchronize_subjects(dataset, Report())[0]

    @classmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[TVxEHR, TVxReport]:
        raise NotImplementedError("Use TrainingSplitGroups.split_dataset()")

    @classmethod
    def subset(cls, tvx_ehr: TVxEHR, group: tuple[str, ...]) -> TVxEHR:
        dataset = cls.sync_dataset(tvx_ehr.dataset, group)
        assert tvx_ehr.subjects is not None, "tvx_ehr.subjects is None"
        subjects = {subject_id: subject for subject_id, subject in tvx_ehr.subjects.items() if subject_id in group}
        tvx_ehr = eqx.tree_at(lambda x: x.dataset, tvx_ehr, dataset)
        tvx_ehr = eqx.tree_at(lambda x: x.subjects, tvx_ehr, subjects)
        tvx_ehr = eqx.tree_at(lambda x: x.splits, tvx_ehr, None, is_leaf=lambda x: x is None)
        return tvx_ehr

    @classmethod
    def split_dataset(
        cls, tvx_ehr: TVxEHR, n_groups: int, seed: int = 0, split_balance: SplitLiteral | None = None
    ) -> tuple[TVxEHR, ...]:
        assert tvx_ehr.config.splits is not None and tvx_ehr.splits is not None, "splits is None"
        training_split = tvx_ehr.splits[0]
        dataset = cls.sync_dataset(tvx_ehr.dataset, training_split)
        split_quantiles = np.linspace(0, 1, n_groups + 1)[1:-1]
        groups = dataset.random_splits(
            splits=split_quantiles.tolist(),
            random_seed=seed,
            balance=split_balance or tvx_ehr.config.splits.balance,  # type: ignore
            discount_first_admission=tvx_ehr.config.splits.discount_first_admission,
        )
        return tuple(cls.subset(tvx_ehr, tuple(group)) for group in groups)


class ZScoreScaler(CodedValueScaler):
    mean: pd.Series
    std: pd.Series

    def __init__(
        self,
        config: ScalerConfig,
        mean: pd.Series = pd.Series(),
        std: pd.Series = pd.Series(),
        table_name: str | None = None,
        code_column: str | None = None,
        value_column: str | None = None,
    ):
        super().__init__(config=config, table_name=table_name, code_column=code_column, value_column=value_column)
        self.mean = mean
        self.std = std

    @property
    def original_dtype(self) -> np.dtype:
        return self.mean.dtype  # type: ignore

    def __call__(self, dataset: Dataset) -> Dataset:
        table = self.table_getter(dataset)

        mean = table.loc[:, self.code_column].map(self.mean)
        std = table.loc[:, self.code_column].map(self.std)
        table.loc[:, self.value_column] = (table.loc[:, self.value_column] - mean) / std
        if self.config.use_float16:
            table = table.astype({self.value_column: np.float16})

        return eqx.tree_at(lambda x: self.table_getter(dataset), dataset, table)

    def unscale(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(self.original_dtype)
        index = np.arange(array.shape[-1])
        return array * self.std.loc[index].values + self.mean.loc[index].values

    def unscale_code(self, array: np.ndarray, code_index: Hashable) -> np.ndarray:
        array = array.astype(self.original_dtype)
        return array * self.std.loc[code_index] + self.mean.loc[code_index]

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> dict[str, pd.Series]:
        stat = df.groupby(c_code)[[c_value]].apply(
            lambda x: pd.Series({"mu": x[c_value].mean(), "sigma": x[c_value].std()})
        )
        return dict(mean=stat.loc[:, "mu"], std=stat.loc[:, "sigma"])


class MaxScaler(CodedValueScaler):
    max_val: pd.Series

    def __init__(
        self,
        config: ScalerConfig,
        max_val: pd.Series = pd.Series(),
        table_name: str | None = None,
        code_column: str | None = None,
        value_column: str | None = None,
    ):
        super().__init__(config=config, table_name=table_name, code_column=code_column, value_column=value_column)
        self.max_val = max_val

    @property
    def original_dtype(self) -> np.dtype:
        return self.max_val.dtype  # type: ignore

    def __call__(self, dataset: Dataset) -> Dataset:
        df = self.table_getter(dataset).copy()

        max_val = df.loc[:, self.code_column].map(self.max_val)
        df.loc[:, self.value_column] = df.loc[:, self.value_column] / max_val
        if self.config.use_float16:
            df = df.astype({self.value_column: np.float16})
        return eqx.tree_at(self.table_getter, dataset, df)

    def unscale(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(self.original_dtype)
        if array.shape[-1] == len(self.max_val):
            index = np.arange(array.shape[-1])
            return array * self.max_val.loc[index].values
        index = np.array(self.max_val.index.values)
        array = array.copy()
        if array.ndim == 1:
            array[index] *= self.max_val.values
        else:
            array[:, index] *= self.max_val.values
        return array

    def unscale_code(self, array: np.ndarray, code_index: Hashable) -> np.ndarray:
        array = array.astype(self.original_dtype)
        return array * self.max_val.loc[code_index]

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> dict[str, pd.Series]:
        stat = df.groupby(c_code)[[c_value]].apply(lambda x: pd.Series({"max": x[c_value].max()}))
        return dict(max_val=stat.loc[:, "max"])


class AdaptiveScaler(CodedValueScaler):
    max_val: pd.Series
    min_val: pd.Series
    mean: pd.Series
    std: pd.Series

    def __init__(
        self,
        config: ScalerConfig,
        max_val: pd.Series = pd.Series(),
        min_val: pd.Series = pd.Series(),
        mean: pd.Series = pd.Series(),
        std: pd.Series = pd.Series(),
        table_name: str | None = None,
        code_column: str | None = None,
        value_column: str | None = None,
    ):
        super().__init__(config=config, table_name=table_name, code_column=code_column, value_column=value_column)
        self.max_val = max_val
        self.min_val = min_val
        self.mean = mean
        self.std = std

    @property
    def original_dtype(self) -> np.dtype:
        return self.max_val.dtype  # type: ignore

    def __call__(self, dataset: Dataset) -> Dataset:
        df = self.table_getter(dataset).copy()

        min_val = df.loc[:, self.code_column].map(self.min_val)
        max_val = df.loc[:, self.code_column].map(self.max_val)
        mean = df.loc[:, self.code_column].map(self.mean)
        std = df.loc[:, self.code_column].map(self.std)

        minmax_scaled = (df.loc[:, self.value_column] - min_val) / max_val
        z_scaled = (df.loc[:, self.value_column] - mean) / std

        df.loc[:, self.value_column] = np.where(min_val >= 0.0, minmax_scaled, z_scaled)
        if self.config.use_float16:
            df = df.astype({self.value_column: np.float16})
        return eqx.tree_at(self.table_getter, dataset, df)

    def unscale(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(self.original_dtype)
        index = np.arange(array.shape[-1])
        mu = self.mean.loc[index].values
        sigma = self.std.loc[index].values
        min_val = self.min_val.loc[index].values
        max_val = self.max_val.loc[index].values
        z_unscaled = array * sigma + mu
        minmax_unscaled = array * max_val + min_val
        return np.where(min_val >= 0.0, minmax_unscaled, z_unscaled)

    def unscale_code(self, array: np.ndarray, code_index: Hashable) -> np.ndarray:
        array = array.astype(self.original_dtype)
        mu = self.mean.loc[code_index]  # type: ignore
        sigma = self.std.loc[code_index]  # type: ignore
        min_val = self.min_val.loc[code_index]  # type: ignore
        max_val = self.max_val.loc[code_index]  # type: ignore
        z_unscaled = array * sigma + mu
        minmax_unscaled = array * max_val + min_val
        return np.where(min_val >= 0.0, minmax_unscaled, z_unscaled)

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> dict[str, pd.Series]:
        stat = df.groupby(c_code)[[c_value]].apply(
            lambda x: pd.Series(
                {"mu": x[c_value].mean(), "sigma": x[c_value].std(), "min": x[c_value].min(), "max": x[c_value].max()}
            )
        )
        return dict(
            mean=stat.loc[:, "mu"], std=stat.loc[:, "sigma"], min_val=stat.loc[:, "min"], max_val=stat.loc[:, "max"]
        )


class IQROutlierRemover(CodedValueProcessor):
    config: IQROutlierRemoverConfig
    min_val: pd.Series
    max_val: pd.Series

    def __init__(
        self,
        config: IQROutlierRemoverConfig,
        min_val: pd.Series = pd.Series(),
        max_val: pd.Series = pd.Series(),
        table_name: str | None = None,
        code_column: str | None = None,
        value_column: str | None = None,
    ):
        super().__init__(config=config, table_name=table_name, code_column=code_column, value_column=value_column)
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, dataset: Dataset) -> Dataset:
        table = self.table_getter(dataset)

        min_val = table.loc[:, self.code_column].map(self.min_val)
        max_val = table.loc[:, self.code_column].map(self.max_val)
        table = table.loc[table.loc[:, self.value_column].between(min_val, max_val)]

        return eqx.tree_at(self.table_getter, dataset, table)

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> dict[str, pd.Series]:
        outlier_q = np.array([self.config.outlier_q1, self.config.outlier_q2])
        q = df.groupby(c_code)[[c_value]].apply(lambda x: x[c_value].quantile(outlier_q))

        q.columns = ["q1", "q2"]
        q["iqr"] = q["q2"] - q["q1"]
        q["out_q1"] = q["q1"] - self.config.outlier_iqr_scale * q["iqr"]
        q["out_q2"] = q["q2"] + self.config.outlier_iqr_scale * q["iqr"]

        stat = df.groupby(c_code)[[c_value]].apply(
            lambda x: pd.Series({"mu": x[c_value].mean(), "sigma": x[c_value].std()})
        )

        stat["out_z1"] = stat["mu"] - self.config.outlier_z1 * stat["sigma"]
        stat["out_z2"] = stat["mu"] + self.config.outlier_z2 * stat["sigma"]
        return dict(min_val=np.minimum(q["out_q1"], stat["out_z1"]), max_val=np.maximum(q["out_q2"], stat["out_z2"]))  # type: ignore


class ObsIQROutlierRemover(TrainableTransformation):
    @classmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[TVxEHR, TVxReport]:
        assert tvx_ehr.config.numerical_processors.outlier_removers is not None, (
            "tvx_ehr.config.numerical_processors.outlier_removers is None"
        )
        config = tvx_ehr.config.numerical_processors.outlier_removers.obs
        if config is None:
            return cls.skip(tvx_ehr, report, "config.numerical_processors.outlier_removers.obs is None")
        remover = IQROutlierRemover(config=config).fit(
            tvx_ehr.dataset,
            cls.get_admission_ids(tvx_ehr),
            table_name="obs",
            code_column=tvx_ehr.dataset.config.columns.obs.code,
            value_column=tvx_ehr.dataset.config.columns.obs.measurement,
        )
        tvx_ehr = eqx.tree_at(
            lambda x: x.numerical_processors.outlier_removers.obs, tvx_ehr, remover, is_leaf=lambda x: x is None
        )
        report = report.add(
            table="obs",
            column=None,
            value_type="type",
            transformation=cls,
            operation="TVxEHR.numerical_processors.outlier_removers.obs <- IQROutlierRemover",
            after=type(remover),
        )
        assert tvx_ehr.dataset.tables.obs is not None, "tvx_ehr.dataset.tables.obs is None"
        n1 = len(tvx_ehr.dataset.tables.obs)
        # TODO: report specific removals stats for each code.
        tvx_ehr = eqx.tree_at(lambda x: x.dataset, tvx_ehr, remover(tvx_ehr.dataset))
        assert tvx_ehr.dataset.tables.obs is not None, "tvx_ehr.dataset.tables.obs is None"
        n2 = len(tvx_ehr.dataset.tables.obs)
        report = report.add(
            table="obs", column=None, value_type="count", transformation=cls, operation="filter", before=n1, after=n2
        )
        return tvx_ehr, report


class ObsAdaptiveScaler(TrainableTransformation):
    @classmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[TVxEHR, TVxReport]:
        assert tvx_ehr.config.numerical_processors.scalers is not None, (
            "tvx_ehr.config.numerical_processors.scalers is None"
        )
        config = tvx_ehr.config.numerical_processors.scalers.obs

        if config is None:
            return cls.skip(tvx_ehr, report, "config.numerical_processors.scalers.obs is None")

        value_column = tvx_ehr.dataset.config.columns.obs.measurement
        scaler = AdaptiveScaler(config=config).fit(
            tvx_ehr.dataset,
            cls.get_admission_ids(tvx_ehr),
            table_name="obs",
            code_column=tvx_ehr.dataset.config.columns.obs.code,
            value_column=value_column,
        )
        tvx_ehr = eqx.tree_at(
            lambda x: x.numerical_processors.scalers.obs, tvx_ehr, scaler, is_leaf=lambda x: x is None
        )
        report = report.add(
            table="obs",
            column=None,
            value_type="type",
            transformation=cls,
            operation="TVxEHR.numerical_processors.scalers.obs <- AdaptiveScaler",
            after=type(scaler),
        )
        assert tvx_ehr.dataset.tables.obs is not None, "tvx_ehr.dataset.tables.obs is None"
        dtype1 = tvx_ehr.dataset.tables.obs[value_column].dtype
        tvx_ehr = eqx.tree_at(lambda x: x.dataset, tvx_ehr, scaler(tvx_ehr.dataset))
        assert tvx_ehr.dataset.tables.obs is not None, "tvx_ehr.dataset.tables.obs is None"
        dtype2 = tvx_ehr.dataset.tables.obs[value_column].dtype
        report = report.add(
            table="obs",
            column=value_column,
            value_type="dtype",
            transformation=cls,
            operation=f"scaled_and_maybe_cast_{scaler.config.use_float16}",
            before=dtype1,
            after=dtype2,
        )
        return tvx_ehr, report


class InputScaler(TrainableTransformation):
    @classmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[TVxEHR, TVxReport]:
        code_column = tvx_ehr.dataset.config.columns.icu_inputs.code
        value_column = tvx_ehr.dataset.config.columns.icu_inputs.derived_normalized_amount_per_hour
        assert tvx_ehr.config.numerical_processors.scalers is not None, (
            "tvx_ehr.config.numerical_processors.scalers is None"
        )
        config = tvx_ehr.config.numerical_processors.scalers.icu_inputs

        if config is None:
            return cls.skip(tvx_ehr, report, "config.numerical_processors.scalers.icu_inputs is None")

        scaler = MaxScaler(config=config).fit(
            tvx_ehr.dataset,
            cls.get_admission_ids(tvx_ehr),
            table_name="icu_inputs",
            code_column=code_column,
            value_column=value_column,
        )

        tvx_ehr = eqx.tree_at(
            lambda x: x.numerical_processors.scalers.icu_inputs, tvx_ehr, scaler, is_leaf=lambda x: x is None
        )
        report = report.add(
            table="icu_inputs",
            column=None,
            value_type="type",
            operation="TVxEHR.numerical_processors.scalers.icu_inputs <- MaxScaler",
            after=type(scaler),
        )
        assert tvx_ehr.dataset.tables.icu_inputs is not None, "tvx_ehr.dataset.tables.icu_inputs is None"
        dtype1 = tvx_ehr.dataset.tables.icu_inputs[value_column].dtype
        tvx_ehr = eqx.tree_at(lambda x: x.dataset, tvx_ehr, scaler(tvx_ehr.dataset))
        assert tvx_ehr.dataset.tables.icu_inputs is not None, "tvx_ehr.dataset.tables.icu_inputs is None"
        dtype2 = tvx_ehr.dataset.tables.icu_inputs[value_column].dtype
        report = report.add(
            table="icu_inputs",
            column=value_column,
            value_type="dtype",
            transformation=cls,
            operation=f"scaled_and_maybe_cast_{scaler.config.use_float16}",
            before=dtype1,
            after=dtype2,
        )
        return tvx_ehr, report


class InterventionSegmentation(TVxTransformation):
    @classmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[SegmentedTVxEHR | TVxEHR, TVxReport]:
        if tvx_ehr.config.interventions_segmentation in (None, False) or not tvx_ehr.config.interventions:
            reason = []
            if tvx_ehr.config.interventions_segmentation in (None, False):
                reason.append("config.interventions_segmentation is None/False")
            if not tvx_ehr.config.interventions:
                reason.append("config.interventions is False")
            return cls.skip(tvx_ehr, report, reason=" + ".join(reason))

        maximum_padding = 100
        segmented_tvx_ehr = SegmentedTVxEHR.from_tvx_ehr(
            tvx_ehr, schemes_context=schemes_context, maximum_padding=maximum_padding
        )
        tvx_concept_path = TVxReportAttributes.admission_attribute_prefix("observables", InpatientObservables)
        report = report.add(
            tvx_concept=tvx_concept_path,
            value_type="concepts_count",
            operation=f"segmentation(maximum_padding={maximum_padding})",
            transformation=cls,
            before=sum(1 for _ in tvx_ehr.iter_obs()),
            after=sum(1 for _ in segmented_tvx_ehr.iter_obs()),
        )
        return segmented_tvx_ehr, report


class ObsTimeBinning(TVxTransformation):
    @classmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[TVxEHR, TVxReport]:
        if tvx_ehr.config.time_binning is None:
            return cls.skip(tvx_ehr, report, reason="config.time_binning is None")

        interval = tvx_ehr.config.time_binning
        obs_scheme = tvx_ehr.scheme_proxy(schemes_context).obs
        assert isinstance(obs_scheme, NumericScheme), "obs_scheme is not NumericScheme"
        assert tvx_ehr.subjects is not None, "tvx_ehr.subjects is None"

        tvx_concept_path = TVxReportAttributes.admission_attribute_prefix("observables", InpatientObservables)
        tvx_binned_ehr = eqx.tree_at(
            lambda x: x.subjects,
            tvx_ehr,
            {
                subject_id: subject.observables_time_binning(interval, obs_scheme.types)
                for subject_id, subject in tvx_ehr.subjects.items()
            },
        )

        report = report.add(
            tvx_concept=tvx_concept_path,
            value_type="concepts_count",
            operation="time_binning",
            transformation=cls,
            before=sum(1 for _ in tvx_ehr.iter_obs()),
            after=sum(1 for _ in tvx_binned_ehr.iter_obs()),
        )
        report = report.add(
            tvx_concept=tvx_concept_path,
            value_type="timestamps_count",
            operation="time_binning",
            transformation=cls,
            before=sum(len(o) for o in tvx_ehr.iter_obs()),
            after=sum(len(o) for o in tvx_binned_ehr.iter_obs()),
        )
        report = report.add(
            tvx_concept=tvx_concept_path,
            transformation=cls,
            value_type="values_count",
            operation="time_binning",
            before=sum(o.count for o in tvx_ehr.iter_obs()),
            after=sum(o.count for o in tvx_binned_ehr.iter_obs()),
        )
        return tvx_binned_ehr, report


class LeadingObservableExtraction(TVxTransformation):
    @classmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[TVxEHR, TVxReport]:
        if tvx_ehr.config.leading_observable is None:
            return cls.skip(tvx_ehr, report, reason="config.leading_observable is None")
        obs_scheme = tvx_ehr.dataset.scheme_proxy(schemes_context).obs
        assert isinstance(obs_scheme, NumericScheme), "obs_scheme is not NumericScheme"
        assert tvx_ehr.subjects is not None, "tvx_ehr.subjects is None"
        extractor = LeadingObservableExtractor(tvx_ehr.config.leading_observable, observable_scheme=obs_scheme)
        tvx_concept_path = TVxReportAttributes.admission_attribute_prefix("leading_observables", InpatientObservables)
        tvx_ehr = eqx.tree_at(
            lambda x: x.subjects,
            tvx_ehr,
            {
                subject_id: subject.extract_leading_observables(extractor)
                for subject_id, subject in tvx_ehr.subjects.items()
            },
        )

        report = report.add(
            tvx_concept=tvx_concept_path,
            value_type="concepts_count",
            operation="LeadingObservableExtractor",
            transformation=cls,
            after=sum(1 for _ in tvx_ehr.iter_lead_obs()),
        )
        report = report.add(
            tvx_concept=tvx_concept_path,
            value_type="timestamps_count",
            operation="LeadingObservableExtractor",
            transformation=cls,
            after=sum(len(lo) for lo in tvx_ehr.iter_lead_obs()),
        )
        report = report.add(
            tvx_concept=tvx_concept_path,
            transformation=cls,
            value_type="values_count",
            operation="LeadingObservableExtractor",
            after=sum(lo.count for lo in tvx_ehr.iter_lead_obs()),
        )
        report = report.add(
            transformation=cls,
            tvx_concept=tvx_concept_path,
            value_type="type",
            operation="LeadingObservableExtractor",
            after=InpatientObservables,
        )
        return tvx_ehr, report


class TVxConcepts(TVxTransformation):
    @classmethod
    def _static_info(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[dict[str, StaticInfo], TVxReport]:
        tvx_scheme_proxy = tvx_ehr.scheme_proxy(schemes_context)
        static = tvx_ehr.dataset.tables.static
        static_config = tvx_ehr.dataset.config.columns.static
        config = tvx_ehr.config.demographic
        c_gender = static_config.gender
        c_date_of_birth = static_config.date_of_birth
        c_ethnicity = static_config.race

        report = report.add(
            transformation=cls,
            table="static",
            column=None,
            value_type="count",
            operation="extract_static_info",
            after=len(static),
        )

        gender, ethnicity, dob = {}, {}, {}
        if c_date_of_birth in static.columns and config.age:
            dob = static[c_date_of_birth].to_dict()
        if tvx_scheme_proxy.gender is not None and config.gender:
            gender_m = tvx_scheme_proxy.gender_mapper(tvx_ehr.dataset.config.scheme)
            assert gender_m is not None, "gender_m is None"
            target_scheme = tvx_scheme_proxy.gender
            gender_dict = {
                subject_id: gender_m.map_codeset({c}) for subject_id, c in static[c_gender].to_dict().items()
            }
            gender = {subject_id: target_scheme.codeset2vec(codes) for subject_id, codes in gender_dict.items()}

        if tvx_scheme_proxy.ethnicity is not None and config.ethnicity:
            ethnicity_m = tvx_scheme_proxy.ethnicity_mapper(tvx_ehr.dataset.config.scheme)
            target_scheme = tvx_scheme_proxy.ethnicity
            assert ethnicity_m is not None, "ethnicity_m is None"
            ethnicity_dict = {
                subject_id: ethnicity_m.map_codeset({c}) for subject_id, c in static[c_ethnicity].to_dict().items()
            }
            ethnicity = {subject_id: target_scheme.codeset2vec(codes) for subject_id, codes in ethnicity_dict.items()}

        static_info = {
            subject_id: StaticInfo(
                date_of_birth=dob.get(subject_id), ethnicity=ethnicity.get(subject_id), gender=gender.get(subject_id)
            )
            for subject_id in static.index
        }
        report = report.add(
            tvx_concept=TVxReportAttributes.static_info_prefix(),
            transformation=cls,
            column=None,
            value_type="count",
            operation="extract_static_info",
            after=len(static_info),
        )
        return static_info, report

    @staticmethod
    def _dx_discharge(
        tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager
    ) -> tuple[dict[str, CodesVector], dict[str, frozenset[str]]]:
        tvx_scheme_proxy = tvx_ehr.scheme_proxy(schemes_context)
        c_adm_id = tvx_ehr.dataset.config.columns.dx_discharge.admission_id
        c_code = tvx_ehr.dataset.config.columns.dx_discharge.code
        df = tvx_ehr.dataset.tables.dx_discharge
        assert isinstance(df, pd.DataFrame), "tvx_ehr.dataset.tables.dx_discharge is not a DataFrame"
        dx_mapper = tvx_scheme_proxy.dx_mapper(tvx_ehr.dataset.config.scheme)
        assert dx_mapper is not None, "dx_mapper is None"
        target_scheme = tvx_scheme_proxy.dx_discharge
        assert target_scheme is not None, "target_scheme is None"
        n1 = len(df)
        dx_discharge = df[df[c_code].isin(dx_mapper.data)]
        n2 = len(dx_discharge)
        if n1 != n2:
            unique_codes_a = set(df[c_code])
            unique_codes_b = set(df[c_code]) & set(dx_mapper.data.keys())
            unique_removed = unique_codes_a - unique_codes_b
            n_uniq_a = len(unique_codes_a)
            n_uniq_rem = len(unique_removed)
            source_scheme = tvx_ehr.dataset.scheme_proxy(schemes_context).dx_discharge
            assert source_scheme is not None, "source_scheme is None"
            dataframe_log.info(
                f"In mapping ({dx_mapper.source_name}->{dx_mapper.target_name}), "
                f"some codes are not mapped to the target scheme.\n"
                f"{n1 - n2} / {n1} = {(n1 - n2) / n1: .2f} rows were removed.\n"
                f"{n_uniq_rem} / {n_uniq_a} = {n_uniq_rem / n_uniq_a: .2f} "
                f"unique codes were removed (see report).",
                dataframe=pd.DataFrame(
                    [(code, source_scheme.desc[code]) for code in unique_removed],
                    columns=pd.Series(["code", "description"]),
                ),
                tag="lost_dx_discharge_codes_unique",
            )

        dx_codes_set = dx_discharge.groupby(c_adm_id)[c_code].apply(set).to_dict()
        dx_codes_set = {k: dx_mapper.map_codeset(v) for k, v in dx_codes_set.items()}
        dx_codes_set = {adm_id: dx_codes_set.get(adm_id, frozenset()) for adm_id in tvx_ehr.admission_ids}

        return {adm_id: target_scheme.codeset2vec(codeset) for adm_id, codeset in dx_codes_set.items()}, dx_codes_set

    @staticmethod
    def _dx_discharge_history(
        tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, dx_discharge: dict[str, CodesVector]
    ) -> dict[str, CodesVector]:
        # TODO: test anti causality.
        dx_scheme = tvx_ehr.scheme_proxy(schemes_context).dx_discharge
        assert dx_scheme is not None, "tvx_ehr.scheme_proxy(...).dx_discharge is None"
        # For each subject accumulate previous dx_discharge codes.
        dx_discharge_history = dict()
        initial_history = dx_scheme.codeset2vec(set())
        # For each subject get the list of adm sorted by admission date.
        for subject_id, adm_ids in tvx_ehr.subjects_sorted_admission_ids.items():
            current_history = initial_history
            for adm_id in adm_ids:
                dx_discharge_history[adm_id] = current_history
                current_history = dx_discharge[adm_id].union(current_history)
        return dx_discharge_history

    @staticmethod
    def _outcome(
        tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, dx_discharge: dict[str, frozenset[str]]
    ) -> dict[str, CodesVector]:
        tvx_scheme_proxy = tvx_ehr.scheme_proxy(schemes_context)
        f_outcome = tvx_scheme_proxy.outcome
        assert f_outcome is not None, "tvx_scheme_proxy.outcome is None"
        return {str(adm_id): f_outcome(codeset) for adm_id, codeset in dx_discharge.items()}

    @staticmethod
    def _icu_inputs(tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager) -> dict[str, InpatientInput]:
        table_config = tvx_ehr.dataset.config.columns.icu_inputs
        c_admission_id = table_config.admission_id
        c_code = table_config.code
        c_rate = table_config.derived_normalized_amount_per_hour
        c_start = table_config.start_time
        c_end = table_config.end_time

        # Here we avoid deep copy, and we can still replace
        # a new column without affecting the original table.
        assert tvx_ehr.dataset.tables.icu_inputs is not None, "tvx_ehr.dataset.tables.icu_inputs is None"
        table = tvx_ehr.dataset.tables.icu_inputs.iloc[:, :]
        icu_scheme = tvx_ehr.dataset.scheme_proxy(schemes_context).icu_inputs
        assert icu_scheme is not None, "tvx_ehr.scheme_proxy(...).icu_inputs is None"
        table[c_code] = table[c_code].map(icu_scheme.index)
        assert not table[c_code].isnull().any(), "Some codes are not in the target scheme."
        return {
            str(adm_id): InpatientInput(
                code_index=np.array(x[c_code].to_numpy(), dtype=np.int64),
                rate=x[c_rate].to_numpy(),
                starttime=x[c_start].to_numpy(),
                endtime=x[c_end].to_numpy(),
            )
            for adm_id, x in table.groupby(c_admission_id)
        }

    @staticmethod
    def _procedures(
        schemes_context: CodingSchemesManager,
        table: pd.DataFrame,
        config: AdmissionIntervalEventsTableColumns | AdmissionIntervalRatesTableColumns,
        code_map: CodeMap,
    ) -> dict[str, InpatientInput]:
        c_admission_id = config.admission_id
        c_code = config.code
        c_start_time = config.start_time
        c_end_time = config.end_time

        table = code_map.map_dataframe(table, c_code)
        target_index = code_map.target_index(schemes_context.scheme[code_map.target_name])
        table.loc[:, c_code] = table.loc[:, c_code].map(target_index)
        assert not table.loc[:, c_code].isnull().any(), "Some codes are not in the target scheme."

        return {
            str(adm_id): InpatientInput(
                code_index=np.array(x[c_code].to_numpy(), dtype=np.int64),
                rate=np.ones_like(x[c_code].to_numpy(), dtype=bool),
                starttime=x[c_start_time].to_numpy(),
                endtime=x[c_end_time].to_numpy(),
            )
            for adm_id, x in table.groupby(c_admission_id)
        }

    @staticmethod
    def _hosp_procedures(tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager) -> dict[str, InpatientInput]:
        assert tvx_ehr.dataset.tables.hosp_procedures is not None, "tvx_ehr.dataset.tables.hosp_procedures is None"
        m = tvx_ehr.hosp_procedures_mapper(tvx_ehr.scheme_proxy(schemes_context))
        assert m is not None, "tvx_ehr.hosp_procedures_mapper(...) is None"
        return TVxConcepts._procedures(
            schemes_context,
            tvx_ehr.dataset.tables.hosp_procedures,
            tvx_ehr.dataset.config.columns.hosp_procedures,
            m,
        )

    @staticmethod
    def _icu_procedures(tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager) -> dict[str, InpatientInput]:
        assert tvx_ehr.dataset.tables.icu_procedures is not None, "tvx_ehr.dataset.tables.icu_procedures is None"
        m = tvx_ehr.icu_procedures_mapper(tvx_ehr.scheme_proxy(schemes_context))
        assert m is not None, "tvx_ehr.icu_procedures_mapper(...) is None"
        return TVxConcepts._procedures(
            schemes_context,
            tvx_ehr.dataset.tables.icu_procedures,
            tvx_ehr.dataset.config.columns.icu_procedures,
            m,
        )

    @classmethod
    def _interventions(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[dict[str, InpatientInterventions], TVxReport]:
        concept_path = TVxReportAttributes.inpatient_input_prefix
        hosp_procedures = cls._hosp_procedures(tvx_ehr, schemes_context)
        report = report.add(
            tvx_concept=concept_path("hosp_procedures"),
            transformation=cls,
            table="hosp_procedures",
            column=None,
            value_type="count",
            operation="extract_hosp_procedures",
            after=len(hosp_procedures),
        )

        icu_procedures = cls._icu_procedures(tvx_ehr, schemes_context)
        report = report.add(
            tvx_concept=concept_path("icu_procedures"),
            table="icu_procedures",
            transformation=cls,
            column=None,
            value_type="count",
            operation="extract_icu_procedures",
            after=len(icu_procedures),
        )

        icu_inputs = cls._icu_inputs(tvx_ehr, schemes_context)
        report = report.add(
            tvx_concept=concept_path("icu_inputs"),
            table="icu_inputs",
            transformation=cls,
            column=None,
            value_type="count",
            operation="extract_icu_inputs",
            after=len(icu_inputs),
        )
        interventions = {
            admission_id: InpatientInterventions(
                hosp_procedures=hosp_procedures.get(admission_id),
                icu_procedures=icu_procedures.get(admission_id),
                icu_inputs=icu_inputs.get(admission_id),
            )
            for admission_id in tvx_ehr.admission_ids
        }
        return interventions, report

    @classmethod
    def _observables(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[dict[str, InpatientObservables], TVxReport]:
        c_admission_id = tvx_ehr.dataset.config.columns.obs.admission_id
        c_code = tvx_ehr.dataset.config.columns.obs.code
        c_value = tvx_ehr.dataset.config.columns.obs.measurement
        c_timestamp = tvx_ehr.dataset.config.columns.obs.time

        obs_scheme = tvx_ehr.scheme_proxy(schemes_context).obs
        table = tvx_ehr.dataset.tables.obs
        assert isinstance(obs_scheme, NumericScheme), "obs_scheme is not NumericScheme"
        assert table is not None, "tvx_ehr.dataset.tables.obs is None"
        table = table.assign(**{c_code: table.loc[:, c_code].map(obs_scheme.index)})
        assert not table.loc[:, c_code].isnull().any(), "Some codes are not in the target scheme."
        obs_dim = len(obs_scheme)
        tvx_concept_path = TVxReportAttributes.admission_attribute_prefix("observables", InpatientObservables)

        def time_values(index: np.ndarray, values: np.ndarray) -> np.ndarray:
            value = np.zeros(obs_dim, dtype=np.float16)
            value[index] = values
            return value

        def time_mask(index: np.ndarray) -> np.ndarray:
            mask = np.zeros(obs_dim, dtype=bool)
            mask[index] = True
            return mask

        def inpatient_obs_data(admission_df: pd.DataFrame) -> Generator[tuple[Hashable, np.ndarray, np.ndarray]]:
            admission_df = admission_df.sort_values(c_timestamp)
            for timestamp, time_df in admission_df.groupby(c_timestamp):
                index = time_df.loc[:, c_code].values
                yield timestamp, time_values(index, time_df[c_value].values), time_mask(index)  # type: ignore

        def make_inpatient_obs(admission_df: pd.DataFrame) -> InpatientObservables:
            time, value, mask = zip(*inpatient_obs_data(admission_df))
            return InpatientObservables(
                time=np.stack(time, axis=0),
                value=np.stack(value, dtype=np.float16, axis=0),
                mask=np.stack(mask, dtype=bool, axis=0),
            )

        inpatient_observables = table.groupby(c_admission_id)[[c_timestamp, c_code, c_value]].apply(make_inpatient_obs)  # type: ignore
        assert len(inpatient_observables.index.tolist()) == len(set(inpatient_observables.index.tolist())), (
            "Duplicate admission ids in obs"
        )
        inpatient_observables = inpatient_observables.to_dict()
        empty_obs = InpatientObservables.empty(
            size=obs_dim,
            time_dtype=next(iter(inpatient_observables.values())).time.dtype,
            value_dtype=np.float16,
            mask_dtype=bool,
        )
        empty_obs_dict = {adm_id: empty_obs for adm_id in tvx_ehr.admission_ids if adm_id not in inpatient_observables}

        report = report.add(table="obs", value_type="table_size", operation="extract_observables", after=len(table))
        report = report.add(
            tvx_concept=tvx_concept_path,
            table="obs",
            value_type="concepts_count",
            operation="extract_observables",
            transformation=cls,
            after=len(inpatient_observables),
        )
        report = report.add(
            tvx_concept=tvx_concept_path,
            table="obs",
            value_type="empty_concepts_count",
            operation="extract_observables",
            transformation=cls,
            after=len(empty_obs_dict),
        )
        report = report.add(
            tvx_concept=TVxReportAttributes.admission_attribute_prefix("observables", InpatientObservables),
            table="obs",
            value_type="timestamps_count",
            operation="extract_observables",
            transformation=cls,
            after=sum(len(o) for o in inpatient_observables.values()),
        )
        report = report.add(
            tvx_concept=TVxReportAttributes.admission_attribute_prefix("observables", InpatientObservables),
            transformation=cls,
            table="obs",
            value_type="values_count",
            operation="extract_observables",
            after=sum(o.count for o in inpatient_observables.values()),
        )

        return inpatient_observables | empty_obs_dict, report

    @classmethod
    def apply(
        cls, tvx_ehr: TVxEHR, schemes_context: CodingSchemesManager, report: TVxReport
    ) -> tuple[TVxEHR, TVxReport]:
        subject_admissions = tvx_ehr.subjects_sorted_admission_ids
        static_info, report = cls._static_info(tvx_ehr, schemes_context, report)

        dx_discharge, dx_discharge_codeset = cls._dx_discharge(tvx_ehr, schemes_context)
        dx_discharge_history = cls._dx_discharge_history(tvx_ehr, schemes_context, dx_discharge)
        outcome = cls._outcome(tvx_ehr, schemes_context, dx_discharge_codeset)
        if tvx_ehr.config.interventions:
            interventions, report = cls._interventions(tvx_ehr, schemes_context, report)
        else:
            interventions = None
        if tvx_ehr.config.observables:
            observables, report = cls._observables(tvx_ehr, schemes_context, report)
        else:
            observables = None

        def _admissions(admission_ids: list[str]) -> list[Admission]:
            return [
                Admission(
                    admission_id=i,
                    admission_dates=tvx_ehr.admission_dates[i],
                    dx_codes=dx_discharge[i],
                    dx_codes_history=dx_discharge_history[i],
                    outcome=outcome[i],
                    observables=observables[i] if observables else None,
                    interventions=interventions[i] if interventions else None,
                )
                for i in admission_ids
            ]

        subjects = {
            subject_id: Patient(
                subject_id=subject_id, admissions=_admissions(admission_ids), static_info=static_info[subject_id]
            )
            for subject_id, admission_ids in subject_admissions.items()
        }
        tvx_ehr = eqx.tree_at(lambda x: x.subjects, tvx_ehr, subjects, is_leaf=lambda x: x is None)
        report = report.add(
            tvx_concept=TVxReportAttributes.subjects_prefix(),
            transformation=cls,
            value_type="count",
            operation="extract_subjects",
            after=len(subjects),
        )
        return tvx_ehr, report
