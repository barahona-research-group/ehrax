"""Data Model for Subjects in MIMIC-III and MIMIC-IV"""

import functools
from collections.abc import Callable, Iterator
from functools import cached_property
from typing import ClassVar, Self

import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy

from ._literals import NumericalTypeHint
from .base import AbstractConfig, AbstractVxData
from .coding_scheme import CodesVector, NumericScheme
from .utils import Array, np_module


class InpatientObservables(AbstractVxData):
    """
    Vectorized representation of inpatient observables.

    Attributes:
        time (Array): array of time values.
        value (Array): array of observable values.
        mask (Array): array indicating missing values.

    Methods:
        empty(size: int): creates an empty instance of InpatientObservables.
        __len__(): returns the length of the time array.
        as_dataframe(scheme: AbstractScheme, filter_missing_columns=False): converts the observables
            to a pandas DataFrame.
        groupby_code(index2code: dict[int, str]): groups the observables by code.
        segment(t_sep: Array): splits the observables into segments based on time values.
        concat(observables: Union[InpatientObservables, list[InpatientObservables]]): concatenates multiple
            instances of InpatientObservables.
        time_binning(hours: float): bins the time-series into time-windows and averages the values in each window.
    """

    time: Array  # (time,)
    value: Array  # (time, size, n_layers)
    mask: Array  # (time, size)

    def __init__(self, time: Array, value: Array, mask: Array):
        self.time = time
        self.value = value if value.ndim == 3 else value.reshape(value.shape + (1,))
        self.mask = mask

    def __check_init__(self):
        xnp = np_module(self.time)
        assert self.time.dtype == np.float64, f"Expected time to be of type float64, got {self.time.dtype}"

        assert self.mask.dtype == bool, f"Expected mask to be of type bool, got {self.mask.dtype}"

        assert self.value.ndim == 3, (
            f"Expected value to be 3D (n_timestamps, obs_dim, n_channels) or more, got {self.value.ndim}"
        )
        assert self.mask.ndim == 2, f"Expected mask to be 2D, got {self.mask.ndim}"
        assert self.value.shape[:2] == self.mask.shape, (
            f"Expected value.shape to be {self.mask.shape}, got {self.value.shape}"
        )
        assert self.time.ndim == 1, f"Expected time to be 1D, got {self.time.ndim}"
        assert len(self.time) == len(xnp.unique(self.time)), "Time stamps are not unique."
        assert len(self.time) < 2 or (self.time[:-1] < self.time[1:]).all(), "Time stamps are not sorted."

    @classmethod
    def empty(
        cls,
        size: int,
        time_dtype: type | str = np.float64,
        value_dtype: type | str = np.float16,
        mask_dtype: type | str = bool,
    ) -> Self:
        """
        Create an empty InpatientObservables object.

        Parameters:
        - size (int): the size of the InpatientObservables object.

        Returns:
        - InpatientObservables: an empty InpatientObservables object with zero time, value, and mask arrays.
        """
        return cls(
            time=np.zeros(shape=0, dtype=time_dtype),
            value=np.zeros(shape=(0, size, 1), dtype=value_dtype),
            mask=np.zeros(shape=(0, size), dtype=mask_dtype),
        )

    def __len__(self):
        """
        Returns the length of the 'time' attribute.
        """
        return self.time.shape[0]

    @cached_property
    def count(self) -> int:
        """
        Returns the number of non-missing values in the 'value' attribute.
        """
        return np.sum(self.mask)

    @classmethod
    def concat(cls, observables: list[Self] | tuple[Self, ...]) -> Self:
        """
        Concatenates a list of InpatientObservables into a single InpatientObservables object.

        Args:
            observables (Union[InpatientObservables, list[InpatientObservables]]): the list
                of InpatientObservables to concatenate.

        Returns:
            InpatientObservables: the concatenated InpatientObservables object.
        """

        if len(observables) == 0:
            return cls.empty(0)
        if all(len(o) == 0 for o in observables):
            return observables[0]
        observables = [o for o in observables if len(o) > 0]

        xnp = np_module(observables[0].time)
        time = xnp.concatenate([o.time for o in observables], axis=0)
        mask = xnp.concatenate([o.mask for o in observables], axis=0)
        value = xnp.concatenate([o.value for o in observables], axis=0)
        return cls(time=time, value=value, mask=mask)

    @staticmethod
    def _binary_agg(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.any(x, axis=0, keepdims=True, where=m.reshape(-1, 1, 1)).astype(x.dtype)

    @staticmethod
    def _ordinal_agg(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.max(x[m], axis=0, keepdims=True).astype(x.dtype)

    @staticmethod
    def _categorical_agg(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        # Most frequent value (mode) for categorical.
        return scipy.stats.mode(x[m], axis=0, keepdims=True)[0].astype(x.dtype)

    @staticmethod
    def _continuous_agg(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=0, keepdims=True, where=m.reshape(-1, 1, 1)).astype(x.dtype)

    @staticmethod
    def agg(t: NumericalTypeHint, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        match t:
            case "B":
                return InpatientObservables._binary_agg(x, m)
            case "O":
                return InpatientObservables._ordinal_agg(x, m)
            case "C":
                return InpatientObservables._categorical_agg(x, m)
            case "N":
                return InpatientObservables._continuous_agg(x, m)
            case _:
                raise ValueError(f"Unrecognized aggregation type: {t}.")

    @staticmethod
    def _time_binning_aggregate(
        x: np.ndarray, mask: np.ndarray, types: tuple[NumericalTypeHint, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aggregates the values in a given array based on the type hint.

        Args:
            x (Array): The input array.
            type_hint (npt.NDArray[NumericaltypeHint]): The type hint.

        Returns:
            Array: The aggregated array.
        """
        assert mask.dtype == bool
        assert x.ndim == 3 and mask.ndim == 2, f"Expected x to be 3D, mask to be 2D, got ({x.ndim}, {mask.ndim})"
        assert x.shape[:2] == mask.shape, f"Expected x.shape to be {mask.shape}, got {x.shape}"
        assert x.shape[1] == len(types), f"Expected x.shape[1] to be {len(types)}, got {x.shape[1]}"

        dim_mask = mask.sum(axis=0).astype(mask.dtype)
        if not dim_mask.any():
            v = np.array([0] * np.prod(x.shape[1:]), dtype=x.dtype).reshape(1, *x.shape[1:])
            return v, dim_mask

        def _apply(ti: NumericalTypeHint, xi: np.ndarray, mi: np.ndarray, i: int) -> np.ndarray:
            if dim_mask[i] == 0:
                return np.array([0], dtype=x.dtype).reshape(1, 1, *x.shape[2:])
            return InpatientObservables.agg(ti, xi, mi)

        v = np.concatenate([_apply(ti, x[:, (dim,), :], mask[:, dim], dim) for dim, ti in enumerate(types)], axis=1)
        return v, dim_mask

    def time_binning(self, hours: float, type_hint: tuple[NumericalTypeHint, ...]) -> Self:
        """
        Bin the time-series into time-windows of length `hours`.
        The values are aggregated in each window and assigned to the
        end of the window. Aggregation by default is `mean`, except for observables at binary_indices,
        where the aggregation is `any`.

        Args:
            hours (float): length of the time-windows in hours.
            type_hint (npt.NDArray[NumericaltypeHint]): type hints for the observables vector as
                (B) binary, (O) ordinal, (C) categorical, or (N) numerical.
        Returns:
            InpatientObservables: A new instance of InpatientObservables
                with the time-series binned according to the specified
                time-windows.
        """

        if len(self) == 0:
            return self
        assert self.value.ndim == 3, (
            f"Expected value to be with shape (n_time, obs_dims, n_channels), got {self.value.shape}"
        )

        last_ts = (int(self.time[-1] / hours) + 1) * hours
        new_time = np.arange(0, last_ts + hours, hours) * 1.0
        values = []
        masks = []
        for ti, tf in zip(new_time[:-1], new_time[1:]):
            time_mask = (ti <= self.time) & (self.time < tf)
            value, mask = self._time_binning_aggregate(self.value[time_mask], self.mask[time_mask], type_hint)  # type: ignore
            values.append(value)
            masks.append(mask)

        values = np.concatenate(values, axis=0)
        masks = np.stack(masks, axis=0)

        return type(self)(time=new_time[1:], value=values, mask=masks)

    @cached_property
    def timestamps(self) -> tuple[float]:
        return tuple(self.time.tolist())

    @cached_property
    def time2index(self) -> dict[float, int]:
        return {t: i for i, t in enumerate(self.timestamps)}

    def __iter__(self) -> Iterator[tuple[float, Array, Array]]:
        return zip(self.time, self.value, self.mask)


class SegmentedInpatientObservables(InpatientObservables):
    indexed_split: Array

    # The reason for storing the index split arrays without concretely splitting the observables is to avoid
    # unnecessary computation when the observables are not accessed, inefficient hardly compressed storage of
    # sharded arrays with the accompanied overhead with each sharded item.

    def __init__(self, time: Array, value: Array, mask: Array, indexed_split: Array):
        super().__init__(time=time, value=value, mask=mask)
        self.indexed_split = indexed_split

    @classmethod
    def from_observables(cls, observables: InpatientObservables, time_split: Array) -> Self:
        return cls(
            time=observables.time,
            value=observables.value,
            mask=observables.mask,
            indexed_split=cls.indexed_split_array(observables.time, time_split),
        )

    @staticmethod
    def indexed_split_array(time: Array, time_split: Array) -> Array:
        """
        Generate split indices from the time splits, which will be used to temporally split the InpatientObservables
            arrays.

        Args:
            time (Array): array of time points of the InpatientObservables object.
            time_split (Array): array of time points used to split the InpatientObservables object.

        Returns:
            Array: array of indices used to split the InpatientObservables object.
        """
        return np.searchsorted(time, time_split)

    @property
    def segments(self) -> tuple[InpatientObservables, ...]:
        if len(self.indexed_split) == 0:
            return (InpatientObservables(self.time, self.value, self.mask),)

        time = np.split(self.time, self.indexed_split, axis=0)
        value = np.split(self.value, self.indexed_split, axis=0)
        mask = np.split(self.mask, self.indexed_split, axis=0)
        return tuple(InpatientObservables(t, v, m) for t, v, m in zip(time, value, mask))

    @property
    def n_segments(self) -> int:
        return len(self.indexed_split) + 1

    def __getitem__(self, item: int) -> InpatientObservables:
        return self.segments[item]

    def __iter__(self):
        return iter((t, v, m) for t, v, m in zip(self.time, self.value, self.mask))


class LeadingObservableExtractorConfig(AbstractConfig):
    """
    AbstractConfig for LeadingObservableExtractor.

    Attributes:
        observable_code (str): the observable code to extract the leading window from.
        leading_hours (list[float]): list of leading hours to extract. Must be sorted.
        recovery_window (float): time window in hours to mask out between a nonzero value and zero.
        entry_neglect_window (float): hours to mask out in the beginning.
        minimum_acquisitions (int): minimum number of acquisitions before extracting the leading observable.
        scheme (str): name of the observation coding scheme to use.
    """

    observable_code: str
    scheme: str
    leading_hours: list[float]
    entry_neglect_window: float
    minimum_acquisitions: int  # minimum number of acquisitions to consider
    recovery_window: float

    def __init__(
        self,
        observable_code: str,
        scheme: str,
        leading_hours: list[float] | tuple[float, ...],
        entry_neglect_window: float,
        minimum_acquisitions: int,
        recovery_window: float = 0.0,
    ):
        self.observable_code = observable_code
        self.scheme = scheme
        self.leading_hours = list(leading_hours)
        self.entry_neglect_window = entry_neglect_window
        self.minimum_acquisitions = minimum_acquisitions
        self.recovery_window = recovery_window

    def __check_init__(self):
        # `leading_hours` must be sorted.
        assert all(x <= y for x, y in zip(self.leading_hours[:-1], self.leading_hours[1:])), (
            "leading_hours must be sorted"
        )
        assert isinstance(self.scheme, str), f"Expected scheme to be a string, got {type(self.scheme)}"


class LeadingObservableExtractor(AbstractVxData):
    config: LeadingObservableExtractorConfig
    observable_scheme: NumericScheme

    def __init__(self, config: LeadingObservableExtractorConfig, observable_scheme: NumericScheme):
        self.config = config
        self.observable_scheme = observable_scheme

    def __check_init__(self):
        assert self.type_hint in ("B", "O"), (
            f"LeadingObservableExtractor only supports binary and ordinal observables, "
            f"got {self.type_hint}. Categorical and Numeric types "
            "would require custom aggregation function specific to the observation of interest,"
            "e.g. the mode of categorical or the mean of numerical. In other cases, it could be more "
            "relevant to use max/min aggregation over numeric observables. Create a feature request "
            "if you need this feature."
        )

    @cached_property
    def code_index(self) -> int:
        """
        Returns the index of the observable code.

        Returns:
            int: the index of the observable code.
        """
        return self.observable_scheme.index[self.config.observable_code]

    @cached_property
    def type_hint(self) -> NumericalTypeHint:
        """
        Returns the type hint for the observable.

        Returns:
            NumericaltypeHint: the type hint for the observable.
        """
        return self.observable_scheme.type_hint[self.config.observable_code]

    @cached_property
    def aggregation_name(self) -> str:
        """
        Returns the aggregation function based on the aggregation scheme.

        Returns:
            Callable: the aggregation function.
        """
        if self.type_hint == "B":
            return "any"
        elif self.type_hint == "O":
            return "max"
        else:
            assert False, f"unsupported type hint {self.type_hint}"

    def __len__(self):
        """
        Returns the number of leading hours.

        Returns:
            int: the number of leading hours.
        """
        return len(self.config.leading_hours)

    @cached_property
    def index2code(self):
        """
        Returns the mapping of index to code.

        Returns:
            dict[int, str]: the mapping of index to code.
        """
        desc = self.observable_scheme.desc[self.config.observable_code]
        return dict(
            zip(range(len(self.config.leading_hours)), [f"{desc}_next_{h}hrs" for h in self.config.leading_hours])
        )

    @cached_property
    def index2desc(self):
        """
        Returns the mapping of index to description.

        Returns:
            dict[int, str]: the mapping of index to description.
        """
        return self.index2code

    @cached_property
    def code2index(self):
        """
        Returns the mapping of code to index.

        Returns:
            dict[str, int]: the mapping of code to index.
        """
        return {v: k for k, v in self.index2code.items()}

    def empty(self):
        """
        Returns an empty InpatientObservables object.

        Returns:
            InpatientObservables: an empty InpatientObservables object.
        """
        return InpatientObservables.empty(len(self.config.leading_hours))

    @staticmethod
    def _nan_concat_leading_windows(x: Array) -> Array:
        """
        Generates sliding windows of the input array, padded with NaN values.

        Args:
            x (Array): the input array.

        Returns:
            Array: the sliding windows of the input array, padded with NaN values.
        """
        n = len(x)
        add_arr = np.full(n - 1, np.nan)
        x_ext = np.concatenate((add_arr, x[::-1]))
        strided = np.lib.stride_tricks.as_strided
        nrows = len(x_ext) - n + 1
        s = x_ext.strides[0]
        return strided(x_ext, shape=(nrows, n), strides=(s, s))[::-1, ::-1]

    @staticmethod
    def filter_first_acquisitions(m: int, n: int) -> Array:
        """
        Generates a mask ignoring the first acquisitions in the input array.

        Args:
            n (int): number of acquisitions to neutralize.


        Returns:
            Array: the mask ignoring the first acquisitions in the input array.
        """
        return np.where(np.arange(m) < n, False, True).astype(bool)

    @staticmethod
    def filter_entry_neglect_window(t: Array, neglect_window: float) -> Array:
        """
        Generates a mask that ignores the observations in the beginning of the admission within `neglect_window`.
        The mask suppression is inclusive of the timestamps equal to `neglect_window`.

        Args:
            t (Array): the time array.
            neglect_window (float): number of hours to neglect in the beginning.

        Returns:
            Array: the neutralized input array.
        """
        return np.where(t <= neglect_window, False, True).astype(bool)

    @staticmethod
    def filter_recovery_window(t: Array, x: Array, recovery_window: float) -> Array:
        """
        Generates a mask the ignores the observations within the recovery window.

        Args:
            t (Array): the time array.
            x (Array): the input array.
            recovery_window (float): number of hours to neglect in the beginning.

        Returns:
            Array: the neutralized input array.
        """
        mask = np.ones_like(x).astype(bool)
        if len(t) == 0 or len(t) == 1:
            return mask
        x0 = x[0:-1]
        x1 = x[1:]
        next_recovery = (x0 != 0) & (~np.isnan(x0)) & (x1 == 0)

        for i in np.flatnonzero(next_recovery):
            mask[i + 1 :] = np.where(t[i + 1 :] - t[i] <= recovery_window, 0, 1)
        return mask

    @staticmethod
    def _nan_agg_nonzero(x: Array, axis: int | None) -> Array:
        """
        Aggregates the values in a given array along the specified axis, treating NaN values as zero.

        Args:
            x (Array): The input array.
            axis: The axis along which to aggregate.

        Returns:
            Array: The aggregated array.
        """
        all_nan = np.all(np.isnan(x), axis=axis) * 1.0
        replaced_nan = np.where(np.isnan(x), 0, x)
        return np.where(all_nan, np.nan, np.any(replaced_nan, axis=axis) * 1.0)

    @staticmethod
    def _nan_agg_max(x: Array, axis: int | None):
        """
        Aggregates the values in a given array along the specified axis, treating NaN values as zero.

        Args:
            x (Array): The input array.
            axis: The axis along which to aggregate.

        Returns:
            Array: The aggregated array.
        """
        return np.nanmax(x, axis=axis)

    @classmethod
    def aggregation(cls, aggregation: str) -> Callable:
        """
        Returns the aggregation function based on the aggregation scheme.

        Returns:
            Callable: the aggregation function.
        """
        if aggregation == "max":
            return cls._nan_agg_max
        elif aggregation == "any":
            return cls._nan_agg_nonzero
        else:
            raise ValueError(f"Aggregation {aggregation} not supported")

    @classmethod
    def mask_noisy_observations(
        cls, t: Array, x: Array, entry_neglect_window: float, recovery_window: float, minimum_acquisitions: int
    ) -> Array:
        # neutralize the first acquisitions
        m = cls.filter_first_acquisitions(len(t), minimum_acquisitions)
        # neutralize the observations in the beginning within the entry neglect window.
        m &= cls.filter_entry_neglect_window(t, entry_neglect_window)
        # neutralize the observations within the recovery window.
        m &= cls.filter_recovery_window(t, x, recovery_window)
        return m

    @classmethod
    def extract_leading_window(cls, t: Array, x: Array, leading_hours: list[float], aggregation: str) -> Array:
        agg = cls.aggregation(aggregation)
        # a time-window starting from timestamp_i for each row_i
        # if time = [t0, t1, t2, t3]
        # then t_leads = [[t0, t1, t2, t3],
        #                 [t1, t2, t3, nan],
        #                 [t2, t3, nan, nan],
        #                 [t3, nan, nan, nan]]
        t_leads = cls._nan_concat_leading_windows(t)

        # if time = [t0, t1, t2, t3]
        # then delta_leads = [[0, t1-t0, t2-t0, t3-t0],
        #                     [0, t2-t1, t3-t1, nan],
        #                     [0, t3-t2, nan, nan],
        #                     [0, nan, nan, nan]]
        delta_leads = t_leads - t_leads[:, 0, None]

        # a value-window starting from timestamp_i for each row_i
        # if value = [v0, v1, v2, v3]
        # then v_leads = [[v0, v1, v2, v3],
        #                 [v1, v2, v3, nan],
        #                 [v2, v3, nan, nan],
        #                 [v3, nan, nan, nan]]
        v_leads = cls._nan_concat_leading_windows(x)

        values = []
        for w in leading_hours:
            # select the rows where the time-window is less than t
            # if time = [t0, t1, t2, t3]
            # then mask = [[0 < t, t1 - t0 < t, t2 - t0 < t, t3 - t0 < t],
            #              [0 < t, t2 - t1 < t, t3 - t1 < t, nan < t],
            #              [0 < t, t3 - t2 < t, nan < t, nan < t]],
            #              [0 < t, nan < t, nan < t, nan < t]]
            mask = delta_leads < w
            v_lead = np.where(mask, v_leads, np.nan)
            values.append(agg(v_lead, axis=1).flatten())

        return np.stack(values, axis=1)

    def __call__(self, observables: InpatientObservables):
        """
        Makes a leading observable from the current timestamp.

        Args:
            observables (InpatientObservables): The input observables.

        Returns:
            InpatientObservables: The resulting leading observables.
        """
        if len(observables) == 0:
            return self.empty()

        assert observables.value.shape[2] == 1, "Only one channel is supported."

        time = observables.time
        value = observables.value[:, self.code_index].squeeze(axis=-1)
        mask = observables.mask[:, self.code_index]
        mask &= self.mask_noisy_observations(
            time,
            value,
            entry_neglect_window=self.config.entry_neglect_window,
            recovery_window=self.config.recovery_window,
            minimum_acquisitions=self.config.minimum_acquisitions,
        )
        value = np.where(mask, value, np.nan)
        value = self.extract_leading_window(time, value, self.config.leading_hours, self.aggregation_name)
        return InpatientObservables(time, value, mask=~np.isnan(value))


class InpatientInput(AbstractVxData):
    """
    Represents inpatient input data.

    Attributes:
        code_index (Array): the code index array. Each index value correspond to a unique input code.
        starttime (Array): the start time array.
        endtime (Array): the end time array.
        rate (Optional[Array]): the rate array, if available. If not provided, it is assumed to be a vector of ones.
    """

    code_index: Array
    starttime: Array
    endtime: Array
    rate: Array

    def __init__(self, code_index: Array, starttime: Array, endtime: Array, rate: Array | None = None):
        self.code_index = code_index
        self.starttime = starttime
        self.endtime = endtime
        self.rate = rate if rate is not None else np.ones(len(code_index), dtype=bool)

    def __check_init__(self):
        assert len(self.code_index) == len(self.starttime) == len(self.endtime)
        assert self.code_index.dtype == int and self.starttime.dtype == self.endtime.dtype
        if self.rate is not None:
            assert len(self.code_index) == len(self.rate)
        assert np.all(self.starttime <= self.endtime)
        assert np.all(self.starttime >= 0.0)

    def __len__(self):
        return len(self.code_index)

    def __call__(self, t: float, input_size: int) -> Array:
        """
        Returns the vectorized input at time t.

        Args:
            t (float): the time at which to return the input.
            input_size (int): the size of the input vector.

        Returns:
            Array: the input at time t.
        """

        mask = (self.starttime <= t) & (t < self.endtime)
        index = self.code_index[mask]
        rate = self.rate[mask]
        vec_input = np.zeros(input_size, dtype=rate.dtype)
        vec_input[index] += rate
        return vec_input

    @classmethod
    def empty(cls, code_index_dtype=int, starttime_dtype=float, endtime_dtype=float, rate_dtype=float) -> Self:
        zero_vec = np.zeros(0, dtype=bool)
        return cls(
            code_index=zero_vec.astype(code_index_dtype),
            starttime=zero_vec.astype(starttime_dtype),
            endtime=zero_vec.astype(endtime_dtype),
            rate=zero_vec.astype(rate_dtype),
        )


class InpatientInterventions(AbstractVxData):
    hosp_procedures: InpatientInput | None
    icu_procedures: InpatientInput | None
    icu_inputs: InpatientInput | None

    def __init__(
        self,
        hosp_procedures: InpatientInput | None = None,
        icu_procedures: InpatientInput | None = None,
        icu_inputs: InpatientInput | None = None,
    ):
        self.hosp_procedures = hosp_procedures
        self.icu_procedures = icu_procedures
        self.icu_inputs = icu_inputs

    @cached_property
    def timestamps(self) -> list[float]:
        timestamps = []
        for k in self.__dict__.keys():
            ii = getattr(self, k, None)
            if isinstance(ii, InpatientInput):
                timestamps.extend(ii.starttime)
                timestamps.extend(ii.endtime)
        return list(sorted(set(timestamps)))

    @cached_property
    def final_timestamp(self) -> float:
        return max(self.timestamps)


class SegmentedInpatientInterventions(AbstractVxData):
    time: Array
    icu_inputs: Array | None
    icu_procedures: Array | None
    hosp_procedures: Array | None

    def __init__(
        self,
        time: Array,
        icu_inputs: Array | None = None,
        icu_procedures: Array | None = None,
        hosp_procedures: Array | None = None,
    ):
        self.time = time
        self.icu_inputs = icu_inputs
        self.icu_procedures = icu_procedures
        self.hosp_procedures = hosp_procedures

    @classmethod
    def from_interventions(
        cls,
        inpatient_interventions: InpatientInterventions,
        terminal_time: float,
        hosp_procedures_size: int | None = None,
        icu_procedures_size: int | None = None,
        icu_inputs_size: int | None = None,
        maximum_padding: int = 100,
    ) -> Self:
        timestamps = inpatient_interventions.timestamps
        assert terminal_time >= max(timestamps, default=0.0), (
            f"Terminal time {terminal_time} should be greater than the maximum timestamp {max(timestamps, default=0.0)}"
        )
        assert min(timestamps, default=0.0) >= 0.0, (
            f"Minimum timestamp {min(timestamps, default=0.0)} should be greater than or equal to 0.0"
        )

        time = np.unique(np.array(timestamps + [0.0, terminal_time], dtype=np.float64))
        time = cls.pad_array(time, value=np.nan, maximum_padding=maximum_padding)
        t0 = time[:-1]
        hosp_procedures = None
        icu_procedures = None
        icu_inputs = None
        if inpatient_interventions.hosp_procedures is not None and hosp_procedures_size is not None:
            hosp_procedures = cls._segment(t0, inpatient_interventions.hosp_procedures, hosp_procedures_size)
        if inpatient_interventions.icu_procedures is not None and icu_procedures_size is not None:
            icu_procedures = cls._segment(t0, inpatient_interventions.icu_procedures, icu_procedures_size)
        if inpatient_interventions.icu_inputs is not None and icu_inputs_size is not None:
            icu_inputs = cls._segment(t0, inpatient_interventions.icu_inputs, icu_inputs_size)
        return cls(time=time, hosp_procedures=hosp_procedures, icu_procedures=icu_procedures, icu_inputs=icu_inputs)

    @staticmethod
    def pad_array(array: Array, maximum_padding: int = 100, value: float = 0.0) -> Array:
        """
        Pad array to be a multiple of maximum_padding. This is to
        minimize the number of jit-compiling that is made for the same functions
        when the input shape changes.

        Args:
            array (Array): the array to be padded.
            maximum_padding (int, optional): the maximum padding. Defaults to 100.
            value (float, optional): the value to pad with. Defaults to 0.0.

        Returns:
            Array: the padded array."""

        n = len(array)
        n_pad = maximum_padding - (n % maximum_padding)
        if n_pad == maximum_padding:
            return array

        return np.pad(array, pad_width=(0, n_pad), mode="constant", constant_values=value)

    @staticmethod
    def _segment(t0_padded: Array, inpatient_input: InpatientInput, input_size: int) -> Array:
        """
        Generate segmented procedures and apply the specified procedure transformation.
        Returns:
            numpy.ndarray: the processed segments.
        """
        t = t0_padded[~np.isnan(t0_padded)]
        t_nan = t0_padded[np.isnan(t0_padded)]
        out = np.stack([inpatient_input(ti, input_size) for ti in t], axis=0)
        pad = np.zeros((len(t_nan), out[0].shape[0]), dtype=out.dtype)
        return np.vstack([out, pad])

    @cached_property
    def t0_padded(self):
        return self.time[:-1]

    @property
    def t0(self):
        """Start times for segmenting the interventions"""
        t = self.time
        xnp = np_module(t)
        return t[~xnp.isnan(t)][:-1]

    @property
    def t1_padded(self):
        """End times for segmenting the interventions"""
        return self.time[1:]

    @property
    def t1(self):
        """End times for segmenting the interventions"""
        t = self.time
        xnp = np_module(t)
        return t[~xnp.isnan(t)][1:]


class AdmissionDates(AbstractVxData):
    admission: pd.Timestamp
    discharge: pd.Timestamp

    def __init__(self, admission: pd.Timestamp, discharge: pd.Timestamp):
        self.admission = admission
        self.discharge = discharge

    def __getitem__(self, item) -> pd.Timestamp:
        if item == 0:
            return self.admission
        elif item == 1:
            return self.discharge
        else:
            raise IndexError


class Admission(AbstractVxData):
    """Admission data class representing a hospital admission.

    Attributes:
        admission_id: unique ID for the admission.
        admission_dates: start and end dates for the admission.
        dx_codes: diagnosis codes recorded during the admission.
        dx_codes_history: historical diagnosis codes prior to admission.
        outcome: outcome codes of interest derived from the diagnosis codes.
        observables: timeseries clinical observations data.
        interventions: timeseries clinical interventions data.
        leading_observable: timeseries clinical leading observable (of interest).
    """

    admission_id: str  # Unique ID for each admission
    admission_dates: AdmissionDates
    dx_codes: CodesVector
    dx_codes_history: CodesVector
    outcome: CodesVector
    observables: InpatientObservables | None
    interventions: InpatientInterventions | None
    leading_observable: InpatientObservables | None

    def __init__(
        self,
        admission_id: str,
        admission_dates: AdmissionDates,
        dx_codes: CodesVector,
        dx_codes_history: CodesVector,
        outcome: CodesVector,
        observables: InpatientObservables | None = None,
        interventions: InpatientInterventions | None = None,
        leading_observable: InpatientObservables | None = None,
    ):
        self.admission_id = admission_id
        self.admission_dates = admission_dates
        self.dx_codes = dx_codes
        self.dx_codes_history = dx_codes_history
        self.outcome = outcome
        self.observables = observables
        self.interventions = interventions
        self.leading_observable = leading_observable

    def extract_leading_observable(self, leading_observable_extractor: LeadingObservableExtractor) -> Self:
        """
        Extracts the leading observable from the admission data.

        Args:
            leading_observable_extractor (LeadingObservableExtractor): the leading observable extractor.

        Returns:
            Admission: the admission data with the leading observable extracted.
        """
        assert self.observables is not None, "Observables must be provided to extract leading observable."
        leading_observable = leading_observable_extractor(self.observables)
        return type(self)(
            admission_id=self.admission_id,
            admission_dates=self.admission_dates,
            dx_codes=self.dx_codes,
            dx_codes_history=self.dx_codes_history,
            outcome=self.outcome,
            observables=self.observables,
            leading_observable=leading_observable,
            interventions=self.interventions,
        )

    def observables_time_binning(self, interval: float, types: tuple[NumericalTypeHint, ...]) -> Self:
        """
        Bins the observables data into time intervals of the specified length.

        Args:
            interval (float): the length of the time intervals.
            obs_scheme (NumericScheme): the numeric scheme for the observables, which is used to guide to aggregation
                operations.
        Returns:
            Admission: the admission data with the observables data binned.

        """
        assert self.observables is not None, "Observables must be provided to bin the time series."
        observables = self.observables.time_binning(interval, types)
        return type(self)(
            admission_id=self.admission_id,
            admission_dates=self.admission_dates,
            dx_codes=self.dx_codes,
            dx_codes_history=self.dx_codes_history,
            outcome=self.outcome,
            observables=observables,
            leading_observable=self.leading_observable,
            interventions=self.interventions,
        )

    @cached_property
    def interval_hours(self) -> float:
        """
        Calculates the interval in hours between the admission start and end dates.

        Returns:
            float: the interval in hours.
        """
        return (self.admission_dates[1] - self.admission_dates[0]).total_seconds() / 3600

    @cached_property
    def interval_days(self) -> float:
        """
        Calculates the interval in days based on the interval in hours.

        Returns:
            float: the interval in days.
        """
        return self.interval_hours / 24

    def days_since(self, date: pd.Timestamp) -> tuple[float, float]:
        """
        Calculates the number of days from a reference date to admission and discharge dates, respectively.

        Args:
            date (date): the date to calculate the number of days since.

        Returns:
            tuple[float, float]: the number of days passed at the admission and discharge dates, respectively.
        """
        d1 = (self.admission_dates[0] - date).total_seconds() / 3600 / 24
        d2 = (self.admission_dates[1] - date).total_seconds() / 3600 / 24
        return d1, d2


class SegmentedAdmission(Admission):
    observables: SegmentedInpatientObservables | None
    interventions: SegmentedInpatientInterventions | None
    leading_observable: SegmentedInpatientObservables | None

    def __init__(
        self,
        admission_id: str,
        admission_dates: AdmissionDates,
        dx_codes: CodesVector,
        dx_codes_history: CodesVector,
        outcome: CodesVector,
        observables: SegmentedInpatientObservables | None = None,
        interventions: SegmentedInpatientInterventions | None = None,
        leading_observable: SegmentedInpatientObservables | None = None,
    ):
        super().__init__(
            admission_id=admission_id,
            admission_dates=admission_dates,
            dx_codes=dx_codes,
            dx_codes_history=dx_codes_history,
            outcome=outcome,
            observables=observables,
            interventions=interventions,  # type: ignore
            leading_observable=leading_observable,
        )

    @staticmethod
    def _segment_interventions(
        interventions: InpatientInterventions | None,
        terminal_time: float,
        hosp_procedures_size: int | None,
        icu_procedures_size: int | None,
        icu_inputs_size: int | None,
        maximum_padding: int = 100,
    ) -> SegmentedInpatientInterventions | None:
        if interventions is None:
            return None
        return SegmentedInpatientInterventions.from_interventions(
            interventions,
            terminal_time,
            hosp_procedures_size=hosp_procedures_size,
            icu_procedures_size=icu_procedures_size,
            icu_inputs_size=icu_inputs_size,
            maximum_padding=maximum_padding,
        )

    @staticmethod
    def _segment_observables(
        observables: InpatientObservables | None, interventions: SegmentedInpatientInterventions | None
    ) -> SegmentedInpatientObservables | None:
        if observables is None:
            return None
        if interventions is None:
            return SegmentedInpatientObservables.from_observables(observables, np.array([]))
        time = interventions.time
        t_sep = time[~np.isnan(time)][1:-1]
        return SegmentedInpatientObservables.from_observables(observables, t_sep)

    @staticmethod
    def from_admission(
        admission: Admission,
        hosp_procedures_size: int | None,
        icu_procedures_size: int | None,
        icu_inputs_size: int | None,
        maximum_padding: int = 100,
    ) -> "SegmentedAdmission":
        interventions = SegmentedAdmission._segment_interventions(
            admission.interventions,
            admission.interval_hours,
            hosp_procedures_size=hosp_procedures_size,
            icu_procedures_size=icu_procedures_size,
            icu_inputs_size=icu_inputs_size,
            maximum_padding=maximum_padding,
        )
        observables = SegmentedAdmission._segment_observables(admission.observables, interventions)
        leading_observable = SegmentedAdmission._segment_observables(admission.leading_observable, interventions)
        return SegmentedAdmission(
            admission_id=admission.admission_id,
            admission_dates=admission.admission_dates,
            dx_codes=admission.dx_codes,
            dx_codes_history=admission.dx_codes_history,
            outcome=admission.outcome,
            observables=observables,
            interventions=interventions,
            leading_observable=leading_observable,
        )


class DemographicVectorConfig(AbstractConfig):
    """
    Configuration class for demographic vector.

    Attributes:
        gender (bool): indicates whether gender is included in the vector.
        age (bool): indicates whether age is included in the vector.
        ethnicity (bool): indicates whether ethnicity is included in the vector.
    """

    gender: bool
    age: bool
    ethnicity: bool

    def __init__(self, gender: bool = True, age: bool = True, ethnicity: bool = True):
        self.gender = gender
        self.age = age
        self.ethnicity = ethnicity


class CPRDDemographicVectorConfig(DemographicVectorConfig):
    """
    Configuration class for CPRD demographic vector.
    """

    imd: bool = True


class StaticInfo(AbstractVxData):
    """
    Represents static information about a patient.

    Attributes:
        gender (Optional[CodesVector]): gender information of the patient.
        ethnicity (Optional[CodesVector]): ethnicity information of the patient.
        date_of_birth (Optional[date]): date of birth of the patient.

    Methods:
        age: calculates the age of the patient based on the current date.
        demographic_vector: returns the demographic vector based on the current date.
        _concat: concatenates the age and vector.

    """

    gender: CodesVector | None
    ethnicity: CodesVector | None
    date_of_birth: pd.Timestamp | None

    def __init__(self, gender: CodesVector | None, ethnicity: CodesVector | None, date_of_birth: pd.Timestamp | None):
        self.gender = gender
        self.ethnicity = ethnicity
        self.date_of_birth = date_of_birth

    def constant_attrs_list(self, demographic_vector_config: DemographicVectorConfig) -> list[Array]:
        attrs_vec = []
        if demographic_vector_config.gender:
            assert self.gender is not None and len(self.gender.scheme) > 0, "Gender is not extracted from the dataset"
            attrs_vec.append(self.gender.vec)
        if demographic_vector_config.ethnicity:
            assert self.ethnicity is not None, "Ethnicity is not extracted from the dataset"
            attrs_vec.append(self.ethnicity.vec)
        return attrs_vec

    @functools.lru_cache(maxsize=10, typed=True)
    def constant_vec(self, demographic_vector_config: DemographicVectorConfig) -> Array:
        attrs_vec = self.constant_attrs_list(demographic_vector_config)
        xnp = np_module(attrs_vec[0])
        if len(attrs_vec) == 0:
            return xnp.array([], dtype=xnp.float16)
        else:
            return xnp.hstack(attrs_vec)

    def age(self, current_date: pd.Timestamp) -> float:
        """
        Calculates the age of the patient based on the current date.

        Args:
            current_date (date): the current date.

        Returns:
            float: the age of the patient in years.
        """
        assert self.date_of_birth is not None, "Date of birth is not set"
        return current_date.year - self.date_of_birth.year

    def admission_demographics(self, admission: Admission, demographic_vector_config: DemographicVectorConfig) -> Array:
        """
        Returns the demographic vector for the patient.

        Args:
            demographic_vector_config (DemographicVectorConfig): the demographic vector configuration.

        Returns:
            Array: the demographic vector.
        """
        if demographic_vector_config.age:
            return jnp.hstack((self.age(admission.admission_dates[0]), self.constant_vec(demographic_vector_config)))
        return self.constant_vec(demographic_vector_config)


class Patient(AbstractVxData):
    """
    Represents a patient with demographic information and a clinical history
    of admissions.

    Attributes:
        subject_id: Unique identifier for the patient.
        static_info: Static demographic and geographic data for the patient.
        admissions: list of hospital/clinic admissions for the patient.


    Properties:
        d2d_interval_days: number of days between first and last discharge.


    Methods:
        outcome_frequency_vec: accumulates outcome vectors from all admissions to get
        aggregate vector for patient.
    """

    subject_id: str
    static_info: StaticInfo
    admissions: list[Admission]

    admission_cls: ClassVar[type[Admission]] = Admission

    def __init__(self, subject_id: str, static_info: StaticInfo, admissions: list[Admission]):
        self.subject_id = subject_id
        self.static_info = static_info
        self.admissions = list(sorted(admissions, key=lambda x: x.admission_dates[0]))

    def __len__(self):
        return len(self.admissions)

    def admission_demographics(self, demographic_vector_config: DemographicVectorConfig) -> dict[str, Array]:
        """
        Returns the demographic vector for the patient.

        Args:
            demographic_vector_config (DemographicVectorConfig): the demographic vector configuration.

        Returns:
            Array: the demographic vector.
        """
        return {
            a.admission_id: self.static_info.admission_demographics(a, demographic_vector_config)
            for a in self.admissions
        }

    @cached_property
    def d2d_interval_days(self):
        """
        The interval in days between the first and last discharge dates.

        Returns:
            float: the interval in days.
        """
        d1 = self.admissions[0].admission_dates[1]
        d2 = self.admissions[-1].admission_dates[1]
        return (d2 - d1).total_seconds() / 3600 / 24

    def filter_short_stays(self, min_hours: float) -> Self:
        """
        Filter out admissions that are shorter than a specified duration.

        Args:
            min_hours (float): the minimum duration in hours.
        """
        admissions = [a for a in self.admissions if a.interval_hours >= min_hours]
        return type(self)(subject_id=self.subject_id, static_info=self.static_info, admissions=admissions)

    def outcome_frequency_vec(self) -> Array:
        """
        Calculate the accumulation of outcome vectors for all admissions.

        Returns:
            The total count of outcome code occurrence over all admissions.
        """
        return sum(a.outcome.vec for a in self.admissions)

    def __eq__(self, other) -> bool:
        return self.equals(other)

    def extract_leading_observables(self, leading_observable_extractor: LeadingObservableExtractor) -> Self:
        """
        Extracts the leading observable from all admissions.

        Args:
            leading_observable_extractor (LeadingObservableExtractor): the leading observable extractor.

        Returns:
            Patient: the patient data with the leading observable extracted.
        """
        admissions = [a.extract_leading_observable(leading_observable_extractor) for a in self.admissions]
        return type(self)(subject_id=self.subject_id, static_info=self.static_info, admissions=admissions)

    def observables_time_binning(self, interval: float, types: tuple[NumericalTypeHint, ...]) -> Self:
        """
        Bins the observables for all admissions.

        Args:
            interval (float): the interval in hours to bin the observables.

        Returns:
            Patient: the patient data with the observables binned.
        """
        admissions = [a.observables_time_binning(interval, types) for a in self.admissions]
        return type(self)(subject_id=self.subject_id, static_info=self.static_info, admissions=admissions)


class SegmentedPatient(Patient):
    admissions: list[SegmentedAdmission]
    admission_cls: ClassVar[type[SegmentedAdmission]] = SegmentedAdmission

    def __init__(self, subject_id: str, static_info: StaticInfo, admissions: list[SegmentedAdmission]):
        super().__init__(
            subject_id=subject_id,
            static_info=static_info,
            admissions=admissions,  # type: ignore
        )

    def extract_leading_observables(self, leading_observable_extractor: LeadingObservableExtractor) -> Patient:
        raise NotImplementedError("SegmentedPatient does not support leading observable extraction")

    @staticmethod
    def from_patient(
        patient: Patient,
        hosp_procedures_size: int | None,
        icu_procedures_size: int | None,
        icu_inputs_size: int | None,
        maximum_padding: int = 100,
    ) -> "SegmentedPatient":
        admissions = [
            SegmentedAdmission.from_admission(
                a,
                hosp_procedures_size=hosp_procedures_size,
                icu_procedures_size=icu_procedures_size,
                icu_inputs_size=icu_inputs_size,
                maximum_padding=maximum_padding,
            )
            for a in patient.admissions
        ]
        return SegmentedPatient(subject_id=patient.subject_id, static_info=patient.static_info, admissions=admissions)
