from copy import deepcopy
from typing import cast
from unittest import mock

import ehrax as rx
import equinox as eqx
import numpy as np
import pytest
import tables as tb
from .common_setup import ADMISSION_CONCEPT_MAX_STAY_HOURS
from ehrax.testing.common_setup import (
    BINARY_OBSERVATION_CODE_INDEX,
    CATEGORICAL_OBSERVATION_CODE_INDEX,
    inpatient_binary_input,
    leading_observables_extractor,
    NUMERIC_OBSERVATION_CODE_INDEX,
    ORDINAL_OBSERVATION_CODE_INDEX,
    SCHEMES,
)


class TestInpatientObservables:
    def test_empty(self):
        obs = rx.InpatientObservables.empty(len(SCHEMES["obs"]))
        assert len(obs.time) == 0
        assert len(obs.value) == 0
        assert len(obs.mask) == 0
        assert all(a.shape[:2] == (0, len(SCHEMES["obs"])) for a in [obs.value, obs.mask])

    @pytest.mark.parametrize("time_valid_dtype", [np.float64])
    @pytest.mark.parametrize("mask_valid_dtype", [bool])
    @pytest.mark.parametrize("time_invalid_dtype", [np.int32, np.int64, np.float32, bool, str, object])
    @pytest.mark.parametrize("mask_invalid_dtype", [np.int32, np.int64, np.float32, str, object])
    def test_invalid_init(self, time_invalid_dtype, mask_invalid_dtype, time_valid_dtype, mask_valid_dtype):
        obs = rx.InpatientObservables(
            time=np.array([1.0, 2.0], dtype=time_valid_dtype),
            value=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            mask=np.array([[True, False, True], [False, True, False]], dtype=mask_valid_dtype),
        )

        # Invalid value shape
        with pytest.raises(AssertionError):
            rx.InpatientObservables(time=obs.time[:1], value=obs.value[:1].squeeze(), mask=obs.mask[:1])

        # Invalid mask shape
        with pytest.raises(AssertionError):
            rx.InpatientObservables(time=obs.time[:1], value=obs.value[:1], mask=obs.mask[:1].squeeze())

        # inconsistent time and value shape
        with pytest.raises(AssertionError):
            rx.InpatientObservables(time=obs.time, value=obs.value[:-1], mask=obs.mask)

        # inconsistent time and mask shape
        with pytest.raises(AssertionError):
            rx.InpatientObservables(time=obs.time, value=obs.value, mask=obs.mask[:-1])

        with pytest.raises(AssertionError):
            rx.InpatientObservables(
                time=np.array([1.0, 2.0], dtype=time_invalid_dtype),
                value=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                mask=np.array([[True, False, True], [False, True, False]], dtype=mask_valid_dtype),
            )

        with pytest.raises(AssertionError):
            rx.InpatientObservables(
                time=np.array([1.0, 2.0], dtype=time_valid_dtype),
                value=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                mask=np.array(
                    [[True, False, True], [False, True, False]],
                    dtype=mask_invalid_dtype,
                ),
            )

    def test_time_sorted(self, inpatient_observables: rx.InpatientObservables):
        assert np.all(np.diff(inpatient_observables.time) >= 0)

    def test_len(self, inpatient_observables: rx.InpatientObservables):
        assert len(inpatient_observables) == len(inpatient_observables.time)

    def test_equal(self, inpatient_observables: rx.InpatientObservables):
        assert inpatient_observables.equals(deepcopy(inpatient_observables))
        if len(inpatient_observables) == 0:
            raise pytest.skip("No observations to test")
        time = inpatient_observables.time.copy()
        time[-1] = time[-1] + 1
        c1 = eqx.tree_at(lambda x: x.time, inpatient_observables, time)
        assert not inpatient_observables.equals(c1)

        value = inpatient_observables.value.copy()
        value[-1, -1] = np.nan
        c2 = eqx.tree_at(lambda x: x.value, inpatient_observables, value)
        assert not inpatient_observables.equals(c2)

        mask = inpatient_observables.mask.astype(int)
        c3 = eqx.tree_at(lambda x: x.mask, inpatient_observables, mask)
        assert not inpatient_observables.equals(c3)

    def test_as_dataframe(self):
        pass

    def test_groupby_code(self):
        pass

    @pytest.fixture(
        params=[np.array([5.0]), np.array([0.5, 2.0]), np.array([1.0, 3.0, 5.0])],
        scope="class",
    )
    def sep(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def segmented_inpatient_observables(self, inpatient_observables: rx.InpatientObservables, sep):
        return rx.SegmentedInpatientObservables.from_observables(inpatient_observables, sep)

    def test_segmentation_concat(
        self,
        inpatient_observables: rx.InpatientObservables,
        segmented_inpatient_observables: rx.SegmentedInpatientObservables,
        sep: np.ndarray,
    ):
        seg = segmented_inpatient_observables.segments
        assert len(seg) == len(sep) + 1
        assert sum(len(s) for s in seg) == len(inpatient_observables)
        assert sum(s.time.size + s.value.size + s.mask.size for s in seg) == (
            inpatient_observables.time.size + inpatient_observables.value.size + inpatient_observables.mask.size
        )
        assert inpatient_observables.equals(rx.InpatientObservables.concat(seg))
        assert all(seg[i].time.max() <= sep[i] for i in range(len(sep)) if len(seg[i]) > 0)
        assert all(seg[i + 1].time.min() >= sep[i] for i in range(len(sep)) if len(seg[i + 1]) > 0)

    def test_hf5_group_serialization(self, inpatient_observables: rx.InpatientObservables, hf5_group_writer: tb.Group):
        inpatient_observables.to_hdf_group(hf5_group_writer)
        assert inpatient_observables.equals(rx.InpatientObservables.from_hdf_group(hf5_group_writer))

    def test_hd5_group_serialization_segmented(
        self,
        segmented_inpatient_observables: rx.SegmentedInpatientObservables,
        hf5_group_writer: tb.Group,
    ):
        segmented_inpatient_observables.to_hdf_group(hf5_group_writer)
        assert segmented_inpatient_observables.equals(rx.SegmentedInpatientObservables.from_hdf_group(hf5_group_writer))

    @pytest.mark.parametrize("ntype", ["N", "B", "C", "O"])
    def test_type_aggregator(self, ntype: rx.NumericalTypeHint):
        a = rx.InpatientObservables.agg(ntype, np.array([1, 2, 3]).reshape(-1, 1, 1), np.ones((3,), dtype=bool))
        assert np.size(a) == 1

    # mask = (1, 0, 1)
    # @pytest.mark.serial_test
    @pytest.mark.parametrize(
        "x,ntype,out",
        [
            (np.array([1.0, 2.0, 3.0]), ("N",), 2.0),
            (np.array([1, 0, 1]), ("B",), 1.0),
            (np.array([0, 1, 0]), ("B",), 0.0),
            (np.array([0, 1, 0]), ("C",), 0),
            (np.array([1, 0, 1]), ("C",), 1),
            (np.array([0, 1, 0]), ("O",), 0),
            (np.array([1, 0, 1]), ("O",), 1),
            (np.array([1, 0, 0]), ("O",), 1),
            (
                np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [2.0, 1.0, 6.0]]),
                ("N", "N", "N"),
                np.array([1.5, 1.5, 4.5]),
            ),
            (
                np.array(
                    [
                        [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]],
                        [[2.0, 4.0], [4.0, 0.0], [5.0, 1.0]],
                        [[6.0, 1.0], [7.0, 0.0], [8.0, 0.0]],
                    ]
                ),
                ("N", "N", "N"),
                np.array([[3.5, 0.5], [4.5, 0.5], [5.5, 0.0]]),
            ),
        ],
    )
    def test_time_binning_aggregate(self, x, ntype: tuple[rx.NumericalTypeHint, ...], out):
        if x.ndim == 1:
            x = x.reshape(-1, 1, 1)
        if x.ndim == 2:
            x = x.reshape(x.shape + (1,))

        time_mask = np.broadcast_to(np.array([1, 0, 1]).reshape(-1, 1).astype(bool), x.shape[:2])
        out = np.array([out]).reshape((1,) + x.shape[1:])

        np.testing.assert_equal(rx.InpatientObservables._time_binning_aggregate(x, time_mask, ntype)[0], out)
        # if the input mask is all zeros, then the resulting mask is all zeros:
        v, m = rx.InpatientObservables._time_binning_aggregate(x, np.zeros_like(time_mask, dtype=bool), ntype)
        assert all(m == 0)

        with pytest.raises(AssertionError):
            rx.InpatientObservables._time_binning_aggregate(x, time_mask.astype(int), ntype)

        if any(d == 1 for d in x.shape):
            with pytest.raises(AssertionError):
                rx.InpatientObservables._time_binning_aggregate(x.squeeze(), time_mask, ntype)

    @pytest.mark.parametrize("hours", [1.0, 2.0, 3.0])
    def test_time_binning(self, hours: float, inpatient_observables: rx.InpatientObservables):
        if len(inpatient_observables) == 0:
            raise pytest.skip("No observations to test")

        binned = inpatient_observables.time_binning(hours, SCHEMES["obs"].types)
        assert np.all(binned.time % hours == 0.0)
        assert sorted(binned.time) == binned.time.tolist()

        # test causality. inject np.inf to the time array and ensure that
        # values in future time bins do not leak into the past ones.
        for i, ti in enumerate(inpatient_observables.time):
            val = inpatient_observables.value.copy()
            # NUMERIC_OBSERVATION_CODE_INDEX is aggregated with mean.
            val[i:, NUMERIC_OBSERVATION_CODE_INDEX] = np.inf
            obs = eqx.tree_at(lambda x: x.value, inpatient_observables, val)
            binned = obs.time_binning(hours, SCHEMES["obs"].types)
            assert np.all(binned.value[binned.time < ti] < np.inf)

            mask = inpatient_observables.mask.copy()
            if mask[i:, NUMERIC_OBSERVATION_CODE_INDEX].sum() > 0:
                assert np.any(np.isinf(binned.value[binned.time >= ti]))


class TestLeadingObservableExtractor:
    @pytest.mark.parametrize("leading_hours", [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
    def test_len(self, leading_hours: list[float]):
        extractor = leading_observables_extractor(
            observation_scheme=cast(rx.NumericScheme, SCHEMES["obs"]), leading_hours=leading_hours
        )
        assert len(extractor) == len(leading_hours)

    @pytest.mark.parametrize("leading_hours", [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
    @pytest.mark.parametrize("entry_neglect_window", [0.0, 1.0, 2.0])
    @pytest.mark.parametrize("recovery_window", [0.0, 1.0, 2.0])
    @pytest.mark.parametrize("minimum_acquisitions", [0, 1, 2, 3])
    def test_init(
        self,
        leading_hours: list[float],
        entry_neglect_window: float,
        recovery_window: float,
        minimum_acquisitions: int,
    ):
        if len(leading_hours) < 2:
            raise pytest.skip("Not enough leading hours to test")

        with pytest.raises(AssertionError):
            # leading hours must be sorted
            leading_observables_extractor(
                observation_scheme=cast(rx.NumericScheme, SCHEMES["obs"]),
                leading_hours=list(reversed(leading_hours)),
                entry_neglect_window=entry_neglect_window,
                recovery_window=recovery_window,
                minimum_acquisitions=minimum_acquisitions,
                code_index=BINARY_OBSERVATION_CODE_INDEX,
            )

        with pytest.raises(AssertionError):
            # categorical and numerical codes are not supported, yet.
            leading_observables_extractor(
                observation_scheme=cast(rx.NumericScheme, SCHEMES["obs"]),
                leading_hours=leading_hours,
                entry_neglect_window=entry_neglect_window,
                recovery_window=recovery_window,
                minimum_acquisitions=minimum_acquisitions,
                code_index=CATEGORICAL_OBSERVATION_CODE_INDEX,
            )

        with pytest.raises(AssertionError):
            # categorical and numerical codes are not supported, yet.
            leading_observables_extractor(
                observation_scheme=cast(rx.NumericScheme, SCHEMES["obs"]),
                leading_hours=leading_hours,
                entry_neglect_window=entry_neglect_window,
                recovery_window=recovery_window,
                minimum_acquisitions=minimum_acquisitions,
                code_index=NUMERIC_OBSERVATION_CODE_INDEX,
            )

    def test_index2code(self):
        pass

    def test_index2desc(self):
        pass

    def test_code2index(self):
        pass

    @pytest.mark.parametrize("leading_hours", [[1.0], [2.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
    @pytest.mark.parametrize("code_index", [BINARY_OBSERVATION_CODE_INDEX, ORDINAL_OBSERVATION_CODE_INDEX])
    def test_empty(self, leading_hours: list[float], code_index: int):
        extractor = leading_observables_extractor(
            observation_scheme=cast(rx.NumericScheme, SCHEMES["obs"]),
            leading_hours=leading_hours,
            code_index=code_index,
        )
        empty = extractor.empty()
        assert len(extractor) == len(leading_hours)
        assert empty.value.shape[1] == len(extractor)
        assert empty.mask.shape[1] == len(extractor)

        assert len(empty) == 0
        assert empty.time.size == 0
        assert empty.value.size == 0
        assert empty.mask.size == 0

    @pytest.mark.parametrize(
        "x, y",
        [
            (
                np.array([1.0, 2.0, 3.0]),
                np.array([[1.0, 2.0, 3.0], [2.0, 3.0, np.nan], [3.0, np.nan, np.nan]]),
            )
        ],
    )
    def test_nan_concat_leading_windows(self, x, y):
        assert np.array_equal(
            rx.LeadingObservableExtractor._nan_concat_leading_windows(x),
            y,
            equal_nan=True,
        )

    @pytest.mark.parametrize(
        "x, y",
        [
            (
                np.array([[1.0, 2.0, 3.0], [2.0, 3.0, np.nan], [3.0, np.nan, np.nan]]),
                np.array([1.0, 1.0, 1.0]),
            ),
            (
                np.array([[1.0, 2.0, -3.0], [0.0, 0.0, np.nan], [np.nan, np.nan, np.nan]]),
                np.array([1.0, 0.0, np.nan]),
            ),
        ],
    )
    def test_nan_agg_nonzero(self, x, y):
        assert np.array_equal(rx.LeadingObservableExtractor._nan_agg_nonzero(x, axis=1), y, equal_nan=True)

    @pytest.mark.parametrize(
        "x, y",
        [
            (
                np.array([[1.0, 2.0, 3.0], [2.0, 3.0, np.nan], [3.0, np.nan, np.nan]]),
                np.array([3.0, 3.0, 3.0]),
            ),
            (
                np.array([[1.0, 2.0, -3.0], [0.0, 0.0, np.nan], [np.nan, np.nan, np.nan]]),
                np.array([2.0, 0.0, np.nan]),
            ),
        ],
    )
    def test_nan_agg_max(self, x, y):
        assert np.array_equal(rx.LeadingObservableExtractor._nan_agg_max(x, axis=1), y, equal_nan=True)

    @pytest.mark.parametrize("minimum_acquisitions", [0, 1, 2, 3])
    def test_filter_first_acquisitions(self, inpatient_observables: rx.InpatientObservables, minimum_acquisitions: int):
        lead_extractor = leading_observables_extractor(
            observation_scheme=cast(rx.NumericScheme, SCHEMES["obs"]), minimum_acquisitions=minimum_acquisitions
        )
        m = lead_extractor.filter_first_acquisitions(len(inpatient_observables), minimum_acquisitions)
        n_affected = min(minimum_acquisitions, len(inpatient_observables))
        assert (~m).sum() == n_affected
        assert (~m)[n_affected:].sum() == 0

    @pytest.mark.parametrize("entry_neglect_window", [0.0, 1.0, 2.0])
    def test_neutralize_entry_neglect_window(
        self, inpatient_observables: rx.InpatientObservables, entry_neglect_window: int
    ):
        lead_extractor = leading_observables_extractor(
            observation_scheme=cast(rx.NumericScheme, SCHEMES["obs"]), entry_neglect_window=entry_neglect_window
        )
        t = inpatient_observables.time
        m = lead_extractor.filter_entry_neglect_window(t, entry_neglect_window)
        n_affected = np.sum(t <= entry_neglect_window)
        assert (~m).sum() == n_affected
        assert (~m)[n_affected:].sum() == 0

    @pytest.mark.parametrize("recovery_window", [0.0, 1.0, 2.0])
    def test_neutralize_recovery_window(self, inpatient_observables: rx.InpatientObservables, recovery_window: float):
        if len(inpatient_observables) == 0:
            raise pytest.skip("No observations to test")

        lead_extractor = leading_observables_extractor(
            observation_scheme=cast(rx.NumericScheme, SCHEMES["obs"]), entry_neglect_window=recovery_window
        )
        x = inpatient_observables.value[:, lead_extractor.code_index]
        t = inpatient_observables.time
        m = lead_extractor.filter_recovery_window(t, x, recovery_window)

        arg_x_recovery = np.flatnonzero((x[:-1] != 0) & (x[1:] == 0))
        assert np.isnan(x).sum() == 0
        for i, arg in enumerate(arg_x_recovery):
            last = len(t)
            if i < len(arg_x_recovery) - 1:
                last = arg_x_recovery[i + 1]

            ti = t[arg + 1 : last] - t[arg]
            mi = m[arg + 1 : last]
            n_affected = np.sum(ti <= recovery_window)

            assert (~mi).sum() == n_affected

            assert (~mi)[n_affected:].sum() == 0

    @pytest.mark.parametrize("minimum_acquisitions", [0, 100])
    @pytest.mark.parametrize("entry_neglect_window", [0.0, 1000.0])
    @pytest.mark.parametrize("recovery_window", [0.0, 10000.0])
    def test_mask_noisy_observations(
        self,
        inpatient_observables: rx.InpatientObservables,
        minimum_acquisitions: int,
        entry_neglect_window: int,
        recovery_window: float,
    ):
        lead_extractor = leading_observables_extractor(
            observation_scheme=cast(rx.NumericScheme, SCHEMES["obs"]),
            minimum_acquisitions=minimum_acquisitions,
            entry_neglect_window=entry_neglect_window,
            recovery_window=recovery_window,
        )
        x = inpatient_observables.value[:, lead_extractor.code_index]
        t = inpatient_observables.time
        qual = f"{lead_extractor.__module__}.{type(lead_extractor).__qualname__}"
        with mock.patch(f"{qual}.filter_first_acquisitions") as mocker1:
            with mock.patch(f"{qual}.filter_entry_neglect_window") as mocker2:
                with mock.patch(f"{qual}.filter_recovery_window") as mocker3:
                    lead_extractor.mask_noisy_observations(
                        t,
                        x,
                        minimum_acquisitions=minimum_acquisitions,
                        entry_neglect_window=entry_neglect_window,
                        recovery_window=recovery_window,
                    )
                    mocker1.assert_called_once_with(len(t), minimum_acquisitions)
                    mocker2.assert_called_once_with(t, entry_neglect_window)
                    mocker3.assert_called_once_with(t, x, recovery_window)

    @pytest.mark.parametrize("code_index", [BINARY_OBSERVATION_CODE_INDEX, ORDINAL_OBSERVATION_CODE_INDEX])
    def test_extract_leading_window(self, inpatient_observables: rx.InpatientObservables, code_index: int):
        lead_extractor = leading_observables_extractor(
            observation_scheme=cast(rx.NumericScheme, SCHEMES["obs"]), code_index=code_index
        )
        x = inpatient_observables.value[:, lead_extractor.code_index, :]
        m = inpatient_observables.mask[:, lead_extractor.code_index]
        t = inpatient_observables.time
        lead = lead_extractor(inpatient_observables)

        assert len(lead) == len(inpatient_observables)
        assert lead.value.shape[1] == len(lead_extractor)

        for iw, w in enumerate(lead_extractor.config.leading_hours):
            for i in range(len(t)):
                delta_t = t[i:] - t[i]
                indexer = (delta_t <= w) & m[i:]
                xi = x[i:][indexer]
                yi = lead.value[i, iw]
                # if all are nan, then the lead is nan.
                if np.size(xi) == 0:
                    assert np.isnan(yi)
                elif code_index == BINARY_OBSERVATION_CODE_INDEX:
                    assert yi == xi.any()
                elif code_index == ORDINAL_OBSERVATION_CODE_INDEX:
                    assert yi == xi.max()
                else:
                    assert False

    def test_apply(self):
        pass


class TestInpatientInput:
    def test_init(self):
        pass

    def test_apply(self):
        # TODO: priority.
        pass

    @pytest.mark.parametrize("n", [0, 1, 100, 501])
    @pytest.mark.parametrize("p", [1, 100, 501])
    def test_hf5_group_serialization(self, n: int, p: int, hf5_group_writer: tb.Group):
        input = inpatient_binary_input(n, p, los_hours=ADMISSION_CONCEPT_MAX_STAY_HOURS)
        input.to_hdf_group(hf5_group_writer)
        loaded_input = rx.InpatientInput.from_hdf_group(hf5_group_writer)
        assert len(input) == len(loaded_input)
        assert input.equals(loaded_input)

    def test_empty(self):
        pass


class TestInpatientInterventions:
    def test_hf5_group_serialization(
        self,
        inpatient_interventions_with_a_none: rx.InpatientInterventions,
        hf5_group_writer: tb.Group,
    ):
        inpatient_interventions_with_a_none.to_hdf_group(hf5_group_writer)
        assert inpatient_interventions_with_a_none.equals(rx.InpatientInterventions.from_hdf_group(hf5_group_writer))

    def test_timestamps(self, inpatient_interventions_with_a_none: rx.InpatientInterventions):
        timestamps = inpatient_interventions_with_a_none.timestamps
        assert len(timestamps) == len(set(timestamps))
        assert all(isinstance(t, float) for t in timestamps)
        assert list(sorted(timestamps)) == timestamps
        ts = set()
        if inpatient_interventions_with_a_none.hosp_procedures is not None:
            ts.update(inpatient_interventions_with_a_none.hosp_procedures.starttime.tolist())
            ts.update(inpatient_interventions_with_a_none.hosp_procedures.endtime.tolist())
        if inpatient_interventions_with_a_none.icu_procedures is not None:
            ts.update(inpatient_interventions_with_a_none.icu_procedures.starttime.tolist())
            ts.update(inpatient_interventions_with_a_none.icu_procedures.endtime.tolist())
        if inpatient_interventions_with_a_none.icu_inputs is not None:
            ts.update(inpatient_interventions_with_a_none.icu_inputs.starttime.tolist())
            ts.update(inpatient_interventions_with_a_none.icu_inputs.endtime.tolist())
        assert ts == set(timestamps)


class TestSegmentedInpatientInterventions:
    def test_from_inpatient_interventions(self, inpatient_interventions_with_a_none):
        assert all(
            isinstance(scheme, rx.CodingScheme)
            for scheme in (
                SCHEMES["hosp_procedures"],
                SCHEMES["icu_procedures"],
                SCHEMES["icu_inputs"],
            )
        )
        schemes = {
            "hosp_procedures": SCHEMES["hosp_procedures"],
            "icu_procedures": SCHEMES["icu_procedures"],
            "icu_inputs": SCHEMES["icu_inputs"],
        }
        seg = rx.SegmentedInpatientInterventions.from_interventions(
            inpatient_interventions_with_a_none,
            ADMISSION_CONCEPT_MAX_STAY_HOURS,
            hosp_procedures_size=len(SCHEMES["hosp_procedures"]),
            icu_procedures_size=len(SCHEMES["icu_procedures"]),
            icu_inputs_size=len(SCHEMES["icu_inputs"]),
            maximum_padding=1,
        )
        assert abs(len(seg.time) - len(inpatient_interventions_with_a_none.timestamps)) <= 2
        for k in ("hosp_procedures", "icu_procedures", "icu_inputs"):
            if getattr(inpatient_interventions_with_a_none, k) is not None:
                seg_intervention = getattr(seg, k)
                assert seg_intervention.shape == (len(seg.time) - 1, len(schemes[k]))

            else:
                assert getattr(seg, k) is None

    @pytest.mark.parametrize("array", [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
    @pytest.mark.parametrize("value", [0.0, np.nan])
    @pytest.mark.parametrize("maximum_padding", [1, 2, 3, 5, 10, 100])
    def test_pad_array(self, array, value, maximum_padding):
        y = rx.SegmentedInpatientInterventions.pad_array(array, value=value, maximum_padding=maximum_padding)
        assert len(y) >= len(array)
        assert len(y) <= len(array) + maximum_padding
        assert np.all(y[: len(array)] == array)
        if np.isnan(value):
            assert np.isnan(y[len(array) :]).all()
        else:
            assert np.all(y[len(array) :] == value)

        if len(array) == maximum_padding or maximum_padding == 1:
            assert len(y) == len(array)

    @pytest.mark.parametrize("test_target", ["hosp_procedures", "icu_procedures", "icu_inputs"])
    def test_segmentation(
        self,
        inpatient_interventions_with_a_none: rx.InpatientInterventions,
        test_target: str,
    ):
        inpatient_intervention = getattr(inpatient_interventions_with_a_none, test_target)
        if inpatient_intervention is None or inpatient_intervention.starttime.size == 0:
            raise pytest.skip("No interventions to test")

        scheme = {
            "hosp_procedures": SCHEMES["hosp_procedures"],
            "icu_procedures": SCHEMES["icu_procedures"],
            "icu_inputs": SCHEMES["icu_inputs"],
        }[test_target]
        seg = rx.SegmentedInpatientInterventions._segment(
            inpatient_intervention.starttime, inpatient_intervention, len(scheme)
        )
        for i, t in enumerate(inpatient_intervention.starttime):
            assert np.array_equal(inpatient_intervention(t, len(scheme)), seg[i])

    def test_hf5_group_serialization(
        self,
        segmented_inpatient_interventions: rx.SegmentedInpatientInterventions,
        hf5_group_writer: tb.Group,
    ):
        segmented_inpatient_interventions.to_hdf_group(hf5_group_writer)
        assert segmented_inpatient_interventions.equals(
            rx.SegmentedInpatientInterventions.from_hdf_group(hf5_group_writer)
        )


class TestAdmission:
    def test_hf5_group_serialization(self, admission: rx.Admission, hf5_group_writer: tb.Group):
        admission.to_hdf_group(hf5_group_writer)
        assert admission.equals(rx.Admission.from_hdf_group(hf5_group_writer))

    def test_interval_hours(self):
        pass

    def test_interval_days(self):
        pass

    def test_days_since(self):
        pass


class TestSegmentedAdmission:
    def test_from_admission(self, admission: rx.Admission, segmented_admission: rx.SegmentedAdmission):
        assert segmented_admission.admission_id == admission.admission_id
        assert segmented_admission.admission_dates == admission.admission_dates
        assert segmented_admission.dx_codes == admission.dx_codes
        assert segmented_admission.dx_codes_history == admission.dx_codes_history
        assert segmented_admission.outcome == admission.outcome
        assert segmented_admission.observables is not None
        assert segmented_admission.leading_observable is not None
        if admission.observables is not None:
            assert rx.InpatientObservables.concat(segmented_admission.observables.segments).equals(
                admission.observables
            )
        if admission.leading_observable is not None:
            assert rx.InpatientObservables.concat(segmented_admission.leading_observable.segments).equals(
                admission.leading_observable
            )
        if admission.interventions is None:
            return
        interventions = admission.interventions
        seg_interventions = segmented_admission.interventions
        timestamps = admission.interventions.timestamps

        time = sorted(set([0.0] + timestamps + [ADMISSION_CONCEPT_MAX_STAY_HOURS]))
        for i, (start, end) in enumerate(zip(time[:-1], time[1:])):
            if interventions.hosp_procedures is not None:
                assert seg_interventions is not None and seg_interventions.hosp_procedures is not None
                p = seg_interventions.hosp_procedures.shape[1]
                hosp_procedure = interventions.hosp_procedures(start, p)
                seg_hosp_procedure = seg_interventions.hosp_procedures[i]
                assert np.array_equal(seg_hosp_procedure, hosp_procedure)
            if interventions.icu_procedures is not None:
                assert seg_interventions is not None and seg_interventions.icu_procedures is not None
                p = seg_interventions.icu_procedures.shape[1]
                icu_procedure = interventions.icu_procedures(start, p)
                seg_icu_procedure = seg_interventions.icu_procedures[i]
                assert np.array_equal(seg_icu_procedure, icu_procedure)
            if interventions.icu_inputs is not None:
                assert seg_interventions is not None and seg_interventions.icu_inputs is not None
                p = seg_interventions.icu_inputs.shape[1]
                icu_input = interventions.icu_inputs(start, p)
                seg_icu_input = seg_interventions.icu_inputs[i]
                assert np.array_equal(seg_icu_input, icu_input)

            if admission.observables is not None:
                obs_time = admission.observables.time
                segment_time = segmented_admission.observables[i].time
                assert np.all((start <= segment_time) & (segment_time <= end))
                if i == len(time) - 2:
                    time_mask = (obs_time >= start) & (obs_time <= end)
                else:
                    time_mask = (obs_time >= start) & (obs_time < end)
                segment_val = segmented_admission.observables[i].value
                segment_mask = segmented_admission.observables[i].mask
                val = admission.observables.value[time_mask]
                mask = admission.observables.mask[time_mask]
                assert np.array_equal(segment_val, val)
                assert np.array_equal(segment_mask, mask)

    def test_hf5_group_serialization(self, segmented_admission: rx.SegmentedAdmission, hf5_group_writer: tb.Group):
        segmented_admission.to_hdf_group(hf5_group_writer)
        assert segmented_admission.equals(rx.SegmentedAdmission.from_hdf_group(hf5_group_writer))


class TestStaticInfo:
    def test_hf5_group_serialization(self, static_info: rx.StaticInfo, hf5_group_writer: tb.Group):
        static_info.to_hdf_group(hf5_group_writer)
        assert static_info.equals(rx.StaticInfo.from_hdf_group(hf5_group_writer))

    def test_constant_attributes(self):
        pass

    def test_constant_vec(self):
        pass

    def test_dynamic_attributes(self):
        pass


class TestPatient:
    def test_d2d_interval_days(self):
        pass

    def test_outcome_frequency(self):
        pass

    def test_hf5_group_serialization(self, patient: rx.Patient, tmpdir: str):
        path = f"{tmpdir}/patient.h5"
        with tb.open_file(path, "w") as f:
            patient.to_hdf_group(f.create_group("/", "patient"))

        with tb.open_file(path, "r") as f:
            loaded_patient = rx.Patient.from_hdf_group(f.root.patient)
            assert patient.equals(loaded_patient)


class TestSegmentedPatient:
    def test_d2d_interval_days(self):
        pass

    def test_outcome_frequency(self):
        pass

    def test_hf5_group_serialization(self, segmented_patient: rx.SegmentedPatient, tmpdir: str):
        path = f"{tmpdir}/segmented_patient.h5"
        with tb.open_file(path, "w") as f:
            segmented_patient.to_hdf_group(f.create_group("/", "seg_patient"))

        with tb.open_file(path, "r") as f:
            loaded_segmented_patient = rx.SegmentedPatient.from_hdf_group(f.root.seg_patient)
            assert segmented_patient.equals(loaded_segmented_patient)
