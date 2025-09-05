from abc import abstractmethod
from collections.abc import Callable
from typing import get_args
from unittest import mock

import ehrax as rx
import equinox as eqx
import numpy as np
import pandas as pd
import pytest
import tables as tb
from ehrax._literals import TableAggregationLiteral
from ehrax._stats.dataset import DatasetStatsInterface, TargetHistogram
from ehrax.dataset import AdmissionsTableColumns
from ehrax.testing.common_setup import DATASET_SCHEME_CONF, DATASET_SCHEME_MANAGER, OUTCOME_DATA


@pytest.mark.parametrize(
    "columns, id_cols, code_cols, time_cols, index",
    [
        [rx.dataset.StaticTableColumns(), ("subject_id",), (), ("date_of_birth",), ("subject_id",)],
        [
            rx.dataset.AdmissionsTableColumns(),
            ("subject_id", "admission_id"),
            (),
            ("start_time", "end_time"),
            ("admission_id",),
        ],
        [rx.dataset.AdmissionSummaryTableColumns(), ("admission_id",), ("code",), (), ()],
        [rx.dataset.AdmissionTimeSeriesTableColumns(), ("admission_id",), ("code",), ("time",), ()],
        [
            rx.dataset.AdmissionIntervalEventsTableColumns(),
            ("admission_id",),
            ("code",),
            ("start_time", "end_time"),
            (),
        ],
        [rx.dataset.AdmissionIntervalRatesTableColumns(), ("admission_id",), ("code",), ("start_time", "end_time"), ()],
    ],
)
def test_table_config(
    columns: rx.dataset.TableColumns,
    id_cols: tuple[str, ...],
    code_cols: tuple[str, ...],
    time_cols: tuple[str, ...],
    index: str | None,
):
    assert sorted(columns.time_cols) == sorted(time_cols)
    assert sorted(columns.code_cols) == sorted(code_cols)
    assert sorted(columns.id_cols) == sorted(id_cols)
    assert sorted(columns.index) == sorted(index)


def test_assert_invalid_column_fail():
    with pytest.raises(AssertionError, match="Fields must be one of"):
        c = rx.dataset.DatasetColumns()
        updated = eqx.tree_at(lambda x: x.admissions.subject_id, c, "subject_id_")
        updated.validate()


def test_scheme_dict():
    scheme = rx.DatasetSchemeProxy(config=DATASET_SCHEME_CONF, schemes_context=DATASET_SCHEME_MANAGER)

    for space, scheme_name in DATASET_SCHEME_CONF.as_dict().items():
        assert hasattr(scheme, space)
        if scheme_name is not None:
            assert isinstance(getattr(scheme, space), rx.CodingScheme)


class AbstractTestDataset:
    @pytest.fixture(scope="class")
    @abstractmethod
    def dataset_tables(self, *args) -> rx.DatasetTables:
        raise NotImplementedError()

    @pytest.fixture(scope="class")
    @abstractmethod
    def dataset(self, *args) -> rx.Dataset:
        raise NotImplementedError()

    @abstractmethod
    def pipeline(self) -> list[rx.DatasetTransformation]:
        raise NotImplementedError()

    @pytest.fixture(scope="class")
    def processed_dataset(self, dataset: rx.Dataset, pipeline: list[rx.DatasetTransformation]) -> rx.Dataset:
        return dataset._execute_pipeline(pipeline, DATASET_SCHEME_MANAGER)

    @pytest.fixture(scope="class")
    def dataset_with_zero_pipeline(self, dataset: rx.Dataset):
        return dataset.execute_pipeline(rx.AbstractDatasetPipeline(config=None, transformations=[]), None)

    def test_tables_dict_property(self, dataset_tables: rx.DatasetTables):
        all_tables_keys = (
            "static",
            "admissions",
            "dx_discharge",
            "obs",
            "icu_procedures",
            "icu_inputs",
            "hosp_procedures",
        )

        assert set(dataset_tables.tables_dict.keys()) == set(
            k for k in all_tables_keys if getattr(dataset_tables, k) is not None
        )

    def test_save_load_tables(self, dataset_tables: rx.DatasetTables, tmpdir):
        with tb.open_file(f"{tmpdir}/test_dataset_tables.h5", "w") as hf5:
            dataset_tables.save(hf5.create_group("/", "dataset_tables"))
        with tb.open_file(f"{tmpdir}/test_dataset_tables.h5", "r") as hf5:
            loaded = rx.DatasetTables.load(hf5.root["dataset_tables"])
        assert loaded.equals(dataset_tables)

    def test_execute_pipeline(self, dataset: rx.Dataset, dataset_with_zero_pipeline: rx.Dataset):
        assert isinstance(dataset, rx.Dataset)
        assert isinstance(dataset_with_zero_pipeline, rx.Dataset)
        assert dataset.pipeline_report.equals(pd.DataFrame())

        # Because we use identity pipeline, the dataset columns should be the same
        # but the new dataset should have a different report (metadata).
        assert not dataset_with_zero_pipeline.equals(dataset)
        assert not dataset_with_zero_pipeline.pipeline_report.equals(dataset.pipeline_report)
        assert dataset_with_zero_pipeline.tables.equals(dataset.tables)
        assert len(dataset_with_zero_pipeline.pipeline_report) == 1
        assert dataset_with_zero_pipeline.pipeline_report.loc[0, "transformation"] == "identity"

        with mock.patch("logging.warning") as mocker:
            dataset3 = dataset_with_zero_pipeline.execute_pipeline(
                rx.AbstractDatasetPipeline(config=None, transformations=[]), None
            )
            assert dataset3.equals(dataset_with_zero_pipeline)
            mocker.assert_called_once_with("A pipeline has already been executed. Doing nothing.")

    def test_subject_ids_of_unindexed_dataset(self, dataset: rx.Dataset, dataset_with_zero_pipeline: rx.Dataset):
        with pytest.raises(AssertionError):
            _ = dataset.subject_ids

        with pytest.raises(AssertionError):
            _ = dataset_with_zero_pipeline.subject_ids

    def test_subject_ids_of_indexed_dataset(self, processed_dataset: rx.Dataset):
        assert set(processed_dataset.subject_ids) == set(processed_dataset.tables.static.index.unique())

    @pytest.mark.expensive_test
    def test_save_load(self, dataset: rx.Dataset, dataset_with_zero_pipeline: rx.Dataset, tmpdir: str):
        dataset_with_zero_pipeline.save(f"{tmpdir}/test_dataset")
        loaded = rx.Dataset.load(f"{tmpdir}/test_dataset")
        assert loaded.equals(dataset_with_zero_pipeline)
        assert not loaded.equals(dataset)
        assert loaded.equals(dataset._execute_pipeline([], None))

    @pytest.fixture(scope="class")
    def subject_ids(self, processed_dataset: rx.Dataset):
        return processed_dataset.subject_ids

    @pytest.mark.parametrize("valid_split", [[1.0]])
    @pytest.mark.parametrize("valid_balance", ["subjects", "admissions", "admissions_intervals"])
    @pytest.mark.parametrize("invalid_splits", [[], [0.3, 0.8, 0.7, 0.9], [0.5, 0.2]])  # should be sorted.
    @pytest.mark.parametrize("invalid_balance", ["hi", "unsupported"])
    def test_random_split_invalid_args(
        self,
        processed_dataset: rx.Dataset,
        subject_ids: list[str],
        valid_split: list[float],
        valid_balance: rx.SplitLiteral,
        invalid_splits: list[float],
        invalid_balance: rx.SplitLiteral,
    ):
        if len(subject_ids) == 0:
            with pytest.raises(AssertionError):
                processed_dataset.random_splits(valid_split, balance=valid_balance)
            return

        if len(processed_dataset.tables.admissions) == 0 and "admissions" in valid_balance:
            with pytest.raises(AssertionError):
                processed_dataset.random_splits(valid_split, balance=valid_balance)
            return

        assert set(processed_dataset.random_splits(valid_split, balance=valid_balance)[0]) == set(subject_ids)

        with pytest.raises(AssertionError):
            processed_dataset.random_splits(valid_split, balance=invalid_balance)

        with pytest.raises(AssertionError):
            processed_dataset.random_splits(invalid_splits, balance=valid_balance)

        with pytest.raises(AssertionError):
            processed_dataset.random_splits(invalid_splits, balance=invalid_balance)


class TestDatasetWithoutRecords(AbstractTestDataset):
    @pytest.fixture(scope="class")
    def dataset_tables(self, dataset_tables_without_records: rx.DatasetTables) -> rx.DatasetTables:
        return dataset_tables_without_records

    @pytest.fixture(scope="class")
    def dataset(self, dataset_without_records: rx.Dataset) -> rx.Dataset:
        return dataset_without_records

    @pytest.fixture(scope="class")
    def pipeline(self) -> list[rx.DatasetTransformation]:
        return [rx.SetIndex(), rx.SynchronizeSubjects(), rx.CastTimestamps(), rx.SetAdmissionRelativeTimes()]


class TestDatasetWithRecords(AbstractTestDataset):
    @pytest.fixture(scope="class")
    def dataset_tables(self, dataset_tables_with_records: rx.DatasetTables) -> rx.DatasetTables:
        return dataset_tables_with_records

    @pytest.fixture(scope="class")
    def dataset(self, dataset_with_records: rx.Dataset) -> rx.Dataset:
        return dataset_with_records

    @pytest.fixture(scope="class")
    def pipeline(self):
        return [
            rx.SetIndex(),
            rx.SynchronizeSubjects(),
            rx.CastTimestamps(),
            rx.ICUInputRateUnitConversion(),
            rx.SetAdmissionRelativeTimes(),
        ]

    @pytest.fixture(scope="class")
    def split_measure(self, processed_dataset: rx.Dataset, balance: str):
        return {
            "subjects": lambda x: len(x),
            "admissions": lambda x: sum(processed_dataset.subjects_n_admissions.loc[x]),
            "admissions_intervals": lambda x: sum(processed_dataset.subjects_intervals_sum.loc[x]),
        }[balance]

    @pytest.fixture(params=["subjects", "admissions", "admissions_intervals"], scope="class")
    def balance(self, request):
        return request.param

    @pytest.fixture(params=[[0.1], [0.1, 0.5, 0.7, 0.9]], scope="class")
    def split_quantiles(self, request):
        return request.param

    @pytest.fixture(params=[1, 11, 111], scope="class")
    def subject_splits(
        self, processed_dataset: rx.Dataset, balance: rx.SplitLiteral, split_quantiles: list[float], request
    ):
        random_seed = request.param
        return processed_dataset.random_splits(split_quantiles, balance=balance, random_seed=random_seed)

    def test_random_split(
        self,
        processed_dataset: rx.Dataset,
        subject_ids: list[str],
        subject_splits: list[list[str]],
        split_quantiles: list[float],
    ):
        assert set(i for ii in subject_splits for i in ii) == set(subject_ids)
        assert len(subject_splits) == len(split_quantiles) + 1
        # No overlaps.
        assert sum(len(v) for v in subject_splits) == len(processed_dataset.subject_ids)
        assert set(i for ii in subject_splits for i in ii) == set(processed_dataset.subject_ids)

    @pytest.fixture
    def split_proportions(self, split_quantiles: list[float]):
        return [p1 - p0 for p0, p1 in zip([0] + split_quantiles, split_quantiles + [1])]

    def test_random_split_balance(
        self,
        subject_ids: list[str],
        subject_splits: list[list[str]],
        split_proportions: list[float],
        balance: str,
        split_measure: Callable[[list[str]], float],
    ):
        if len(subject_ids) < 5:
            raise pytest.skip("Not enough subjects to test random split")

        # # test proportionality
        # NOTE: no specified behaviour when splits have equal proportions, so comparing argsorts
        # is not appropriate.
        p_threshold = 1 / len(subject_ids)
        tolerance = max(
            abs(split_measure([i]) - split_measure([j])) for (i, j) in zip(subject_ids[1:], subject_ids[:-1])
        )
        for i in range(len(split_proportions)):
            split_measure_i = split_measure(subject_splits[i])
            for j in range(i + 1, len(split_proportions)):
                if abs(split_proportions[i] - split_proportions[j]) < 2 * p_threshold:
                    if balance == "subjects":
                        # Difference between subjects is at most 1 when balance is applied
                        # on subjects count AND proportions are (almost) equal.
                        assert abs(len(subject_splits[i]) - len(subject_splits[j])) <= 1
                elif split_proportions[i] > split_proportions[j]:
                    assert (split_measure_i - split_measure(subject_splits[j])) >= -tolerance
                else:
                    assert (split_measure_i - split_measure(subject_splits[j])) <= tolerance

    #
    # @pytest.fixture
    # def dataset_hist_interface(self, processed_dataset: rx.Dataset) -> TargetHistogram:
    #     return processed_dataset.stats(DATASET_SCHEME_MANAGER).target_hist
    #
    # #
    # def stats_dx_discharge(self) -> :


class TestTargetHistogram:
    @pytest.fixture(scope="class")
    def stats_interface(self, dataset_with_records: rx.Dataset) -> DatasetStatsInterface:
        return dataset_with_records._execute_pipeline(
            [rx.SetIndex(), rx.SynchronizeSubjects(), rx.CastTimestamps()], DATASET_SCHEME_MANAGER
        ).stats(DATASET_SCHEME_MANAGER)

    @pytest.fixture(scope="class")
    def code_map(self) -> rx.CodeMap:
        m = rx.CodeMap(
            source_name="x",
            target_name="y",
            data=rx.FrozenDict1N({"a": {"x"}, "b": {"x"}, "c": {"y"}, "d": {"x", "y"}}),
        )
        assert m.domain == {"a", "b", "c", "d"}
        assert m.range == {"x", "y"}
        return m

    @pytest.fixture(scope="class")
    def c_code(self) -> str:
        return "code"

    @pytest.fixture(scope="class")
    def c_admission_id(self) -> str:
        return "admission_id"

    @pytest.mark.parametrize(
        "admission_id, code, expected",
        [
            # empty table -> {}
            [(), (), {}],
            # table with no valid code -> {}
            [("1",), ("!",), {}],
            [("1", "2"), ("?", "!"), {}],
            [("1", "1"), ("$", "*"), {}],
            # table with just one valid code.
            [("1",), ("a",), {"x": 1}],
            # aggregation of admissions
            [("1", "1"), ("a", "b"), {"x": 1}],  # a and b both map to x
            [("1", "1"), ("a", "b"), {"x": 1}],
            [("1", "2"), ("a", "a"), {"x": 2}],
            [("1", "2"), ("a", "b"), {"x": 2}],
            [("1", "1"), ("a", "c"), {"x": 1, "y": 1}],
            # rest
            [("1", "2", "3"), ("a", "a", "a"), {"x": 3}],
            [("1", "1", "2"), ("d", "a", "d"), {"x": 2, "y": 2}],
        ],
    )
    def test_hardcoded_table(
        self, admission_id: tuple[str, ...], code: tuple[str, ...], expected: dict[str, int], code_map: rx.CodeMap
    ):
        table = pd.DataFrame({"id": admission_id, "code": code})
        hist = TargetHistogram.compute(table, c_admission_id="id", c_code="code", scheme_mapper=code_map).to_dict()
        assert hist == expected

    @pytest.mark.parametrize(
        "admission_id, subject_id, admission_time, expected_agg_admission, "
        "expected_agg_first_admission, expected_agg_subject",
        [
            [(), (), (), (), (), ()],
            [("1",), ("a",), (0,), ("1",), ("1",), ("a",)],
            [("1", "2"), ("a", "a"), (1, 0), ("1", "2"), ("2",), ("a", "a")],
            [
                ("1", "2", "3", "4"),
                ("a", "a", "b", "b"),
                (0, 1, 4, 3),
                ("1", "2", "3", "4"),
                ("1", "4"),
                ("a", "a", "b", "b"),
            ],
        ],
    )
    def test_aggregation_adapt(
        self,
        admission_id: tuple[str, ...],
        subject_id: tuple[str, ...],
        admission_time: tuple[int, ...],
        expected_agg_admission: tuple[str, ...],
        expected_agg_first_admission: tuple[str, ...],
        expected_agg_subject: tuple[str, ...],
    ):
        adm_cols = AdmissionsTableColumns()
        table = pd.DataFrame(
            {adm_cols.admission_id: admission_id, adm_cols.subject_id: subject_id, adm_cols.start_time: admission_time}
        )
        adms = table.set_index(adm_cols.admission_id)
        adapted1, n1 = TargetHistogram.adapt_aggregation_level(
            adms, adm_cols, table, adm_cols.admission_id, "admission"
        )
        adapted2, n2 = TargetHistogram.adapt_aggregation_level(
            adms, adm_cols, table, adm_cols.admission_id, "first_admission"
        )
        adapted3, n3 = TargetHistogram.adapt_aggregation_level(adms, adm_cols, table, adm_cols.admission_id, "subject")
        assert adapted1[adm_cols.admission_id].tolist() == list(expected_agg_admission)
        assert adapted2[adm_cols.admission_id].tolist() == list(expected_agg_first_admission)
        assert adapted3[adm_cols.admission_id].tolist() == list(expected_agg_subject)
        assert n1 == len(admission_id)
        assert n2 == n3 == len(set(subject_id))

    def test_dataset_stats(self, stats_interface: DatasetStatsInterface):
        assert isinstance(stats_interface, DatasetStatsInterface)
        o = DATASET_SCHEME_MANAGER.outcome_data[OUTCOME_DATA.name]
        o_scheme = o.as_coding_scheme(DATASET_SCHEME_MANAGER.scheme[o.base_name])
        dx_scheme = DATASET_SCHEME_MANAGER.scheme[o.base_name]
        dx_stats, dx_norm = zip(
            *[stats_interface.target_hist.dx_discharge(dx_scheme.name, a) for a in get_args(TableAggregationLiteral)]
        )
        o_stats, o_norm = zip(
            *[stats_interface.target_hist.outcome(o.name, a) for a in get_args(TableAggregationLiteral)]
        )
        assert all(isinstance(dx_stats_i, pd.Series) for dx_stats_i in dx_stats)
        assert all(dx_stats_i.index.tolist() == list(dx_scheme.codes) for dx_stats_i in dx_stats)
        assert all(dx_stats_i.dtype is np.dtype("int") for dx_stats_i in dx_stats)

        assert all(isinstance(o_stats_i, pd.Series) for o_stats_i in o_stats)
        assert all(o_stats_i.index.tolist() == list(o_scheme.codes) for o_stats_i in o_stats)
        assert all(o_stats_i.dtype is np.dtype("int") for o_stats_i in o_stats)
