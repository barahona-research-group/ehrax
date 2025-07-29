from abc import abstractmethod
from typing import Callable, Optional
from unittest import mock

import equinox as eqx
import pandas as pd
import pytest
import tables as tb

import ehrax as rx
from ehrax.testing.common_setup import DATASET_SCHEME_MANAGER, DATASET_SCHEME_CONF


@pytest.mark.parametrize('columns, id_cols, code_cols, time_cols, index', [
    [rx.dataset.StaticTableColumns(), ('subject_id',), (), ('date_of_birth',), ('subject_id',)],
    [rx.dataset.AdmissionsTableColumns(), ('subject_id', 'admission_id'), (), ('start_time', 'end_time'),
     ('admission_id',)],
    [rx.dataset.AdmissionSummaryTableColumns(), ('admission_id',), ('code',), (), ()],
    [rx.dataset.AdmissionTimeSeriesTableColumns(), ('admission_id',), ('code',), ('time',), ()],
    [rx.dataset.AdmissionIntervalEventsTableColumns(), ('admission_id',), ('code',), ('start_time', 'end_time'), ()],
    [rx.dataset.AdmissionIntervalRatesTableColumns(), ('admission_id',), ('code',), ('start_time', 'end_time'), ()],
])
def test_table_config(columns: rx.dataset.TableColumns, id_cols: tuple[str, ...],
                      code_cols: tuple[str, ...], time_cols: tuple[str, ...], index: Optional[str]):
    assert sorted(columns.time_cols) == sorted(time_cols)
    assert sorted(columns.code_cols) == sorted(code_cols)
    assert sorted(columns.id_cols) == sorted(id_cols)
    assert sorted(columns.index) == sorted(index)


def test_assert_invalid_column_fail():
    with pytest.raises(AssertionError, match="Fields must be one of"):
        c = rx.dataset.DatasetColumns()
        updated = eqx.tree_at(lambda x: x.admissions.subject_id, c, f'subject_id_')
        updated.validate()


def test_scheme_dict():
    scheme = rx.DatasetSchemeProxy(config=DATASET_SCHEME_CONF, schemes_context=DATASET_SCHEME_MANAGER)

    for space, scheme_name in DATASET_SCHEME_CONF.as_dict().items():
        assert hasattr(scheme, space)
        if scheme_name is not None:
            assert isinstance(getattr(scheme, space), rx.CodingScheme)


class AbstractTestDataset:
    @pytest.fixture(scope='class')
    @abstractmethod
    def dataset_tables(self, *args) -> rx.DatasetTables:
        raise NotImplementedError()

    @pytest.fixture(scope='class')
    @abstractmethod
    def dataset(self, *args) -> rx.Dataset:
        raise NotImplementedError()

    @abstractmethod
    def pipeline(self) -> list[rx.DatasetTransformation]:
        raise NotImplementedError()

    @pytest.fixture(scope='class')
    def processed_dataset(self, dataset: rx.Dataset, pipeline: list[rx.DatasetTransformation]) -> rx.Dataset:
        return dataset._execute_pipeline(pipeline, DATASET_SCHEME_MANAGER)

    @pytest.fixture(scope='class')
    def dataset_with_zero_pipeline(self, dataset: rx.Dataset):
        return dataset.execute_pipeline(rx.AbstractDatasetPipeline(config=None, transformations=[]), None)

    def test_tables_dict_property(self, dataset_tables: rx.DatasetTables):
        all_tables_keys = ('static', 'admissions', 'dx_discharge', 'obs',
                           'icu_procedures', 'icu_inputs', 'hosp_procedures')

        assert set(dataset_tables.tables_dict.keys()) == set(k for k in all_tables_keys
                                                             if getattr(dataset_tables, k) is not None)

    def test_save_load(self, dataset_tables: rx.DatasetTables, tmpdir):
        with tb.open_file(f'{tmpdir}/test_dataset_tables.h5', 'w') as hf5:
            dataset_tables.save(hf5.create_group('/', 'dataset_tables'))
        with tb.open_file(f'{tmpdir}/test_dataset_tables.h5', 'r') as hf5:
            loaded = rx.DatasetTables.load(hf5.root['dataset_tables'])
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
        assert dataset_with_zero_pipeline.pipeline_report.loc[0, 'transformation'] == 'identity'

        with mock.patch('logging.warning') as mocker:
            dataset3 = dataset_with_zero_pipeline.execute_pipeline(
                rx.AbstractDatasetPipeline(config=None, transformations=[]), None)
            assert dataset3.equals(dataset_with_zero_pipeline)
            mocker.assert_called_once_with("A pipeline has already been executed. Doing nothing.")

    def test_subject_ids_of_unindexed_dataset(self, dataset: rx.Dataset,
                                              dataset_with_zero_pipeline: rx.Dataset):
        with pytest.raises(AssertionError):
            _ = dataset.subject_ids

        with pytest.raises(AssertionError):
            _ = dataset_with_zero_pipeline.subject_ids

    def test_subject_ids_of_indexed_dataset(self, processed_dataset: rx.Dataset):
        assert set(processed_dataset.subject_ids) == set(processed_dataset.tables.static.index.unique())

    @pytest.mark.expensive_test
    def test_save_load(self, dataset: rx.Dataset,
                       dataset_with_zero_pipeline: rx.Dataset,
                       tmpdir: str):
        dataset_with_zero_pipeline.save(f'{tmpdir}/test_dataset')
        loaded = rx.Dataset.load(f'{tmpdir}/test_dataset')
        assert loaded.equals(dataset_with_zero_pipeline)
        assert not loaded.equals(dataset)
        assert loaded.equals(dataset._execute_pipeline([], None))

    @pytest.fixture(scope='class')
    def subject_ids(self, processed_dataset: rx.Dataset):
        return processed_dataset.subject_ids

    @pytest.mark.parametrize('valid_split', [[1.0]])
    @pytest.mark.parametrize('valid_balance', ['subjects', 'admissions', 'admissions_intervals'])
    @pytest.mark.parametrize('invalid_splits', [[], [0.3, 0.8, 0.7, 0.9], [0.5, 0.2]])  # should be sorted.
    @pytest.mark.parametrize('invalid_balance', ['hi', 'unsupported'])
    def test_random_split_invalid_args(self, processed_dataset: rx.Dataset, subject_ids: list[str],
                                       valid_split: list[float], valid_balance: rx.SplitLiteral,
                                       invalid_splits: list[float], invalid_balance: rx.SplitLiteral):
        if len(subject_ids) == 0:
            with pytest.raises(AssertionError):
                processed_dataset.random_splits(valid_split, balance=valid_balance)
            return

        if len(processed_dataset.tables.admissions) == 0 and 'admissions' in valid_balance:
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
    @pytest.fixture(scope='class')
    def dataset_tables(self, dataset_tables_without_records: rx.DatasetTables) -> rx.DatasetTables:
        return dataset_tables_without_records

    @pytest.fixture(scope='class')
    def dataset(self, dataset_without_records: rx.Dataset) -> rx.Dataset:
        return dataset_without_records

    @pytest.fixture(scope='class')
    def pipeline(self) -> list[rx.DatasetTransformation]:
        return [rx.SetIndex(), rx.SynchronizeSubjects(), rx.CastTimestamps(), rx.SetAdmissionRelativeTimes()]


class TestDatasetWithRecords(AbstractTestDataset):
    @pytest.fixture(scope='class')
    def dataset_tables(self, dataset_tables_with_records: rx.DatasetTables) -> rx.DatasetTables:
        return dataset_tables_with_records

    @pytest.fixture(scope='class')
    def dataset(self, dataset_with_records: rx.Dataset) -> rx.Dataset:
        return dataset_with_records

    @pytest.fixture(scope='class')
    def pipeline(self):
        return [rx.SetIndex(), rx.SynchronizeSubjects(), rx.CastTimestamps(), rx.ICUInputRateUnitConversion(),
                rx.SetAdmissionRelativeTimes()]

    @pytest.fixture(scope='class')
    def split_measure(self, processed_dataset: rx.Dataset, balance: str):
        return {
            'subjects': lambda x: len(x),
            'admissions': lambda x: sum(processed_dataset.subjects_n_admissions.loc[x]),
            'admissions_intervals': lambda x: sum(processed_dataset.subjects_intervals_sum.loc[x])
        }[balance]

    @pytest.fixture(params=['subjects', 'admissions', 'admissions_intervals'], scope='class')
    def balance(self, request):
        return request.param

    @pytest.fixture(params=[[0.1], [0.1, 0.5, 0.7, 0.9]], scope='class')
    def split_quantiles(self, request):
        return request.param

    @pytest.fixture(params=[1, 11, 111], scope='class')
    def subject_splits(self, processed_dataset: rx.Dataset, balance: rx.SplitLiteral,
                       split_quantiles: list[float], request):
        random_seed = request.param
        return processed_dataset.random_splits(split_quantiles, balance=balance, random_seed=random_seed)

    def test_random_split(self, processed_dataset: rx.Dataset, subject_ids: list[str],
                          subject_splits: list[list[str]],
                          split_quantiles: list[float]):
        assert set.union(*list(map(set, subject_splits))) == set(subject_ids)
        assert len(subject_splits) == len(split_quantiles) + 1
        # No overlaps.
        assert sum(len(v) for v in subject_splits) == len(processed_dataset.subject_ids)
        assert set.union(*[set(v) for v in subject_splits]) == set(processed_dataset.subject_ids)

    @pytest.fixture
    def split_proportions(self, split_quantiles: list[float]):
        return [p1 - p0 for p0, p1 in zip([0] + split_quantiles, split_quantiles + [1])]

    def test_random_split_balance(self, subject_ids: list[str],
                                  subject_splits: list[list[str]],
                                  split_proportions: list[float],
                                  balance: str,
                                  split_measure: Callable[[list[str]], float]):
        if len(subject_ids) < 5:
            raise pytest.skip("Not enough subjects to test random split")

        # # test proportionality
        # NOTE: no specified behaviour when splits have equal proportions, so comparing argsorts
        # is not appropriate.
        p_threshold = 1 / len(subject_ids)
        tolerance = max(
            abs(split_measure([i]) - split_measure([j])) for (i, j) in zip(subject_ids[1:], subject_ids[:-1]))
        for i in range(len(split_proportions)):
            split_measure_i = split_measure(subject_splits[i])
            for j in range(i + 1, len(split_proportions)):
                if abs(split_proportions[i] - split_proportions[j]) < 2 * p_threshold:
                    if balance == 'subjects':
                        # Difference between subjects is at most 1 when balance is applied
                        # on subjects count AND proportions are (almost) equal.
                        assert abs(len(subject_splits[i]) - len(subject_splits[j])) <= 1
                elif split_proportions[i] > split_proportions[j]:
                    assert (split_measure_i - split_measure(subject_splits[j])) >= -tolerance
                else:
                    assert (split_measure_i - split_measure(subject_splits[j])) <= tolerance
