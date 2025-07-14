import random
import string
from collections import defaultdict

import equinox as eqx
import numpy as np
import pandas as pd
import pytest

from ehrax.base import AbstractConfig
from ehrax.dataset import ReportAttributes, Report, Dataset
from ehrax.transformations import (DatasetTransformation, FilterUnsupportedCodes, SetAdmissionRelativeTimes,
                                     ProcessOverlappingAdmissions,
                                     FilterClampTimestampsToAdmissionInterval, FilterSubjectsNegativeAdmissionLengths,
                                     CastTimestamps)
from test.common_setup import ALIAS, DATASET_SCHEME_MANAGER


class TestDatasetTransformation:

    @pytest.fixture(scope='class')
    def removed_subject_admissions_dataset(self, indexed_dataset: Dataset, sample_subject_id: str):
        admissions = indexed_dataset.tables.admissions
        admissions = admissions[admissions[ALIAS['subject_id']] != sample_subject_id]
        return eqx.tree_at(lambda x: x.tables.admissions, indexed_dataset, admissions)

    @pytest.fixture(scope='class')
    def removed_no_admission_subjects_RESULTS(self, removed_subject_admissions_dataset: Dataset):
        filtered_dataset, report = DatasetTransformation.filter_no_admission_subjects(
            removed_subject_admissions_dataset, Report())
        return filtered_dataset, report

    @pytest.fixture(scope='class')
    def removed_no_admission_subjects(self, removed_no_admission_subjects_RESULTS) -> Dataset:
        return removed_no_admission_subjects_RESULTS[0]

    @pytest.fixture(scope='class')
    def removed_no_admission_subjects_REPORT(self, removed_no_admission_subjects_RESULTS) -> Report:
        return removed_no_admission_subjects_RESULTS[1]

    @pytest.fixture(scope='class')
    def removed_subject_dataset_unsync(self, has_admissions_dataset: Dataset, sample_subject_id: str):
        static = has_admissions_dataset.tables.static
        static = static.drop(index=sample_subject_id)
        return eqx.tree_at(lambda x: x.tables.static, has_admissions_dataset, static)

    @pytest.fixture(scope='class')
    def removed_admission_dataset_unsync(self, has_admissions_dataset: Dataset, sample_admission_id: str):
        admissions = has_admissions_dataset.tables.admissions
        admissions = admissions.drop(index=sample_admission_id)
        return eqx.tree_at(lambda x: x.tables.admissions, has_admissions_dataset, admissions)

    @pytest.fixture(scope='class')
    def removed_subject_dataset_sync_RESULTS(self, removed_subject_dataset_unsync):
        return DatasetTransformation.synchronize_subjects(removed_subject_dataset_unsync, Report())

    @pytest.fixture(scope='class')
    def removed_subject_dataset_sync(self, removed_subject_dataset_sync_RESULTS) -> Dataset:
        return removed_subject_dataset_sync_RESULTS[0]

    @pytest.fixture(scope='class')
    def removed_subject_dataset_sync_REPORT(self, removed_subject_dataset_sync_RESULTS) -> Report:
        return removed_subject_dataset_sync_RESULTS[1]

    @pytest.fixture(scope='class')
    def removed_admission_dataset_sync_RESUTLS(self, removed_admission_dataset_unsync):
        return DatasetTransformation.synchronize_index(removed_admission_dataset_unsync,
                                                       'admissions', ALIAS['admission_id'], Report())

    @pytest.fixture(scope='class')
    def removed_admission_dataset_sync(self, removed_admission_dataset_sync_RESUTLS) -> Dataset:
        return removed_admission_dataset_sync_RESUTLS[0]

    @pytest.fixture(scope='class')
    def removed_admission_dataset_sync_REPORT(self, removed_admission_dataset_sync_RESUTLS) -> Report:
        return removed_admission_dataset_sync_RESUTLS[1]

    def test_filter_no_admissions_subjects(self, removed_no_admission_subjects: Dataset,
                                           sample_subject_id: str):
        assert sample_subject_id not in removed_no_admission_subjects.tables.static

    def test_synchronize_index_subjects(self,
                                        has_admissions_dataset: Dataset,
                                        removed_subject_dataset_unsync: Dataset,
                                        removed_subject_dataset_sync: Dataset,
                                        sample_subject_id: str):
        assert sample_subject_id in has_admissions_dataset.tables.admissions[ALIAS['subject_id']].values
        assert sample_subject_id in removed_subject_dataset_unsync.tables.admissions[ALIAS['subject_id']].values
        assert sample_subject_id not in removed_subject_dataset_sync.tables.admissions[ALIAS['subject_id']].values
        assert set(removed_subject_dataset_sync.tables.static.index) == set(
            removed_subject_dataset_sync.tables.admissions[ALIAS['subject_id']])

    def test_synchronize_index_admissions(self, has_admissions_dataset: Dataset,
                                          removed_admission_dataset_unsync: Dataset,
                                          removed_admission_dataset_sync: Dataset,
                                          sample_admission_id: str):
        for table_name, table in removed_admission_dataset_unsync.tables.tables_dict.items():
            if table is not None and ALIAS['admission_id'] in table.columns and len(table) > 0:
                assert sample_admission_id in table[ALIAS['admission_id']].values
                synced_table = getattr(removed_admission_dataset_sync.tables, table_name)
                assert sample_admission_id not in synced_table[ALIAS['admission_id']].values
                assert set(synced_table[ALIAS['admission_id']]).issubset(
                    set(removed_admission_dataset_sync.tables.admissions.index))

    def test_generated_report1(self, removed_no_admission_subjects_REPORT: Report):
        assert isinstance(removed_no_admission_subjects_REPORT, Report)
        assert len(removed_no_admission_subjects_REPORT) > 0
        assert isinstance(removed_no_admission_subjects_REPORT[0], ReportAttributes)

    def test_serializable_report1(self, removed_no_admission_subjects_REPORT: Report):
        assert all(isinstance(v.as_dict(), dict) for v in removed_no_admission_subjects_REPORT)
        assert all([AbstractConfig.from_dict(v.to_dict()).equals(v) for v in removed_no_admission_subjects_REPORT])


class TestCastTimestamps:
    @pytest.fixture(scope='class')
    def str_timestamps_dataset(self, indexed_dataset: Dataset):

        for table_name, time_cols in indexed_dataset.config.tables.time_cols.items():
            table = indexed_dataset.tables.tables_dict[table_name].copy()
            for col in time_cols:
                table[col] = table[col].astype(str)
            indexed_dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), indexed_dataset, table)
        return indexed_dataset

    @pytest.fixture(scope='class')
    def casted_timestamps_dataset(self, str_timestamps_dataset: Dataset):
        return CastTimestamps.apply(str_timestamps_dataset, DATASET_SCHEME_MANAGER, Report())[0]

    def test_cast_timestamps(self, str_timestamps_dataset: Dataset,
                             casted_timestamps_dataset: Dataset):
        for table_name, time_cols in str_timestamps_dataset.config.tables.time_cols.items():
            table1 = str_timestamps_dataset.tables.tables_dict[table_name]
            table2 = casted_timestamps_dataset.tables.tables_dict[table_name]
            for col in time_cols:
                assert table1[col].dtype == np.dtype('O')
                assert table2[col].dtype == np.dtype('datetime64[ns]')


class TestFilterUnsupportedCodes:
    @pytest.fixture(scope='class')
    def dataset_with_unsupported_codes(self, has_codes_dataset: Dataset) -> tuple[Dataset, dict[str, set[str]]]:
        unsupported_codes = {}
        for table_name, code_col in has_codes_dataset.config.tables.code_column.items():
            table = has_codes_dataset.tables.tables_dict[table_name]
            unsupported_code = f'UNSUPPORTED_CODE_{"".join(random.choices(string.ascii_uppercase, k=5))}'
            table.loc[table.index[0], code_col] = unsupported_code
            unsupported_codes[table_name] = unsupported_code
            has_codes_dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), has_codes_dataset, table)
        return has_codes_dataset, unsupported_codes

    @pytest.fixture(scope='class')
    def filtered_dataset(self, dataset_with_unsupported_codes: tuple[Dataset, dict[str, set[str]]]) -> Dataset:
        dataset, _ = dataset_with_unsupported_codes
        return FilterUnsupportedCodes.apply(dataset, DATASET_SCHEME_MANAGER, Report())[0]

    def test_filter_unsupported_codes(self, dataset_with_unsupported_codes: tuple[Dataset, dict[str, set[str]]],
                                      filtered_dataset: Dataset):
        unfiltered_dataset, unsupported_codes = dataset_with_unsupported_codes
        for table_name, code_col in filtered_dataset.config.tables.code_column.items():
            assert unsupported_codes[table_name] in getattr(unfiltered_dataset.tables, table_name)[code_col].values
            assert unsupported_codes[table_name] not in getattr(filtered_dataset.tables, table_name)[code_col].values


class TestSetRelativeTimes:

    @pytest.fixture(scope='class')
    def relative_times_dataset(self, has_obs_dataset: Dataset):
        return SetAdmissionRelativeTimes.apply(has_obs_dataset, DATASET_SCHEME_MANAGER, Report())[0]

    @pytest.fixture(scope='class')
    def admission_los_table(self, has_obs_dataset: Dataset):
        admissions = has_obs_dataset.tables.admissions.copy()
        admissions['los_hours'] = (admissions[ALIAS['discharge_time']] - admissions[
            ALIAS['admission_time']]).dt.total_seconds() / (60 * 60)
        return admissions[['los_hours']]

    def test_set_relative_times(self, has_obs_dataset: Dataset,
                                relative_times_dataset: Dataset,
                                admission_los_table: pd.DataFrame):

        for table_name, time_cols in has_obs_dataset.config.tables.time_cols.items():
            if table_name in ('admissions', 'static'):
                continue
            table = getattr(relative_times_dataset.tables, table_name)
            table = table.merge(admission_los_table,
                                left_on=ALIAS['admission_id'],
                                right_index=True)
            for col in time_cols:
                assert table[col].dtype == float
                assert table[col].min() >= 0
                assert all(table[col] <= table['los_hours'])


class TestFilterSubjectsWithNegativeAdmissionInterval:
    @pytest.fixture(scope='class')
    def dataset_with_negative_admission(self, has_admissions_dataset: Dataset,
                                        sample_admission_id: str) -> Dataset:
        admissions = has_admissions_dataset.tables.admissions.copy()
        c_admittime = has_admissions_dataset.config.tables.admissions.admission_time_alias
        c_dischtime = has_admissions_dataset.config.tables.admissions.discharge_time_alias
        admittime = admissions.loc[sample_admission_id, c_admittime]
        dischtime = admissions.loc[sample_admission_id, c_dischtime]
        admissions.loc[sample_admission_id, c_admittime] = dischtime
        admissions.loc[sample_admission_id, c_dischtime] = admittime
        return eqx.tree_at(lambda x: x.tables.admissions, has_admissions_dataset, admissions)

    @pytest.fixture(scope='class')
    def filtered_dataset(self, dataset_with_negative_admission: Dataset):
        return FilterSubjectsNegativeAdmissionLengths.apply(dataset_with_negative_admission, DATASET_SCHEME_MANAGER,
                                                            Report())[0]

    def test_filter_subjects_negative_admission_length(self, dataset_with_negative_admission: Dataset,
                                                       filtered_dataset: Dataset):
        admissions0 = dataset_with_negative_admission.tables.admissions
        static0 = dataset_with_negative_admission.tables.static
        admissions1 = filtered_dataset.tables.admissions
        static1 = filtered_dataset.tables.static

        assert admissions0.shape[0] > admissions1.shape[0]
        assert static0.shape[0] == static1.shape[0] + 1
        assert admissions0.loc[admissions0.index[0], ALIAS['admission_time']] > admissions0.loc[
            admissions0.index[0], ALIAS['discharge_time']]
        assert any(admissions0[ALIAS['admission_time']] > admissions0[ALIAS['discharge_time']])
        assert all(admissions1[ALIAS['admission_time']] <= admissions1[ALIAS['discharge_time']])
        assert admissions0.index[0] not in admissions1.index


class TestOverlappingAdmissions:
    def _generate_admissions_from_pattern(self, pattern: list[str]) -> pd.DataFrame:
        if len(pattern) == 0:
            return pd.DataFrame(columns=[ALIAS['admission_time'], ALIAS['discharge_time']])
        random_monotonic_positive_integers = np.random.randint(1, 30, size=len(pattern)).cumsum()
        sequence_dates = list(
            map(lambda x: pd.Timestamp.today().normalize() + pd.Timedelta(days=x), random_monotonic_positive_integers))
        event_times = dict(zip(pattern, sequence_dates))
        admission_time = {k: v for k, v in event_times.items() if k.startswith('A')}
        discharge_time = {k.replace('D', 'A'): v for k, v in event_times.items() if k.startswith('D')}
        admissions = pd.DataFrame(index=list(admission_time.keys()))
        admissions[ALIAS['admission_time']] = admissions.index.map(admission_time)
        admissions[ALIAS['discharge_time']] = admissions.index.map(discharge_time)
        # shuffle the rows shuffled.
        return admissions.sample(frac=1)

    @pytest.fixture(params=[
        # Each line below has an ordered list of events labeled Ak and Dk, where Ak is the k-th admission event
        # and Dk is the discharge of k-th admission; and a dictionary of the
        # expected {super_interval: [merged_subintervals]}.
        # In the database, it is possible to have overlapping admissions, so we merge them.
        # A1 and D1 represent the admission and discharge time of a particular admission record in the database.
        # A2 and D2 mean the same, and that A2 happened after A1 (and not necessarily after D1).
        (['A1', 'A2', 'A3', 'D1', 'D3', 'D2'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'A3', 'D1', 'D2', 'D3'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'A3', 'D2', 'D1', 'D3'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'A3', 'D2', 'D3', 'D1'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'A3', 'D3', 'D1', 'D2'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'A3', 'D3', 'D2', 'D1'], {'A1': ['A2', 'A3']}),
        ##
        (['A1', 'A2', 'D1', 'A3', 'D3', 'D2'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'D1', 'A3', 'D2', 'D3'], {'A1': ['A2', 'A3']}),
        ##
        (['A1', 'A2', 'D1', 'D2', 'A3', 'D3'], {'A1': ['A2']}),
        (['A1', 'A2', 'D1', 'D2'], {'A1': ['A2']}),
        ##
        (['A1', 'A2', 'D2', 'A3', 'D1', 'D3'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'D2', 'A3', 'D3', 'D1'], {'A1': ['A2', 'A3']}),
        (['A1', 'A2', 'D2', 'D1', 'A3', 'D3'], {'A1': ['A2']}),
        ##
        (['A1', 'D1', 'A2', 'A3', 'D2', 'D3'], {'A2': ['A3']}),
        (['A1', 'D1', 'A2', 'A3', 'D3', 'D2'], {'A2': ['A3']}),
        (['A1', 'D1', 'A2', 'D2', 'A3', 'D3'], {}),
        ##
        (['A1', 'D1'], {}),
        ([], {}),
        (['A1', 'D1', 'A2', 'A3', 'D3', 'A4', 'D2', 'D4'], {'A2': ['A3', 'A4']}),
        (['A1', 'A2', 'D2', 'D1', 'A3', 'A4', 'D3', 'D4'], {'A1': ['A2'], 'A3': ['A4']}),
    ], scope='class')
    def admission_pattern_with_expected_out(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def admission_pattern(self, admission_pattern_with_expected_out):
        return admission_pattern_with_expected_out[0]

    @pytest.fixture(scope='class')
    def expected_out(self, admission_pattern_with_expected_out):
        return admission_pattern_with_expected_out[1]

    @pytest.fixture(scope='class')
    def admissions_table(self, admission_pattern) -> pd.DataFrame:
        return self._generate_admissions_from_pattern(admission_pattern)

    @pytest.fixture(scope='class')
    def superset_admissions_dictionary(self, admissions_table: pd.DataFrame) -> dict[str, list[str]]:
        sub2sup = ProcessOverlappingAdmissions._collect_overlaps(admissions_table,
                                                                 ALIAS['admission_time'],
                                                                 ALIAS['discharge_time'])
        sup2sub = defaultdict(list)
        for sub, sup in sub2sup.items():
            sup2sub[sup].append(sub)
        return sup2sub

    @pytest.fixture(scope='class')
    def large_admissions_dataset(self, has_admissions_dataset: Dataset):
        if len(has_admissions_dataset.tables.admissions) < 10:
            raise pytest.skip("Not enough admissions for the test in dataset.")
        return has_admissions_dataset

    def test_overlapping_cases(self, superset_admissions_dictionary, expected_out):
        assert superset_admissions_dictionary == expected_out

    @pytest.fixture(scope='class')
    def sample_admission_ids_map(self, large_admissions_dataset: Dataset):
        index = large_admissions_dataset.tables.admissions.index
        return {
            index[1]: index[0],
            index[2]: index[0],
            index[3]: index[0],
            index[5]: index[4],
            index[6]: index[4],
        }

    @pytest.fixture(scope='class')
    def merged_admissions_dataset(self, large_admissions_dataset: Dataset, sample_admission_ids_map: dict[str, str]):
        return ProcessOverlappingAdmissions._merge_overlapping_admissions(large_admissions_dataset,
                                                                          sample_admission_ids_map, Report())[0]

    def test_map_admission_ids(self, large_admissions_dataset: Dataset,
                               merged_admissions_dataset: Dataset,
                               sample_admission_ids_map: dict[str, str]):
        admissions0 = large_admissions_dataset.tables.admissions
        admissions1 = merged_admissions_dataset.tables.admissions

        assert len(admissions0) == len(admissions1) + len(sample_admission_ids_map)
        assert set(admissions1.index).issubset(set(admissions0.index))
        for table_name, table1 in merged_admissions_dataset.tables.tables_dict.items():
            table0 = getattr(large_admissions_dataset.tables, table_name)
            if ALIAS['admission_id'] in table1.columns:
                assert len(table1) == len(table0)
                assert set(table1[ALIAS['admission_id']]) - set(admissions1.index.values) == set()
                assert set(table0[ALIAS['admission_id']]) - set(admissions0.index.values) == set()
                assert set(table1[ALIAS['admission_id']]) - set(table0[ALIAS['admission_id']]) == set()

    @pytest.fixture(scope='class')
    def large_dataset_overlaps_dictionary(self, large_admissions_dataset: Dataset):
        admissions = large_admissions_dataset.tables.admissions

        sub2sup = {adm_id: super_adm_id for _, subject_adms in admissions.groupby(ALIAS['subject_id'])
                   for adm_id, super_adm_id in ProcessOverlappingAdmissions._collect_overlaps(subject_adms,
                                                                                              ALIAS['admission_time'],
                                                                                              ALIAS[
                                                                                                  'discharge_time']).items()}

        if len(sub2sup) == 0:
            raise pytest.skip("No overlapping admissions in dataset.")

        return sub2sup

    @pytest.fixture(scope='class')
    def merged_overlapping_admission_dataset(self, large_admissions_dataset: Dataset):
        large_admissions_dataset = eqx.tree_at(lambda x: x.config.overlapping_admissions, large_admissions_dataset,
                                               "merge")
        return ProcessOverlappingAdmissions.apply(large_admissions_dataset, DATASET_SCHEME_MANAGER, Report())[0]

    @pytest.fixture(scope='class')
    def removed_overlapping_admission_subjects_dataset(self, large_admissions_dataset: Dataset):
        large_admissions_dataset = eqx.tree_at(lambda x: x.config.overlapping_admissions, large_admissions_dataset,
                                               "remove")
        return ProcessOverlappingAdmissions.apply(large_admissions_dataset, DATASET_SCHEME_MANAGER, Report())[0]

    def test_process_overlapping_admissions(self, large_admissions_dataset: Dataset,
                                            large_dataset_overlaps_dictionary: dict[str, str],
                                            merged_overlapping_admission_dataset: Dataset,
                                            removed_overlapping_admission_subjects_dataset: Dataset):

        admissions0 = large_admissions_dataset.tables.admissions
        admissions_m = merged_overlapping_admission_dataset.tables.admissions
        admissions_r = removed_overlapping_admission_subjects_dataset.tables.admissions

        assert len(admissions0) == len(admissions_m) + len(large_dataset_overlaps_dictionary)
        assert len(admissions_m) > len(admissions_r)
        assert len(merged_overlapping_admission_dataset.tables.static) > len(
            removed_overlapping_admission_subjects_dataset.tables.static)

        for table_name, table0 in large_admissions_dataset.tables.tables_dict.items():
            if table_name in ('static', 'admissions'):
                continue
            table_m = getattr(merged_overlapping_admission_dataset.tables, table_name)
            table_r = getattr(removed_overlapping_admission_subjects_dataset.tables, table_name)
            assert len(table_m) == len(table0)
            assert set(table_m[ALIAS['admission_id']]) - set(admissions_m.index.values) == set()
            assert set(table0[ALIAS['admission_id']]) - set(admissions0.index.values) == set()
            assert set(table_m[ALIAS['admission_id']]) - set(table0[ALIAS['admission_id']]) == set()
            assert len(table_r) <= len(table0)
            assert set(table_r[ALIAS['admission_id']]).intersection(
                set(large_dataset_overlaps_dictionary.keys()) | set(
                    large_dataset_overlaps_dictionary.values())) == set()


def test_select_subjects_with_observation(indexed_dataset: Dataset):
    # assert False
    pass


class TestClampTimestamps:
    @pytest.fixture(scope='class')
    def shifted_timestamps_dataset(self, indexed_dataset: Dataset):
        if any(len(getattr(indexed_dataset.tables, k)) == 0 for k in indexed_dataset.config.tables.time_cols.keys()):
            raise pytest.skip("No temporal data in dataset.")

        admissions = indexed_dataset.tables.admissions

        c_admission_id = indexed_dataset.config.tables.admissions.admission_id_alias
        c_admittime = indexed_dataset.config.tables.admissions.admission_time_alias
        c_dischtime = indexed_dataset.config.tables.admissions.discharge_time_alias
        admission_id = admissions.index[0]
        admittime = admissions.loc[admission_id, c_admittime]
        dischtime = admissions.loc[admission_id, c_dischtime]
        if 'obs' in indexed_dataset.tables.tables_dict:
            assert indexed_dataset.config.tables.obs is not None
            c_time = indexed_dataset.config.tables.obs.time_alias

            obs = indexed_dataset.tables.obs.copy()
            admission_obs = obs[obs[c_admission_id] == admission_id]
            if len(admission_obs) > 0:
                obs.loc[admission_obs.index[0], c_time] = dischtime + pd.Timedelta(days=1)
            if len(admission_obs) > 1:
                obs.loc[admission_obs.index[1], c_time] = admittime + pd.Timedelta(days=-1)
            indexed_dataset = eqx.tree_at(lambda x: x.tables.obs, indexed_dataset, obs)
        for k in ('hosp_procedures', 'icu_procedures', 'icu_inputs'):
            if k in indexed_dataset.tables.tables_dict:
                c_starttime = getattr(indexed_dataset.config.tables, k).start_time_alias
                c_endtime = getattr(indexed_dataset.config.tables, k).end_time_alias
                procedures = getattr(indexed_dataset.tables, k).copy()

                admission_procedures = procedures[procedures[c_admission_id] == admission_id]
                if len(admission_procedures) > 0:
                    procedures.loc[admission_procedures.index[0], c_starttime] = admittime + pd.Timedelta(days=-1)
                    procedures.loc[admission_procedures.index[0], c_endtime] = dischtime + pd.Timedelta(days=1)

                if len(admission_procedures) > 1:
                    procedures.loc[admission_procedures.index[1], c_starttime] = dischtime + pd.Timedelta(days=1)
                    procedures.loc[admission_procedures.index[1], c_endtime] = dischtime + pd.Timedelta(days=2)

                if len(admission_procedures) > 2:
                    procedures.loc[admission_procedures.index[2], c_starttime] = admittime + pd.Timedelta(days=-2)
                    procedures.loc[admission_procedures.index[2], c_endtime] = admittime + pd.Timedelta(days=-1)

                indexed_dataset = eqx.tree_at(lambda x: getattr(x.tables, k), indexed_dataset, procedures)

        return indexed_dataset

    @pytest.fixture(scope='class')
    def fixed_dataset(self, shifted_timestamps_dataset: Dataset):
        return \
            FilterClampTimestampsToAdmissionInterval.apply(shifted_timestamps_dataset, DATASET_SCHEME_MANAGER,
                                                           Report())[0]

    def test_clamp_timestamps_to_admission_interval(self, shifted_timestamps_dataset: Dataset,
                                                    fixed_dataset: Dataset):
        admissions = shifted_timestamps_dataset.tables.admissions

        admission_id = admissions.index[0]
        admittime = admissions.loc[admission_id, ALIAS['admission_time']]
        dischtime = admissions.loc[admission_id, ALIAS['discharge_time']]

        if 'obs' in shifted_timestamps_dataset.tables.tables_dict:
            assert shifted_timestamps_dataset.config.tables.obs is not None

            c_time = shifted_timestamps_dataset.config.tables.obs.time_alias

            obs0 = shifted_timestamps_dataset.tables.obs
            obs1 = fixed_dataset.tables.obs
            assert obs0 is not None
            assert obs1 is not None

            admission_obs0 = obs0[obs0[ALIAS['admission_id']] == admission_id]
            admission_obs1 = obs1[obs1[ALIAS['admission_id']] == admission_id]
            if len(admission_obs0) > 0:
                assert len(admission_obs0) > len(admission_obs1)
                assert not admission_obs0[c_time].between(admittime, dischtime).all()
                assert admission_obs1[c_time].between(admittime, dischtime).all()

        for k in ('hosp_procedures', 'icu_procedures', 'icu_inputs'):
            if k in shifted_timestamps_dataset.tables.tables_dict:
                c_starttime = getattr(shifted_timestamps_dataset.config.tables, k).start_time_alias
                c_endtime = getattr(shifted_timestamps_dataset.config.tables, k).end_time_alias
                procedures0 = getattr(shifted_timestamps_dataset.tables, k)
                procedures1 = getattr(fixed_dataset.tables, k)

                admission_procedures0 = procedures0[procedures0[ALIAS['admission_id']] == admission_id]
                admission_procedures1 = procedures1[procedures1[ALIAS['admission_id']] == admission_id]
                if len(admission_procedures0) > 0:
                    assert len(admission_procedures0) >= len(admission_procedures1)
                    assert admission_procedures1[c_starttime].between(admittime, dischtime).all()
                    assert admission_procedures1[c_endtime].between(admittime, dischtime).all()

                if len(admission_procedures0) > 1:
                    assert len(admission_procedures0) > len(admission_procedures1)
