import random
import string
from collections import defaultdict

import ehrax as rx
import equinox as eqx
import numpy as np
import pandas as pd
import pytest
from ehrax.testing.common_setup import DATASET_SCHEME_MANAGER


@pytest.fixture(scope="module")
def indexed_dataset(large_dataset: rx.Dataset) -> rx.Dataset:
    return large_dataset._execute_pipeline([rx.SetIndex()], DATASET_SCHEME_MANAGER)


@pytest.fixture(scope="module")
def sample_subject_id(indexed_dataset: rx.Dataset) -> str:
    return random.choice(indexed_dataset.tables.static.index)


@pytest.fixture(scope="module")
def sample_admission_id(indexed_dataset: rx.Dataset) -> str:
    # Get an admission id that exists in all columns.
    candidates = set(indexed_dataset.tables.admissions.index)
    for _, table in indexed_dataset.tables.tables_dict.items():
        if str(rx.COLUMN.admission_id) in table.columns:
            candidates &= set(table[str(rx.COLUMN.admission_id)].values)
    return random.choice(list(candidates))


class TestDatasetTransformation:
    @pytest.fixture(scope="class")
    def removed_subject_admissions_dataset(self, indexed_dataset: rx.Dataset, sample_subject_id: str):
        admissions = indexed_dataset.tables.admissions
        admissions = admissions[admissions[str(rx.COLUMN.subject_id)] != sample_subject_id]
        return eqx.tree_at(lambda x: x.tables.admissions, indexed_dataset, admissions)

    @pytest.fixture(scope="class")
    def removed_no_admission_subjects_RESULTS(self, removed_subject_admissions_dataset: rx.Dataset):
        filtered_dataset, report = rx.DatasetTransformation.filter_no_admission_subjects(
            removed_subject_admissions_dataset, rx.Report()
        )
        return filtered_dataset, report

    @pytest.fixture(scope="class")
    def removed_no_admission_subjects(self, removed_no_admission_subjects_RESULTS) -> rx.Dataset:
        return removed_no_admission_subjects_RESULTS[0]

    @pytest.fixture(scope="class")
    def removed_no_admission_subjects_REPORT(self, removed_no_admission_subjects_RESULTS) -> rx.Report:
        return removed_no_admission_subjects_RESULTS[1]

    @pytest.fixture(scope="class")
    def removed_subject_dataset_unsync(self, indexed_dataset: rx.Dataset, sample_subject_id: str):
        static = indexed_dataset.tables.static
        static = static.drop(index=sample_subject_id)
        return eqx.tree_at(lambda x: x.tables.static, indexed_dataset, static)

    @pytest.fixture(scope="class")
    def removed_admission_dataset_unsync(self, indexed_dataset: rx.Dataset, sample_admission_id: str):
        admissions = indexed_dataset.tables.admissions
        admissions = admissions.drop(index=sample_admission_id)
        return eqx.tree_at(lambda x: x.tables.admissions, indexed_dataset, admissions)

    @pytest.fixture(scope="class")
    def removed_subject_dataset_sync_RESULTS(self, removed_subject_dataset_unsync):
        return rx.DatasetTransformation.synchronize_subjects(removed_subject_dataset_unsync, rx.Report())

    @pytest.fixture(scope="class")
    def removed_subject_dataset_sync(self, removed_subject_dataset_sync_RESULTS) -> rx.Dataset:
        return removed_subject_dataset_sync_RESULTS[0]

    @pytest.fixture(scope="class")
    def removed_subject_dataset_sync_REPORT(self, removed_subject_dataset_sync_RESULTS) -> rx.Report:
        return removed_subject_dataset_sync_RESULTS[1]

    @pytest.fixture(scope="class")
    def removed_admission_dataset_sync_RESUTLS(self, removed_admission_dataset_unsync):
        return rx.DatasetTransformation.synchronize_index(
            removed_admission_dataset_unsync, "admissions", str(rx.COLUMN.admission_id), rx.Report()
        )

    @pytest.fixture(scope="class")
    def removed_admission_dataset_sync(self, removed_admission_dataset_sync_RESUTLS) -> rx.Dataset:
        return removed_admission_dataset_sync_RESUTLS[0]

    @pytest.fixture(scope="class")
    def removed_admission_dataset_sync_REPORT(self, removed_admission_dataset_sync_RESUTLS) -> rx.Report:
        return removed_admission_dataset_sync_RESUTLS[1]

    def test_filter_no_admissions_subjects(self, removed_no_admission_subjects: rx.Dataset, sample_subject_id: str):
        assert sample_subject_id not in removed_no_admission_subjects.tables.static

    def test_synchronize_index_subjects(
        self,
        indexed_dataset: rx.Dataset,
        removed_subject_dataset_unsync: rx.Dataset,
        removed_subject_dataset_sync: rx.Dataset,
        sample_subject_id: str,
    ):
        assert sample_subject_id in indexed_dataset.tables.admissions[str(rx.COLUMN.subject_id)].values
        assert sample_subject_id in removed_subject_dataset_unsync.tables.admissions[str(rx.COLUMN.subject_id)].values
        assert sample_subject_id not in removed_subject_dataset_sync.tables.admissions[str(rx.COLUMN.subject_id)].values
        assert set(removed_subject_dataset_sync.tables.static.index) == set(
            removed_subject_dataset_sync.tables.admissions[str(rx.COLUMN.subject_id)]
        )

    def test_synchronize_index_admissions(
        self,
        removed_admission_dataset_unsync: rx.Dataset,
        removed_admission_dataset_sync: rx.Dataset,
        sample_admission_id: str,
    ):
        for table_name, table in removed_admission_dataset_unsync.tables.tables_dict.items():
            if str(rx.COLUMN.admission_id) in table.columns:
                assert sample_admission_id in table[str(rx.COLUMN.admission_id)].values
                synced_table = getattr(removed_admission_dataset_sync.tables, table_name)
                assert sample_admission_id not in synced_table[str(rx.COLUMN.admission_id)].values
                assert set(synced_table[str(rx.COLUMN.admission_id)]).issubset(
                    set(removed_admission_dataset_sync.tables.admissions.index)
                )

    def test_generated_report1(self, removed_no_admission_subjects_REPORT: rx.Report):
        assert isinstance(removed_no_admission_subjects_REPORT, rx.Report)
        assert len(removed_no_admission_subjects_REPORT) > 0
        assert isinstance(removed_no_admission_subjects_REPORT[0], rx.ReportAttributes)

    def test_serializable_report1(self, removed_no_admission_subjects_REPORT: rx.Report):
        assert all(isinstance(v.as_dict(), dict) for v in removed_no_admission_subjects_REPORT)
        assert all([rx.AbstractConfig.from_dict(v.to_dict()).equals(v) for v in removed_no_admission_subjects_REPORT])


class TestCastTimestamps:
    @pytest.fixture(scope="class")
    def str_timestamps_dataset(self, indexed_dataset: rx.Dataset):
        for table_name, time_cols in indexed_dataset.config.columns.time_cols.items():
            table = indexed_dataset.tables.tables_dict[table_name].copy()
            for col in time_cols:
                table[col] = table[col].astype(str)
            indexed_dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), indexed_dataset, table)
        return indexed_dataset

    @pytest.fixture(scope="class")
    def casted_timestamps_dataset(self, str_timestamps_dataset: rx.Dataset):
        return rx.CastTimestamps.apply(str_timestamps_dataset, DATASET_SCHEME_MANAGER, rx.Report())[0]

    def test_cast_timestamps(self, str_timestamps_dataset: rx.Dataset, casted_timestamps_dataset: rx.Dataset):
        for table_name, time_cols in str_timestamps_dataset.config.columns.time_cols.items():
            table1 = str_timestamps_dataset.tables.tables_dict[table_name]
            table2 = casted_timestamps_dataset.tables.tables_dict[table_name]
            for col in time_cols:
                assert table1[col].dtype == np.dtype("O")
                assert table2[col].dtype == np.dtype("datetime64[ns]")


class TestFilterUnsupportedCodes:
    @pytest.fixture(scope="class")
    def dataset_with_unsupported_codes(self, indexed_dataset: rx.Dataset) -> tuple[rx.Dataset, dict[str, set[str]]]:
        unsupported_codes = {}
        for table_name, code_col in indexed_dataset.config.columns.code_column.items():
            table = indexed_dataset.tables.tables_dict[table_name]
            unsupported_code = f"UNSUPPORTED_CODE_{''.join(random.choices(string.ascii_uppercase, k=5))}"
            table.loc[table.index[0], code_col] = unsupported_code
            unsupported_codes[table_name] = unsupported_code
            indexed_dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), indexed_dataset, table)
        return indexed_dataset, unsupported_codes

    @pytest.fixture(scope="class")
    def filtered_dataset(self, dataset_with_unsupported_codes: tuple[rx.Dataset, dict[str, set[str]]]) -> rx.Dataset:
        dataset, _ = dataset_with_unsupported_codes
        return rx.FilterUnsupportedCodes.apply(dataset, DATASET_SCHEME_MANAGER, rx.Report())[0]

    def test_filter_unsupported_codes(
        self, dataset_with_unsupported_codes: tuple[rx.Dataset, dict[str, set[str]]], filtered_dataset: rx.Dataset
    ):
        unfiltered_dataset, unsupported_codes = dataset_with_unsupported_codes
        for table_name, code_col in filtered_dataset.config.columns.code_column.items():
            (code_col,) = code_col
            assert unsupported_codes[table_name] in getattr(unfiltered_dataset.tables, table_name)[code_col].values
            assert unsupported_codes[table_name] not in getattr(filtered_dataset.tables, table_name)[code_col].values


class TestSetRelativeTimes:
    @pytest.fixture(scope="class")
    def relative_times_dataset(self, indexed_dataset: rx.Dataset):
        return rx.SetAdmissionRelativeTimes.apply(indexed_dataset, DATASET_SCHEME_MANAGER, rx.Report())[0]

    @pytest.fixture(scope="class")
    def admission_los_table(self, indexed_dataset: rx.Dataset):
        admissions = indexed_dataset.tables.admissions.copy()
        admissions["los_hours"] = (
            admissions[str(rx.COLUMN.end_time)] - admissions[str(rx.COLUMN.start_time)]
        ).dt.total_seconds() / (60 * 60)
        return admissions[["los_hours"]]

    def test_set_relative_times(
        self, indexed_dataset: rx.Dataset, relative_times_dataset: rx.Dataset, admission_los_table: pd.DataFrame
    ):
        for table_name, time_cols in indexed_dataset.config.columns.time_cols.items():
            if table_name in ("admissions", "static"):
                continue
            table = getattr(relative_times_dataset.tables, table_name)
            table = table.merge(admission_los_table, left_on=str(rx.COLUMN.admission_id), right_index=True)
            for col in time_cols:
                assert table[col].dtype == float
                assert table[col].min() >= 0
                assert all(table[col] <= table["los_hours"])


class TestFilterSubjectsWithNegativeAdmissionInterval:
    @pytest.fixture(scope="class")
    def dataset(self, indexed_dataset: rx.Dataset) -> rx.Dataset:
        return rx.FilterSubjectsNegativeAdmissionLengths.apply(indexed_dataset, DATASET_SCHEME_MANAGER, rx.Report())[0]

    @pytest.fixture(scope="class")
    def dataset_inverted_admission(self, dataset: rx.Dataset, sample_admission_id: str) -> rx.Dataset:
        admissions = dataset.tables.admissions.copy()
        c_admittime = dataset.config.columns.admissions.start_time
        c_dischtime = dataset.config.columns.admissions.end_time
        admittime = admissions.loc[sample_admission_id, c_admittime]
        dischtime = admissions.loc[sample_admission_id, c_dischtime]
        admissions.loc[sample_admission_id, c_admittime] = dischtime
        admissions.loc[sample_admission_id, c_dischtime] = admittime
        return eqx.tree_at(lambda x: x.tables.admissions, dataset, admissions)

    @pytest.fixture(scope="class")
    def filtered_dataset(self, dataset_inverted_admission: rx.Dataset):
        return rx.FilterSubjectsNegativeAdmissionLengths.apply(
            dataset_inverted_admission, DATASET_SCHEME_MANAGER, rx.Report()
        )[0]

    def test_filter_subjects_negative_admission_length(
        self, dataset_inverted_admission: rx.Dataset, filtered_dataset: rx.Dataset, sample_admission_id: str
    ):
        admissions0 = dataset_inverted_admission.tables.admissions
        static0 = dataset_inverted_admission.tables.static
        admissions1 = filtered_dataset.tables.admissions
        static1 = filtered_dataset.tables.static

        assert admissions0.shape[0] > admissions1.shape[0]
        assert static0.shape[0] == static1.shape[0] + 1
        assert (
            admissions0.loc[sample_admission_id, str(rx.COLUMN.start_time)]
            > admissions0.loc[sample_admission_id, str(rx.COLUMN.end_time)]
        )
        assert any(admissions0[str(rx.COLUMN.start_time)] > admissions0[str(rx.COLUMN.end_time)])
        assert all(admissions1[str(rx.COLUMN.start_time)] <= admissions1[str(rx.COLUMN.end_time)])
        assert sample_admission_id in admissions0.index
        assert sample_admission_id not in admissions1.index


class TestOverlappingAdmissions:
    def _generate_admissions_from_pattern(self, pattern: list[str]) -> pd.DataFrame:
        if len(pattern) == 0:
            return pd.DataFrame(columns=[str(rx.COLUMN.start_time), str(rx.COLUMN.end_time)])
        random_monotonic_positive_integers = np.random.randint(1, 30, size=len(pattern)).cumsum()
        sequence_dates = list(
            map(lambda x: pd.Timestamp.today().normalize() + pd.Timedelta(days=x), random_monotonic_positive_integers)
        )
        event_times = dict(zip(pattern, sequence_dates))
        admission_time = {k: v for k, v in event_times.items() if k.startswith("A")}
        discharge_time = {k.replace("D", "A"): v for k, v in event_times.items() if k.startswith("D")}
        admissions = pd.DataFrame(index=list(admission_time.keys()))
        admissions[str(rx.COLUMN.start_time)] = admissions.index.map(admission_time)
        admissions[str(rx.COLUMN.end_time)] = admissions.index.map(discharge_time)
        # shuffle the rows shuffled.
        return admissions.sample(frac=1)

    @pytest.fixture(
        params=[
            # Each line below has an ordered list of events labeled Ak and Dk, where Ak is the k-th admission event
            # and Dk is the discharge of k-th admission; and a dictionary of the
            # expected {super_interval: [merged_subintervals]}.
            # In the database, it is possible to have overlapping admissions, so we merge them.
            # A1 and D1 represent the admission and discharge time of a particular admission record in the database.
            # A2 and D2 mean the same, and that A2 happened after A1 (and not necessarily after D1).
            (["A1", "A2", "A3", "D1", "D3", "D2"], {"A1": ["A2", "A3"]}),
            (["A1", "A2", "A3", "D1", "D2", "D3"], {"A1": ["A2", "A3"]}),
            (["A1", "A2", "A3", "D2", "D1", "D3"], {"A1": ["A2", "A3"]}),
            (["A1", "A2", "A3", "D2", "D3", "D1"], {"A1": ["A2", "A3"]}),
            (["A1", "A2", "A3", "D3", "D1", "D2"], {"A1": ["A2", "A3"]}),
            (["A1", "A2", "A3", "D3", "D2", "D1"], {"A1": ["A2", "A3"]}),
            ##
            (["A1", "A2", "D1", "A3", "D3", "D2"], {"A1": ["A2", "A3"]}),
            (["A1", "A2", "D1", "A3", "D2", "D3"], {"A1": ["A2", "A3"]}),
            ##
            (["A1", "A2", "D1", "D2", "A3", "D3"], {"A1": ["A2"]}),
            (["A1", "A2", "D1", "D2"], {"A1": ["A2"]}),
            ##
            (["A1", "A2", "D2", "A3", "D1", "D3"], {"A1": ["A2", "A3"]}),
            (["A1", "A2", "D2", "A3", "D3", "D1"], {"A1": ["A2", "A3"]}),
            (["A1", "A2", "D2", "D1", "A3", "D3"], {"A1": ["A2"]}),
            ##
            (["A1", "D1", "A2", "A3", "D2", "D3"], {"A2": ["A3"]}),
            (["A1", "D1", "A2", "A3", "D3", "D2"], {"A2": ["A3"]}),
            (["A1", "D1", "A2", "D2", "A3", "D3"], {}),
            ##
            (["A1", "D1"], {}),
            ([], {}),
            (["A1", "D1", "A2", "A3", "D3", "A4", "D2", "D4"], {"A2": ["A3", "A4"]}),
            (["A1", "A2", "D2", "D1", "A3", "A4", "D3", "D4"], {"A1": ["A2"], "A3": ["A4"]}),
        ],
        scope="class",
    )
    def admission_pattern_with_expected_out(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def admission_pattern(self, admission_pattern_with_expected_out):
        return admission_pattern_with_expected_out[0]

    @pytest.fixture(scope="class")
    def expected_out(self, admission_pattern_with_expected_out):
        return admission_pattern_with_expected_out[1]

    @pytest.fixture(scope="class")
    def admissions_table(self, admission_pattern) -> pd.DataFrame:
        return self._generate_admissions_from_pattern(admission_pattern)

    @pytest.fixture(scope="class")
    def superset_admissions_dictionary(self, admissions_table: pd.DataFrame) -> dict[str, list[str]]:
        sub2sup = rx.MergeOverlappingAdmissions._collect_overlaps(admissions_table)
        sup2sub = defaultdict(list)
        for sub, sup in sub2sup.items():
            sup2sub[sup].append(sub)
        return sup2sub

    def test_overlapping_cases(self, superset_admissions_dictionary, expected_out):
        assert superset_admissions_dictionary == expected_out

    @pytest.fixture(scope="class")
    def sample_admission_ids_map(self, indexed_dataset: rx.Dataset):
        index = indexed_dataset.tables.admissions.index
        return {
            index[1]: index[0],
            index[2]: index[0],
            index[3]: index[0],
            index[5]: index[4],
            index[6]: index[4],
        }

    @pytest.fixture(scope="class")
    def merged_admissions_dataset(self, indexed_dataset: rx.Dataset, sample_admission_ids_map: dict[str, str]):
        return rx.MergeOverlappingAdmissions._admissions_map_admission_ids(
            indexed_dataset, sample_admission_ids_map, rx.Report()
        )[0]

    @pytest.fixture(scope="class")
    def mapped_tables_dataset(self, indexed_dataset: rx.Dataset, sample_admission_ids_map: dict[str, str]):
        return rx.MergeOverlappingAdmissions._tables_map_admission_ids(
            indexed_dataset, sample_admission_ids_map, rx.Report()
        )[0]

    def test_merge_admission_ids(
        self,
        indexed_dataset: rx.Dataset,
        merged_admissions_dataset: rx.Dataset,
        sample_admission_ids_map: dict[str, str],
    ):
        admissions0 = indexed_dataset.tables.admissions
        admissions1 = merged_admissions_dataset.tables.admissions

        assert len(admissions0) == len(admissions1) + len(sample_admission_ids_map)
        assert set(admissions1.index).issubset(set(admissions0.index))

    def test_map_admission_ids(
        self, indexed_dataset: rx.Dataset, mapped_tables_dataset: rx.Dataset, sample_admission_ids_map: dict[str, str]
    ):
        admissions0 = indexed_dataset.tables.admissions
        admissions1 = mapped_tables_dataset.tables.admissions
        for table_name, table1 in mapped_tables_dataset.tables.tables_dict.items():
            table0 = getattr(indexed_dataset.tables, table_name)
            if str(rx.COLUMN.admission_id) in table1.columns:
                assert len(table1) == len(table0)
                assert set(table1[str(rx.COLUMN.admission_id)]) - set(admissions1.index.values) == set()
                assert set(table0[str(rx.COLUMN.admission_id)]) - set(admissions0.index.values) == set()
                assert set(table1[str(rx.COLUMN.admission_id)]) - set(table0[str(rx.COLUMN.admission_id)]) == set()

    @pytest.fixture(scope="class")
    def large_dataset_overlaps_dictionary(self, indexed_dataset: rx.Dataset):
        admissions = indexed_dataset.tables.admissions

        sub2sup = {
            adm_id: super_adm_id
            for _, subject_adms in admissions.groupby(str(rx.COLUMN.subject_id))
            for adm_id, super_adm_id in rx.MergeOverlappingAdmissions._collect_overlaps(subject_adms).items()
        }

        if len(sub2sup) == 0:
            assert 0, "No overlapping admissions in rx.Dataset."

        return sub2sup

    @pytest.fixture(scope="class")
    def merged_overlapping_admission_dataset(self, indexed_dataset: rx.Dataset):
        return rx.MergeOverlappingAdmissions.apply(indexed_dataset, DATASET_SCHEME_MANAGER, rx.Report())[0]

    @pytest.fixture(scope="class")
    def removed_overlapping_admission_subjects_dataset(self, indexed_dataset: rx.Dataset):
        return rx.RemoveSubjectsWithOverlappingAdmissions.apply(indexed_dataset, DATASET_SCHEME_MANAGER, rx.Report())[0]

    def test_process_overlapping_admissions(
        self,
        indexed_dataset: rx.Dataset,
        large_dataset_overlaps_dictionary: dict[str, str],
        merged_overlapping_admission_dataset: rx.Dataset,
        removed_overlapping_admission_subjects_dataset: rx.Dataset,
    ):
        admissions0 = indexed_dataset.tables.admissions
        admissions_m = merged_overlapping_admission_dataset.tables.admissions
        admissions_r = removed_overlapping_admission_subjects_dataset.tables.admissions

        assert len(admissions0) == len(admissions_m) + len(large_dataset_overlaps_dictionary)
        assert len(admissions_m) > len(admissions_r)
        assert len(merged_overlapping_admission_dataset.tables.static) > len(
            removed_overlapping_admission_subjects_dataset.tables.static
        )

        for table_name, table0 in indexed_dataset.tables.tables_dict.items():
            if table_name in ("static", "admissions"):
                continue
            table_m = getattr(merged_overlapping_admission_dataset.tables, table_name)
            table_r = getattr(removed_overlapping_admission_subjects_dataset.tables, table_name)
            assert len(table_m) == len(table0)
            assert set(table_m[str(rx.COLUMN.admission_id)]) - set(admissions_m.index.values) == set()
            assert set(table0[str(rx.COLUMN.admission_id)]) - set(admissions0.index.values) == set()
            assert set(table_m[str(rx.COLUMN.admission_id)]) - set(table0[str(rx.COLUMN.admission_id)]) == set()
            assert len(table_r) <= len(table0)
            assert (
                set(table_r[str(rx.COLUMN.admission_id)]).intersection(
                    set(large_dataset_overlaps_dictionary.keys()) | set(large_dataset_overlaps_dictionary.values())
                )
                == set()
            )


def test_select_subjects_with_observation(indexed_dataset: rx.Dataset):
    # assert False
    pass


class TestClampTimestamps:
    @pytest.fixture(scope="class")
    def shifted_timestamps_dataset(self, indexed_dataset: rx.Dataset):
        if any(len(getattr(indexed_dataset.tables, k)) == 0 for k in indexed_dataset.config.columns.time_cols.keys()):
            raise pytest.skip("No temporal data in rx.Dataset.")

        admissions = indexed_dataset.tables.admissions

        c_admission_id = indexed_dataset.config.columns.admissions.admission_id
        c_admittime = indexed_dataset.config.columns.admissions.start_time
        c_dischtime = indexed_dataset.config.columns.admissions.end_time
        admission_id = admissions.index[0]
        admittime = admissions.loc[admission_id, c_admittime]
        dischtime = admissions.loc[admission_id, c_dischtime]
        if "obs" in indexed_dataset.tables.tables_dict:
            assert indexed_dataset.config.columns.obs is not None
            c_time = indexed_dataset.config.columns.obs.time

            obs = indexed_dataset.tables.obs.copy()
            admission_obs = obs[obs[c_admission_id] == admission_id]
            if len(admission_obs) > 0:
                obs.loc[admission_obs.index[0], c_time] = dischtime + pd.Timedelta(days=1)
            if len(admission_obs) > 1:
                obs.loc[admission_obs.index[1], c_time] = admittime + pd.Timedelta(days=-1)
            indexed_dataset = eqx.tree_at(lambda x: x.tables.obs, indexed_dataset, obs)
        for k in ("hosp_procedures", "icu_procedures", "icu_inputs"):
            if k in indexed_dataset.tables.tables_dict:
                c_starttime = getattr(indexed_dataset.config.columns, k).start_time
                c_endtime = getattr(indexed_dataset.config.columns, k).end_time
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

    @pytest.fixture(scope="class")
    def fixed_dataset(self, shifted_timestamps_dataset: rx.Dataset):
        return rx.FilterClampTimestampsToAdmissionInterval.apply(
            shifted_timestamps_dataset, DATASET_SCHEME_MANAGER, rx.Report()
        )[0]

    def test_clamp_timestamps_to_admission_interval(
        self, shifted_timestamps_dataset: rx.Dataset, fixed_dataset: rx.Dataset
    ):
        admissions = shifted_timestamps_dataset.tables.admissions

        admission_id = admissions.index[0]
        admittime = admissions.loc[admission_id, str(rx.COLUMN.start_time)]
        dischtime = admissions.loc[admission_id, str(rx.COLUMN.end_time)]

        if "obs" in shifted_timestamps_dataset.tables.tables_dict:
            assert shifted_timestamps_dataset.config.columns.obs is not None

            c_time = shifted_timestamps_dataset.config.columns.obs.time

            obs0 = shifted_timestamps_dataset.tables.obs
            obs1 = fixed_dataset.tables.obs
            assert obs0 is not None
            assert obs1 is not None

            admission_obs0 = obs0[obs0[str(rx.COLUMN.admission_id)] == admission_id]
            admission_obs1 = obs1[obs1[str(rx.COLUMN.admission_id)] == admission_id]
            if len(admission_obs0) > 0:
                assert len(admission_obs0) > len(admission_obs1)
                assert not admission_obs0[c_time].between(admittime, dischtime).all()
                assert admission_obs1[c_time].between(admittime, dischtime).all()

        for k in ("hosp_procedures", "icu_procedures", "icu_inputs"):
            if k in shifted_timestamps_dataset.tables.tables_dict:
                c_starttime = getattr(shifted_timestamps_dataset.config.columns, k).start_time
                c_endtime = getattr(shifted_timestamps_dataset.config.columns, k).end_time
                procedures0 = getattr(shifted_timestamps_dataset.tables, k)
                procedures1 = getattr(fixed_dataset.tables, k)

                admission_procedures0 = procedures0[procedures0[str(rx.COLUMN.admission_id)] == admission_id]
                admission_procedures1 = procedures1[procedures1[str(rx.COLUMN.admission_id)] == admission_id]
                if len(admission_procedures0) > 0:
                    assert len(admission_procedures0) >= len(admission_procedures1)
                    assert admission_procedures1[c_starttime].between(admittime, dischtime).all()
                    assert admission_procedures1[c_endtime].between(admittime, dischtime).all()

                if len(admission_procedures0) > 1:
                    assert len(admission_procedures0) > len(admission_procedures1)


class TestFilterAdmissionsWithNoDiagnoses:
    ADMISSION_ID = list(map(str, range(10)))
    ADMISSION_ID_WITH_NO_DIAGNOSIS = ["2", "5"]

    @pytest.fixture(
        scope="class", params=[(ADMISSION_ID, ADMISSION_ID_WITH_NO_DIAGNOSIS), (ADMISSION_ID, []), ([], [])]
    )
    def admission_id_admission_id_with_no_diagnosis(self, request) -> tuple[list[str], list[str]]:
        return request.param

    @pytest.fixture(scope="class")
    def admission_id(self, admission_id_admission_id_with_no_diagnosis: tuple[list[str], list[str]]) -> list[str]:
        return admission_id_admission_id_with_no_diagnosis[0]

    @pytest.fixture(scope="class")
    def admission_id_with_no_diagnosis(
        self, admission_id_admission_id_with_no_diagnosis: tuple[list[str], list[str]]
    ) -> list[str]:
        return admission_id_admission_id_with_no_diagnosis[1]

    @pytest.fixture(scope="class")
    def static(self) -> pd.DataFrame:
        return pd.DataFrame({rx.COLUMN.subject_id: list(map(str, range(100, 110)))}).set_index(rx.COLUMN.subject_id)

    @pytest.fixture(scope="class")
    def admissions(self, static: pd.DataFrame, admission_id: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                rx.COLUMN.subject_id: random.choices(static.index, k=len(admission_id)),
                rx.COLUMN.admission_id: admission_id,
            }
        ).set_index(rx.COLUMN.admission_id)

    @pytest.fixture(scope="class")
    def dx_discharge(
        self, admissions: pd.DataFrame, admission_id: list[str], admission_id_with_no_diagnosis: list[str]
    ) -> pd.DataFrame:
        return pd.DataFrame({rx.COLUMN.admission_id: list(set(admission_id) - set(admission_id_with_no_diagnosis))})

    @pytest.fixture(scope="class")
    def dataset(self, static: pd.DataFrame, dx_discharge: pd.DataFrame, admissions: pd.DataFrame) -> rx.Dataset:
        return rx.Dataset(
            tables=rx.DatasetTables(static=static, admissions=admissions, dx_discharge=dx_discharge),
            config=rx.DatasetConfig(scheme=rx.DatasetSchemeConfig()),
        )

    @pytest.fixture(scope="class")
    def filtered_dataset(self, dataset: rx.Dataset) -> rx.Dataset:
        updated_dataset, _ = rx.FilterAdmissionsWithNoDiagnoses.apply(dataset, None, rx.Report())
        return updated_dataset

    def test_transformation(
        self,
        dataset: rx.Dataset,
        filtered_dataset: rx.Dataset,
        admission_id: list[str],
        admission_id_with_no_diagnosis: list[str],
    ):
        assert set(admission_id).issubset(dataset.tables.admissions.index)
        assert set(filtered_dataset.tables.admissions.index) == set(admission_id) - set(admission_id_with_no_diagnosis)


class TestFilterSubjectsWithSingleOrNoAdmission:
    SUBJECT_ID = list(map(str, range(10)))
    SUBJECT_ID_WITH_NO_ADMISSIONS = ["4", "5"]

    @pytest.fixture(scope="class", params=[(SUBJECT_ID, SUBJECT_ID_WITH_NO_ADMISSIONS), (SUBJECT_ID, []), ([], [])])
    def subject_id_subject_id_with_1_admissions(self, request) -> tuple[list[str], list[str]]:
        return request.param

    @pytest.fixture(scope="class")
    def subject_id(self, subject_id_subject_id_with_1_admissions: tuple[list[str], list[str]]) -> list[str]:
        return subject_id_subject_id_with_1_admissions[0]

    @pytest.fixture(scope="class")
    def subject_id_with_1_admission(
        self, subject_id_subject_id_with_1_admissions: tuple[list[str], list[str]]
    ) -> list[str]:
        return subject_id_subject_id_with_1_admissions[1]

    @pytest.fixture(scope="class")
    def static(self, subject_id: list[str]) -> pd.DataFrame:
        return pd.DataFrame({rx.COLUMN.subject_id: subject_id}).set_index(rx.COLUMN.subject_id)

    @pytest.fixture(scope="class")
    def admissions(
        self, static: pd.DataFrame, subject_id: list[str], subject_id_with_1_admission: list[str]
    ) -> pd.DataFrame:
        admission_id_1 = list(map(str, range(len(subject_id))))
        df_1 = pd.DataFrame({rx.COLUMN.subject_id: subject_id, rx.COLUMN.admission_id: admission_id_1})
        n1 = len(subject_id)
        subject_w_2_admissions = set(subject_id) - set(subject_id_with_1_admission)
        admission_id_2 = list(map(str, range(n1, n1 + len(subject_w_2_admissions))))
        df_2 = pd.DataFrame(
            {rx.COLUMN.subject_id: list(subject_w_2_admissions), rx.COLUMN.admission_id: admission_id_2}
        )

        return pd.concat([df_1, df_2], ignore_index=True).set_index(rx.COLUMN.admission_id)

    @pytest.fixture(scope="class")
    def dataset(self, static: pd.DataFrame, admissions: pd.DataFrame) -> rx.Dataset:
        return rx.Dataset(
            tables=rx.DatasetTables(static=static, admissions=admissions),
            config=rx.DatasetConfig(scheme=rx.DatasetSchemeConfig()),
        )

    @pytest.fixture(scope="class")
    def filtered_dataset(self, dataset: rx.Dataset) -> rx.Dataset:
        updated_dataset, _ = rx.FilterSubjectsWithSingleOrNoAdmission.apply(dataset, None, rx.Report())
        return updated_dataset

    def test_transformation(
        self,
        dataset: rx.Dataset,
        filtered_dataset: rx.Dataset,
        subject_id: list[str],
        subject_id_with_1_admission: list[str],
    ):
        assert set(subject_id).issubset(dataset.tables.static.index)
        assert set(filtered_dataset.tables.static.index) == set(subject_id) - set(subject_id_with_1_admission)


class TestFilterSubjectsWithLongAdmission:
    SUBJECT_ID = list(map(str, range(10)))
    SUBJECT_ID_WITH_LONG_ADMISSIONS = ["4", "5"]
    MAX_LOS_DAYS = 10.0

    @pytest.fixture(scope="class", params=[(SUBJECT_ID, SUBJECT_ID_WITH_LONG_ADMISSIONS), (SUBJECT_ID, []), ([], [])])
    def subject_id_subject_id_with_long_admissions(self, request) -> tuple[list[str], list[str]]:
        return request.param

    @pytest.fixture(scope="class")
    def subject_id(self, subject_id_subject_id_with_long_admissions: tuple[list[str], list[str]]) -> list[str]:
        return subject_id_subject_id_with_long_admissions[0]

    @pytest.fixture(scope="class")
    def subject_id_with_long_admission(
        self, subject_id_subject_id_with_long_admissions: tuple[list[str], list[str]]
    ) -> list[str]:
        return subject_id_subject_id_with_long_admissions[1]

    @pytest.fixture(scope="class")
    def static(self, subject_id: list[str]) -> pd.DataFrame:
        return pd.DataFrame({rx.COLUMN.subject_id: subject_id}).set_index(rx.COLUMN.subject_id)

    @pytest.fixture(scope="class")
    def admissions(
        self, static: pd.DataFrame, subject_id: list[str], subject_id_with_long_admission: list[str]
    ) -> pd.DataFrame:
        admission_id_1 = list(map(str, range(len(subject_id))))
        df_1 = pd.DataFrame(
            {
                rx.COLUMN.subject_id: subject_id,
                rx.COLUMN.admission_id: admission_id_1,
                rx.COLUMN.start_time: pd.Timestamp.now(),
                rx.COLUMN.end_time: pd.Timestamp.now()
                + pd.Timedelta(days=random.uniform(0.0, self.MAX_LOS_DAYS * 0.99)),
            }
        )
        n1 = len(subject_id)
        admission_id_2 = list(map(str, range(n1, n1 + len(subject_id_with_long_admission))))
        df_2 = pd.DataFrame(
            {
                rx.COLUMN.subject_id: list(subject_id_with_long_admission),
                rx.COLUMN.admission_id: admission_id_2,
                rx.COLUMN.start_time: pd.Timestamp.now(),
                rx.COLUMN.end_time: pd.Timestamp.now()
                + pd.Timedelta(days=random.uniform(self.MAX_LOS_DAYS * 1.01, self.MAX_LOS_DAYS * 2.0)),
            }
        )
        return pd.concat([df_1, df_2], ignore_index=True).set_index(rx.COLUMN.admission_id)

    @pytest.fixture(scope="class")
    def dataset_no_config(self, static: pd.DataFrame, admissions: pd.DataFrame) -> rx.Dataset:
        return rx.Dataset(
            tables=rx.DatasetTables(static=static, admissions=admissions),
            config=rx.DatasetConfig(scheme=rx.DatasetSchemeConfig()),
        )

    @pytest.fixture(scope="class")
    def dataset(self, dataset_no_config: rx.Dataset) -> rx.Dataset:
        return eqx.tree_at(
            lambda x: x.config.select_subjects_with_short_admissions,
            dataset_no_config,
            self.MAX_LOS_DAYS,
            is_leaf=lambda x: x is None,
        )

    @pytest.fixture(scope="class")
    def filtered_dataset_no_config(self, dataset_no_config: rx.Dataset) -> rx.Dataset:
        updated, _ = rx.FilterSubjectsWithLongAdmission.apply(dataset_no_config, None, rx.Report())
        return updated

    @pytest.fixture(scope="class")
    def filtered_dataset(self, dataset: rx.Dataset) -> rx.Dataset:
        updated_dataset, _ = rx.FilterSubjectsWithLongAdmission.apply(dataset, None, rx.Report())
        return updated_dataset

    def test_transformation(
        self,
        dataset: rx.Dataset,
        filtered_dataset: rx.Dataset,
        filtered_dataset_no_config: rx.Dataset,
        subject_id: list[str],
        subject_id_with_long_admission: list[str],
    ):
        assert set(subject_id).issubset(dataset.tables.static.index)
        assert set(filtered_dataset.tables.static.index) == set(subject_id) - set(subject_id_with_long_admission)
        assert dataset.tables.equals(filtered_dataset_no_config.tables)


class TestSqueezeToStandardColumns:
    @pytest.fixture(scope="class")
    def static(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                rx.COLUMN.subject_id: list(range(100)),
                rx.COLUMN.gender: list(reversed(range(100))),
                rx.COLUMN.date_of_birth: ["x"] * 100,
                "other_column": ["y"] * 100,
            }
        )

    @pytest.fixture(scope="class")
    def indexed_static(self, static: pd.DataFrame) -> pd.DataFrame:
        return static.set_index(rx.COLUMN.subject_id)

    @pytest.fixture(scope="class")
    def dataset(self, static: pd.DataFrame) -> rx.Dataset:
        return rx.Dataset(
            tables=rx.DatasetTables(static=static, admissions=pd.DataFrame(columns=["admission_id"])),
            config=rx.DatasetConfig(scheme=rx.DatasetSchemeConfig()),
        )

    @pytest.fixture(scope="class")
    def indexed_dataset(self, indexed_static: pd.DataFrame) -> rx.Dataset:
        return rx.Dataset(
            tables=rx.DatasetTables(static=indexed_static, admissions=pd.DataFrame(columns=["admission_id"])),
            config=rx.DatasetConfig(scheme=rx.DatasetSchemeConfig()),
        )

    @pytest.fixture(scope="class")
    def indexed_transformed(self, indexed_dataset: rx.Dataset) -> rx.Dataset:
        updated, _ = rx.SqueezeToStandardColumns.apply(indexed_dataset, None, rx.Report())
        return updated

    @pytest.fixture(scope="class")
    def transformed(self, dataset: rx.Dataset) -> rx.Dataset:
        updated, _ = rx.SqueezeToStandardColumns.apply(dataset, None, rx.Report())
        return updated

    def test_transformation(
        self, transformed: rx.Dataset, indexed_transformed: rx.Dataset, dataset: rx.Dataset, indexed_dataset: rx.Dataset
    ):
        for ds in (dataset, indexed_dataset):
            assert "other_column" in ds.tables.static.columns
        for ds in (transformed, indexed_transformed):
            assert "other_column" not in ds.tables.static.columns
        assert set(transformed.tables.static.columns) - set(indexed_transformed.tables.static.columns) == {"subject_id"}
        assert set(dataset.tables.static.columns) - set(transformed.tables.static.columns) == {"other_column"}
        assert set(dataset.tables.static.columns) - set(indexed_transformed.tables.static.columns) == {
            "other_column",
            "subject_id",
        }
