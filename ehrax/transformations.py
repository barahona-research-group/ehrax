import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import equinox as eqx
import numpy as np
import pandas as pd

from .coding_scheme import CodingSchemesManager, CodingSchemeWithUOM
from .dataset import (
    AbstractTransformation,
    COLUMN,
    Dataset,
    Report,
    SECONDS_TO_DAYS_SCALER,
    SECONDS_TO_HOURS_SCALER,
)


class DatasetTransformation(AbstractTransformation, metaclass=ABCMeta):
    @staticmethod
    def synchronize_index(
        dataset: Dataset, indexed_table_name: str, index_name: str, report: Report
    ) -> tuple[Dataset, Report]:
        tables_dict = dataset.tables.tables_dict

        target_tables = {  # columns that have admission_id as column
            k: v for k, v in tables_dict.items() if k != indexed_table_name and index_name in v.columns
        }

        index = tables_dict[indexed_table_name].index
        tables = dataset.tables
        for table_name, table in target_tables.items():
            n1 = len(table)
            table = table[table[index_name].isin(pd.Series(index))]
            n2 = len(table)
            report = report.add(
                table=table_name, column=index_name, before=n1, after=n2, value_type="count", operation="sync_index"
            )
            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)

        return eqx.tree_at(lambda x: x.tables, dataset, tables), report

    @staticmethod
    def filter_subjects_with_less_than_n_admissions(
        static: pd.DataFrame, admissions: pd.DataFrame, n: int
    ) -> pd.DataFrame:
        n_admissions = static.index.map(admissions.groupby(COLUMN.subject_id).size()).fillna(0)
        return static.loc[n_admissions >= n]

    @staticmethod
    def filter_no_admission_subjects(dataset: Dataset, report: Report) -> tuple[Dataset, Report]:
        static = dataset.tables.static
        admissions = dataset.tables.admissions
        n1 = len(static)
        static = DatasetTransformation.filter_subjects_with_less_than_n_admissions(static, admissions, 1)
        n2 = len(static)
        report = report.add(
            table="static",
            column=static.index.name,
            before=n1,
            after=n2,
            value_type="count",
            operation="filter_no_admission_subjects",
        )
        return eqx.tree_at(lambda x: x.tables.static, dataset, static), report

    @classmethod
    def synchronize_admissions(cls, dataset: Dataset, report: Report) -> tuple[Dataset, Report]:
        dataset, report = cls.synchronize_index(
            dataset, "admissions", dataset.config.columns.admissions.admission_id, report
        )
        return cls.filter_no_admission_subjects(dataset, report)

    @classmethod
    def synchronize_subjects(cls, dataset: Dataset, report: Report) -> tuple[Dataset, Report]:
        # Synchronizing subjects might entail synchronizing admissions, so we need to call it first
        dataset, report = cls.synchronize_index(dataset, "static", dataset.config.columns.static.subject_id, report)
        return cls.synchronize_admissions(dataset, report)

    @classmethod
    @abstractmethod
    def apply(
        cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report
    ) -> tuple[Dataset, Report]: ...


class SynchronizeAdmissions(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        return cls.synchronize_admissions(dataset, report)


class SynchronizeSubjects(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        return cls.synchronize_subjects(dataset, report)


class SetIndex(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        tables_dict = dataset.tables.tables_dict
        for indexed_table_name, index_name, table in (
            (table_name, index_name, tables_dict[table_name])
            for table_name, index_name in dataset.config.columns.indices.items()
            if table_name in tables_dict
        ):
            (index_name,) = index_name
            index1 = table.index.name
            table = table.set_index(index_name)
            index2 = table.index.name
            report = report.add(
                table=indexed_table_name,
                column=index_name,
                before=index1,
                after=index2,
                value_type="index_name",
                operation="set_index",
            )
            dataset = eqx.tree_at(lambda x: getattr(x.tables, indexed_table_name), dataset, table)
        return dataset, report


class CastTimestamps(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        tables = dataset.tables
        tables_dict = tables.tables_dict
        for table_name, time_cols, table in (
            (name, cols, tables_dict[name])
            for name, cols in dataset.config.columns.time_cols.items()
            if name in tables_dict
        ):
            table = table.iloc[:, :]
            for time_col in time_cols:
                assert time_col in table.columns, f"{time_col} not found in {table_name}"
                assert isinstance(time_col, str)

                if table[time_col].dtype == "datetime64[ns]":
                    logging.debug(f"{table_name}[{time_col}] already in datetime64[ns]")
                    continue
                dtype1 = table[time_col].dtype
                table[time_col] = pd.to_datetime(table[time_col], errors="raise")
                dtype2 = table[time_col].dtype
                report = report.add(
                    table=table_name, column=time_col, before=dtype1, after=dtype2, value_type="dtype", operation="cast"
                )

            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)
        return eqx.tree_at(lambda x: x.tables, dataset, tables), report


class SqueezeToStandardColumns(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        tables_dict = dataset.tables.tables_dict
        for table_name, columns, table in (
            (name, cols, tables_dict[name])
            for name, cols in dataset.config.columns.columns_dict().items()
            if name in tables_dict
        ):
            columns = tuple(c for c in columns if c in table.columns)
            if columns == tuple(table.columns):
                continue
            elif set(columns) == set(table.columns):
                operation = "columns reordering"
            else:
                operation = "columns subsetting & reordering"

            report = report.add(
                table=table_name,
                before=", ".join(table.columns),
                after=", ".join(columns),
                value_type="columns",
                operation=operation,
            )
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, table[list(columns)])
        return dataset, report


class SetAdmissionRelativeTimes(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        time_cols = {
            k: v
            for k, v in dataset.config.columns.time_cols.items()
            if dataset.config.columns.temporal_admission_linked_table(k)
        }

        c_admittime = dataset.config.columns.admissions.start_time
        c_admission_id = dataset.config.columns.admissions.admission_id
        admissions = dataset.tables.admissions[[c_admittime]]
        tables_dict = dataset.tables.tables_dict

        for table_name, table_time_cols, table in (
            (name, cols, tables_dict[name]) for name, cols in time_cols.items() if name in tables_dict
        ):
            df = pd.merge(
                table, admissions, left_on=c_admission_id, right_index=True, suffixes=(None, "_y"), how="left"
            )
            admittime_col = f"{c_admittime}_y" if c_admittime in table.columns else c_admittime
            for time_col in table_time_cols:
                update = {time_col: (df[time_col] - df[admittime_col]).dt.total_seconds() * SECONDS_TO_HOURS_SCALER}
                df = df.assign(**update)
                report = report.add(
                    table=table_name,
                    column=time_col,
                    before=table[time_col].dtype,
                    after=df[time_col].dtype,
                    value_type="dtype",
                    operation="set_admission_relative_times",
                )

            df = df[table.columns]
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, df)

        return dataset, report


class FilterSubjectsNegativeAdmissionLengths(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        table_config = dataset.config.columns.admissions
        c_subject_id = table_config.subject_id
        c_dischtime = table_config.end_time
        c_admittime = table_config.start_time
        adms = dataset.tables.admissions

        # assert dtypes are datetime64[ns]
        assert adms[c_admittime].dtype == "datetime64[ns]" and adms[c_dischtime].dtype == "datetime64[ns]", (
            f"{c_admittime} and {c_dischtime} must be datetime64[ns]"
        )

        static = dataset.tables.static
        neg_los_sub = adms.loc[adms.loc[:, c_dischtime] < adms.loc[:, c_admittime], c_subject_id].unique()
        n_before = len(static)
        static = static[~static.index.isin(neg_los_sub)]
        n_after = len(static)
        report = report.add(
            table="static", column=c_subject_id, value_type="count", operation="filter", before=n_before, after=n_after
        )
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return cls.synchronize_subjects(dataset, report)


class FilterUnsupportedCodes(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        tables_dict = dataset.tables.tables_dict
        for table_name, code_column, table in (
            (name, col, tables_dict[name])
            for name, col in dataset.config.columns.code_column.items()
            if name in tables_dict
        ):
            (code_column,) = code_column
            coding_scheme = getattr(dataset.scheme_proxy(schemes_context), table_name)
            n1 = len(table)
            table = table[table[code_column].isin(coding_scheme.codes)]
            n2 = len(table)
            report = report.add(
                table=table_name, column=code_column, before=n1, after=n2, value_type="count", operation="filter"
            )
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, table)
        return dataset, report


class FilterAdmissionsWithNoDiagnoses(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        dx_discharge = dataset.tables.dx_discharge
        assert isinstance(dx_discharge, pd.DataFrame)
        admissions = dataset.tables.admissions
        c_admission_id = dataset.config.columns.dx_discharge.admission_id
        selected_admission_id = set(dx_discharge[c_admission_id].tolist())
        n1 = len(admissions)
        admissions = admissions[admissions.index.isin(selected_admission_id)]
        n2 = len(admissions)
        report = report.add(
            table="admissions",
            column=admissions.index.name,
            before=n1,
            after=n2,
            value_type="count",
            operation="filter",
        )
        dataset = eqx.tree_at(lambda x: x.tables.admissions, dataset, admissions)
        return cls.synchronize_admissions(dataset, report)


class FilterAdmissionsWithNoObservables(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        obs = dataset.tables.obs
        assert isinstance(obs, pd.DataFrame)
        admissions = dataset.tables.admissions
        c_admission_id = dataset.config.columns.obs.admission_id
        selected_admission_id = set(obs[c_admission_id].tolist())
        n1 = len(admissions)
        admissions = admissions[admissions.index.isin(selected_admission_id)]
        n2 = len(admissions)
        report = report.add(
            table="admissions",
            column=admissions.index.name,
            before=n1,
            after=n2,
            value_type="count",
            operation="filter",
        )
        dataset = eqx.tree_at(lambda x: x.tables.admissions, dataset, admissions)
        return cls.synchronize_admissions(dataset, report)


class FilterSubjectsWithSingleOrNoAdmission(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        n1 = len(dataset.tables.static)
        static = cls.filter_subjects_with_less_than_n_admissions(dataset.tables.static, dataset.tables.admissions, 2)
        n2 = len(static)
        report = report.add(
            table="static", column=static.index.name, before=n1, after=n2, value_type="count", operation="filter"
        )
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return cls.synchronize_subjects(dataset, report)


class FilterSubjectsWithLongAdmission(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        max_days = dataset.config.select_subjects_with_short_admissions
        if max_days is None:
            return cls.skip(dataset, report, reason="select_subjects_with_short_admissions is not configured.")

        a_df = dataset.tables.admissions.copy()
        static = dataset.tables.static
        a_df["los"] = (a_df[COLUMN.end_time] - a_df[COLUMN.start_time]).dt.total_seconds() * SECONDS_TO_DAYS_SCALER
        max_admission_los = a_df.groupby(COLUMN.subject_id)["los"].max().loc[static.index]
        n1 = len(static)
        static = static[max_admission_los < max_days]
        n2 = len(static)
        report = report.add(
            table="static", column=static.index.name, before=n1, after=n2, value_type="count", operation="filter"
        )
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return cls.synchronize_subjects(dataset, report)


class FilterShortAdmissions(DatasetTransformation):  # without removing the corresponding subjects.
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        min_days = dataset.config.admission_minimum_los
        if min_days is None:
            return cls.skip(dataset, report, reason="select_subjects_with_short_admissions is not configured.")
        a_df = dataset.tables.admissions
        los = (a_df[COLUMN.end_time] - a_df[COLUMN.start_time]).dt.total_seconds() * SECONDS_TO_DAYS_SCALER
        n1 = len(a_df)
        a_df = a_df.loc[los >= min_days]
        n2 = len(a_df)
        report = report.add(
            table="admissions", column=a_df.index.name, before=n1, after=n2, value_type="count", operation="filter"
        )
        dataset = eqx.tree_at(lambda x: x.tables.admissions, dataset, a_df)
        return cls.synchronize_admissions(dataset, report)


class ProcessOverlappingAdmissions(DatasetTransformation):
    @staticmethod
    def _collect_overlaps(admissions: pd.DataFrame) -> dict[str, str]:
        """
        Collect overlapping admissions for a subject.
        """
        # Sort by admission time.
        if len(admissions) == 0:
            return dict()

        admissions = admissions.sort_values(COLUMN.start_time)
        intervals = list(zip(admissions.index, admissions[COLUMN.start_time], admissions[COLUMN.end_time]))
        new_admission = dict()
        new_admission_id, _, last_discharge = intervals[0]
        for admission_id, admission_time, discharge_time in intervals:
            if last_discharge < admission_time:
                new_admission_id = admission_id
            if new_admission_id != admission_id:
                new_admission[admission_id] = new_admission_id
            last_discharge = max(last_discharge, discharge_time)
        return dict(new_admission)


class MergeOverlappingAdmissions(ProcessOverlappingAdmissions):
    @staticmethod
    def _tables_map_admission_ids(dataset: Dataset, sub2sup: dict[str, str], report: Report) -> tuple[Dataset, Report]:
        tables_dict = dataset.tables.tables_dict
        c_admission_id = dataset.config.columns.admissions.admission_id

        target_tables = {  # columns that have admission_id as column
            k: v for k, v in tables_dict.items() if k != "admissions" and c_admission_id in v.columns
        }

        tables = dataset.tables
        for table_name, table in target_tables.items():
            n1 = table[c_admission_id].nunique()
            table.loc[:, c_admission_id] = table.loc[:, c_admission_id].map(lambda i: sub2sup.get(i, i))
            n2 = table[c_admission_id].nunique()
            report = report.add(
                table=table_name,
                column=c_admission_id,
                before=n1,
                after=n2,
                value_type="nunique",
                operation="map_admission_id",
            )
            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)

        return eqx.tree_at(lambda x: x.tables, dataset, tables), report

    @staticmethod
    def _admissions_map_admission_ids(
        dataset: Dataset, sub2sup: dict[str, str], report: Report
    ) -> tuple[Dataset, Report]:
        admissions = dataset.tables.admissions

        # Step 1: Map from super-admissions to its sub-admissions.
        sup2sub = defaultdict(list)
        for sub, sup in sub2sup.items():
            sup2sub[sup].append(sub)

        # Step 2: Merge overlapping admissions by extending discharge time to the maximum discharge
        # time of its sub-admissions.
        for super_idx, sub_indices in sup2sub.items():
            current_dischtime = admissions.loc[super_idx, COLUMN.end_time]
            new_dischtime = max(max(admissions.loc[sub_indices, COLUMN.end_time].values), current_dischtime)
            admissions.loc[super_idx, COLUMN.end_time] = new_dischtime

        # Step 3: Remove sub-admissions.
        n1 = len(admissions)
        admissions = admissions.drop(list(sub2sup.keys()), axis="index")
        n2 = len(admissions)
        dataset = eqx.tree_at(lambda x: x.tables.admissions, dataset, admissions)
        report = report.add(
            table="admissions",
            column=COLUMN.admission_id,
            value_type="count",
            operation="merge_overlapping_admissions",
            before=n1,
            after=n2,
        )
        return dataset, report

    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        # Step 1: Collect overlapping admissions
        # NOTE: assumes unique admissions globally. See ISSUE_ADM_UNIQ in dataset.py.
        sub2sup = {
            adm_id: super_adm_id
            for _, subject_admissions in dataset.tables.admissions.groupby(COLUMN.subject_id)
            for adm_id, super_adm_id in cls._collect_overlaps(subject_admissions).items()
        }
        # Step 2: Merge in admissions table
        dataset, report = cls._admissions_map_admission_ids(dataset, sub2sup, report)
        # Step 3: Map admission ids in other tables.
        return cls._tables_map_admission_ids(dataset, sub2sup, report)


class RemoveSubjectsWithOverlappingAdmissions(ProcessOverlappingAdmissions):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        admissions = dataset.tables.admissions
        # Step 1: Collect overlapping admissions
        # Map from sub-admissions to the new super-admissions.
        sub2sup = {
            adm_id: super_adm_id
            for _, subject_admissions in admissions.groupby(COLUMN.subject_id)
            for adm_id, super_adm_id in cls._collect_overlaps(subject_admissions).items()
        }

        # Step 2: Collect subjects with at least one overlapping admission and remove them entirely.
        subject_ids = admissions.loc[sub2sup.keys(), COLUMN.subject_id].unique()
        static = dataset.tables.static
        n1 = len(static)
        static = static.drop(subject_ids, axis="index")
        n2 = len(static)
        report = report.add(
            table="static",
            column=COLUMN.subject_id,
            value_type="count",
            operation="filter_problematic_subjects",
            before=n1,
            after=n2,
        )
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        # Step 4: synchronize subjects
        return cls.synchronize_subjects(dataset, report)


class FilterClampTimestampsToAdmissionInterval(DatasetTransformation):
    @classmethod
    def _filter_timestamped_tables(cls, dataset: Dataset, report: Report) -> tuple[Dataset, Report]:
        tables_dict = dataset.tables.tables_dict
        timestamped_tables_conf = dataset.config.columns.timestamped_tables_config_dict
        timestamped_tables = {name: tables_dict[name] for name in timestamped_tables_conf.keys() if name in tables_dict}

        table_config = dataset.config.columns.admissions
        c_admission_id = table_config.admission_id
        c_dischtime = table_config.end_time
        c_admittime = table_config.start_time
        admissions = dataset.tables.admissions[[c_admittime, c_dischtime]]

        for name, table in timestamped_tables.items():
            c_time = timestamped_tables_conf[name].time
            df = pd.merge(
                table, admissions, how="left", left_on=c_admission_id, right_index=True, suffixes=(None, "_y")
            )
            admittime_col = f"{c_admittime}_y" if c_admittime in table.columns else c_admittime
            dischtime_col = f"{c_dischtime}_y" if c_dischtime in table.columns else c_dischtime

            index = df[df[c_time].between(df[admittime_col], df[dischtime_col])].index
            n1 = len(table)
            table = table.loc[index]
            n2 = len(table)
            report = report.add(table=name, column=c_time, value_type="count", operation="filter", before=n1, after=n2)
            dataset = eqx.tree_at(lambda x: getattr(x.tables, name), dataset, table)

        return dataset, report

    @classmethod
    def _filter_interval_based_tables(cls, dataset: Dataset, report: Report) -> tuple[Dataset, Report]:
        tables_dict = dataset.tables.tables_dict
        interval_based_tables_conf = dataset.config.columns.interval_based_table_config_dict
        interval_based_tables: dict[str, pd.DataFrame] = {
            name: tables_dict[name] for name in interval_based_tables_conf.keys() if name in tables_dict
        }
        table_config = dataset.config.columns.admissions
        c_admission_id = table_config.admission_id
        c_dischtime = table_config.end_time
        c_admittime = table_config.start_time
        admissions = dataset.tables.admissions[[c_admittime, c_dischtime]]

        for name, table in interval_based_tables.items():
            c_start_time = interval_based_tables_conf[name].start_time
            c_end_time = interval_based_tables_conf[name].end_time
            df = pd.merge(
                table, admissions, how="left", left_on=c_admission_id, right_index=True, suffixes=(None, "_y")
            )
            admittime_col = f"{c_admittime}_y" if c_admittime in table.columns else c_admittime
            dischtime_col = f"{c_dischtime}_y" if c_dischtime in table.columns else c_dischtime
            # Step 1: Filter out intervals that are entirely outside admission interval.
            index = df[
                df[c_start_time].between(df[admittime_col], df[dischtime_col])
                | df[c_end_time].between(df[admittime_col], df[dischtime_col])
            ].index
            n1 = len(df)
            df = df.loc[index]
            n2 = len(df)
            report = report.add(
                table=name,
                column=(c_start_time, c_end_time),
                value_type="count",
                operation="filter",
                before=n1,
                after=n2,
            )

            # Step 2: Clamp intervals to admission interval if either side is outside.
            n_to_clamp = np.sum((df[c_start_time] < df[admittime_col]) | (df[c_end_time] > df[dischtime_col]))
            report = report.add(
                table=name,
                column=(c_start_time, c_end_time),
                value_type="count",
                operation="clamp",
                before=None,
                after=n_to_clamp,
            )
            df[c_start_time] = df[c_start_time].clip(lower=df[admittime_col], upper=df[dischtime_col])
            df[c_end_time] = df[c_end_time].clip(lower=df[admittime_col], upper=df[dischtime_col])
            df = df[table.columns]
            dataset = eqx.tree_at(lambda x: getattr(x.tables, name), dataset, df)

        return dataset, report

    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        dataset, report = cls._filter_timestamped_tables(dataset, report)
        return cls._filter_interval_based_tables(dataset, report)


class SelectSubjectsWithObservation(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        c_code = dataset.config.columns.obs.code
        c_admission_id = dataset.config.columns.obs.admission_id
        c_subject = dataset.config.columns.static.subject_id
        obs = dataset.tables.obs
        assert isinstance(obs, pd.DataFrame)

        code = dataset.config.select_subjects_with_observation
        assert isinstance(code, str), "No code provided for filtering subjects"

        admission_ids = obs.loc[obs[c_code] == code, c_admission_id].unique()
        assert len(admission_ids) > 0, f"No observations for code {code}"

        subjects = dataset.tables.admissions.loc[admission_ids, c_subject].unique()
        static = dataset.tables.static
        n1 = len(static)
        static = static[static.index.isin(subjects)]
        n2 = len(static)
        report = report.add(
            table="static",
            column=c_subject,
            value_type="count",
            operation=f"select_subjects(has({code}))",
            before=n1,
            after=n2,
        )
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return cls.synchronize_subjects(dataset, report)


class FilterInvalidInputRatesSubjects(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        c_rate = dataset.config.columns.icu_inputs.derived_normalized_amount_per_hour
        c_admission_id = dataset.config.columns.admissions.admission_id
        c_subject_id = dataset.config.columns.admissions.subject_id

        icu_inputs = dataset.tables.icu_inputs
        assert isinstance(icu_inputs, pd.DataFrame)
        static = dataset.tables.static
        admissions = dataset.tables.admissions

        nan_input_rates = icu_inputs[icu_inputs[c_rate].isnull()]
        n_nan_inputs = len(nan_input_rates)
        nan_adm_ids = nan_input_rates.loc[:, c_admission_id].unique()
        n_nan_adms = len(nan_adm_ids)

        nan_subject_ids = admissions.loc[admissions.index.isin(nan_adm_ids), c_subject_id].unique()
        n_nan_subjects = len(nan_subject_ids)

        report = report.add(
            table=("icu_inputs", "admissions", "static"),
            column=(c_rate, c_admission_id, c_subject_id),
            value_type="nan_counts",
            before=(n_nan_inputs, n_nan_adms, n_nan_subjects),
            after=None,
            operation="filter_invalid_input_rates_subjects",
        )

        n1 = len(static)
        static = static[~static.index.isin(nan_subject_ids)]
        n2 = len(static)
        report = report.add(
            table="static",
            column=c_subject_id,
            value_type="count",
            before=n1,
            after=n2,
            operation="filter_invalid_input_rates_subjects",
        )
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return cls.synchronize_subjects(dataset, report)


class FilterSubjectsWithInvalidInputInterval(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        c_admission_id = dataset.config.columns.admissions.admission_id
        c_subject_id = dataset.config.columns.admissions.subject_id
        c_start = dataset.config.columns.icu_inputs.start_time
        c_end = dataset.config.columns.icu_inputs.end_time

        icu_inputs = dataset.tables.icu_inputs
        assert isinstance(icu_inputs, pd.DataFrame)
        static = dataset.tables.static
        admissions = dataset.tables.admissions

        invalid_input_interval = icu_inputs[icu_inputs[c_start] > icu_inputs[c_end]]
        n_invalid_inputs = len(invalid_input_interval)
        invalidated_adm_ids = invalid_input_interval.loc[:, c_admission_id].unique()
        n_invalidated_adms = len(invalidated_adm_ids)

        invalidated_subject_ids = admissions.loc[admissions.index.isin(invalidated_adm_ids), c_subject_id].unique()
        n_nan_subjects = len(invalidated_subject_ids)

        report = report.add(
            table=("icu_inputs", "admissions", "static"),
            column=(c_start, c_admission_id, c_subject_id),
            value_type="invalid_counts",
            before=(n_invalid_inputs, n_invalidated_adms, n_nan_subjects),
            after=None,
            operation="filter_invalid_input_intervals_subjects",
        )

        n1 = len(static)
        static = static[~static.index.isin(invalidated_subject_ids)]
        n2 = len(static)
        report = report.add(
            table="static",
            column=c_subject_id,
            value_type="count",
            before=n1,
            after=n2,
            operation="filter_invalid_input_intervals_subjects",
        )
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return cls.synchronize_subjects(dataset, report)


class ICUInputRateUnitConversion(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, schemes_context: CodingSchemesManager, report: Report) -> tuple[Dataset, Report]:
        ds_config = dataset.config
        tables_config = ds_config.columns
        table_config = tables_config.icu_inputs
        c_code = table_config.code
        c_amount = table_config.amount
        c_start_time = table_config.start_time
        c_end_time = table_config.end_time
        c_amount_unit = table_config.amount_unit
        c_normalized_amount = table_config.derived_normalized_amount
        c_normalized_amount_per_hour = table_config.derived_normalized_amount_per_hour
        c_universal_unit = table_config.derived_universal_unit
        c_normalization_factor = table_config.derived_unit_normalization_factor
        icu_inputs = dataset.tables.icu_inputs
        assert isinstance(icu_inputs, pd.DataFrame)

        scheme = dataset.scheme_proxy(schemes_context).icu_inputs
        assert isinstance(scheme, CodingSchemeWithUOM), f"Expected CodingSchemeWithUOM but got {type(scheme)}"
        _derived_columns = [c_normalized_amount, c_normalized_amount_per_hour, c_universal_unit, c_normalization_factor]

        assert (c in icu_inputs.columns for c in [c_code, c_amount, c_amount_unit]), (
            f"Some columns in: {c_code}, {c_amount}, {c_amount_unit}, not found in icu_inputs table"
        )
        assert all(c not in icu_inputs.columns for c in _derived_columns), (
            f"Some of these columns [{', '.join(_derived_columns)}] already exists in icu_inputs table"
        )
        df = icu_inputs.iloc[:, :]
        df[c_universal_unit] = df[c_code].map(lambda c: scheme.universal_unit[c])
        df[c_normalization_factor] = [
            scheme.uom_normalization_factor[code][unit] for code, unit in zip(df[c_code], df[c_amount_unit].str.lower())
        ]

        delta_hours = (df[c_end_time] - df[c_start_time]).dt.total_seconds() * SECONDS_TO_HOURS_SCALER
        df[c_normalized_amount] = df[c_amount] * df[c_normalization_factor]
        df[c_normalized_amount_per_hour] = df[c_normalized_amount] / delta_hours
        df = df[icu_inputs.columns.tolist() + _derived_columns]
        dataset = eqx.tree_at(lambda x: x.tables.icu_inputs, dataset, df)
        report = report.add(
            table="icu_inputs",
            column=None,
            value_type="columns",
            operation="new_columns",
            before=icu_inputs.columns.tolist(),
            after=df.columns.tolist(),
        )

        return dataset, report
