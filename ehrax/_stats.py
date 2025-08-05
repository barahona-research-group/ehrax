from typing import TYPE_CHECKING, get_args

import pandas as pd

from ._literals import TableAggregationLiteral
from .coding_scheme import CodeMap, CodingSchemesManager

if TYPE_CHECKING:
    from .dataset import (AdmissionsTableColumns, Dataset, DatasetSchemeProxy)  # type: ignore


class TargetHistogram:
    def __init__(self, dataset: 'Dataset', schemes_manager: CodingSchemesManager):
        self.dataset = dataset
        self.schemes_manager = schemes_manager

    @staticmethod
    def compute(table: pd.DataFrame, c_admission_id: str, c_code: str, scheme_mapper: CodeMap) -> pd.Series:
        s_table = table[[c_admission_id, c_code]].drop_duplicates()
        s_table = s_table[s_table[c_code].isin(scheme_mapper.domain)]
        # group by admission_id to collapse duplicate target codes within a single admission.
        t_table = s_table.groupby(c_admission_id)[c_code].apply(scheme_mapper.map_codeset)
        return t_table.explode().value_counts()

    @staticmethod
    def adapt_aggregation_level(admissions: pd.DataFrame, admissions_cols: 'AdmissionsTableColumns',
                                table: pd.DataFrame, c_admission_id: str,
                                aggregation_level: TableAggregationLiteral) -> pd.DataFrame:
        c_subject_id = admissions_cols.subject_id
        match aggregation_level:
            case 'admission':
                # do nothing.
                return table
            case 'first_admission':
                # Apply the statistics only on the first admission for each subject.
                # Collect the first admission id for each subject and remove the rest.
                admissions = admissions.sort_values(by=admissions_cols.start_time, ascending=True)
                admission_index = str(admissions.index.name)
                assert admission_index == c_admission_id
                first_admissions = admissions.reset_index(drop=False).groupby(c_subject_id)[admission_index].first()
                return table[table[c_admission_id].isin(first_admissions)]
            case 'subject':
                # Apply the statistics on the level of each subject as a whole.
                # to adapt to the same function of `compute`, we just rename admission ids of each subject
                # to have the same dummy value. We just set the values of admission ids to the subject ids.
                return table.assign(**{c_admission_id: table[c_admission_id].map(admissions[c_subject_id].to_dict())})
            case _:
                raise ValueError(f"Unknown aggregation level '{aggregation_level}'. "
                                 f"Expected one of: {get_args(TableAggregationLiteral)}.")

    def _dx_discharge(self, codemap: CodeMap, target_codes: tuple[str, ...],
                      aggregation_level: TableAggregationLiteral = 'admission') -> pd.Series:
        table = self.dataset.tables.dx_discharge
        cols = self.dataset.config.columns.dx_discharge
        table = self.adapt_aggregation_level(self.dataset.tables.admissions, self.dataset.config.columns.admissions,
                                             table, cols.admission_id, aggregation_level)
        hist = self.compute(table, cols.admission_id, cols.code, codemap).to_dict()
        return pd.Series(lambda c: hist.get(c, 0), index=target_codes)


    def dx_discharge(self, target_scheme: str, aggregation_level: TableAggregationLiteral = 'admission') -> pd.Series:
        codemap = self.schemes_manager.map[self.dataset.config.scheme.dx_discharge, target_scheme]
        codes = self.schemes_manager.scheme[target_scheme].codes
        return self._dx_discharge(codemap, codes, aggregation_level)


    def outcome(self, outcome: str, aggregation_level: TableAggregationLiteral = 'admission') -> pd.Series:
        o = self.schemes_manager.outcome[self.dataset.config.scheme.dx_discharge, outcome]
        return self._dx_discharge(o.codemap, o.scheme.codes, aggregation_level)


class DatasetStatsInterface:
    def __init__(self, dataset: 'Dataset', schemes_manager: CodingSchemesManager):
        self.dataset = dataset
        self.schemes_manager = schemes_manager

    @property
    def schemes_proxy(self) -> 'DatasetSchemeProxy':
        return self.dataset.scheme_proxy(self.schemes_manager)

    @property
    def target_hist(self) -> TargetHistogram:
        return TargetHistogram(self.dataset, self.schemes_manager)
