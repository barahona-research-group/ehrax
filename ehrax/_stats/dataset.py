from collections.abc import Mapping
from dataclasses import dataclass
from typing import get_args, TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats.distributions as dist

from .._literals import TableAggregationLiteral
from ..coding_scheme import CodeMap, CodingSchemesManager


if TYPE_CHECKING:
    from ..dataset import AdmissionsTableColumns, Dataset, DatasetSchemeProxy  # type: ignore


@dataclass
class TargetHistogram:
    dataset: "Dataset"
    schemes_manager: CodingSchemesManager

    def __init__(self, dataset: "Dataset", schemes_manager: CodingSchemesManager):
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
    def adapt_aggregation_level(
        admissions: pd.DataFrame,
        admissions_cols: "AdmissionsTableColumns",
        table: pd.DataFrame,
        c_admission_id: str,
        aggregation_level: TableAggregationLiteral,
    ) -> tuple[pd.DataFrame, int]:
        c_subject_id = admissions_cols.subject_id
        match aggregation_level:
            case "admission":
                # do nothing.
                return table, len(admissions)
            case "first_admission":
                # Apply the statistics only on the first admission for each subject.
                # Collect the first admission id for each subject and remove the rest.
                admissions = admissions.sort_values(by=admissions_cols.start_time, ascending=True)
                admission_index = str(admissions.index.name)
                assert admission_index == c_admission_id
                first_admissions = admissions.reset_index(drop=False).groupby(c_subject_id)[admission_index].first()
                return table[table[c_admission_id].isin(first_admissions)], len(first_admissions)
            case "subject":
                # Apply the statistics on the level of each subject as a whole.
                # To adapt to the same function of `compute`, we just rename admission ids of each subject
                # to have the same dummy value. We just set the values of admission ids to the subject ids.
                table = table.assign(**{c_admission_id: table[c_admission_id].map(admissions[c_subject_id].to_dict())})
                return table, admissions[c_subject_id].nunique()
            case _:
                raise ValueError(
                    f"Unknown aggregation level '{aggregation_level}'. "
                    f"Expected one of: {get_args(TableAggregationLiteral)}."
                )

    def _dx_discharge(
        self, codemap: CodeMap, target_codes: tuple[str, ...], aggregation_level: TableAggregationLiteral = "admission"
    ) -> tuple[pd.Series, int]:
        table = self.dataset.tables.dx_discharge
        cols = self.dataset.config.columns.dx_discharge
        table, n = self.adapt_aggregation_level(
            self.dataset.tables.admissions,
            self.dataset.config.columns.admissions,
            table,
            cols.admission_id,
            aggregation_level,
        )
        hist = self.compute(table, cols.admission_id, cols.code, codemap)
        return hist.reindex(pd.Index(target_codes, name=cols.code), fill_value=0), n

    def dx_discharge(
        self, target_scheme: str | tuple[str, ...], aggregation_level: TableAggregationLiteral = "admission"
    ) -> tuple[pd.Series, int]:
        if isinstance(target_scheme, str):
            codemap = self.schemes_manager.map[self.dataset.config.scheme.dx_discharge, target_scheme]
        elif isinstance(target_scheme, tuple):
            path = (self.dataset.config.scheme.dx_discharge,) + target_scheme
            codemap = self.schemes_manager.make_chained_map(path)
            target_scheme = target_scheme[-1]
        else:
            raise ValueError(f"Expected a string or a tuple of strings, got {type(target_scheme)}.")

        codes = self.schemes_manager.scheme[target_scheme].codes
        return self._dx_discharge(codemap, codes, aggregation_level)

    def outcome(
        self, outcome: str | tuple[str, ...], aggregation_level: TableAggregationLiteral = "admission"
    ) -> tuple[pd.Series, int]:
        dx_scheme = self.dataset.config.scheme.dx_discharge
        if isinstance(outcome, str):
            o = self.schemes_manager.outcome[dx_scheme, outcome]
        elif isinstance(outcome, tuple):
            chain, outcome = outcome[:-1], outcome[-1]
            chain = (dx_scheme, *chain, self.schemes_manager.outcome_data[outcome].base_name)
            m = self.schemes_manager.make_chained_map(chain)
            o = self.schemes_manager.add_map(m, overwrite=True).outcome[dx_scheme, outcome]
        else:
            raise ValueError(f"Expected a string or a tuple of strings, got {type(outcome)}.")
        return self._dx_discharge(o.codemap, o.scheme.codes, aggregation_level)


@dataclass
class MultiDatasetTargetHistogram:
    target_hist: tuple[TargetHistogram, ...]
    schemes_manager: CodingSchemesManager

    def __init__(self, datasets: tuple["Dataset", ...], schemes_manager: CodingSchemesManager):
        self.target_hist = tuple(TargetHistogram(dataset, schemes_manager) for dataset in datasets)
        self.schemes_manager = schemes_manager

    @staticmethod
    def compile(results: tuple[pd.Series, ...], normalize: tuple[int, ...]) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.concat(results, axis=1)
        df.columns = [f"D{i}" for i in range(len(results))]
        df.index.name = results[0].index.name
        return df, pd.Series(normalize, index=df.columns)

    def dx_discharge(
        self, target_scheme: str, aggregation_level: TableAggregationLiteral = "admission"
    ) -> tuple[pd.DataFrame, pd.Series]:
        results, norm = zip(*[h.dx_discharge(target_scheme, aggregation_level) for h in self.target_hist])
        return self.compile(results, norm)

    def outcome(
        self, target_scheme: str, aggregation_level: TableAggregationLiteral = "admission"
    ) -> tuple[pd.DataFrame, pd.Series]:
        results, norm = zip(*[h.outcome(target_scheme, aggregation_level) for h in self.target_hist])
        return self.compile(results, norm)


@dataclass
class DatasetStatsInterface:
    dataset: "Dataset"
    schemes_manager: CodingSchemesManager

    def __init__(self, dataset: "Dataset", schemes_manager: CodingSchemesManager):
        self.dataset = dataset
        self.schemes_manager = schemes_manager

    def age_at_first_admission(self) -> pd.Series:
        c_subject_id = self.dataset.config.columns.admissions.subject_id
        c_admittime = self.dataset.config.columns.admissions.start_time
        c_dob = self.dataset.config.columns.static.date_of_birth
        admissions = self.dataset.tables.admissions.sort_values(by=c_admittime, ascending=True)
        first_admission = admissions.groupby(c_subject_id, as_index=False)[[c_subject_id, c_admittime]].first()
        admission_year = first_admission[c_admittime].dt.year.values
        birth_year = self.dataset.tables.static.loc[first_admission[c_subject_id], c_dob].dt.year.values
        return pd.Series(admission_year - birth_year, index=first_admission[c_subject_id])

    @property
    def schemes_proxy(self) -> "DatasetSchemeProxy":
        return self.dataset.scheme_proxy(self.schemes_manager)

    @property
    def target_hist(self) -> TargetHistogram:
        return TargetHistogram(self.dataset, self.schemes_manager)


@dataclass
class TwoDatasetsTargetHistogram(MultiDatasetTargetHistogram):
    def __init__(self, datasets: tuple["Dataset", ...], schemes_manager: CodingSchemesManager):
        super().__init__(datasets, schemes_manager)
        assert len(datasets) == 2


@dataclass
class MultiDatasetsStatsInterface:
    datasets: tuple["Dataset", ...]
    schemes_manager: CodingSchemesManager

    def __init__(self, *datasets: "Dataset", schemes_manager: CodingSchemesManager):
        self.datasets = datasets
        self.schemes_manager = schemes_manager

    @property
    def target_hist(self) -> MultiDatasetTargetHistogram:
        return MultiDatasetTargetHistogram(self.datasets, self.schemes_manager)


@dataclass
class TwoSamplesTest:
    target_hist: TwoDatasetsTargetHistogram

    def __init__(self, target_histogram: TwoDatasetsTargetHistogram):
        self.target_hist = target_histogram

    @staticmethod
    def proportion_tests(counts: pd.DataFrame, n: pd.Series, code_description: Mapping[str, str]) -> pd.DataFrame:
        assert counts.shape[1] == n.shape[0] == 2
        # z-test two sample proportion: https://mverbakel.github.io/2021-02-13/two-sample-proportions
        x1, x2 = counts.iloc[:, 0], counts.iloc[:, 1]
        n1, n2 = n.loc[counts.columns].tolist()
        avg_p = (x1 + x2) / n.sum()
        z_val = (x1 / n1 - x2 / n2) / np.sqrt(avg_p * (1 - avg_p) * (1 / n1 + 1 / n2))
        z_prob = pd.Series(-np.abs(z_val)).map(dist.norm.cdf)
        normal_assumption = (x1 >= 10) & (x2 >= 10) & (n1 - x1 >= 10) & (n2 - x2 >= 10)  # same blog post.
        return pd.DataFrame(
            {
                "code_description": list(map(code_description.get, counts.index)),
                "p_val_two_sided": 2 * z_prob,
                "normal_assumption": normal_assumption,
                "D0_counts": x1,
                "D1_counts": x2,
                "D0_p": x1 / n1,
                "D1_p": x2 / n2,
            },
            index=counts.index,
        )

    @staticmethod
    def summerise(stats: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
        divergent = (stats["normal_assumption"]) & (stats["p_val_two_sided"] < alpha)
        convergent = (stats["normal_assumption"]) & (stats["p_val_two_sided"] > alpha)
        n_divergent = divergent.sum()
        n_convergent = convergent.sum()
        n_skip_test = (~stats["normal_assumption"]).sum()
        n = stats.shape[0]
        return pd.Series({"total": n, "divergent": n_divergent, "convergent": n_convergent, "skip_test": n_skip_test})

    def dx_discharge(
        self, target_scheme: str | tuple[str, ...], aggregation_level: TableAggregationLiteral = "admission"
    ) -> pd.DataFrame:
        counts, n = self.target_hist.dx_discharge(target_scheme, aggregation_level)
        if isinstance(target_scheme, tuple):
            target_scheme = target_scheme[-1]
        return self.proportion_tests(counts, n, self.target_hist.schemes_manager.scheme[target_scheme].desc)

    def outcome(
        self, target_scheme: str | tuple[str, ...], aggregation_level: TableAggregationLiteral = "admission"
    ) -> pd.DataFrame:
        counts, n = self.target_hist.outcome(target_scheme, aggregation_level)
        if isinstance(target_scheme, tuple):
            target_scheme = target_scheme[-1]
        return self.proportion_tests(counts, n, self.target_hist.schemes_manager.outcome_scheme[target_scheme].desc)


@dataclass
class TwoDatasetsStatsInterface:
    datasets: tuple["Dataset", ...]
    schemes_manager: CodingSchemesManager

    def __init__(self, *datasets: "Dataset", schemes_manager: CodingSchemesManager):
        self.datasets = datasets
        self.schemes_manager = schemes_manager
        assert len(datasets) == 2

    @property
    def target_hist(self) -> TwoDatasetsTargetHistogram:
        return TwoDatasetsTargetHistogram(self.datasets, self.schemes_manager)

    @property
    def target_p_tests(self) -> TwoSamplesTest:
        return TwoSamplesTest(self.target_hist)

    @staticmethod
    def summerise_p_tests(stats: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
        return TwoSamplesTest.summerise(stats, alpha)
