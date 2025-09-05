"""."""

import warnings
from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, cast, Self

import equinox as eqx
import numpy as np
import pandas as pd

from ..base import AbstractConfig
from ..coding_scheme import (
    CodeMap,
    CodingScheme,
    CodingSchemesManager,
    CodingSchemeWithUOM,
    FrozenDict11,
    NumericScheme,
    ReducedCodeMapN1,
)
from ..dataset import (
    AdmissionIntervalEventsTableColumns,
    AdmissionIntervalRatesTableColumns,
    AdmissionSummaryTableColumns,
    AdmissionTimeSeriesTableColumns,
    COLUMN,
    Dataset,
    DatasetConfig,
    DatasetSchemeConfig,
    DatasetTables,
    MultivariateTimeSeriesTableMeta,
    SECONDS_TO_HOURS_SCALER,
    StaticTableColumns,
    TableColumns,
)
from ..example_schemes.icd_ccs_integration import setup_standard_icd_ccs
from ..example_schemes.mixed_icd import MultiVersionScheme
from ..utils import resources_path


warnings.filterwarnings("error", category=RuntimeWarning, message=r"overflow encountered in cast")


class TableResource(AbstractConfig):
    columns: TableColumns

    def __init__(self, columns: TableColumns):
        self.columns = columns

    @staticmethod
    def _coerce_columns_to_str(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        is_str = pd.api.types.is_string_dtype
        # coerce to integers then fix as strings.
        int_dtypes = {k: int for k in columns if k in df.columns and not is_str(df.dtypes[k])}
        str_dtypes = {k: str for k in columns if k in df.columns and not is_str(df.dtypes[k])}
        return df.astype(int_dtypes).astype(str_dtypes)

    def _coerce_id_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer ids in the database when downloaded are stored as floats.
        A fix is to coerce them to integers then fix as strings.
        """
        return self._coerce_columns_to_str(df, self.columns.id_cols)

    def _coerce_code_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer codes in the database when downloaded are stored as floats or integers.
        A fix is to coerce them to integers then fix as strings.
        """
        return self._coerce_columns_to_str(df, self.columns.code_cols)

    @staticmethod
    def _filter_null_codes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Some code cells are empty (e.g., ROW_ID=7969 in MIMIC-III v1.3 DIAGNOSES_ICD.csv.gz). Filter these rows.
        """
        not_null = df[COLUMN.code].notnull() & (df[COLUMN.code] != "")
        return df.loc[not_null]

    @property
    def pipeline(self) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return (self._coerce_id_to_str,)

    @staticmethod
    def apply_pipeline(pipeline: tuple[Callable[[pd.DataFrame], pd.DataFrame], ...], table):
        for f in pipeline:
            table = f(table)
        return table

    @abstractmethod
    def load_standard_columns_table(self, data_connection: Any, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()

    def __call__(self, data_connection: Any, **kwargs) -> pd.DataFrame:
        return self.apply_pipeline(self.pipeline, self.load_standard_columns_table(data_connection, **kwargs))


CodedColumns = (
    AdmissionSummaryTableColumns
    | AdmissionTimeSeriesTableColumns
    | AdmissionIntervalEventsTableColumns
    | AdmissionIntervalRatesTableColumns
)


class CodedTableResource(TableResource):
    def __check_init__(self):
        assert all(c in self.columns for c in (COLUMN.code, COLUMN.description))

    @property
    def pipeline(self) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return self._coerce_id_to_str, self._filter_null_codes, self._coerce_code_to_str

    def space(self, data_connection: Any) -> pd.DataFrame:
        return self.apply_pipeline(self.pipeline, self.load_space_table(data_connection))

    @abstractmethod
    def load_space_table(self, data_connection: Any):
        raise NotImplementedError("This method should be implemented in subclasses.")


class StaticTableResource(TableResource):
    columns: StaticTableColumns

    def __init__(self):
        super().__init__(StaticTableColumns())

    @staticmethod
    def substitute_null_function(column: str, replace: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
        def _apply(df: pd.DataFrame) -> pd.DataFrame:
            return df.assign(**{column: df[column].fillna(value=replace).astype(str)})

        return _apply

    @abstractmethod
    def load_gender_space_table(self, data_connection: Any) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def load_ethnicity_space_table(self, data_connection: Any) -> pd.DataFrame:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def derive_deshifted_date_of_birth(cls, patients: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Different procedures to implement for MIMIC-III and MIMIC-IV
        raise NotImplementedError()

    @classmethod
    def _add_deshifted_date_of_birth(cls, admissions: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
        def _add(df: pd.DataFrame) -> pd.DataFrame:
            df[COLUMN.date_of_birth] = cls.derive_deshifted_date_of_birth(df, admissions=admissions)
            return df

        return _add

    @property
    def pipeline(self) -> None:
        return None

    def gender_space(self, date_source: Any) -> pd.DataFrame:
        p = self.substitute_null_function(str(COLUMN.gender), "MISSING_GENDER")
        return p(self.load_gender_space_table(date_source))

    def ethnicity_space(self, data_connection: Any) -> pd.DataFrame:
        p = self.substitute_null_function(str(COLUMN.race), "MISSING_ETHNICITY")
        return p(self.load_ethnicity_space_table(data_connection))

    def __call__(self, data_connection: Any, **kwargs) -> pd.DataFrame:
        assert "admissions" in kwargs, "Pass the processed admissions table."
        admissions = kwargs.pop("admissions")
        pipeline = (
            self._coerce_id_to_str,
            self._add_deshifted_date_of_birth(admissions=admissions),
            self.substitute_null_function(str(COLUMN.race), "MISSING_ETHNICITY"),
            self.substitute_null_function(str(COLUMN.gender), "MISSING_GENDER"),
        )
        return self.apply_pipeline(pipeline, self.load_standard_columns_table(data_connection, **kwargs))


class StaticTableResource_MIMICIV(StaticTableResource):
    @classmethod
    def derive_deshifted_date_of_birth(cls, patients: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(
            list(
                map(
                    lambda dt, age: dt + pd.DateOffset(years=-age),
                    pd.to_datetime(patients[COLUMN.anchor_year], format="%Y").dt.normalize(),
                    patients[COLUMN.anchor_age].astype(int),
                )
            ),
            index=patients.index,
        )


class StaticTableResource_MIMICIII(StaticTableResource):
    @classmethod
    def derive_deshifted_date_of_birth(cls, patients: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Important comment from MIMIC-III documentation at \
            https://mimic.mit.edu/docs/iii/tables/patients/
        > DOB is the date of birth of the given patient. Patients who are \
            older than 89 years old at any time in the database have had their\
            date of birth shifted to obscure their age and comply with HIPAA.\
            The shift process was as follows: the patient’s age at their \
            first admission was determined. The date of birth was then set to\
            exactly 300 years before their first admission.

        # TODO: check https://mimic.mit.edu/docs/iii/about/time/
        """
        assert "admissions" in kwargs, "Should pass the admissions table as keyword argument."
        admissions = kwargs.pop("admissions").loc[:, [COLUMN.subject_id, COLUMN.start_time, COLUMN.end_time]]
        admissions[COLUMN.start_time] = pd.to_datetime(admissions[COLUMN.start_time])
        admissions[COLUMN.end_time] = pd.to_datetime(admissions[COLUMN.end_time])
        dob = pd.to_datetime(patients[COLUMN.date_of_birth])
        last_disch_date = admissions.groupby(COLUMN.subject_id)[COLUMN.end_time].max()
        first_admit_date = admissions.groupby(COLUMN.subject_id)[COLUMN.start_time].min()
        last_disch_date = last_disch_date.loc[patients[COLUMN.subject_id]]
        first_admit_date = first_admit_date.loc[patients[COLUMN.subject_id]]
        uncertainty = (last_disch_date.dt.year.values - first_admit_date.dt.year.values) // 2
        age_before_shift = uncertainty + 89
        deshifted = np.array(list(map(lambda dt, s: dt + pd.DateOffset(-s), first_admit_date, age_before_shift)))
        adjusted_dob = np.where((last_disch_date.dt.year.values - dob.dt.year.values) > 150, deshifted, dob.values)
        return pd.Series(pd.to_datetime(adjusted_dob), index=dob.index).dt.normalize()


class MixedVersionICDSummaryTableColumns(TableColumns):
    admission_id: str = str(COLUMN.admission_id)
    code: str = str(COLUMN.code)
    version: str = str(COLUMN.version)
    description: str = str(COLUMN.description)


class MixedICDTableResource(CodedTableResource):
    columns: MixedVersionICDSummaryTableColumns

    def __init__(self):
        super().__init__(MixedVersionICDSummaryTableColumns())

    def setup_schemes(
        self,
        manager: CodingSchemesManager,
        name: str,
        component_schemes: dict[str, str],
        infer_maps: tuple[str, ...],
        target_name: str | None,
        mapping: pd.DataFrame | None,
        selection: pd.DataFrame | None,
        *,
        c_code: str,
        c_version: str,
        c_target_code: str,
        c_target_desc: str,
    ) -> CodingSchemesManager:
        scheme = self.register_scheme(
            name=name,
            selection=selection,
            component_schemes={k: manager.scheme[v] for k, v in component_schemes.items()},
        )
        manager = manager.add_scheme(scheme)
        if target_name is not None and mapping is not None:
            manager = scheme.register_map(
                manager=manager,
                target_name=target_name,
                mapping=mapping,
                c_code=c_code,
                c_version=c_version,
                c_target_code=c_target_code,
                c_target_desc=c_target_desc,
            )
        for target in infer_maps:
            manager = scheme.register_infer_map(manager, target)
        return manager

    @staticmethod
    def register_scheme(
        name: str, component_schemes: dict[str, MultiVersionScheme], selection: pd.DataFrame | None
    ) -> MultiVersionScheme:
        return MultiVersionScheme.from_selection(name, selection, component_schemes=component_schemes)

    @staticmethod
    @abstractmethod
    def _add_version_column_if_not_exists(df: pd.DataFrame) -> pd.DataFrame:
        # This is specifically added for MIMIC-III, pure ICD-9 codes.
        raise NotImplementedError("Override this method in subclass")

    @classmethod
    def _coerce_version_to_str(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer codes in the database when downloaded are stored as floats or integers.
        A fix is to coerce them to integers then fix as strings.
        """
        return cls._coerce_columns_to_str(df, (str(COLUMN.version),))

    @staticmethod
    def _strip_icd_codes(df: pd.DataFrame) -> pd.DataFrame:
        df[COLUMN.code] = df[COLUMN.code].str.strip()
        return df

    @staticmethod
    def _mixed_code_format(
        mixed_scheme_name: str, schemes_manager: CodingSchemesManager
    ) -> Callable[[pd.DataFrame], pd.DataFrame]:
        scheme = cast(MultiVersionScheme, schemes_manager.scheme[mixed_scheme_name])

        def _transform(df: pd.DataFrame) -> pd.DataFrame:
            return scheme.mixed_code_format_table(schemes_manager, df)

        return _transform

    @property
    def pipeline(self) -> None:
        return None

    def space(self, data_connection: Any) -> pd.DataFrame:
        pipeline = (
            self._coerce_code_to_str,
            self._filter_null_codes,
            self._strip_icd_codes,
            self._add_version_column_if_not_exists,
            self._coerce_version_to_str,
        )
        return self.apply_pipeline(pipeline, self.load_space_table(data_connection))

    def _custom_pipeline(
        self, mixed_scheme_name: str, schemes_manager: CodingSchemesManager
    ) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return (
            self._coerce_id_to_str,
            self._filter_null_codes,
            self._coerce_code_to_str,
            self._strip_icd_codes,
            self._add_version_column_if_not_exists,
            self._coerce_version_to_str,
            self._mixed_code_format(mixed_scheme_name, schemes_manager),
        )

    def __call__(self, data_connection: Any, **kwargs) -> pd.DataFrame:
        mixed_scheme_name = kwargs.pop("mixed_scheme_name")
        schemes_manager = kwargs.pop("schemes_manager")
        pipeline = self._custom_pipeline(mixed_scheme_name, schemes_manager)
        return self.apply_pipeline(pipeline, self.load_standard_columns_table(data_connection, **kwargs))


class MixedICDTableResource_MIMICIII(MixedICDTableResource):
    @staticmethod
    def _add_version_column_if_not_exists(df: pd.DataFrame) -> pd.DataFrame:
        assert str(COLUMN.version) not in df
        df = df.iloc[:]
        df[COLUMN.version] = "9"
        return df


class MixedICDTableResource_MIMICIV(MixedICDTableResource):
    @staticmethod
    def _add_version_column_if_not_exists(df: pd.DataFrame) -> pd.DataFrame:
        assert str(COLUMN.version) in df.columns
        return df


class MultivariateTimeSeriesTableResource(CodedTableResource):
    config: MultivariateTimeSeriesTableMeta
    columns: AdmissionTimeSeriesTableColumns

    def __init__(self, config: MultivariateTimeSeriesTableMeta):
        super().__init__(AdmissionTimeSeriesTableColumns())
        self.config = config

    @staticmethod
    def _validate_columns(table: pd.DataFrame, attributes: tuple[str, ...]) -> pd.DataFrame:
        # Perform any necessary validations on the table.
        assert all(attr not in COLUMN for attr in attributes), f"Attributes {attributes} must not be in COLUMN enum."
        assert len(set(attributes)) == len(attributes), f"Duplicate attributes {attributes} found."
        assert all(c in COLUMN for c in set(table.columns) - set(attributes)), (
            f"Some columns {set(table.columns) - set(attributes)} are not in COLUMN enum."
        )
        return table

    @staticmethod
    def _melt_attributes(table: pd.DataFrame) -> pd.DataFrame:
        melted_obs_df = table.melt(
            id_vars=[str(COLUMN.admission_id), str(COLUMN.time)],
            var_name=str(COLUMN.code),
            value_name=str(COLUMN.measurement),
        )
        return melted_obs_df[melted_obs_df[COLUMN.measurement].notnull()]

    @staticmethod
    def _coerce_value_to_real(df: pd.DataFrame) -> pd.DataFrame:
        """
        Some values in the measurement column were found befoer to be stored and loaded as strings.
        """
        return df.astype({str(COLUMN.measurement): float})

    def _rename_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        df[COLUMN.code] = pd.Series([self.config.name] * len(df)) + "." + df[COLUMN.code]
        return df

    @property
    def pipeline(self) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return (
            lambda df: self._validate_columns(df, self.config.attributes),
            self._coerce_id_to_str,
            self._melt_attributes,
            self._rename_attributes,
            self._coerce_code_to_str,
            self._coerce_value_to_real,
        )

    def load_space_table(self, data_sourece: None = None) -> pd.DataFrame:
        space = pd.DataFrame(
            [(self.config.name, a, self.config.type_hint[i]) for i, a in enumerate(self.config.attributes)],
            columns=["group", "attribute", "type_hint"],
        )
        space["code"] = space["group"] + "." + space["attribute"]
        return space.sort_values("code")

    def space(self, data_connection: None = None):
        # to skip applying pipeline
        return self.load_space_table(None)


class GroupedMultivariateTimeSeriesTableResource(CodedTableResource):
    groups: tuple[MultivariateTimeSeriesTableResource, ...]

    def __init__(self, groups: tuple[MultivariateTimeSeriesTableResource, ...]):
        super().__init__(AdmissionTimeSeriesTableColumns())
        self.groups = groups

    @staticmethod
    def _time_stat(code_table: pd.DataFrame) -> pd.Series:
        c_admission_id = str(COLUMN.admission_id)
        c_time = str(COLUMN.time)
        timestamps = code_table[[c_admission_id, c_time]].sort_values([c_admission_id, c_time])
        time_deltas = (timestamps[c_time].diff().dt.total_seconds() * SECONDS_TO_HOURS_SCALER).iloc[1:]
        in_admission = pd.Series(timestamps[c_admission_id] == timestamps[c_admission_id].shift()).iloc[1:]
        time_deltas_stats = time_deltas[in_admission].describe()
        return time_deltas_stats.rename(index={k: f"time_delta_{k}" for k in time_deltas_stats.index})

    @staticmethod
    def _stats(code: str, code_table: pd.DataFrame) -> pd.DataFrame:
        values = code_table[COLUMN.measurement]
        stats = values.describe()
        stats["nunique"] = values.nunique()
        time_stats = GroupedMultivariateTimeSeriesTableResource._time_stat(code_table)
        stats = pd.concat([stats, time_stats])
        stats = stats.to_frame().T
        stats[COLUMN.code] = code
        return stats.set_index(str(COLUMN.code))

    def load_standard_columns_table(self, data_connection: Any, **kwargs) -> pd.DataFrame:
        return pd.concat([g(data_connection, **kwargs) for g in self.groups], axis=0)

    @property
    def pipeline(self) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return tuple([lambda df: df.reset_index(drop=True)])

    def stats(self, data_connection: Any) -> pd.DataFrame:
        dfs = []
        for g in self.groups:
            for code, code_table in g(data_connection).groupby(str(COLUMN.code)):
                dfs.append(self._stats(code, code_table))
        return pd.concat(dfs, axis=0)

    def load_space_table(self, data_sourece: None = None) -> pd.DataFrame:
        return pd.concat([c.space(None) for c in self.groups]).sort_values(["code"])

    def space(self, data_connection: None = None) -> pd.DataFrame:
        return self.load_space_table(None)

    @staticmethod
    def register_scheme(
        name: str, space_table: pd.DataFrame, attributes_selection: pd.DataFrame | None
    ) -> CodingSchemesManager:
        if attributes_selection is None:
            attributes_selection = space_table
        else:
            if "type_hint" not in attributes_selection.columns:
                attributes_selection = pd.merge(
                    attributes_selection,
                    space_table,
                    left_on=["group", "attribute"],
                    right_on=["group", "attribute"],
                    suffixes=(None, "_y"),
                    how="inner",
                )

        # format codes to be of the form 'group.attribute'
        df = attributes_selection.astype({"attribute": str, "type_hint": str, "group": str})
        codes = tuple(df["group"] + "." + df["attribute"].tolist())
        desc = FrozenDict11(dict(zip(codes, codes)))
        group = FrozenDict11(dict(zip(codes, df.index.astype(str).values)))
        type_hint = FrozenDict11(dict(zip(codes, df["type_hint"].tolist())))
        scheme = NumericScheme(name=name, codes=tuple(sorted(codes)), group=group, desc=desc, type_hint=type_hint)  # noqa
        return CodingSchemesManager().add_scheme(scheme)


class DatasetSchemeMapsFileNames(AbstractConfig):
    gender: str | None = "gender.csv"
    ethnicity: str | None = "ethnicity.csv"
    icu_inputs: str | None = "icu_inputs.csv"
    icu_procedures: str | None = "icu_procedures.csv"
    hosp_procedures: str | None = "hosp_procedures.csv"
    dx_discharge: str | None = "dx_discharge.csv"


class ExternalMapResources(AbstractConfig):
    filenames: DatasetSchemeMapsFileNames
    resources_dir: str

    def __init__(self, resources_dir: str, filenames: DatasetSchemeMapsFileNames = DatasetSchemeMapsFileNames()):
        self.filenames = filenames
        self.resources_dir = resources_dir

    def map_file(self, filename: str) -> pd.DataFrame | None:
        try:
            return pd.read_csv(resources_path(self.resources_dir, filename)).astype(str)
        except FileNotFoundError:
            return None

    @property
    def dx_discharge(self) -> pd.DataFrame | None:
        return self.map_file(self.filenames.dx_discharge)

    @property
    def gender(self) -> pd.DataFrame | None:
        return self.map_file(self.filenames.gender)

    @property
    def ethnicity(self) -> pd.DataFrame | None:
        return self.map_file(self.filenames.ethnicity)

    @property
    def icu_inputs(self) -> pd.DataFrame | None:
        return self.map_file(self.filenames.icu_inputs)

    @property
    def icu_procedures(self) -> pd.DataFrame | None:
        return self.map_file(self.filenames.icu_procedures)

    @property
    def hosp_procedures(self) -> pd.DataFrame | None:
        return self.map_file(self.filenames.hosp_procedures)


class DatasetSchemeSelectionFiles(AbstractConfig):
    gender: str | None = "gender.csv"
    ethnicity: str | None = "ethnicity.csv"
    icu_inputs: str | None = "icu_inputs.csv"
    icu_procedures: str | None = "icu_procedures.csv"
    hosp_procedures: str | None = "hosp_procedures.csv"
    obs: str | None = "obs.csv"
    dx_discharge: str | None = "dx_discharge.csv"


class ExternalSelectionResources(AbstractConfig):
    filenames: DatasetSchemeSelectionFiles
    resources_dir: str

    def __init__(self, resources_dir: str, filenames: DatasetSchemeSelectionFiles = DatasetSchemeSelectionFiles()):
        self.filenames = filenames
        self.resources_dir = resources_dir

    def selection_file(self, path: str) -> pd.DataFrame | None:
        try:
            return pd.read_csv(resources_path(self.resources_dir, path)).astype(str)
        except FileNotFoundError:
            return None

    @property
    def gender(self) -> pd.DataFrame | None:
        return self.selection_file(self.filenames.gender)

    @property
    def ethnicity(self) -> pd.DataFrame | None:
        return self.selection_file(self.filenames.ethnicity)

    @property
    def icu_inputs(self) -> pd.DataFrame | None:
        return self.selection_file(self.filenames.icu_inputs)

    @property
    def icu_procedures(self) -> pd.DataFrame | None:
        return self.selection_file(self.filenames.icu_procedures)

    @property
    def hosp_procedures(self) -> pd.DataFrame | None:
        return self.selection_file(self.filenames.hosp_procedures)

    @property
    def obs(self) -> pd.DataFrame | None:
        df = self.selection_file(self.filenames.obs)
        return df.set_index("group", drop=True).sort_values("attribute").sort_index()

    @property
    def dx_discharge(self) -> pd.DataFrame | None:
        return self.selection_file(self.filenames.dx_discharge)


class MIMICDatasetSchemeSuffixes(AbstractConfig):
    gender: str | None
    ethnicity: str | None
    dx_discharge: str | None
    obs: str | None
    icu_inputs: str | None
    icu_procedures: str | None
    hosp_procedures: str | None

    def __init__(
        self,
        gender: str | None = None,
        ethnicity: str | None = None,
        dx_discharge: str | None = None,
        obs: str | None = None,
        icu_inputs: str | None = None,
        icu_procedures: str | None = None,
        hosp_procedures: str | None = None,
    ):
        self.gender = gender
        self.ethnicity = ethnicity
        self.dx_discharge = dx_discharge
        self.obs = obs
        self.icu_inputs = icu_inputs
        self.icu_procedures = icu_procedures
        self.hosp_procedures = hosp_procedures


class ScopedSchemeNames(AbstractConfig):
    suffixes: MIMICDatasetSchemeSuffixes
    name_separator: str
    name_prefix: str
    global_infix: tuple[str, ...] = ()

    def __init__(
        self,
        name_separator: str = ".",
        name_prefix: str = "",
        suffixes: MIMICDatasetSchemeSuffixes = MIMICDatasetSchemeSuffixes(),
        global_infix: tuple[str, ...] = (),
    ):
        self.suffixes = suffixes
        self.name_separator = name_separator
        self.name_prefix = name_prefix
        self.global_infix = global_infix

    def _scheme_name(self, k: str) -> str | None:
        suffix = getattr(self.suffixes, k)
        if suffix is None:
            return None
        return self.name_separator.join((self.name_prefix,) + self.global_infix + (suffix,))

    @property
    def gender(self) -> str | None:
        return self._scheme_name("gender")

    @property
    def ethnicity(self) -> str | None:
        return self._scheme_name("ethnicity")

    @property
    def icu_inputs(self) -> str | None:
        return self._scheme_name("icu_inputs")

    @property
    def icu_procedures(self) -> str | None:
        return self._scheme_name("icu_procedures")

    @property
    def hosp_procedures(self) -> str | None:
        return self._scheme_name("hosp_procedures")

    @property
    def dx_discharge(self) -> str | None:
        return self._scheme_name("dx_discharge")

    @property
    def obs(self) -> str | None:
        return self._scheme_name("obs")

    @property
    def mapped(self) -> Self:
        return eqx.tree_at(lambda x: x.global_infix, self, self.global_infix + ("mapped",))

    def column_name(self, key: str) -> str:
        return self.name_separator.join(self.global_infix + (key,))


class MIMICDatasetAuxiliaryResources(AbstractConfig):
    scoped_names: ScopedSchemeNames
    maps: ExternalMapResources
    selections: ExternalSelectionResources
    icu_inputs_uom_normalization: str | None
    icu_inputs_aggregation_column: str | None

    def __init__(
        self,
        maps: ExternalMapResources,
        selections: ExternalSelectionResources,
        scoped_names: ScopedSchemeNames = ScopedSchemeNames(),
        icu_inputs_uom_normalization: str | None = None,
        icu_inputs_aggregation_column: str | None = None,
    ):
        self.scoped_names = scoped_names
        self.maps = maps
        self.selections = selections
        self.icu_inputs_uom_normalization = icu_inputs_uom_normalization
        self.icu_inputs_aggregation_column = icu_inputs_aggregation_column

    @classmethod
    def make_resources(
        cls,
        name_prefix: str,
        resources_root: str,
        suffixes: MIMICDatasetSchemeSuffixes = MIMICDatasetSchemeSuffixes(),
        name_separator: str = ".",
        selection_subdir: str = "selection",
        map_subdir: str = "map",
        map_files: DatasetSchemeMapsFileNames = DatasetSchemeMapsFileNames(),
        selection_files: DatasetSchemeSelectionFiles = DatasetSchemeSelectionFiles(),
        icu_inputs_uom_normalization: tuple[str, ...] | None = None,
        icu_inputs_aggregation_column: str | None = None,
    ):
        scoped_names = ScopedSchemeNames(suffixes=suffixes, name_separator=name_separator, name_prefix=name_prefix)
        maps = ExternalMapResources(resources_path(resources_root, map_subdir), filenames=map_files)
        selections = ExternalSelectionResources(
            resources_path(resources_root, selection_subdir), filenames=selection_files
        )
        if icu_inputs_uom_normalization is not None:
            icu_inputs_uom_normalization = resources_path(resources_root, *icu_inputs_uom_normalization)

        icu_inputs_aggregation_column = icu_inputs_aggregation_column
        return cls(
            scoped_names=scoped_names,
            maps=maps,
            selections=selections,
            icu_inputs_uom_normalization=icu_inputs_uom_normalization,
            icu_inputs_aggregation_column=icu_inputs_aggregation_column,
        )

    @property
    def icu_inputs_uom_normalization_table(self) -> pd.DataFrame:
        assert self.icu_inputs_uom_normalization is not None, "icu_inputs_uom_normalization is not set"
        return pd.read_csv(self.icu_inputs_uom_normalization).astype(str)


class DatasetTablesResources(AbstractConfig):
    static: StaticTableResource
    admissions: TableResource
    dx_discharge: MixedICDTableResource
    obs: GroupedMultivariateTimeSeriesTableResource
    hosp_procedures: MixedICDTableResource
    icu_procedures: CodedTableResource
    icu_inputs: CodedTableResource


class MIMICSchemeResources(AbstractConfig):
    tables: DatasetTablesResources
    scheme: DatasetSchemeConfig
    aux: MIMICDatasetAuxiliaryResources

    def __init__(
        self, tables: DatasetTablesResources, scheme: DatasetSchemeConfig, aux: MIMICDatasetAuxiliaryResources
    ):
        self.scheme = scheme
        self.tables = tables
        self.aux = aux

    def _make_demographic_scheme(
        self,
        name: str,
        space_table: pd.DataFrame,
        c_code: str,
        selection: pd.DataFrame,
        target_name: str | None = None,
        map_table: pd.DataFrame | None = None,
    ) -> CodingSchemesManager:
        # TODO: handle missing values. Options:
        # 1. A missingness-aware CodingScheme (not preferred, requires new class, new logic, new tests).
        # 2. Hard-code replacement here (preferred, just replace 'nan' with 'MISSING').
        source_scheme = CodingScheme.from_table(
            name=name, table=space_table, code_selection=selection, c_code=c_code, c_desc=c_code
        )
        manager = CodingSchemesManager().add_scheme(source_scheme)
        if map_table is not None and target_name is not None:
            names = self.aux.scoped_names
            target_scheme = CodingScheme.from_table(
                name=target_name,
                table=map_table,
                c_code=names.mapped.column_name(c_code),
                c_desc=names.mapped.column_name(c_code),
            )
            code_map = CodeMap.from_table(
                source_scheme,
                target_scheme,
                c_source_code=c_code,
                c_target_code=names.mapped.column_name(c_code),
                table=map_table,
            )
            manager = manager.add_scheme(target_scheme).add_map(code_map)
        return manager

    def make_gender_scheme(self, data_connection: Any) -> CodingSchemesManager:
        gender_space_table = self.tables.static.gender_space(data_connection)
        return self._make_demographic_scheme(
            name=self.scheme.gender,
            space_table=gender_space_table,
            c_code=COLUMN.gender,
            selection=self.aux.selections.gender,
        )

    def make_ethnicity_scheme(self, data_connection: Any) -> CodingSchemesManager:
        race_space_table = self.tables.static.ethnicity_space(data_connection)
        return self._make_demographic_scheme(
            name=self.scheme.ethnicity,
            space_table=race_space_table,
            c_code=COLUMN.race,
            selection=self.aux.selections.ethnicity,
            target_name=self.aux.scoped_names.mapped.ethnicity,
            map_table=self.aux.maps.ethnicity,
        )

    def make_obs_scheme(self) -> CodingSchemesManager:
        obs = self.tables.obs
        return obs.register_scheme(
            name=self.scheme.obs, space_table=obs.space(), attributes_selection=self.aux.selections.obs
        )

    def make_icu_inputs_scheme(self) -> CodingSchemesManager:
        c_code = str(COLUMN.code)
        c_desc = str(COLUMN.description)
        scheme = CodingSchemeWithUOM.from_table(
            name=self.scheme.icu_inputs,
            table=self.aux.icu_inputs_uom_normalization_table,
            c_code=c_code,
            c_desc=c_desc,
            c_universal_unit=str(COLUMN.derived_universal_unit),
            c_unit=str(COLUMN.amount_unit),
            c_normalization_factor=str(COLUMN.derived_unit_normalization_factor),
            code_selection=self.aux.selections.icu_inputs,
        )
        manager = CodingSchemesManager().add_scheme(scheme)
        c_aggregation = self.aux.icu_inputs_aggregation_column
        map_data = self.aux.maps.icu_inputs
        if map_data is not None and c_aggregation is not None:
            target_names = self.aux.scoped_names.mapped
            c_target_code = target_names.column_name(c_code)
            c_target_desc = target_names.column_name(c_desc)
            target_scheme = CodingScheme.from_table(
                name=target_names.icu_inputs, table=map_data, c_code=c_target_code, c_desc=c_target_desc
            )
            codemap = ReducedCodeMapN1.from_table(
                scheme,
                target_scheme,
                c_source_code=c_code,
                c_target_code=c_target_code,
                c_target_agg=c_aggregation,
                table=map_data,
            )
            manager = manager.add_scheme(target_scheme).add_map(codemap)

        return manager

    def make_icu_procedures_scheme(self, data_connection: Any) -> CodingSchemesManager:
        space_table = self.tables.icu_procedures.space(data_connection)
        source_scheme = CodingScheme.from_table(
            name=self.scheme.icu_procedures,
            table=space_table,
            code_selection=self.aux.selections.icu_procedures,
            c_code=str(COLUMN.code),
            c_desc=str(COLUMN.description),
        )
        manager = CodingSchemesManager().add_scheme(source_scheme)
        map_table = self.aux.maps.icu_procedures
        if map_table is not None:
            target_names = self.aux.scoped_names.mapped
            target_scheme = CodingScheme.from_table(
                name=target_names.icu_procedures,
                table=map_table,
                c_code=target_names.column_name(str(COLUMN.code)),
                c_desc=target_names.column_name(str(COLUMN.description)),
            )
            code_map = CodeMap.from_table(
                source_scheme,
                target_scheme,
                c_source_code=str(COLUMN.code),
                c_target_code=target_names.column_name(str(COLUMN.code)),
                table=map_table,
            )
            manager = manager.add_scheme(target_scheme).add_map(code_map)
        return manager

    def make_hosp_procedures_scheme(self, manager: CodingSchemesManager) -> CodingSchemesManager:
        table = self.tables.hosp_procedures
        name = self.scheme.hosp_procedures
        target_name = self.aux.scoped_names.mapped.hosp_procedures
        mapping = self.aux.maps.hosp_procedures
        selection = self.aux.selections.hosp_procedures
        c_target_code = self.aux.scoped_names.mapped.column_name(str(COLUMN.code))
        c_target_desc = self.aux.scoped_names.mapped.column_name(str(COLUMN.description))
        return table.setup_schemes(
            manager,
            name=name,
            component_schemes={"9": "icd9pcs", "10": "icd10pcs"},
            infer_maps=("icd9pcs", "icd10pcs", "pr_ccs", "pr_flat_ccs"),
            target_name=target_name,
            mapping=mapping,
            selection=selection,
            c_code=COLUMN.code,
            c_version=COLUMN.version,
            c_target_code=c_target_code,
            c_target_desc=c_target_desc,
        )

    def make_dx_discharge_scheme(self, manager: CodingSchemesManager) -> CodingSchemesManager:
        table = self.tables.dx_discharge
        name = self.scheme.dx_discharge
        target_name = self.aux.scoped_names.mapped.dx_discharge
        mapping = self.aux.maps.dx_discharge
        selection = self.aux.selections.dx_discharge
        c_target_code = self.aux.scoped_names.mapped.column_name(str(COLUMN.code))
        c_target_desc = self.aux.scoped_names.mapped.column_name(str(COLUMN.description))
        m = table.setup_schemes(
            manager,
            name=name,
            component_schemes={"9": "icd9cm", "10": "icd10cm"},
            infer_maps=(
                "icd9cm",
                "icd10cm",  # 'dx_ccs',
                "dx_flat_ccs",
            ),
            target_name=target_name,
            mapping=mapping,
            selection=selection,
            c_code=COLUMN.code,
            c_version=COLUMN.version,
            c_target_code=c_target_code,
            c_target_desc=c_target_desc,
        )
        # OVERRIDE the low-quality mapping icd10cm->dx_ccs with a chained map icd10cm->icd9cm->dx_ccs
        m = m.add_chained_map(name, "icd9cm", "dx_ccs", overwrite=True)
        return m

    def make_all_schemes(self, data_connection: Any) -> CodingSchemesManager:
        # make standard ones.
        manager = setup_standard_icd_ccs()
        if self.scheme.hosp_procedures is not None:
            manager += self.make_hosp_procedures_scheme(manager)
        if self.scheme.dx_discharge is not None:
            manager += self.make_dx_discharge_scheme(manager)
        if self.scheme.gender is not None:
            manager += self.make_gender_scheme(data_connection)
        if self.scheme.ethnicity is not None:
            manager += self.make_ethnicity_scheme(data_connection)
        if self.scheme.icu_inputs is not None:
            manager += self.make_icu_inputs_scheme()
        if self.scheme.icu_procedures is not None:
            manager += self.make_icu_procedures_scheme(data_connection)
        if self.scheme.obs is not None:
            manager += self.make_obs_scheme()
        return manager


class MIMICResourceExploratory(AbstractConfig):
    tables: DatasetTablesResources

    def __init__(self, tables: DatasetTablesResources):
        self.tables = tables

    def supported_gender(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.static.gender_space(data_connection)

    def supported_ethnicity(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.static.ethnicity_space(data_connection)

    def obs_stats(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.obs.stats(data_connection)

    def supported_obs_variables(self, data_connection: None = None) -> pd.DataFrame:
        return self.tables.obs.space(data_connection)

    def supported_icu_procedures(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.icu_procedures.space(data_connection)

    def supported_icu_inputs(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.icu_inputs.space(data_connection)

    def supported_hosp_procedures(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.hosp_procedures.space(data_connection)

    def supported_dx_discharge(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.dx_discharge.space(data_connection)


class MIMICDatasetCompiler(AbstractConfig):
    tables: DatasetTablesResources
    scheme: DatasetSchemeConfig

    def __init__(self, tables: DatasetTablesResources, scheme: DatasetSchemeConfig):
        self.tables = tables
        self.scheme = scheme

    def load_static(self, data_connection: Any, admissions: pd.DataFrame) -> pd.DataFrame:
        return self.tables.static(data_connection, admissions=admissions)

    def load_admissions(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.admissions(data_connection)

    def load_dx_discharge(self, data_connection: Any, schemes_manager: CodingSchemesManager) -> pd.DataFrame:
        table = self.tables.dx_discharge
        return table(data_connection, schemes_manager=schemes_manager, mixed_scheme_name=self.scheme.dx_discharge)

    def load_obs(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.obs(data_connection)

    def load_icu_procedures(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.icu_procedures(data_connection)

    def load_icu_inputs(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.icu_inputs(data_connection)

    def load_hosp_procedures(self, data_connection: Any, schemes_manager: CodingSchemesManager) -> pd.DataFrame:
        table = self.tables.hosp_procedures
        return table(data_connection, schemes_manager=schemes_manager, mixed_scheme_name=self.scheme.hosp_procedures)

    def load_tables(self, data_connection: Any, schemes_manager: CodingSchemesManager) -> DatasetTables:
        S = self.scheme
        hosp_procedures = self.load_hosp_procedures(data_connection, schemes_manager) if S.hosp_procedures else None
        icu_procedures = self.load_icu_procedures(data_connection) if S.icu_procedures else None
        icu_inputs = self.load_icu_inputs(data_connection) if S.icu_inputs else None
        obs = self.load_obs(data_connection) if S.obs else None
        admissions = self.load_admissions(data_connection)
        static = self.load_static(data_connection, admissions=admissions)
        dx_discharge = self.load_dx_discharge(data_connection, schemes_manager)
        return DatasetTables(
            static=static,
            admissions=admissions,
            dx_discharge=dx_discharge,
            obs=obs,
            icu_procedures=icu_procedures,
            icu_inputs=icu_inputs,
            hosp_procedures=hosp_procedures,
        )


def load_scheme_manager(
    tables: DatasetTablesResources,
    scheme: DatasetSchemeConfig,
    aux: MIMICDatasetAuxiliaryResources,
    data_connection: Any,
) -> CodingSchemesManager:
    schemes_resources = MIMICSchemeResources(tables=tables, scheme=scheme, aux=aux)
    return schemes_resources.make_all_schemes(data_connection=data_connection)


def load_tables(
    tables: DatasetTablesResources,
    scheme: DatasetSchemeConfig,
    schemes_manager: CodingSchemesManager,
    data_connection: Any,
) -> DatasetTables:
    compiler = MIMICDatasetCompiler(tables=tables, scheme=scheme)
    return compiler.load_tables(data_connection=data_connection, schemes_manager=schemes_manager)


def load_mimic(
    config: DatasetConfig, tables: DatasetTablesResources, aux: MIMICDatasetAuxiliaryResources, data_connection: Any
) -> tuple[Dataset, CodingSchemesManager]:
    manager = load_scheme_manager(tables, config.scheme, aux, data_connection)
    tables = load_tables(tables, config.scheme, manager, data_connection)
    return Dataset(tables=tables, config=config), manager
