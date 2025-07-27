"""."""

import warnings
from abc import abstractmethod
from typing import Optional, Iterable, Any, Callable, Self

import equinox as eqx
import pandas as pd

from ..base import AbstractConfig
from ..coding_scheme import (CodingScheme, resources_dir, CodingSchemesManager, FrozenDict11, CodingSchemeWithUOM,
                             ReducedCodeMapN1, NumericScheme, CodeMap)
from ..dataset import COLUMN, SECONDS_TO_HOURS_SCALER, AdmissionSummaryTableColumns, \
    AdmissionIntervalEventsTableColumns, AdmissionIntervalRatesTableColumns, AdmissionTimeSeriesTableColumns, \
    include_cols, MultivariateTimeSeriesTableMeta, DatasetConfig
from ..dataset import (StaticTableColumns,
                       TableColumns,
                       DatasetTables, DatasetSchemeConfig, Dataset, AbstractDatasetPipelineConfig)
from ..example_schemes.icd import setup_standard_icd_ccs, CCSICDSchemeSelection, CCSICDOutcomeSelection
from ..example_schemes.mimiciv_icd import MixedICDScheme

warnings.filterwarnings('error', category=RuntimeWarning, message=r'overflow encountered in cast')


class TableResource(AbstractConfig):
    columns: TableColumns

    def __init__(self, columns: TableColumns):
        self.columns = columns

    # TODO: Document this class.
    @staticmethod
    def _coerce_columns_to_str(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        # TODO: test this method.
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
        return self._coerce_columns_to_str(df, self.columns.id_dict.values())

    def _coerce_code_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer codes in the database when downloaded are stored as floats or integers.
        A fix is to coerce them to integers then fix as strings.
        """
        return self._coerce_columns_to_str(df, self.columns.code_cols)

    @property
    def pipeline(self) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return (self._coerce_id_to_str,)

    def preprocess(self, table):
        for f in self.pipeline:
            table = f(table)
        return table

    @abstractmethod
    def load_standard_columns_table(self, data_connection: Any, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()

    def __call__(self, data_connection: Any, *args, **kwargs) -> pd.DataFrame:
        return self.preprocess(self.load_standard_columns_table(data_connection, *args, **kwargs))


CodedColumns = AdmissionSummaryTableColumns | AdmissionTimeSeriesTableColumns | AdmissionIntervalEventsTableColumns | AdmissionIntervalRatesTableColumns


class CodedTableResource(TableResource):

    def __check_init__(self):
        assert all(c in self.columns for c in (COLUMN.code, COLUMN.description))

    @property
    def pipeline(self) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return (self._coerce_id_to_str, self._coerce_code_to_str)

    def space(self, data_connection: Any) -> pd.DataFrame:
        return self.preprocess(self.load_space_table(data_connection))

    @abstractmethod
    def load_space_table(self, data_connection: Any):
        raise NotImplementedError("This method should be implemented in subclasses.")


class StaticTableResource(TableResource):
    columns: StaticTableColumns

    def __init__(self):
        super().__init__(StaticTableColumns())

    @abstractmethod
    def load_gender_space_table(self, data_connection: Any) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def load_ethnicity_space_table(self, data_connection: Any) -> pd.DataFrame:
        raise NotImplementedError()

    def _derive_data_of_birth(self, df: pd.DataFrame) -> pd.DataFrame:
        anchor_date = pd.to_datetime(df[str(COLUMN.anchor_year)], format='%Y').dt.normalize()
        anchor_age = df[str(COLUMN.anchor_age)].map(lambda y: pd.DateOffset(years=-y))
        df[str(COLUMN.date_of_birth)] = anchor_date + anchor_age
        return df

    def pipeline(self) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return (self._coerce_id_to_str, self._derive_data_of_birth)

    def gender_space(self, date_source: Any) -> pd.DataFrame:
        return self.load_gender_space_table(date_source)

    def ethnicity_space(self, data_connection: Any) -> pd.DataFrame:
        return self.load_ethnicity_space_table(data_connection)


@include_cols(COLUMN.admission_id, COLUMN.code, COLUMN.version, COLUMN.description)
class MixedVersionICDSummaryTableColumns(TableColumns):
    pass


class MixedICDTableResource(CodedTableResource):
    columns: MixedVersionICDSummaryTableColumns

    def __init__(self):
        super().__init__(MixedVersionICDSummaryTableColumns())

    @staticmethod
    def _register_scheme(manager: CodingSchemesManager,
                         name: str,
                         icd_version_schemes: FrozenDict11,
                         supported_space: pd.DataFrame,
                         icd_version_selection: Optional[pd.DataFrame]) -> CodingSchemesManager:
        c_code = str(COLUMN.code)
        c_version = str(COLUMN.version)
        c_desc = str(COLUMN.description)

        # TODO: test this method.
        if icd_version_selection is None:
            icd_version_selection = supported_space[[c_version, c_code, c_desc]].drop_duplicates()
            icd_version_selection = icd_version_selection.astype(str)
        else:
            if c_desc not in icd_version_selection.columns:
                icd_version_selection = pd.merge(icd_version_selection,
                                                 supported_space[[c_version, c_code, c_desc]],
                                                 on=[c_version, c_code], how='left')
            icd_version_selection = icd_version_selection[[c_version, c_code, c_desc]]
            icd_version_selection = icd_version_selection.drop_duplicates()
            icd_version_selection = icd_version_selection.astype(str)
            for version, codes in icd_version_selection.groupby(c_version):
                support_subset = supported_space[supported_space[c_version] == version]
                unsupported_codes = codes[~codes[c_code].isin(support_subset[c_code])]

                assert len(unsupported_codes) == 0, f'Codes {unsupported_codes} are not supported for version {version}'

        manager = manager.add_scheme(MixedICDScheme.from_selection(manager, name, icd_version_selection,
                                                                   icd_version_schemes=icd_version_schemes))
        scheme: MixedICDScheme = manager.scheme[name]
        return scheme.register_standard_icd_maps(manager)

    def register_scheme(self, manager: CodingSchemesManager,
                        name: str,
                        space_table: pd.DataFrame,
                        icd_version_schemes: FrozenDict11,
                        icd_version_selection: pd.DataFrame,
                        target_name: Optional[str],
                        mapping: Optional[pd.DataFrame]) -> CodingSchemesManager:
        manager = self._register_scheme(manager=manager,
                                        name=name, icd_version_schemes=icd_version_schemes,
                                        supported_space=space_table,
                                        icd_version_selection=icd_version_selection)
        if target_name is not None and mapping is not None:
            mixed_icd_scheme: MixedICDScheme = manager.scheme[name]
            manager = mixed_icd_scheme.register_map(manager=manager, target_name=target_name, mapping=mapping)
        return manager

    def _coerce_version_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the integer codes in the database when downloaded are stored as floats or integers.
        A fix is to coerce them to integers then fix as strings.
        """
        return self._coerce_columns_to_str(df, (str(COLUMN.version),))

    def _strip_icd_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        df[str(COLUMN.code)] = df[str(COLUMN.code)].str.strip()
        return df

    @property
    def pipeline(self) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return super().pipeline + (self._strip_icd_codes, self._coerce_version_to_str)

    def __call__(self, data_connection: Any, schemes_manager: CodingSchemesManager, mixed_scheme_name: str, *args,
                 **kwargs):
        table = super()(data_connection, *args, **kwargs)
        scheme: MixedICDScheme = schemes_manager.scheme[mixed_scheme_name]
        return scheme.mixedcode_format_table(schemes_manager, table)


class MultivariateTimeSeriesTableResource(CodedTableResource):
    # TODO: Document this class.
    config: MultivariateTimeSeriesTableMeta
    columns: AdmissionTimeSeriesTableColumns

    def __init__(self, config: MultivariateTimeSeriesTableMeta):
        super().__init__(AdmissionTimeSeriesTableColumns())
        self.config = config

    @staticmethod
    def _validate_columns(table: pd.DataFrame, attributes: tuple[str, ...]) -> pd.DataFrame:
        # Perform any necessary validations on the table.
        assert all(attr not in COLUMN for attr in attributes), (f"Attributes {attributes} must not be in COLUMN enum.")
        assert len(set(attributes)) == len(attributes), (f"Duplicate attributes {attributes} found.")
        assert all(c in COLUMN for c in set(table.columns) - set(attributes)), (
            f"Some columns {set(table.columns) - set(attributes)} are not in COLUMN enum.")
        return table

    @staticmethod
    def _melt_attributes(table: pd.DataFrame) -> pd.DataFrame:
        melted_obs_df = table.melt(id_vars=[str(COLUMN.admission_id), str(COLUMN.time)],
                                   var_name=str(COLUMN.code), value_name=str(COLUMN.measurement))
        return melted_obs_df[melted_obs_df[str(COLUMN.measurement)].notnull()]

    def _coerce_value_to_real(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Some of the values in the measurement column might be stored as strings.
        """
        return df.astype({str(COLUMN.measurement): float})

    def _rename_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        df[str(COLUMN.code)] = pd.Series([self.config.name for _ in range(len(df))]) + '.' + df[str(COLUMN.code)]
        return df

    @property
    def pipeline(self) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return (lambda df: self._validate_columns(df, self.config.attributes),
                self._coerce_id_to_str, self._melt_attributes, self._rename_attributes,
                self._coerce_code_to_str, self._coerce_value_to_real)

    def load_space_table(self, data_sourece: None = None) -> pd.DataFrame:
        space = pd.DataFrame(
            [(self.config.name, a, self.config.type_hint[i]) for i, a in enumerate(self.config.attributes)],
            columns=['group', 'attribute', 'type_hint'])
        space['code'] = space['group'] + '.' + space['attribute']
        return space.sort_values('code')

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
        return time_deltas_stats.rename(index={k: f'time_delta_{k}' for k in time_deltas_stats.index})

    @staticmethod
    def _stats(code: str, code_table: pd.DataFrame) -> pd.DataFrame:
        values = code_table[str(COLUMN.measurement)]
        stats = values.describe()
        stats['nunique'] = values.nunique()
        time_stats = GroupedMultivariateTimeSeriesTableResource._time_stat(code_table)
        stats = pd.concat([stats, time_stats])
        stats = stats.to_frame().T
        stats[str(COLUMN.code)] = code
        return stats.set_index(str(COLUMN.code))

    def load_standard_columns_table(self, data_connection: Any, *args, **kwargs) -> pd.DataFrame:
        return pd.concat([g(data_connection, *args, **kwargs) for g in self.groups], axis=0)

    def pipeline(self) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], ...]:
        return (lambda df: df.reset_index(drop=True),)

    def stats(self, data_connection: Any) -> pd.DataFrame:
        dfs = []
        for g in self.groups:
            for code, code_table in g(data_connection).groupby(str(COLUMN.code)):
                dfs.append(self._stats(code, code_table))
        return pd.concat(dfs, axis=0)

    def load_space_table(self, data_sourece: None = None) -> pd.DataFrame:
        return pd.concat([c.space(None) for c in self.groups]).sort_values(['code'])

    def space(self, data_connection: None = None) -> pd.DataFrame:
        return self.load_space_table(None)

    def register_scheme(self,
                        name: str,
                        space_table: pd.DataFrame,
                        attributes_selection: Optional[pd.DataFrame]) -> CodingSchemesManager:
        if attributes_selection is None:
            attributes_selection = space_table
        else:
            if 'type' not in attributes_selection.columns:
                attributes_selection = pd.merge(attributes_selection, space_table,
                                                left_on=['group', 'attribute'],
                                                right_on=['group', 'attribute'],
                                                suffixes=(None, '_y'),
                                                how='inner')

        # format codes to be of the form 'table_name.attribute'
        df = attributes_selection.astype({'attribute': str, 'type_hint': str})
        codes = tuple(sorted(df.index + '.' + df['attribute'].tolist()))
        desc = FrozenDict11(dict(zip(codes, codes)))
        group = FrozenDict11(dict(zip(codes, df.index)))
        type_hint = FrozenDict11(dict(zip(codes, df['type_hint'].tolist())))
        return CodingSchemesManager().add_scheme(NumericScheme(name=name,
                                                               codes=codes,
                                                               group=group,
                                                               desc=desc,
                                                               type_hint=type_hint))


class DatasetSchemeMapsFileNames(AbstractConfig):
    gender: Optional[str] = 'gender.csv'
    ethnicity: Optional[str] = 'ethnicity.csv'
    icu_inputs: Optional[str] = 'icu_inputs.csv'
    icu_procedures: Optional[str] = 'icu_procedures.csv'
    hosp_procedures: Optional[str] = 'hosp_procedures.csv'
    dx_discharge: Optional[str] = 'dx_discharge.csv'


class ExternalMapResources(AbstractConfig):
    filenames: DatasetSchemeMapsFileNames
    resources_dir: str

    def __init__(self, resources_dir: str, filenames: DatasetSchemeMapsFileNames = DatasetSchemeMapsFileNames()):
        self.filenames = filenames
        self.resources_dir = resources_dir

    def map_file(self, filename: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(resources_dir(self.resources_dir, filename)).astype(str)
        except FileNotFoundError:
            return None

    @property
    def dx_discharge(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.filenames.dx_discharge)

    @property
    def gender(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.filenames.gender)

    @property
    def ethnicity(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.filenames.ethnicity)

    @property
    def icu_inputs(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.filenames.icu_inputs)

    @property
    def icu_procedures(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.filenames.icu_procedures)

    @property
    def hosp_procedures(self) -> Optional[pd.DataFrame]:
        return self.map_file(self.filenames.hosp_procedures)


class DatasetSchemeSelectionFiles(AbstractConfig):
    gender: Optional[str] = 'gender.csv'
    ethnicity: Optional[str] = 'ethnicity.csv'
    icu_inputs: Optional[str] = 'icu_inputs.csv'
    icu_procedures: Optional[str] = 'icu_procedures.csv'
    hosp_procedures: Optional[str] = 'hosp_procedures.csv'
    obs: Optional[str] = 'obs.csv'
    dx_discharge: Optional[str] = 'dx_discharge.csv'


class ExternalSelectionResources(AbstractConfig):
    filenames: DatasetSchemeSelectionFiles
    resources_dir: str

    def __init__(self, resources_dir: str, filenames: DatasetSchemeSelectionFiles = DatasetSchemeSelectionFiles()):
        self.filenames = filenames
        self.resources_dir = resources_dir

    def selection_file(self, path: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(resources_dir(self.resources_dir, path)).astype(str)
        except FileNotFoundError:
            return None

    @property
    def gender(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.filenames.gender)

    @property
    def ethnicity(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.filenames.ethnicity)

    @property
    def icu_inputs(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.filenames.icu_inputs)

    @property
    def icu_procedures(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.filenames.icu_procedures)

    @property
    def hosp_procedures(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.filenames.hosp_procedures)

    @property
    def obs(self) -> Optional[pd.DataFrame]:
        df = self.selection_file(self.filenames.obs)
        return df.set_index('table_name', drop=True).sort_values('attribute').sort_index()

    @property
    def dx_discharge(self) -> Optional[pd.DataFrame]:
        return self.selection_file(self.filenames.dx_discharge)


class MIMICDatasetSchemeSuffixes(AbstractConfig):
    gender: str = 'bin_gender'
    ethnicity: str = 'ethnicity'
    dx_discharge: str = 'dx_mixed_icd'
    obs: str = 'obs'
    icu_inputs: str = 'icu_inputs'
    icu_procedures: str = 'icu_procedures'
    hosp_procedures: str = 'pr_mixed_icd'


class ScopedSchemeNames(AbstractConfig):
    suffixes: MIMICDatasetSchemeSuffixes
    name_separator: str
    name_prefix: str
    global_suffix: tuple[str, ...] = ()

    def __init__(self, name_separator: str = '.', name_prefix: str = '',
                 suffixes: MIMICDatasetSchemeSuffixes = MIMICDatasetSchemeSuffixes(),
                 global_suffix: tuple[str, ...] = ()):
        self.suffixes = suffixes
        self.name_separator = name_separator
        self.name_prefix = name_prefix
        self.global_suffix = global_suffix

    def _scheme_name(self, k: str) -> str:
        return f'{self.name_prefix}{self.name_separator}{"_".join(self.global_suffix + (getattr(self.suffixes, k),))}'

    @property
    def gender(self) -> str:
        return self._scheme_name('gender')

    @property
    def ethnicity(self) -> str:
        return self._scheme_name('ethnicity')

    @property
    def icu_inputs(self) -> str:
        return self._scheme_name('icu_inputs')

    @property
    def icu_procedures(self) -> str:
        return self._scheme_name('icu_procedures')

    @property
    def hosp_procedures(self) -> str:
        return self._scheme_name('hosp_procedures')

    @property
    def dx_discharge(self) -> str:
        return self._scheme_name('dx_discharge')

    @property
    def obs(self) -> str:
        return self._scheme_name('obs')

    @property
    def target(self) -> Self:
        return eqx.tree_at(lambda x: x.global_suffix, self, 'mapped')

    def column_name(self, key: str) -> str:
        return '_'.join(self.global_suffix + (key,))


class MIMICDatasetAuxiliaryResources(AbstractConfig):
    scoped_names: ScopedSchemeNames
    maps: ExternalMapResources
    selections: ExternalSelectionResources
    icu_inputs_uom_normalization: Optional[str]
    icu_inputs_aggregation_column: Optional[str]

    def __init__(self, maps: ExternalMapResources,
                 selections: ExternalSelectionResources,
                 scoped_names: ScopedSchemeNames = ScopedSchemeNames(),
                 icu_inputs_uom_normalization: Optional[str] = None,
                 icu_inputs_aggregation_column: Optional[str] = None):
        self.scoped_names = scoped_names
        self.maps = maps
        self.selections = selections
        self.icu_inputs_uom_normalization = icu_inputs_uom_normalization
        self.icu_inputs_aggregation_column = icu_inputs_aggregation_column

    @classmethod
    def make_resources(cls, suffixes: MIMICDatasetSchemeSuffixes = MIMICDatasetSchemeSuffixes(),
                       name_separator: str = '.', name_prefix: str = 'mimiciv', resources_root: str = 'mimiciv',
                       selection_subdir: str = 'selection',
                       map_subdir: str = 'map', map_files: DatasetSchemeMapsFileNames = DatasetSchemeMapsFileNames(),
                       selection_files: DatasetSchemeSelectionFiles = DatasetSchemeSelectionFiles(),
                       icu_inputs_uom_normalization: Optional[tuple[str]] = ("uom_normalization", "icu_inputs.csv"),
                       icu_inputs_aggregation_column: Optional[str] = "aggregation"):
        scoped_names = ScopedSchemeNames(suffixes=suffixes, name_separator=name_separator, name_prefix=name_prefix)
        maps = ExternalMapResources(resources_dir(resources_root, map_subdir), filenames=map_files)
        selections = ExternalSelectionResources(resources_dir(resources_root, selection_subdir),
                                                filenames=selection_files)
        icu_inputs_uom_normalization = resources_dir(resources_root, *icu_inputs_uom_normalization)
        icu_inputs_aggregation_column = icu_inputs_aggregation_column
        return cls(scoped_names=scoped_names, maps=maps, selections=selections,
                   icu_inputs_uom_normalization=icu_inputs_uom_normalization,
                   icu_inputs_aggregation_column=icu_inputs_aggregation_column)

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
    # TODO: Document this class.
    tables: DatasetTablesResources
    scheme: DatasetSchemeConfig
    aux: MIMICDatasetAuxiliaryResources

    def __init__(self, tables: DatasetTablesResources, scheme: DatasetSchemeConfig,
                 aux: MIMICDatasetAuxiliaryResources):
        self.scheme = scheme
        self.tables = tables
        self.aux = aux

    def _make_demographic_scheme(self, name: str, space_table: pd.DataFrame,
                                 c_code: str, selection: pd.DataFrame, target_name: str,
                                 map_table: Optional[pd.DataFrame] = None) -> CodingSchemesManager:
        source_scheme = CodingScheme.from_table(name=name,
                                                table=space_table,
                                                code_selection=selection,
                                                c_code=c_code,
                                                c_desc=c_code)
        manager = CodingSchemesManager().add_scheme(source_scheme)
        if map_table is not None:
            names = self.aux.scoped_names
            target_scheme = CodingScheme.from_table(name=target_name, table=map_table,
                                                    c_code=names.target.column_name(c_code),
                                                    c_desc=names.target.column_name(c_code))
            code_map = CodeMap.from_table(source_scheme, target_scheme, c_source_code=c_code,
                                          c_target_code=names.target.column_name(c_code))
            manager = manager.add_scheme(target_scheme).add_map(code_map)
        return manager

    def make_gender_scheme(self, data_connection: Any) -> CodingSchemesManager:
        gender_space_table = self.tables.static.load_gender_space_table(data_connection)
        return self._make_demographic_scheme(name=self.scheme.gender, space_table=gender_space_table,
                                             c_code=str(COLUMN.gender),
                                             selection=self.aux.selections.gender,
                                             target_name=self.aux.scoped_names.gender)

    def make_ethnicity_scheme(self, data_connection: Any) -> CodingSchemesManager:
        race_space_table = self.tables.static.load_ethnicity_space_table(data_connection)
        return self._make_demographic_scheme(name=self.scheme.ethnicity, space_table=race_space_table,
                                             c_code=str(COLUMN.race),
                                             selection=self.aux.selections.ethnicity,
                                             target_name=self.aux.scoped_names.ethnicity,
                                             map_table=self.aux.maps.ethnicity)

    def make_obs_scheme(self) -> CodingSchemesManager:
        obs = self.tables.obs
        return obs.register_scheme(name=self.scheme.obs,
                                   space_table=obs.space(),
                                   attributes_selection=self.aux.selections.obs)

    def make_icu_inputs_scheme(self) -> CodingSchemesManager:
        c_code = str(COLUMN.code)
        c_desc = str(COLUMN.description)
        scheme = CodingSchemeWithUOM.from_table(name=self.scheme.icu_inputs,
                                                table=self.aux.icu_inputs_uom_normalization_table,
                                                c_code=c_code, c_desc=c_desc,
                                                c_universal_unit=str(COLUMN.derived_universal_unit),
                                                c_unit=str(COLUMN.amount_unit),
                                                c_normalization_factor=str(COLUMN.derived_unit_normalization_factor),
                                                code_selection=self.aux.selections.icu_inputs)
        manager = CodingSchemesManager().add_scheme(scheme)
        c_aggregation = self.aux.icu_inputs_aggregation_column
        map_data = self.aux.maps.icu_inputs
        if map_data is not None and c_aggregation is not None:
            target_names = self.aux.scoped_names.target
            c_target_code = target_names.column_name(c_code)
            c_target_desc = target_names.column_name(c_desc)
            target_scheme = CodingScheme.from_table(name=target_names.icu_inputs,
                                                    table=map_data,
                                                    c_code=c_target_code,
                                                    c_desc=c_target_desc)
            codemap = ReducedCodeMapN1.from_table(scheme, target_scheme, c_source_code=c_code,
                                                  c_target_code=c_target_code, c_target_agg=c_aggregation,
                                                  table=map_data)
            manager = manager.add_scheme(target_scheme).add_map(codemap)

        return manager

    def make_icu_procedures_scheme(self, data_connection: Any) -> CodingSchemesManager:
        space_table = self.tables.icu_procedures.space(data_connection)
        source_scheme = CodingScheme.from_table(name=self.scheme.icu_procedures,
                                                table=space_table,
                                                code_selection=self.aux.selections.icu_procedures,
                                                c_code=str(COLUMN.code),
                                                c_desc=str(COLUMN.description))
        manager = CodingSchemesManager().add_scheme(source_scheme)
        map_table = self.aux.maps.icu_procedures
        if map_table is not None:
            target_names = self.aux.scoped_names.target
            target_scheme = CodingScheme.from_table(name=target_names.icu_procedures, table=map_table,
                                                    c_code=target_names.column_name(str(COLUMN.code)),
                                                    c_desc=target_names.column_name(str(COLUMN.description)))
            code_map = CodeMap.from_table(source_scheme, target_scheme, c_source_code=str(COLUMN.code),
                                          c_target_code=target_names.column_name(str(COLUMN.code)))
            manager = manager.add_scheme(target_scheme).add_map(code_map)
        return manager

    def make_hosp_procedures_scheme(self, manager: CodingSchemesManager, data_connection: Any) -> CodingSchemesManager:
        target_names = self.aux.scoped_names.target
        table = self.tables.hosp_procedures
        return table.register_scheme(manager, name=self.scheme.hosp_procedures,
                                     space_table=self.tables.hosp_procedures.space(data_connection),
                                     icd_version_schemes=FrozenDict11({'9': 'pr_icd9', '10': 'pr_flat_icd10'}),
                                     icd_version_selection=self.aux.selections.hosp_procedures,
                                     target_name=target_names.hosp_procedures,
                                     mapping=self.aux.maps.hosp_procedures)

    def make_dx_discharge_scheme(self, manager: CodingSchemesManager, data_connection: Any) -> CodingSchemesManager:
        target_names = self.aux.scoped_names.target
        table = self.tables.dx_discharge
        return table.register_scheme(manager, name=self.scheme.dx_discharge,
                                     space_table=self.tables.dx_discharge.space(data_connection),
                                     icd_version_schemes=FrozenDict11({'9': 'dx_icd9', '10': 'dx_flat_icd10'}),
                                     icd_version_selection=self.aux.selections.hosp_procedures,
                                     target_name=target_names.dx_discharge,
                                     mapping=self.aux.maps.dx_discharge)

    def make_all_schemes(self, data_connection: Any) -> CodingSchemesManager:
        # make standard ones.
        manager = setup_standard_icd_ccs(CodingSchemesManager(),
                                         scheme_selection=CCSICDSchemeSelection.all(),
                                         outcome_selection=CCSICDOutcomeSelection.all())
        manager = self.make_hosp_procedures_scheme(manager, data_connection)
        manager = self.make_dx_discharge_scheme(manager, data_connection)
        return (manager + self.make_gender_scheme(data_connection) + self.make_ethnicity_scheme(
            data_connection) + self.make_icu_inputs_scheme() + self.make_icu_procedures_scheme(
            data_connection) + self.make_obs_scheme())


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
        return self.tables.obs.space(None)

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

    def load_static(self, data_connection: Any) -> pd.DataFrame:
        return self.tables.static(data_connection)

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
        static = self.load_static(data_connection)
        admissions = self.load_admissions(data_connection)
        dx_discharge = self.load_dx_discharge(data_connection, schemes_manager)
        return DatasetTables(static=static, admissions=admissions, dx_discharge=dx_discharge, obs=obs,
                             icu_procedures=icu_procedures, icu_inputs=icu_inputs, hosp_procedures=hosp_procedures)


class MIMICDatasetPipelineConfig(AbstractDatasetPipelineConfig):
    overlap_merge: bool = True


class MIMICDataset(Dataset):

    @staticmethod
    def load_scheme_manager(tables: DatasetTablesResources,
                            scheme: DatasetSchemeConfig,
                            aux: MIMICDatasetAuxiliaryResources, data_connection: Any) -> CodingSchemesManager:
        schemes_resources = MIMICSchemeResources(tables=tables, scheme=scheme, aux=aux)
        return schemes_resources.make_all_schemes(data_connection=data_connection)

    @staticmethod
    def load_tables(tables: DatasetTablesResources,
                    scheme: DatasetSchemeConfig,
                    schemes_manager: CodingSchemesManager,
                    data_connection: Any) -> DatasetTables:
        compiler = MIMICDatasetCompiler(tables=tables, scheme=scheme)
        return compiler.load_tables(data_connection=data_connection, schemes_manager=schemes_manager)

    @classmethod
    def compile(cls, config: DatasetConfig, tables: DatasetTablesResources,
                aux: MIMICDatasetAuxiliaryResources, data_connection: Any) -> tuple[Self, CodingSchemesManager]:
        manager = cls.load_scheme_manager(tables, config.scheme, aux, data_connection)
        tables = cls.load_tables(tables, config.scheme, manager, data_connection)
        return cls(tables=tables, config=config), manager
