from typing import Literal, cast

import pandas as pd

from .mimic_resources import TableResource, CodedTableResource, CodedColumns, StaticTableResource, \
    MixedICDTableResource, DatasetTablesResources, MIMICDataset, MIMICDatasetAuxiliaryResources, ScopedSchemeNames, \
    ExternalMapResources, ExternalSelectionResources, MIMICDatasetSchemeSuffixes, DatasetSchemeMapsFileNames, \
    DatasetSchemeSelectionFiles
from ..base import AbstractVxData, AbstractConfig
from ..dataset import COLUMN, TableColumns, StaticTableColumns, DatasetSchemeConfig, DatasetConfig
from ..freezer import FrozenDict11

TableFileTitle = Literal['patients', 'admissions', 'diagnoses_icd', 'd_icd_diagnoses']


class InMemoryMIMICTableFiles(AbstractVxData):
    patients: pd.DataFrame
    admissions: pd.DataFrame
    diagnoses_icd: pd.DataFrame
    d_icd_diagnoses: pd.DataFrame

    def __init__(self, patients: pd.DataFrame, admissions: pd.DataFrame, diagnoses_icd: pd.DataFrame,
                 d_icd_diagnoses: pd.DataFrame):
        self.patients = patients
        self.admissions = admissions
        self.diagnoses_icd = diagnoses_icd
        self.d_icd_diagnoses = d_icd_diagnoses

    @classmethod
    def from_path(cls, patients: str, admissions: str, diagnoses_icd: str, d_icd_diagnoses: str):
        return InMemoryMIMICTableFiles(
            patients=pd.read_csv(patients),
            admissions=pd.read_csv(admissions),
            diagnoses_icd=pd.read_csv(diagnoses_icd),
            d_icd_diagnoses=pd.read_csv(d_icd_diagnoses),
        )


class TableInterface(AbstractConfig):
    table_name: TableFileTitle
    column_map: FrozenDict11[str]

    def __init__(self, table_name: TableFileTitle, column_map: FrozenDict11[str]):
        self.table_name = table_name
        self.column_map = column_map

    def load_standard_columns_table(self, in_memory_tables: InMemoryMIMICTableFiles) -> pd.DataFrame:
        table = cast(pd.DataFrame, getattr(in_memory_tables, self.table_name))
        assert set(self.column_map.keys()).issubset(table.columns)
        return table.rename(columns=self.column_map)


class CodedTableInterface(TableInterface):
    table_name: TableFileTitle
    column_map: FrozenDict11[str]
    space_table_name: TableFileTitle
    space_column_map: FrozenDict11[str]

    def __init__(self, table_name: TableFileTitle, column_map: FrozenDict11[str], space_table_name: TableFileTitle,
                 space_column_map: FrozenDict11[str]):
        super().__init__(table_name=table_name, column_map=column_map)
        self.space_table_name = space_table_name
        self.space_column_map = space_column_map

    def load_space_table(self, in_memory_tables: InMemoryMIMICTableFiles) -> pd.DataFrame:
        table = getattr(in_memory_tables, self.space_table_name)
        assert set(self.space_column_map.keys()).issubset(table.columns)
        return table.rename(columns=self.space_column_map)[list(self.space_column_map.values())]


class StaticTableInterface(TableInterface):
    admissions_column_map: FrozenDict11[str]

    def __init__(self, static_column_map: FrozenDict11[str], admissions_colmap: FrozenDict11[str]):
        super().__init__(table_name='patients', column_map=static_column_map)
        self.admissions_column_map = admissions_colmap

    def load_standard_columns_table(self, in_memory_tables: InMemoryMIMICTableFiles) -> pd.DataFrame:
        assert set(self.column_map.keys()).issubset(in_memory_tables.patients)
        assert set(self.admissions_column_map.keys()).issubset(in_memory_tables.admissions)
        patients = in_memory_tables.patients.rename(columns=self.column_map)
        admissions = in_memory_tables.admissions.rename(columns=self.admissions_column_map)
        ethno_map = admissions.set_index(str(COLUMN.subject_id))[str(COLUMN.race)].to_dict()
        patients[str(COLUMN.race)] = patients[str(COLUMN.subject_id)].map(ethno_map)
        return patients

    def load_gender_space_table(self, in_memory_tables: InMemoryMIMICTableFiles) -> pd.DataFrame:
        table = self.load_standard_columns_table(in_memory_tables)
        return table[[str(COLUMN.gender)]].drop_duplicates()

    def load_ethnicity_space_table(self, in_memory_tables: InMemoryMIMICTableFiles) -> pd.DataFrame:
        table = self.load_standard_columns_table(in_memory_tables)
        return table[[str(COLUMN.race)]].drop_duplicates()


class InMemoryTableResource(TableResource):
    interface: TableInterface

    def __init__(self, columns: TableColumns, table_name: TableFileTitle, column_map: FrozenDict11[str]):
        super().__init__(columns)
        self.interface = TableInterface(table_name=table_name, column_map=column_map)

    def load_standard_columns_table(self, in_memory_tables: InMemoryMIMICTableFiles, *args, **kwargs) -> pd.DataFrame:
        return self.interface.load_standard_columns_table(in_memory_tables)


class InMemoryCodedTableResource(CodedTableResource):
    interface: CodedTableInterface

    def __init__(self, columns: CodedColumns, table_name: TableFileTitle, column_map: FrozenDict11[str],
                 space_table_name: TableFileTitle,
                 space_column_map: FrozenDict11[str]):
        super().__init__(columns)
        self.interface = CodedTableInterface(table_name=table_name, column_map=column_map,
                                             space_table_name=space_table_name, space_column_map=space_column_map)

    def load_standard_columns_table(self, in_memory_tables: InMemoryMIMICTableFiles, *args, **kwargs) -> pd.DataFrame:
        return self.interface.load_standard_columns_table(in_memory_tables)

    def load_space_table(self, in_memory_tables: InMemoryMIMICTableFiles) -> pd.DataFrame:
        return self.interface.load_space_table(in_memory_tables)


class InMemoryStaticTableResource(StaticTableResource):
    columns: StaticTableColumns
    interface: StaticTableInterface

    def __init__(self, column_map: FrozenDict11[str], admissions_column_map: FrozenDict11[str]):
        super().__init__()
        self.interface = StaticTableInterface(admissions_column_map=admissions_column_map, column_map=column_map)

    def load_standard_columns_table(self, in_memory_tables: InMemoryMIMICTableFiles, *args, **kwargs) -> pd.DataFrame:
        return self.interface.load_standard_columns_table(in_memory_tables)

    def load_gender_space_table(self, in_memory_tables: InMemoryMIMICTableFiles, *args, **kwargs) -> pd.DataFrame:
        return self.interface.load_gender_space_table(in_memory_tables)

    def load_ethnicity_space_table(self, in_memory_tables: InMemoryMIMICTableFiles, *args, **kwargs) -> pd.DataFrame:
        return self.interface.load_ethnicity_space_table(in_memory_tables)


class InMemoryMixedICDTableResource(MixedICDTableResource):
    interface: CodedTableInterface

    def __init__(self, table_name: TableFileTitle, column_map: FrozenDict11[str], space_table_name: TableFileTitle,
                 space_column_map: FrozenDict11[str]):
        super().__init__()
        self.interface = CodedTableInterface(table_name=table_name, column_map=column_map,
                                             space_table_name=space_table_name, space_column_map=space_column_map)

    def load_standard_columns_table(self, in_memory_tables: InMemoryMIMICTableFiles, *args, **kwargs) -> pd.DataFrame:
        return self.interface.load_standard_columns_table(in_memory_tables)

    def load_space_table(self, in_memory_tables: InMemoryMIMICTableFiles) -> pd.DataFrame:
        return self.interface.load_space_table(in_memory_tables)


class InMemoryMIMICDatasetAuxiliaryResources(MIMICDatasetAuxiliaryResources):
    scoped_names: ScopedSchemeNames
    maps: ExternalMapResources
    selections: ExternalSelectionResources
    icu_inputs_uom_normalization: None
    icu_inputs_aggregation_column: None

    @classmethod
    def make_resources(cls, suffixes: MIMICDatasetSchemeSuffixes = MIMICDatasetSchemeSuffixes(),
                       name_separator: str = '.', name_prefix: str = 'mimic', resources_root: str = 'mimic',
                       selection_subdir: str = 'selection',
                       map_subdir: str = 'map', map_files: DatasetSchemeMapsFileNames = DatasetSchemeMapsFileNames(),
                       selection_files: DatasetSchemeSelectionFiles = DatasetSchemeSelectionFiles(), *args, **kwargs):
        return super().make_resources(suffixes=suffixes, name_separator=name_separator, name_prefix=name_prefix,
                                      resources_root=resources_root, selection_subdir=selection_subdir,
                                      map_subdir=map_subdir, map_files=map_files,
                                      selection_files=selection_files, icu_inputs_aggregation_column=None,
                                      icu_inputs_uom_normalization=None)


class MIMICTablesResources(DatasetTablesResources):
    static: StaticTableResource
    admissions: TableResource
    dx_discharge: MixedICDTableResource
    obs: None
    hosp_procedures: None
    icu_procedures: None
    icu_inputs: None

    def __init__(self, static: StaticTableResource,
                 admissions: TableResource,
                 dx_discharge: MixedICDTableResource):
        super().__init__(self, static=static, admissions=admissions, dx_discharge=dx_discharge,
                         obs=None, icu_procedures=None, icu_inputs=None,
                         hosp_procedures=None)
