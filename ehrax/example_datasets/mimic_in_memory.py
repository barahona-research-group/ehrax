from typing import Literal, cast

import pandas as pd

from .mimic import TableResource, CodedTableResource, CodedColumns, StaticTableResource, \
    MixedICDTableResource, DatasetTablesResources, MIMICDatasetAuxiliaryResources, ScopedSchemeNames, \
    ExternalMapResources, ExternalSelectionResources, MIMICDatasetSchemeSuffixes, DatasetSchemeMapsFileNames, \
    DatasetSchemeSelectionFiles, StaticTableResource_MIMICIII, StaticTableResource_MIMICIV, \
    MixedICDTableResource_MIMICIII, MixedICDTableResource_MIMICIV
from ..base import AbstractVxData, AbstractConfig
from ..dataset import COLUMN, TableColumns, StaticTableColumns, AdmissionsTableColumns
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


class InMemoryStaticTableResource:  # mixin
    columns: StaticTableColumns
    interface: StaticTableInterface

    def __init__(self, column_map: FrozenDict11[str], admissions_column_map: FrozenDict11[str]):
        self.interface = StaticTableInterface(admissions_column_map=admissions_column_map, column_map=column_map)

    def load_standard_columns_table(self, in_memory_tables: InMemoryMIMICTableFiles, *args, **kwargs) -> pd.DataFrame:
        return self.interface.load_standard_columns_table(in_memory_tables)

    def load_gender_space_table(self, in_memory_tables: InMemoryMIMICTableFiles, *args, **kwargs) -> pd.DataFrame:
        return self.interface.load_gender_space_table(in_memory_tables)

    def load_ethnicity_space_table(self, in_memory_tables: InMemoryMIMICTableFiles, *args, **kwargs) -> pd.DataFrame:
        return self.interface.load_ethnicity_space_table(in_memory_tables)


class InMemoryStaticTableResource_MIMICIII(InMemoryStaticTableResource, StaticTableResource_MIMICIII):
    pass


class InMemoryStaticTableResource_MIMICIV(InMemoryStaticTableResource, StaticTableResource_MIMICIV):
    pass


class InMemoryMixedICDTableResource:  # mixin
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


class InMemoryMixedICDTableResource_MIMICIII(InMemoryMixedICDTableResource, MixedICDTableResource_MIMICIII):
    pass


class InMemoryMixedICDTableResource_MIMICIV(InMemoryMixedICDTableResource, MixedICDTableResource_MIMICIV):
    pass


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

# The configurations below adapt to MIMIC-III v1.4
MIMICIII_STATIC_COLMAP = FrozenDict11({'DOB': str(COLUMN.date_of_birth),
                                       'SUBJECT_ID': str(COLUMN.subject_id),
                                       'GENDER': str(COLUMN.gender), })
MIMICIII_ADMISSIONS_COLMAP = FrozenDict11({'HADM_ID': str(COLUMN.admission_id),
                                           'SUBJECT_ID': str(COLUMN.subject_id),
                                           'ADMITTIME': str(COLUMN.start_time),
                                           'DISCHTIME': str(COLUMN.end_time),
                                           'ETHNICITY': str(COLUMN.race)})
MIMICIII_DIAGNOSES_ICD_COLMAP = FrozenDict11({'HADM_ID': str(COLUMN.admission_id),
                                              'ICD9_CODE': str(COLUMN.code)})
MIMICIII_D_ICD_DIAGNOSES_COLMAP = FrozenDict11({'ICD9_CODE': str(COLUMN.code),
                                                'LONG_TITLE': str(COLUMN.description)})

MIMICIII_STATIC_RESOURCES = InMemoryStaticTableResource_MIMICIII(MIMICIII_STATIC_COLMAP, MIMICIII_ADMISSIONS_COLMAP)

MIMICIII_ADMISSIONS_RESOURCES = InMemoryTableResource(AdmissionsTableColumns(), 'admissions',
                                                      MIMICIII_ADMISSIONS_COLMAP)
MIMICIII_DX_DISCHARGE_RESOURCES = InMemoryMixedICDTableResource_MIMICIII('diagnoses_icd', MIMICIII_DIAGNOSES_ICD_COLMAP,
                                                                         'd_icd_diagnoses',
                                                                         MIMICIII_D_ICD_DIAGNOSES_COLMAP)
MIMICIII_TABLES_RESOURCES = MIMICTablesResources(static=MIMICIII_STATIC_RESOURCES,
                                                 admissions=MIMICIII_ADMISSIONS_RESOURCES,
                                                 dx_discharge=MIMICIII_DX_DISCHARGE_RESOURCES)

# The configurations below adapt to MIMIC-IV v3.1
MIMICIV_STATIC_COLMAP = FrozenDict11({'subject_id': str(COLUMN.subject_id),
                                      'gender': str(COLUMN.gender),
                                      'anchor_age': str(COLUMN.anchor_age),
                                      'anchor_year': str(COLUMN.anchor_year)})

MIMICIV_ADMISSIONS_COLMAP = FrozenDict11({'hadm_id': str(COLUMN.admission_id),
                                          'subject_id': str(COLUMN.subject_id),
                                          'admittime': str(COLUMN.start_time),
                                          'dischtime': str(COLUMN.end_time),
                                          'race': str(COLUMN.race)})

MIMICIV_DIAGNOSES_ICD_COLMAP = FrozenDict11({'hadm_ic': str(COLUMN.admission_id),
                                             'icd_code': str(COLUMN.code),
                                             'icd_version': str(COLUMN.version)})

MIMICIV_D_ICD_DIAGNOSES_COLMAP = FrozenDict11({'icd_code': str(COLUMN.code),
                                               'icd_version': str(COLUMN.version),
                                               'long_title': str(COLUMN.description)})

MIMICIV_STATIC_RESOURCES = InMemoryStaticTableResource_MIMICIV(MIMICIV_STATIC_COLMAP, MIMICIV_ADMISSIONS_COLMAP)

MIMICIV_ADMISSIONS_RESOURCES = InMemoryTableResource(AdmissionsTableColumns(), 'admissions',
                                                     MIMICIV_ADMISSIONS_COLMAP)
MIMICIV_DX_DISCHARGE_RESOURCES = InMemoryMixedICDTableResource_MIMICIV('diagnoses_icd', MIMICIV_DIAGNOSES_ICD_COLMAP,
                                                                       'd_icd_diagnoses',
                                                                       MIMICIV_D_ICD_DIAGNOSES_COLMAP)
MIMICIV_TABLES_RESOURCES = MIMICTablesResources(static=MIMICIV_STATIC_RESOURCES,
                                                admissions=MIMICIV_ADMISSIONS_RESOURCES,
                                                dx_discharge=MIMICIV_DX_DISCHARGE_RESOURCES)
