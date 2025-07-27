from typing import Optional, Final

import pandas as pd
from sqlalchemy import Engine

from .mimiciv import CodedTableResource, TableResource, CodedColumns, StaticTableResource, \
    MultivariateTimeSeriesTableResource, GroupedMultivariateTimeSeriesTableResource
from ..base import AbstractConfig
from ..coding_scheme import resources_dir
from ..dataset import COLUMN, StaticTableColumns, MultivariateTimeSeriesTableMeta, \
    AdmissionTimeSeriesTableColumns


class SQLTableInterface(AbstractConfig):
    # resource file.
    # TODO: Add an attribute for the version using the last relevant git commit hash.
    query_template: Optional[str]

    def __init__(self, query_template: Optional[str]):
        self.query_template = query_template

    @property
    def query(self) -> str:
        assert self.query_template is not None, "Query template must be set."
        return open(resources_dir(self.query_template), "r").read()

    def load_standard_columns_table(self, engine: Engine):
        query = self.query.format(**COLUMN.as_dict())
        return pd.read_sql(query, engine, coerce_float=False)


class SQLCodedTableInterface(SQLTableInterface):
    query_template: Optional[str]

    def __init__(self, query_template: Optional[str] = None,
                 space_query_template: Optional[str] = None):
        super().__init__(query_template=query_template)
        self.space_query_template = space_query_template

    @property
    def space_query(self) -> str:
        assert self.space_query_template is not None, "Space query template must be set."
        return open(resources_dir(self.space_query_template), "r").read()

    def load_space_table(self, engine: Engine) -> pd.DataFrame:
        """
        Load the space table for the coded table.
        """
        query = self.space_query.format(**COLUMN.as_dict())
        return pd.read_sql(query, engine, coerce_float=False)


class SQLStaticTableInterface(SQLTableInterface):
    query: str
    gender_space_query_template: str
    race_space_query_template: str

    def __init__(self, query_template: Optional[str],
                 gender_space_query_template: str, race_space_query_template: str):
        super().__init__(query_template=query_template)
        self.gender_space_query_template = gender_space_query_template
        self.race_space_query_template = race_space_query_template

    @property
    def gender_space_query(self) -> str:
        return open(resources_dir(self.gender_space_query_template), "r").read()

    @property
    def race_space_query(self) -> str:
        return open(resources_dir(self.race_space_query_template), "r").read()

    def load_gender_space_table(self, engine: Engine):
        query = self.gender_space_query.format(**COLUMN.as_dict())
        return pd.read_sql(query, engine)

    def load_ethnicity_space_table(self, engine: Engine):
        query = self.race_space_query.format(**COLUMN.as_dict())
        return pd.read_sql(query, engine)


class SQLTableResource(TableResource):
    sql_interface: SQLTableInterface

    def __init__(self, query_template: Optional[str] = None):
        self.sql_interface = SQLTableInterface(query_template=query_template)

    def load_standard_columns_table(self, engine: Engine, *args, **kwargs) -> pd.DataFrame:
        return self.sql_interface.load_standard_columns_table(engine)


class SQLCodedTableResource(CodedTableResource):
    sql_interface: SQLCodedTableInterface

    def __init__(self, columns: CodedColumns, query_template: Optional[str] = None,
                 space_query_template: Optional[str] = None):
        self.columns = columns
        self.sql_interface = SQLCodedTableInterface(query_template=query_template,
                                                    space_query_template=space_query_template)

    def load_standard_columns_table(self, engine: Engine, *args, **kwargs) -> pd.DataFrame:
        return self.sql_interface.load_standard_columns_table(engine)

    def load_space_table(self, engine: Engine) -> pd.DataFrame:
        return self.sql_interface.load_space_table(engine)


class SQLStaticTableResource(StaticTableResource):
    columns: StaticTableColumns
    sql_interface: SQLStaticTableInterface

    def __init__(self, columns: StaticTableColumns,
                 query_template: Optional[str],
                 gender_space_query_template: str, race_space_query_template: str):
        self.columns = columns
        self.sql_interface = SQLStaticTableInterface(query_template=query_template,
                                                     gender_space_query_template=gender_space_query_template,
                                                     race_space_query_template=race_space_query_template)

    def load_standard_columns_table(self, engine: Engine, *args, **kwargs) -> pd.DataFrame:
        return self.sql_interface.load_standard_columns_table(engine)

    def load_gender_space_table(self, engine: Engine):
        return self.sql_interface.load_gender_space_table(engine)

    def load_ethnicity_space_table(self, engine: Engine):
        return self.sql_interface.load_ethnicity_space_table(engine)


class SQLMultivariateTimeSeriesResource(MultivariateTimeSeriesTableResource):
    sql_interface: SQLCodedTableInterface
    melted_columns: AdmissionTimeSeriesTableColumns

    def __init__(self, meta: MultivariateTimeSeriesTableMeta,
                 query_template: Optional[str] = None):
        self.meta = meta
        self.sql_interface = SQLCodedTableInterface(query_template=query_template)

    def load_standard_columns_table(self, engine: Engine, *args, **kwargs) -> pd.DataFrame:
        return self.sql_interface.load_standard_columns_table(engine)


class SQLGroupedMultivariateTimeSeriesTableResource(GroupedMultivariateTimeSeriesTableResource):
    groups: tuple[SQLMultivariateTimeSeriesResource, ...]
    space_query_template: Optional[str]


ENV_MIMICIV_HOST: Final[str] = 'MIMICIV_HOST'
ENV_MIMICIV_PORT: Final[str] = 'MIMICIV_PORT'
ENV_MIMICIV_USER: Final[str] = 'MIMICIV_USER'
ENV_MIMICIV_PASSWORD: Final[str] = 'MIMICIV_PASSWORD'
ENV_MIMICIV_DBNAME: Final[str] = 'MIMICIV_DBNAME'
ENV_MIMICIV_URL: Final[str] = 'MIMICIV_URL'


class MIMICIVSQLTablesConfig(DatasetTablesConfig):
    static: StaticSQLTableConfig
    admissions: AdmissionSQLTableConfig
    dx_discharge: AdmissionMixedICDSQLTableConfig
    obs: AdmissionTimestampedCodedValueSQLTableConfig
    icu_procedures: IntervalICUProcedureSQLTableConfig
    icu_inputs: RatedInputSQLTableConfig
    hosp_procedures: AdmissionIntervalBasedMixedICDTableConfig

    def __init__(self, static: StaticSQLTableConfig = STATIC_CONF,
                 admissions: AdmissionSQLTableConfig = ADMISSIONS_CONF,
                 dx_discharge: AdmissionMixedICDSQLTableConfig = DX_DISCHARGE_CONF,
                 obs: AdmissionTimestampedCodedValueSQLTableConfig = OBS_TABLE_CONFIG,
                 icu_procedures: IntervalICUProcedureSQLTableConfig = ICU_PROCEDURES_CONF,
                 icu_inputs: RatedInputSQLTableConfig = ICU_INPUTS_CONF,
                 hosp_procedures: AdmissionIntervalBasedMixedICDTableConfig = HOSP_PROCEDURES_CONF):
        DatasetTablesConfig.__init__(self, static=static, admissions=admissions, dx_discharge=dx_discharge,
                                     obs=obs, icu_procedures=icu_procedures, icu_inputs=icu_inputs,
                                     hosp_procedures=hosp_procedures)

    @staticmethod
    def url() -> str:
        if ENV_MIMICIV_URL in os.environ:
            return os.environ[ENV_MIMICIV_URL]
        elif all(e in os.environ for e in
                 [ENV_MIMICIV_USER, ENV_MIMICIV_PASSWORD, ENV_MIMICIV_HOST, ENV_MIMICIV_PORT, ENV_MIMICIV_DBNAME]):
            return MIMICIVSQLTablesConfig.url_from_credentials(
                user=os.environ[ENV_MIMICIV_USER],
                password=os.environ[ENV_MIMICIV_PASSWORD],
                host=os.environ[ENV_MIMICIV_HOST],
                port=os.environ[ENV_MIMICIV_PORT],
                dbname=os.environ[ENV_MIMICIV_DBNAME]
            )
        else:
            credentials_env_list = [ENV_MIMICIV_USER, ENV_MIMICIV_PASSWORD, ENV_MIMICIV_HOST, ENV_MIMICIV_PORT,
                                    ENV_MIMICIV_DBNAME]
            raise ValueError(f"Environment variables ({ENV_MIMICIV_URL}) or "
                             f"({', '.join(credentials_env_list)}) "
                             f"are not set.")

    @staticmethod
    def url_from_credentials(user: str, password: str, host: str, port: str, dbname: str) -> str:
        return f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}'


ADMISSIONS_CONF = AdmissionSQLTableConfig(query_template="mimiciv/sql/admissions.tsql")
STATIC_CONF = StaticSQLTableConfig(query_template="mimiciv/sql/static.tsql",
                                   gender_space_query_template="mimiciv/sql/static_gender_space.tsql",
                                   race_space_query_template="mimiciv/sql/static_race_space.tsql")
DX_DISCHARGE_CONF = AdmissionMixedICDSQLTableConfig(query_template="mimiciv/sql/dx_discharge.tsql",
                                                    space_query_template="mimiciv/sql/dx_discharge_space.tsql")

RENAL_OUT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_out",
                                                               attributes=('uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'),
                                                               query_template="mimiciv/sql/renal_out.tsql")

RENAL_CREAT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_creat",
                                                                 attributes=('creat',),
                                                                 query_template="mimiciv/sql/renal_creat.tsql")

RENAL_AKI_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="renal_aki",
                                                               attributes=('aki_stage_smoothed', 'aki_binary'),
                                                               query_template="mimiciv/sql/renal_aki.tsql",
                                                               type_hint=('O', 'B'))  # Ordinal, Binary.

SOFA_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="sofa",
                                                          attributes=("sofa_24hours",),
                                                          query_template="mimiciv/sql/sofa.tsql",
                                                          default_type_hint='O')  # Ordinal.
BLOOD_GAS_ATTRIBUTES = ('so2', 'po2', 'pco2', 'fio2',
                        'fio2_chartevents',
                        'aado2', 'aado2_calc',
                        'pao2fio2ratio', 'ph',
                        'baseexcess', 'bicarbonate', 'totalco2',
                        'hematocrit',
                        'hemoglobin',
                        'carboxyhemoglobin', 'methemoglobin',
                        'chloride', 'calcium', 'temperature',
                        'potassium',
                        'sodium', 'lactate', 'glucose')
BLOOD_GAS_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_gas",
                                                               attributes=BLOOD_GAS_ATTRIBUTES,
                                                               query_template="mimiciv/sql/blood_gas.tsql")

BLOOD_CHEMISTRY_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_chemistry",
                                                                     attributes=('albumin', 'globulin',
                                                                                 'total_protein',
                                                                                 'aniongap', 'bicarbonate', 'bun',
                                                                                 'calcium', 'chloride',
                                                                                 'creatinine', 'glucose', 'sodium',
                                                                                 'potassium'),
                                                                     query_template="mimiciv/sql/blood_chemistry.tsql")

CARDIAC_MARKER_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="cardiac_marker",
                                                                    attributes=('troponin_t2', 'ntprobnp', 'ck_mb'),
                                                                    query_template="mimiciv/sql/cardiac_marker.tsql")

WEIGHT_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="weight",
                                                            attributes=('weight',),
                                                            query_template="mimiciv/sql/weight.tsql")

CBC_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="cbc",
                                                         attributes=('hematocrit', 'hemoglobin', 'mch', 'mchc',
                                                                     'mcv',
                                                                     'platelet',
                                                                     'rbc', 'rdw', 'wbc'),
                                                         query_template="mimiciv/sql/cbc.tsql")

VITAL_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="vital",
                                                           attributes=('heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni',
                                                                       'dbp_ni',
                                                                       'mbp_ni', 'resp_rate',
                                                                       'temperature', 'spo2',
                                                                       'glucose'),
                                                           query_template="mimiciv/sql/vital.tsql")

# Glasgow Coma Scale, a measure of neurological function
GCS_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="gcs",
                                                         attributes=('gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes',
                                                                     'gcs_unable'),
                                                         query_template="mimiciv/sql/gcs.tsql",
                                                         default_type_hint='O')  # Ordinal.

# Intracranial pressure
ICP_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="icp",
                                                         attributes=('icp',),
                                                         query_template="mimiciv/sql/icp.tsql")

# Inflammation
INFLAMMATION_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="inflammation",
                                                                  attributes=('crp',),
                                                                  query_template="mimiciv/sql/inflammation.tsql")

# Coagulation
COAGULATION_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="coagulation",
                                                                 attributes=('pt', 'ptt', 'inr', 'd_dimer',
                                                                             'fibrinogen', 'thrombin'),
                                                                 query_template="mimiciv/sql/coagulation.tsql")
# Blood differential
BLOOD_DIFF_ATTRIBUTES = ('neutrophils', 'lymphocytes',
                         'monocytes',
                         'eosinophils', 'basophils',
                         'atypical_lymphocytes',
                         'bands', 'immature_granulocytes',
                         'metamyelocytes',
                         'nrbc',
                         'basophils_abs', 'eosinophils_abs',
                         'lymphocytes_abs',
                         'monocytes_abs', 'neutrophils_abs')
BLOOD_DIFF_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="blood_diff",
                                                                attributes=BLOOD_DIFF_ATTRIBUTES,
                                                                query_template="mimiciv/sql/blood_diff.tsql")
ENZYMES_ATTRIBUTES = ('ast', 'alt', 'alp', 'ld_ldh', 'ck_cpk',
                      'ck_mb',
                      'amylase', 'ggt', 'bilirubin_direct',
                      'bilirubin_total', 'bilirubin_indirect')
# Enzymes
ENZYMES_CONF = AdmissionTimestampedMultiColumnSQLTableConfig(name="enzymes",
                                                             attributes=ENZYMES_ATTRIBUTES,
                                                             query_template="mimiciv/sql/enzymes.tsql")

OBS_COMPONENTS = (
    RENAL_OUT_CONF,
    RENAL_CREAT_CONF,
    RENAL_AKI_CONF,
    SOFA_CONF,
    BLOOD_GAS_CONF,
    BLOOD_CHEMISTRY_CONF,
    CARDIAC_MARKER_CONF,
    WEIGHT_CONF,
    CBC_CONF,
    VITAL_CONF,
    # GCS_CONF,
    ICP_CONF,
    INFLAMMATION_CONF,
    COAGULATION_CONF,
    BLOOD_DIFF_CONF,
    ENZYMES_CONF
)
OBS_TABLE_CONFIG = AdmissionTimestampedCodedValueSQLTableConfig(components=OBS_COMPONENTS)

ICU_INPUTS_CONF = RatedInputSQLTableConfig(query_template="mimiciv/sql/icu_inputs.tsql",
                                           space_query_template="mimiciv/sql/icu_inputs_space.tsql")
ICU_PROCEDURES_CONF = IntervalICUProcedureSQLTableConfig(query_template="mimiciv/sql/icu_procedures.tsql",
                                                         space_query_template="mimiciv/sql/icu_procedures_space.tsql")
HOSP_PROCEDURES_CONF = AdmissionIntervalBasedMixedICDTableConfig(
    query_template="mimiciv/sql/hosp_procedures.tsql", space_query_template="mimiciv/sql/hosp_procedures_space.tsql")
