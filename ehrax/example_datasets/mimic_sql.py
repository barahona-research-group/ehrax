import os
from collections.abc import Mapping
from typing import Final

import pandas as pd
import sqlalchemy

from .._literals import NumericalTypeHint
from ..base import AbstractConfig
from ..dataset import (
    AdmissionIntervalEventsTableColumns,
    AdmissionIntervalRatesTableColumns,
    AdmissionsTableColumns,
    COLUMN,
    MultivariateTimeSeriesTableMeta,
    StaticTableColumns,
    TableColumns,
)
from ..utils import resources_path
from .mimic import (
    CodedColumns,
    CodedTableResource,
    DatasetTablesResources,
    GroupedMultivariateTimeSeriesTableResource,
    MixedICDTableResource_MIMICIV,
    MultivariateTimeSeriesTableResource,
    StaticTableResource_MIMICIV,
    TableResource,
)


class SQLTableInterface(AbstractConfig):
    # resource file.
    query_template: str | None
    substitutes: Mapping[str, str]

    def __init__(self, query_template: str | None, substitutes: dict[str, str] | None = None):
        self.substitutes = substitutes or {}
        self.query_template = query_template

    @property
    def query(self) -> str:
        assert self.query_template is not None, "Query template must be set."
        return open(resources_path(self.query_template)).read()

    def load_standard_columns_table(self, engine: sqlalchemy.Engine):
        sub = COLUMN.as_dict() | dict(self.substitutes)
        query = self.query.format(**sub)
        return pd.read_sql(query, engine, coerce_float=False)


class SQLCodedTableInterface(SQLTableInterface):
    query_template: str | None
    space_query_template: str | None

    def __init__(
        self,
        query_template: str | None = None,
        space_query_template: str | None = None,
        substitutes: dict[str, str] | None = None,
    ):
        super().__init__(query_template=query_template, substitutes=substitutes)
        self.space_query_template = space_query_template

    @property
    def space_query(self) -> str:
        assert self.space_query_template is not None, "Space query template must be set."
        return open(resources_path(self.space_query_template)).read()

    def load_space_table(self, engine: sqlalchemy.Engine) -> pd.DataFrame:
        """
        Load the space table for the coded table.
        """
        query = self.space_query.format(**COLUMN.as_dict())
        return pd.read_sql(query, engine, coerce_float=False)


class SQLStaticTableInterface(SQLTableInterface):
    query: str
    gender_space_query_template: str
    race_space_query_template: str

    def __init__(self, query_template: str | None, gender_space_query_template: str, race_space_query_template: str):
        super().__init__(query_template=query_template)
        self.gender_space_query_template = gender_space_query_template
        self.race_space_query_template = race_space_query_template

    @property
    def gender_space_query(self) -> str:
        return open(resources_path(self.gender_space_query_template)).read()

    @property
    def race_space_query(self) -> str:
        return open(resources_path(self.race_space_query_template)).read()

    def load_gender_space_table(self, engine: sqlalchemy.Engine):
        query = self.gender_space_query.format(**COLUMN.as_dict())
        return pd.read_sql(query, engine)

    def load_ethnicity_space_table(self, engine: sqlalchemy.Engine):
        query = self.race_space_query.format(**COLUMN.as_dict())
        return pd.read_sql(query, engine)


class SQLTableResource(TableResource):
    sql_interface: SQLTableInterface

    def __init__(self, columns: TableColumns, query_template: str | None = None):
        super().__init__(columns)
        self.sql_interface = SQLTableInterface(query_template=query_template)

    def load_standard_columns_table(self, engine: sqlalchemy.Engine, *args, **kwargs) -> pd.DataFrame:
        return self.sql_interface.load_standard_columns_table(engine)


class SQLCodedTableResource(CodedTableResource):
    sql_interface: SQLCodedTableInterface

    def __init__(
        self, columns: CodedColumns, query_template: str | None = None, space_query_template: str | None = None
    ):
        super().__init__(columns)
        self.sql_interface = SQLCodedTableInterface(
            query_template=query_template, space_query_template=space_query_template
        )

    def load_standard_columns_table(self, engine: sqlalchemy.Engine, *args, **kwargs) -> pd.DataFrame:
        return self.sql_interface.load_standard_columns_table(engine)

    def load_space_table(self, engine: sqlalchemy.Engine) -> pd.DataFrame:
        return self.sql_interface.load_space_table(engine)


class SQLStaticTableResource(StaticTableResource_MIMICIV):
    columns: StaticTableColumns
    sql_interface: SQLStaticTableInterface

    def __init__(self, query_template: str | None, gender_space_query_template: str, race_space_query_template: str):
        super().__init__()
        self.sql_interface = SQLStaticTableInterface(
            query_template=query_template,
            gender_space_query_template=gender_space_query_template,
            race_space_query_template=race_space_query_template,
        )

    def load_standard_columns_table(self, engine: sqlalchemy.Engine, *args, **kwargs) -> pd.DataFrame:
        return self.sql_interface.load_standard_columns_table(engine)

    def load_gender_space_table(self, engine: sqlalchemy.Engine):
        return self.sql_interface.load_gender_space_table(engine)

    def load_ethnicity_space_table(self, engine: sqlalchemy.Engine):
        return self.sql_interface.load_ethnicity_space_table(engine)


class SQLMixedICDTableResource(MixedICDTableResource_MIMICIV):
    sql_interface: SQLCodedTableInterface

    def __init__(self, query_template: str, space_query_template: str):
        super().__init__()
        self.sql_interface = SQLCodedTableInterface(
            query_template=query_template, space_query_template=space_query_template
        )

    def load_standard_columns_table(self, engine: sqlalchemy.Engine, *args, **kwargs) -> pd.DataFrame:
        return self.sql_interface.load_standard_columns_table(engine)

    def load_space_table(self, engine: sqlalchemy.Engine) -> pd.DataFrame:
        return self.sql_interface.load_space_table(engine)


class SQLMultivariateTimeSeriesResource(MultivariateTimeSeriesTableResource):
    sql_interface: SQLCodedTableInterface

    def __init__(
        self,
        name: str,
        attributes: tuple[str, ...],
        type_hint: tuple[NumericalTypeHint, ...] | None = None,
        default_type_hint: NumericalTypeHint = "N",
        query_template: str | None = None,
    ):
        super().__init__(
            MultivariateTimeSeriesTableMeta(
                name=name, attributes=attributes, type_hint=type_hint, default_type_hint=default_type_hint
            )
        )
        self.sql_interface = SQLCodedTableInterface(
            query_template=query_template, substitutes=dict(attributes=", ".join(attributes))
        )

    def load_standard_columns_table(self, engine: sqlalchemy.Engine, *args, **kwargs) -> pd.DataFrame:
        return self.sql_interface.load_standard_columns_table(engine)


class SQLGroupedMultivariateTimeSeriesTableResource(GroupedMultivariateTimeSeriesTableResource):
    groups: tuple[SQLMultivariateTimeSeriesResource, ...]


ADMISSIONS_CONF = SQLTableResource(query_template="mimiciv/sql/admissions.tsql", columns=AdmissionsTableColumns())
STATIC_CONF = SQLStaticTableResource(
    query_template="mimiciv/sql/static.tsql",
    gender_space_query_template="mimiciv/sql/static_gender_space.tsql",
    race_space_query_template="mimiciv/sql/static_race_space.tsql",
)
DX_DISCHARGE_CONF = SQLMixedICDTableResource(
    query_template="mimiciv/sql/dx_discharge.tsql", space_query_template="mimiciv/sql/dx_discharge_space.tsql"
)

RENAL_OUT_CONF = SQLMultivariateTimeSeriesResource(
    name="renal_out", attributes=("uo_rt_6hr", "uo_rt_12hr", "uo_rt_24hr"), query_template="mimiciv/sql/renal_out.tsql"
)

RENAL_CREAT_CONF = SQLMultivariateTimeSeriesResource(
    name="renal_creat", attributes=("creat",), query_template="mimiciv/sql/renal_creat.tsql"
)

RENAL_AKI_CONF = SQLMultivariateTimeSeriesResource(
    name="renal_aki",
    attributes=("aki_stage_smoothed", "aki_binary"),
    query_template="mimiciv/sql/renal_aki.tsql",
    type_hint=(
        "O",
        "B",
    ),
)  # Ordinal, Binary.

SOFA_CONF = SQLMultivariateTimeSeriesResource(
    name="sofa", attributes=("sofa_24hours",), query_template="mimiciv/sql/sofa.tsql", default_type_hint="O"
)  # Ordinal.
BLOOD_GAS_ATTRIBUTES = (
    "so2",
    "po2",
    "pco2",
    "fio2",
    "fio2_chartevents",
    "aado2",
    "aado2_calc",
    "pao2fio2ratio",
    "ph",
    "baseexcess",
    "bicarbonate",
    "totalco2",
    "hematocrit",
    "hemoglobin",
    "carboxyhemoglobin",
    "methemoglobin",
    "chloride",
    "calcium",
    "temperature",
    "potassium",
    "sodium",
    "lactate",
    "glucose",
)
BLOOD_GAS_CONF = SQLMultivariateTimeSeriesResource(
    name="blood_gas", attributes=BLOOD_GAS_ATTRIBUTES, query_template="mimiciv/sql/blood_gas.tsql"
)

BLOOD_CHEMISTRY_CONF = SQLMultivariateTimeSeriesResource(
    name="blood_chemistry",
    attributes=(
        "albumin",
        "globulin",
        "total_protein",
        "aniongap",
        "bicarbonate",
        "bun",
        "calcium",
        "chloride",
        "creatinine",
        "glucose",
        "sodium",
        "potassium",
    ),
    query_template="mimiciv/sql/blood_chemistry.tsql",
)

CARDIAC_MARKER_CONF = SQLMultivariateTimeSeriesResource(
    name="cardiac_marker",
    attributes=("troponin_t2", "ntprobnp", "ck_mb"),
    query_template="mimiciv/sql/cardiac_marker.tsql",
)

WEIGHT_CONF = SQLMultivariateTimeSeriesResource(
    name="weight", attributes=("weight",), query_template="mimiciv/sql/weight.tsql"
)

CBC_CONF = SQLMultivariateTimeSeriesResource(
    name="cbc",
    attributes=("hematocrit", "hemoglobin", "mch", "mchc", "mcv", "platelet", "rbc", "rdw", "wbc"),
    query_template="mimiciv/sql/cbc.tsql",
)

VITAL_CONF = SQLMultivariateTimeSeriesResource(
    name="vital",
    attributes=(
        "heart_rate",
        "sbp",
        "dbp",
        "mbp",
        "sbp_ni",
        "dbp_ni",
        "mbp_ni",
        "resp_rate",
        "temperature",
        "spo2",
        "glucose",
    ),
    query_template="mimiciv/sql/vital.tsql",
)

# Glasgow Coma Scale, a measure of neurological function
GCS_CONF = SQLMultivariateTimeSeriesResource(
    name="gcs",
    attributes=("gcs", "gcs_motor", "gcs_verbal", "gcs_eyes", "gcs_unable"),
    query_template="mimiciv/sql/gcs.tsql",
    default_type_hint="O",
)  # Ordinal.

# Intracranial pressure
ICP_CONF = SQLMultivariateTimeSeriesResource(name="icp", attributes=("icp",), query_template="mimiciv/sql/icp.tsql")

# Inflammation
INFLAMMATION_CONF = SQLMultivariateTimeSeriesResource(
    name="inflammation", attributes=("crp",), query_template="mimiciv/sql/inflammation.tsql"
)

# Coagulation
COAGULATION_CONF = SQLMultivariateTimeSeriesResource(
    name="coagulation",
    attributes=("pt", "ptt", "inr", "d_dimer", "fibrinogen", "thrombin"),
    query_template="mimiciv/sql/coagulation.tsql",
)
# Blood differential
BLOOD_DIFF_ATTRIBUTES = (
    "neutrophils",
    "lymphocytes",
    "monocytes",
    "eosinophils",
    "basophils",
    "atypical_lymphocytes",
    "bands",
    "immature_granulocytes",
    "metamyelocytes",
    "nrbc",
    "basophils_abs",
    "eosinophils_abs",
    "lymphocytes_abs",
    "monocytes_abs",
    "neutrophils_abs",
)
BLOOD_DIFF_CONF = SQLMultivariateTimeSeriesResource(
    name="blood_diff", attributes=BLOOD_DIFF_ATTRIBUTES, query_template="mimiciv/sql/blood_diff.tsql"
)
ENZYMES_ATTRIBUTES = (
    "ast",
    "alt",
    "alp",
    "ld_ldh",
    "ck_cpk",
    "ck_mb",
    "amylase",
    "ggt",
    "bilirubin_direct",
    "bilirubin_total",
    "bilirubin_indirect",
)
# Enzymes
ENZYMES_CONF = SQLMultivariateTimeSeriesResource(
    name="enzymes", attributes=ENZYMES_ATTRIBUTES, query_template="mimiciv/sql/enzymes.tsql"
)

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
    ENZYMES_CONF,
)
OBS_TABLE_CONFIG = SQLGroupedMultivariateTimeSeriesTableResource(groups=OBS_COMPONENTS)

ICU_INPUTS_CONF = SQLCodedTableResource(
    query_template="mimiciv/sql/icu_inputs.tsql",
    space_query_template="mimiciv/sql/icu_inputs_space.tsql",
    columns=AdmissionIntervalRatesTableColumns(),
)
ICU_PROCEDURES_CONF = SQLCodedTableResource(
    query_template="mimiciv/sql/icu_procedures.tsql",
    space_query_template="mimiciv/sql/icu_procedures_space.tsql",
    columns=AdmissionIntervalEventsTableColumns(),
)
HOSP_PROCEDURES_CONF = SQLMixedICDTableResource(
    query_template="mimiciv/sql/hosp_procedures.tsql", space_query_template="mimiciv/sql/hosp_procedures_space.tsql"
)

ENV_MIMICIV_HOST: Final[str] = "MIMICIV_HOST"
ENV_MIMICIV_PORT: Final[str] = "MIMICIV_PORT"
ENV_MIMICIV_USER: Final[str] = "MIMICIV_USER"
ENV_MIMICIV_PASSWORD: Final[str] = "MIMICIV_PASSWORD"
ENV_MIMICIV_DBNAME: Final[str] = "MIMICIV_DBNAME"
ENV_MIMICIV_URL: Final[str] = "MIMICIV_URL"


class SQLMIMICTablesResources(DatasetTablesResources):
    static: SQLStaticTableResource
    admissions: SQLTableResource
    dx_discharge: SQLMixedICDTableResource
    obs: SQLGroupedMultivariateTimeSeriesTableResource
    hosp_procedures: SQLMixedICDTableResource
    icu_procedures: SQLCodedTableResource
    icu_inputs: SQLCodedTableResource

    def __init__(
        self,
        static: SQLStaticTableResource = STATIC_CONF,
        admissions: SQLTableResource = ADMISSIONS_CONF,
        dx_discharge: SQLMixedICDTableResource = DX_DISCHARGE_CONF,
        obs: SQLGroupedMultivariateTimeSeriesTableResource = OBS_TABLE_CONFIG,
        icu_procedures: SQLCodedTableResource = ICU_PROCEDURES_CONF,
        icu_inputs: SQLCodedTableResource = ICU_INPUTS_CONF,
        hosp_procedures: SQLMixedICDTableResource = HOSP_PROCEDURES_CONF,
    ):
        super().__init__(
            static=static,
            admissions=admissions,
            dx_discharge=dx_discharge,
            obs=obs,
            icu_procedures=icu_procedures,
            icu_inputs=icu_inputs,
            hosp_procedures=hosp_procedures,
        )

    @classmethod
    def url(cls) -> str:
        if ENV_MIMICIV_URL in os.environ:
            return os.environ[ENV_MIMICIV_URL]
        elif all(
            e in os.environ
            for e in [ENV_MIMICIV_USER, ENV_MIMICIV_PASSWORD, ENV_MIMICIV_HOST, ENV_MIMICIV_PORT, ENV_MIMICIV_DBNAME]
        ):
            return cls.url_from_credentials(
                user=os.environ[ENV_MIMICIV_USER],
                password=os.environ[ENV_MIMICIV_PASSWORD],
                host=os.environ[ENV_MIMICIV_HOST],
                port=os.environ[ENV_MIMICIV_PORT],
                dbname=os.environ[ENV_MIMICIV_DBNAME],
            )
        else:
            credentials_env_list = [
                ENV_MIMICIV_USER,
                ENV_MIMICIV_PASSWORD,
                ENV_MIMICIV_HOST,
                ENV_MIMICIV_PORT,
                ENV_MIMICIV_DBNAME,
            ]
            raise ValueError(
                f"Environment variables ({ENV_MIMICIV_URL}) or ({', '.join(credentials_env_list)}) are not set."
            )

    @staticmethod
    def url_from_credentials(user: str, password: str, host: str, port: str, dbname: str) -> str:
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
