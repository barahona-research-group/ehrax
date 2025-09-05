from collections import namedtuple
from typing import Any, cast

import ehrax as rx
import numpy as np
import pandas as pd
import pytest
from ehrax import CodingScheme, CodingSchemesManager, COLUMN, FrozenDict1N, FrozenDict11
from ehrax.example_datasets.mimic import (
    MIMICDatasetSchemeSuffixes,
    MixedICDTableResource,
    MixedICDTableResource_MIMICIII,
    ScopedSchemeNames,
    TableResource,
)
from ehrax.example_datasets.mimic_in_memory import MIMICIII_DX_DISCHARGE_RESOURCES
from ehrax.example_schemes.icd9 import ICD9CM
from ehrax.example_schemes.icd10 import ICD10CM
from ehrax.example_schemes.icd_ccs_integration import ICDSchemeSelection, setup_standard_icd_ccs
from ehrax.example_schemes.mixed_icd import MultiVersionScheme
from ehrax.utils import resources_path


@pytest.fixture(scope="module")
def standard_icd_manager() -> CodingSchemesManager:
    return setup_standard_icd_ccs(icd_selection=ICDSchemeSelection(icd9cm=True, icd10cm=True))


@pytest.fixture(scope="module")
def icd9cm(standard_icd_manager: CodingSchemesManager) -> CodingScheme:
    return standard_icd_manager.scheme["icd9cm"]


@pytest.fixture(scope="module")
def icd10cm(standard_icd_manager: CodingSchemesManager) -> CodingScheme:
    return standard_icd_manager.scheme["icd10cm"]


class TestTableResource:
    @pytest.mark.parametrize(
        "df, columns",
        [
            (pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), ["a", "b"]),
            (pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), ["a"]),
            (pd.DataFrame({"a": [], "b": []}), ["a"]),
            (pd.DataFrame({"a": ["1", "2"], "b": [4, 5]}), ["a"]),
        ],
    )
    def test_coerce_columns_to_str(self, df: pd.DataFrame, columns: list[str]):
        df2 = TableResource._coerce_columns_to_str(df, columns)
        assert all(pd.api.types.is_string_dtype(df2.dtypes[c]) for c in columns)

    def test_preprocess_pipeline(self):
        add1 = lambda x: x + 1
        assert TableResource.apply_pipeline((add1, add1, add1), 2) == 5


ICD9_KEY = "9"
ICD10_KEY = "10"
N_CODES_PER_SCHEME = 10
MIXED_SCHEME_NAME = "tesTooOO"


class TestMixedICDTableResource:
    @pytest.fixture(scope="class")
    def icd_version_schemes(self) -> FrozenDict11:
        return FrozenDict11({ICD9_KEY: "icd9cm", ICD10_KEY: "icd10cm"})

    @pytest.fixture(scope="class")
    def supported_space(
        self, standard_icd_manager: CodingSchemesManager, icd_version_schemes: FrozenDict11
    ) -> pd.DataFrame:
        def codes_df(version: str, codes: tuple[str, ...], desc: tuple[str, ...]) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    str(rx.COLUMN.code): codes,
                    str(rx.COLUMN.version): [version] * len(codes),
                    str(rx.COLUMN.description): desc,
                }
            )

        S = standard_icd_manager.scheme
        codes = {name: S[name].codes[:N_CODES_PER_SCHEME] for name in icd_version_schemes.values()}
        return pd.concat(
            [
                codes_df(v, codes[name], tuple(map(S[name].desc.get, codes[name])))
                for v, name in icd_version_schemes.items()
            ]
        )

    @pytest.fixture(scope="class")
    def registered_schemes(
        self,
        standard_icd_manager: CodingSchemesManager,
        icd_version_schemes: FrozenDict11,
        supported_space: pd.DataFrame,
    ) -> CodingSchemesManager:
        scheme = MixedICDTableResource.register_scheme(
            MIXED_SCHEME_NAME,
            {k: standard_icd_manager.scheme[v] for k, v in icd_version_schemes.items()},
            supported_space,
        )
        manager = standard_icd_manager.add_scheme(scheme)
        for name in icd_version_schemes.values():
            manager = scheme.register_infer_map(manager, name)
        return manager

    @pytest.fixture(scope="class")
    def mixed_scheme(self, registered_schemes: CodingSchemesManager) -> MultiVersionScheme:
        return cast(MultiVersionScheme, registered_schemes.scheme[MIXED_SCHEME_NAME])

    @pytest.fixture(scope="class")
    def mixed_code_space(
        self, registered_schemes: CodingSchemesManager, mixed_scheme: MultiVersionScheme, supported_space: pd.DataFrame
    ) -> pd.DataFrame:
        return mixed_scheme.mixed_code_format_table(registered_schemes, supported_space)

    def test_mixed_code_space(
        self,
        supported_space: pd.DataFrame,
        mixed_code_space: pd.DataFrame,
        mixed_scheme: MultiVersionScheme,
        icd9cm: CodingScheme,
        icd10cm: CodingScheme,
    ):
        assert supported_space.shape == mixed_code_space.shape
        codes = supported_space[COLUMN.code]
        assert (codes.isin(icd9cm.codes).astype(int) + codes.isin(icd10cm.codes).astype(int) == 1).all()
        assert mixed_code_space[COLUMN.code].isin(mixed_scheme.codes).all()

    def test_mixed_scheme_properties(self, mixed_scheme: MultiVersionScheme):
        assert isinstance(mixed_scheme, MultiVersionScheme)
        assert len(mixed_scheme) == N_CODES_PER_SCHEME * 2

    def test_mixed_schemes_maps(
        self,
        mixed_scheme: MultiVersionScheme,
        registered_schemes: CodingSchemesManager,
        icd9cm: CodingScheme,
        icd10cm: CodingScheme,
    ):
        assert (mixed_scheme.name, icd9cm.name) in registered_schemes.map
        assert (mixed_scheme.name, icd10cm.name) in registered_schemes.map

    def test_mixed_codes_reversal(
        self,
        mixed_scheme: MultiVersionScheme,
        registered_schemes: CodingSchemesManager,
        mixed_code_space: pd.DataFrame,
        supported_space: pd.DataFrame,
    ):
        mixed_codes = mixed_code_space[COLUMN.code]
        mixed_as_icd9 = mixed_codes.map(registered_schemes.map[MIXED_SCHEME_NAME, "icd9cm"].data).to_numpy()
        mixed_as_icd10 = mixed_codes.map(registered_schemes.map[MIXED_SCHEME_NAME, "icd10cm"].data).to_numpy()
        versions = supported_space[COLUMN.version].to_numpy()

        df = supported_space.copy()
        codes = np.where(versions == ICD9_KEY, mixed_as_icd9, mixed_as_icd10)
        assert all(map(lambda c: len(c) == 1, codes))
        codes = list(map(lambda c: next(iter(c)), codes))
        df[COLUMN.code] = codes
        assert df.equals(supported_space)


class TestScopedNames:
    @pytest.fixture(scope="class")
    def suffixes(self) -> MIMICDatasetSchemeSuffixes:
        return MIMICDatasetSchemeSuffixes(
            gender="bin_gender",
            ethnicity="ethnicity",
            dx_discharge="dx_mixed_icd",
            obs="obs",
            icu_inputs="icu_inputs",
            icu_procedures=None,
            hosp_procedures="pr_mixed_icd",
        )

    @pytest.fixture(scope="class", params=[".", ":", "xxx", "**", "-", "_", "!", "%", "/", "no justice no peace"])
    def prefix(self, request) -> str:
        return request.param

    @pytest.fixture(scope="class", params=["_", ":", "***", "-"])
    def name_separator(self, request) -> str:
        return request.param

    @pytest.fixture(scope="class", params=[(), ("why",)])
    def global_infix(self, request) -> tuple[str, ...]:
        return request.param

    @pytest.fixture(scope="class")
    def variable_scoped_name(
        self, suffixes: MIMICDatasetSchemeSuffixes, global_infix: tuple[str, ...], name_separator: str, prefix: str
    ) -> ScopedSchemeNames:
        return ScopedSchemeNames(
            suffixes=suffixes, global_infix=global_infix, name_separator=name_separator, name_prefix=prefix
        )

    @pytest.fixture(scope="class")
    def constant_scoped_name(self, suffixes) -> ScopedSchemeNames:
        return ScopedSchemeNames(suffixes=suffixes, global_infix=(), name_separator="_", name_prefix="SSS")

    def test_scoped_names0(self, constant_scoped_name: ScopedSchemeNames, suffixes: MIMICDatasetSchemeSuffixes):
        assert constant_scoped_name.gender == "SSS_bin_gender"
        assert constant_scoped_name.ethnicity == "SSS_ethnicity"
        assert constant_scoped_name.dx_discharge == "SSS_dx_mixed_icd"
        assert constant_scoped_name.icu_procedures is None

    def test_scoped_names(
        self,
        variable_scoped_name: ScopedSchemeNames,
        suffixes: MIMICDatasetSchemeSuffixes,
        global_infix: tuple[str, ...],
        prefix: str,
        name_separator: str,
    ):
        infix = f"{name_separator.join(global_infix)}{name_separator}" if len(global_infix) > 0 else ""
        assert variable_scoped_name.gender == f"{prefix}{name_separator}{infix}{suffixes.gender}"

    def test_scoped_names_mapped(
        self,
        variable_scoped_name: ScopedSchemeNames,
        suffixes: MIMICDatasetSchemeSuffixes,
        global_infix: tuple[str, ...],
        prefix: str,
        name_separator: str,
    ):
        infix = f"{name_separator.join(global_infix)}{name_separator}" if len(global_infix) > 0 else ""
        mapped = variable_scoped_name.mapped
        assert mapped.gender == f"{prefix}{name_separator}{infix}mapped{name_separator}{suffixes.gender}"


MIXED_SCHEME_NAME = "XYZ"


class TestMIMIC3Diagnoses:
    @pytest.fixture(scope="class")
    def diagnoses_icd(self) -> pd.DataFrame:
        return pd.read_csv(resources_path("testing", "DIAGNOSES_ICD.csv"))

    @pytest.fixture(scope="class")
    def d_icd_diagnoses(self, diagnoses_icd: pd.DataFrame) -> pd.DataFrame:
        codes = diagnoses_icd[diagnoses_icd["ICD9_CODE"].notnull()]["ICD9_CODE"].unique()
        df = pd.DataFrame({"ICD9_CODE": codes})
        df["ROW_ID"] = range(len(df))
        df["SHORT_TITLE"] = "bla"
        df["LONG_TITLE"] = "blabla"
        return df

    @pytest.fixture(scope="class")
    def in_memory_data(self, diagnoses_icd: pd.DataFrame, d_icd_diagnoses: pd.DataFrame) -> Any:
        C = namedtuple("InMemoryData", ["diagnoses_icd", "d_icd_diagnoses"])
        return C(diagnoses_icd, d_icd_diagnoses)

    @pytest.fixture(scope="class")
    def diagnoses_icd_std(self, in_memory_data: Any, table_resource: MixedICDTableResource_MIMICIII) -> pd.DataFrame:
        return table_resource.load_standard_columns_table(in_memory_data)

    @pytest.fixture(scope="class")
    def d_icd_diagnoses_std(self, in_memory_data: Any, table_resource: MixedICDTableResource) -> pd.DataFrame:
        return table_resource.load_space_table(in_memory_data)

    @pytest.fixture(scope="class")
    def table_resource(self) -> MixedICDTableResource_MIMICIII:
        return MIMICIII_DX_DISCHARGE_RESOURCES

    @pytest.fixture(scope="class")
    def light_icd9_scheme(self, d_icd_diagnoses: pd.DataFrame) -> CodingScheme:
        codes = d_icd_diagnoses["ICD9_CODE"]
        codes = codes[codes.notnull()].tolist()
        codes = list(map(ICD9CM.format, codes))
        return ICD9CM(
            name="light_icd9", codes=tuple(sorted(codes)), ch2pt=FrozenDict1N({c: {codes[-1]} for c in codes[:-1]})
        )

    @pytest.fixture(scope="class")
    def light_icd10_scheme(self, d_icd_diagnoses: pd.DataFrame) -> CodingScheme:
        return ICD10CM(name="light_icd10", codes=(), ch2pt=FrozenDict1N({}))

    @pytest.fixture(scope="class")
    def init_manager(self, light_icd9_scheme: CodingScheme, light_icd10_scheme: CodingScheme) -> CodingSchemesManager:
        return CodingSchemesManager().add_scheme(light_icd9_scheme).add_scheme(light_icd10_scheme)

    @pytest.fixture(scope="class")
    def updated_manager(
        self, init_manager: CodingSchemesManager, in_memory_data: Any, table_resource: MixedICDTableResource
    ) -> CodingSchemesManager:
        scheme = table_resource.register_scheme(
            name=MIXED_SCHEME_NAME,
            component_schemes={"9": init_manager.scheme["light_icd9"]},
            selection=table_resource.space(in_memory_data),
        )
        manager = init_manager.add_scheme(scheme)
        manager = scheme.register_infer_map(manager, "light_icd9")
        return manager

    @pytest.fixture(scope="class")
    def after_coerce_id2str(
        self, diagnoses_icd_std: pd.DataFrame, table_resource: MixedICDTableResource_MIMICIII
    ) -> pd.DataFrame:
        return table_resource._coerce_id_to_str(diagnoses_icd_std)

    @pytest.fixture(scope="class")
    def after_nullcode_filter(self, diagnoses_icd_std: pd.DataFrame) -> pd.DataFrame:
        return TableResource._filter_null_codes(diagnoses_icd_std)

    @pytest.fixture(scope="class")
    def after_add_version_column(self, diagnoses_icd_std: pd.DataFrame):
        return MixedICDTableResource_MIMICIII._add_version_column_if_not_exists(diagnoses_icd_std)

    @pytest.fixture(scope="class")
    def mixed_format(
        self,
        diagnoses_icd_std: pd.DataFrame,
        updated_manager: CodingSchemesManager,
        table_resource: MixedICDTableResource_MIMICIII,
    ) -> CodingSchemesManager:
        pipeline = (
            MixedICDTableResource_MIMICIII._filter_null_codes,
            table_resource._coerce_code_to_str,
            MixedICDTableResource_MIMICIII._strip_icd_codes,
            MixedICDTableResource_MIMICIII._add_version_column_if_not_exists,
            MixedICDTableResource_MIMICIII._coerce_version_to_str,
            MixedICDTableResource_MIMICIII._mixed_code_format(MIXED_SCHEME_NAME, updated_manager),
        )
        return TableResource.apply_pipeline(pipeline, diagnoses_icd_std)

    def test_std_icd_diagnoses(self, diagnoses_icd_std: pd.DataFrame, diagnoses_icd: pd.DataFrame):
        # no change in content.
        assert diagnoses_icd.shape == diagnoses_icd_std.shape
        df_a = diagnoses_icd_std.iloc[:]
        df_b = diagnoses_icd.iloc[:]
        df_a.columns = df_b.columns
        assert df_a.equals(df_b)
        # columns updated to COLUMN type.
        assert {COLUMN.admission_id, COLUMN.code}.issubset(diagnoses_icd_std.columns)
        assert len({COLUMN.admission_id, COLUMN.code}.intersection(diagnoses_icd.columns)) == 0

    def test_std_d_icd_diagnoses(self, d_icd_diagnoses: pd.DataFrame, d_icd_diagnoses_std: pd.DataFrame):
        # no content changed.
        assert len(d_icd_diagnoses) == len(d_icd_diagnoses_std)
        df_a = d_icd_diagnoses_std.loc[:, [COLUMN.code, COLUMN.description]]
        df_b = d_icd_diagnoses.loc[:, ["ICD9_CODE", "LONG_TITLE"]]
        df_a.columns = df_b.columns
        assert df_a.equals(df_b)

    def test_coerce_id2str(self, diagnoses_icd_std: pd.DataFrame, after_coerce_id2str: pd.DataFrame):
        # does not change column names
        assert all(after_coerce_id2str.columns == after_coerce_id2str.columns)
        # content type changes
        assert not diagnoses_icd_std.equals(after_coerce_id2str)
        assert any(diagnoses_icd_std.dtypes != after_coerce_id2str.dtypes)
        assert all(type(i) is int for i in diagnoses_icd_std[COLUMN.admission_id])
        assert all(type(i) is str for i in after_coerce_id2str[COLUMN.admission_id])

    def test_filter_null_codes(self, diagnoses_icd_std: pd.DataFrame, after_nullcode_filter: pd.DataFrame):
        # no change in column names.
        assert all(after_nullcode_filter.columns == after_nullcode_filter.columns)
        # no change in dtypes
        assert all(after_nullcode_filter.dtypes == after_nullcode_filter.dtypes)
        # one row contained an empty code.
        assert len(after_nullcode_filter) + 1 == len(diagnoses_icd_std)

    def test_add_version_column(self, diagnoses_icd_std: pd.DataFrame, after_add_version_column: pd.DataFrame):
        assert set(diagnoses_icd_std.columns) == set(after_add_version_column.columns) - {COLUMN.version}
        assert all(after_add_version_column[COLUMN.version] == "9")
        assert diagnoses_icd_std.equals(after_add_version_column[diagnoses_icd_std.columns])

    def test_coerce_version2str(
        self,
        diagnoses_icd_std: pd.DataFrame,
    ):
        df = diagnoses_icd_std.iloc[:]
        df[COLUMN.version] = 9
        df2 = MixedICDTableResource_MIMICIII._coerce_version_to_str(df)
        assert all(type(v) is str for v in df2[COLUMN.version])

    def test_mixed_format(
        self, mixed_format: pd.DataFrame, diagnoses_icd_std: pd.DataFrame, updated_manager: CodingSchemesManager
    ):
        pipeline = (
            MixedICDTableResource_MIMICIII._filter_null_codes,
            lambda df: MixedICDTableResource_MIMICIII._coerce_columns_to_str(df, (COLUMN.code,)),
            MixedICDTableResource_MIMICIII._strip_icd_codes,
            MixedICDTableResource_MIMICIII._add_version_column_if_not_exists,
            MixedICDTableResource_MIMICIII._coerce_version_to_str,
        )
        pure_format = TableResource.apply_pipeline(pipeline, diagnoses_icd_std)
        assert pure_format.shape == mixed_format.shape
        assert all(pure_format.columns == mixed_format.columns)
        assert all(pure_format.dtypes == mixed_format.dtypes)
        rest_cols = list(set(pure_format.columns) - {COLUMN.code})
        assert pure_format[rest_cols].reset_index(drop=True).equals(mixed_format[rest_cols].reset_index(drop=True))
        mix_map = updated_manager.map[(MIXED_SCHEME_NAME, "light_icd9")]
        pure_codes = tuple(map(ICD9CM.format, pure_format[COLUMN.code].tolist()))
        mixed_codes = mixed_format[COLUMN.code].tolist()
        (unmixed_codes,) = zip(*list(map(mix_map.data.get, mixed_codes)))
        assert unmixed_codes == pure_codes

    def test_pipeline_without_null_code_filtration(
        self, updated_manager: CodingSchemesManager, diagnoses_icd_std: pd.DataFrame
    ):
        bad_pipeline = (
            lambda df: MixedICDTableResource_MIMICIII._coerce_columns_to_str(df, (COLUMN.code,)),
            MixedICDTableResource_MIMICIII._strip_icd_codes,
            MixedICDTableResource_MIMICIII._add_version_column_if_not_exists,
            MixedICDTableResource_MIMICIII._coerce_version_to_str,
            MixedICDTableResource_MIMICIII._mixed_code_format(MIXED_SCHEME_NAME, updated_manager),
        )
        good_pipeline = (MixedICDTableResource_MIMICIII._filter_null_codes,) + bad_pipeline
        with pytest.raises(Exception):
            _ = TableResource.apply_pipeline(bad_pipeline, diagnoses_icd_std)
        df = TableResource.apply_pipeline(good_pipeline, diagnoses_icd_std)
        assert isinstance(df, pd.DataFrame)
