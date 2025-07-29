from typing import cast

import numpy as np
import pandas as pd
import pytest

import ehrax as rx
from ehrax import CodingSchemesManager, FrozenDict11, CodingScheme, COLUMN
from ehrax.example_datasets.mimic import TableResource, MixedICDTableResource
from ehrax.example_schemes.icd import setup_standard_icd_ccs, CCSICDSchemeSelection
from ehrax.example_schemes.mixed_icd import MixedICDScheme


@pytest.fixture(scope="module")
def standard_icd_manager() -> CodingSchemesManager:
    return setup_standard_icd_ccs(scheme_selection=CCSICDSchemeSelection(dx_icd9=True, dx_icd10=True))


@pytest.fixture(scope="module")
def dx_icd9(standard_icd_manager: CodingSchemesManager) -> CodingScheme:
    return standard_icd_manager.scheme['dx_icd9']


@pytest.fixture(scope="module")
def dx_icd10(standard_icd_manager: CodingSchemesManager) -> CodingScheme:
    return standard_icd_manager.scheme['dx_icd10']


class TestTableResource:
    @pytest.mark.parametrize("df, columns", [
        (pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), ["a", "b"]),
        (pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), ["a"]),
        (pd.DataFrame({"a": [], "b": []}), ["a"]),
        (pd.DataFrame({"a": ["1", "2"], "b": [4, 5]}), ["a"]),
    ])
    def test_coerce_columns_to_str(self, df: pd.DataFrame, columns: list[str]):
        df2 = TableResource._coerce_columns_to_str(df, columns)
        assert all(pd.api.types.is_string_dtype(df2.dtypes[c]) for c in columns)

    def test_preprocess_pipeline(self):
        add1 = lambda x: x + 1
        assert TableResource.apply_pipeline((add1, add1, add1), 2) == 5


ICD9_KEY = "9"
ICD10_KEY = "10"
N_CODES_PER_SCHEME = 10
MIXED_SCHEME_NAME = 'tesTooOO'


class TestMixedICDTableResource:
    @pytest.fixture(scope='class')
    def icd_version_schemes(self) -> FrozenDict11:
        return FrozenDict11({ICD9_KEY: 'dx_icd9', ICD10_KEY: 'dx_icd10'})

    @pytest.fixture(scope='class')
    def supported_space(self, standard_icd_manager: CodingSchemesManager,
                        icd_version_schemes: FrozenDict11) -> pd.DataFrame:
        def codes_df(version: str, codes: tuple[str, ...], desc: tuple[str, ...]) -> pd.DataFrame:
            return pd.DataFrame({str(rx.COLUMN.code): codes, str(rx.COLUMN.version): [version] * len(codes),
                                 str(rx.COLUMN.description): desc})

        S = standard_icd_manager.scheme
        codes = {name: S[name].codes[:N_CODES_PER_SCHEME] for name in icd_version_schemes.values()}
        return pd.concat([codes_df(v, codes[name], tuple(map(S[name].desc.get, codes[name]))) for v, name in
                          icd_version_schemes.items()])

    @pytest.fixture(scope='class')
    def registered_schemes(self, standard_icd_manager: CodingSchemesManager, icd_version_schemes: FrozenDict11,
                           supported_space: pd.DataFrame) -> CodingSchemesManager:
        return MixedICDTableResource._register_scheme(standard_icd_manager, MIXED_SCHEME_NAME, icd_version_schemes,
                                                      supported_space, None)

    @pytest.fixture(scope='class')
    def mixed_scheme(self, registered_schemes: CodingSchemesManager) -> MixedICDScheme:
        return cast(MixedICDScheme, registered_schemes.scheme[MIXED_SCHEME_NAME])

    @pytest.fixture(scope='class')
    def mixed_code_space(self, registered_schemes: CodingSchemesManager, mixed_scheme: MixedICDScheme,
                         supported_space: pd.DataFrame) -> pd.DataFrame:
        return mixed_scheme.mixed_code_format_table(registered_schemes, supported_space)

    def test_mixed_code_space(self, supported_space: pd.DataFrame, mixed_code_space: pd.DataFrame,
                              mixed_scheme: MixedICDScheme, dx_icd9: CodingScheme, dx_icd10: CodingScheme):
        assert supported_space.shape == mixed_code_space.shape
        codes = supported_space[str(COLUMN.code)]
        assert (codes.isin(dx_icd9.codes).astype(int) + codes.isin(dx_icd10.codes).astype(int) == 1).all()
        assert mixed_code_space[COLUMN.code].isin(mixed_scheme.codes).all()

    def test_mixed_scheme_properties(self, mixed_scheme: MixedICDScheme):
        assert isinstance(mixed_scheme, MixedICDScheme)
        assert len(mixed_scheme) == N_CODES_PER_SCHEME * 2

    def test_mixed_schemes_maps(self, mixed_scheme: MixedICDScheme, registered_schemes: CodingSchemesManager,
                                dx_icd9: CodingScheme, dx_icd10: CodingScheme):
        assert (mixed_scheme.name, dx_icd9.name) in registered_schemes.map
        assert (mixed_scheme.name, dx_icd10.name) in registered_schemes.map

    def test_mixed_codes_reversal(self, mixed_scheme: MixedICDScheme, registered_schemes: CodingSchemesManager,
                                  mixed_code_space: pd.DataFrame, supported_space: pd.DataFrame):
        mixed_codes = mixed_code_space[str(COLUMN.code)]
        mixed_as_icd9 = mixed_codes.map(registered_schemes.map[MIXED_SCHEME_NAME, 'dx_icd9'].data).to_numpy()
        mixed_as_icd10 =mixed_codes.map(registered_schemes.map[MIXED_SCHEME_NAME, 'dx_icd10'].data).to_numpy()
        versions = supported_space[str(COLUMN.version)].to_numpy()

        df = supported_space.copy()
        codes = np.where(versions == ICD9_KEY, mixed_as_icd9, mixed_as_icd10)
        assert all(map(lambda c: len(c) == 1 , codes))
        codes = list(map(lambda c: next(iter(c)), codes))
        df[str(COLUMN.code)] = codes
        assert df.equals(supported_space)

