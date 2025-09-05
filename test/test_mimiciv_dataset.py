import ehrax as rx
import equinox as eqx
import numpy as np
import pandas as pd
import pytest
from ehrax.testing.common_setup import DATASET_CONFIG, DATASET_SCHEME_MANAGER, SCHEMES

from .conftest import Dataset


@pytest.fixture(scope="module")
def mimiciv_dataset_without_uom_normalization(
    dataset_tables_with_records: rx.DatasetTables, unit_converter_table: pd.DataFrame
) -> Dataset:
    config = eqx.tree_at(
        lambda x: x.scheme, DATASET_CONFIG, rx.DatasetSchemeConfig(**DATASET_CONFIG.scheme.scheme_fields())
    )
    ds = Dataset(tables=dataset_tables_with_records, config=config)
    return ds._execute_pipeline([rx.SetIndex(), rx.SynchronizeSubjects(), rx.CastTimestamps()], DATASET_SCHEME_MANAGER)


class TestUnitConversionAndFilterInvalidInputRates:
    @pytest.fixture(scope="class")
    def fixed_dataset(self, mimiciv_dataset_without_uom_normalization: Dataset) -> Dataset:
        return rx.ICUInputRateUnitConversion.apply(
            mimiciv_dataset_without_uom_normalization, DATASET_SCHEME_MANAGER, rx.Report()
        )[0]

    @pytest.fixture(scope="class")
    def icu_inputs_unfixed(self, mimiciv_dataset_without_uom_normalization: Dataset):
        return mimiciv_dataset_without_uom_normalization.tables.icu_inputs

    @pytest.fixture(scope="class")
    def icu_inputs_fixed(self, fixed_dataset: Dataset):
        return fixed_dataset.tables.icu_inputs

    @pytest.fixture(scope="class")
    def derived_icu_inputs_cols(self):
        assert DATASET_CONFIG.columns.icu_inputs is not None, (
            "Dataset configuration does not have icu_inputs table defined."
        )
        c = DATASET_CONFIG.columns.icu_inputs
        return [
            c.derived_unit_normalization_factor,
            c.derived_universal_unit,
            c.derived_normalized_amount,
            c.derived_normalized_amount_per_hour,
        ]

    def test_icu_input_rate_unit_conversion(
        self, icu_inputs_fixed: pd.DataFrame, icu_inputs_unfixed: pd.DataFrame, derived_icu_inputs_cols: list[str]
    ):
        assert all(c not in icu_inputs_unfixed.columns for c in derived_icu_inputs_cols)
        assert all(c in icu_inputs_fixed.columns for c in derived_icu_inputs_cols)
        assert DATASET_CONFIG.columns.icu_inputs is not None, (
            "Dataset configuration does not have icu_inputs table defined."
        )
        c = DATASET_CONFIG.columns.icu_inputs

        scheme: rx.CodingSchemeWithUOM = SCHEMES["icu_inputs"]
        # For every (code, unit) pair, a unique normalization factor and universal unit is assigned.
        for (code, unit), inputs_df in icu_inputs_fixed.groupby([c.code, c.amount_unit]):
            assert inputs_df[c.derived_universal_unit].unique() == scheme.universal_unit[code]
            assert (
                inputs_df[c.derived_unit_normalization_factor].unique() == scheme.uom_normalization_factor[code][unit]
            )
            assert inputs_df[c.derived_normalized_amount].equals(
                inputs_df[c.amount] * scheme.uom_normalization_factor[code][unit]
            )

    @pytest.fixture(scope="class")
    def nan_inputs_dataset(self, fixed_dataset: pd.DataFrame):
        icu_inputs = fixed_dataset.tables.icu_inputs.copy()
        assert DATASET_CONFIG.columns.icu_inputs is not None, (
            "Dataset configuration does not have icu_inputs table defined."
        )
        c = DATASET_CONFIG.columns.icu_inputs
        admission_id = icu_inputs.iloc[0][c.admission_id]
        icu_inputs.loc[icu_inputs[c.admission_id] == admission_id, c.derived_normalized_amount_per_hour] = np.nan
        return eqx.tree_at(lambda x: x.tables.icu_inputs, fixed_dataset, icu_inputs)

    @pytest.fixture(scope="class")
    def filtered_dataset(self, nan_inputs_dataset: Dataset):
        return rx.FilterInvalidInputRatesSubjects.apply(nan_inputs_dataset, DATASET_SCHEME_MANAGER, rx.Report())[0]

    def test_filter_invalid_input_rates_subjects(self, nan_inputs_dataset: Dataset, filtered_dataset: Dataset):
        icu_inputs0 = nan_inputs_dataset.tables.icu_inputs
        admissions0 = nan_inputs_dataset.tables.admissions
        static0 = nan_inputs_dataset.tables.static

        icu_inputs1 = filtered_dataset.tables.icu_inputs
        admissions1 = filtered_dataset.tables.admissions
        static1 = filtered_dataset.tables.static
        assert DATASET_CONFIG.columns.icu_inputs is not None, (
            "Dataset configuration does not have icu_inputs table defined."
        )
        assert DATASET_CONFIG.columns.admissions is not None, (
            "Dataset configuration does not have admissions table defined."
        )
        assert icu_inputs0 is not None, "ICU inputs table is not present in the dataset."
        assert icu_inputs1 is not None, "Filtered ICU inputs table is not present in the dataset."
        cicu = DATASET_CONFIG.columns.icu_inputs
        cadm = DATASET_CONFIG.columns.admissions

        admission_id = icu_inputs0.iloc[0][cicu.admission_id]
        subject_id = static0[static0.index == admissions0.loc[admission_id, cadm.subject_id]].index[0]
        subject_admissions = admissions0[admissions0[cadm.subject_id] == subject_id]

        assert subject_id not in static1.index
        assert not subject_admissions.index.isin(admissions1.index).all()
        assert not subject_admissions.index.isin(icu_inputs1[cicu.admission_id]).all()
        assert icu_inputs0[cicu.derived_normalized_amount_per_hour].isna().any()
        assert not icu_inputs1[cicu.derived_normalized_amount_per_hour].isna().any()
