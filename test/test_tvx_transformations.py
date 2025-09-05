from collections.abc import Callable
from typing import Any

import ehrax as rx
import equinox as eqx
import numpy as np
import pandas as pd
import pytest
import tables as tb
from ehrax.testing.common_setup import BINARY_OBSERVATION_CODE_INDEX, DATASET_SCHEME_MANAGER


def test_serialization_multi_subjects(tvx_ehr: rx.TVxEHR, tmpdir: str):
    path = f"{tmpdir}/tvx_ehr"
    tvx_ehr.save(path)
    loaded_ehr = rx.TVxEHR.load(path)
    assert tvx_ehr.equals(loaded_ehr)


class TestSampleSubjects:
    @pytest.fixture(params=[(1, 3), (111, 5)], scope="class")
    def sampled_tvx_ehr(self, tvx_ehr: rx.TVxEHR, request):
        seed, offset = request.param
        n_subjects = len(tvx_ehr.dataset.tables.static) // 5
        sample = rx.TVxEHRSampleConfig(seed=seed, n_subjects=n_subjects, offset=offset)
        tvx_ehr = eqx.tree_at(lambda x: x.config.sample, tvx_ehr, sample, is_leaf=lambda x: x is None)
        return rx.SampleSubjects.apply(tvx_ehr, DATASET_SCHEME_MANAGER, rx.TVxReport())[0]

    def test_sample_subjects(self, tvx_ehr: rx.TVxEHR, sampled_tvx_ehr: rx.TVxEHR):
        original_subjects = tvx_ehr.dataset.tables.static.index
        sampled_subjects = sampled_tvx_ehr.dataset.tables.static.index
        assert len(sampled_subjects) == len(original_subjects) // 5
        assert len(set(sampled_subjects)) == len(sampled_subjects)
        assert set(sampled_subjects).issubset(set(original_subjects))

    def test_ehr_serialization(self, tvx_ehr: rx.TVxEHR, sampled_tvx_ehr: rx.TVxEHR, tmpdir: str):
        path1 = f"{tmpdir}/tvx_ehr"
        path2 = f"{tmpdir}/sampled_tvx_ehr"

        tvx_ehr.save(path1)
        loaded_tvx_ehr = rx.TVxEHR.load(path1)

        sampled_tvx_ehr.save(path2)
        loaded_sampled_tvx_ehr = rx.TVxEHR.load(path2)

        assert not tvx_ehr.equals(sampled_tvx_ehr)
        assert not tvx_ehr.equals(loaded_sampled_tvx_ehr)
        assert not loaded_tvx_ehr.equals(sampled_tvx_ehr)
        assert not loaded_tvx_ehr.equals(loaded_sampled_tvx_ehr)
        assert sampled_tvx_ehr.equals(loaded_sampled_tvx_ehr)


class TestTrainableTransformer:
    @pytest.fixture(params=["obs", "icu_inputs"], scope="class")
    def scalable_table_name(self, request):
        return request.param

    @pytest.fixture(params=[True, False], scope="class")
    def use_float16(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def scaler_class(self, scalable_table_name) -> type[rx.TrainableTransformation]:
        return {"obs": rx.ObsAdaptiveScaler, "icu_inputs": rx.InputScaler}[scalable_table_name]

    @pytest.fixture(params=[True, False], scope="class")
    def numerical_processor_config(self, request) -> rx.DatasetNumericalProcessorsConfig:
        null = request.param
        if null:
            return rx.DatasetNumericalProcessorsConfig()
        scalers_conf = rx.ScalersConfig(
            obs=rx.ScalerConfig(use_float16=True), icu_inputs=rx.ScalerConfig(use_float16=True)
        )
        outliers_conf = rx.OutlierRemoversConfig(obs=rx.IQROutlierRemoverConfig())
        return rx.DatasetNumericalProcessorsConfig(scalers_conf, outliers_conf)

    @pytest.fixture(scope="class")
    def large_scalable_split_ehr(self, tvx_ehr: rx.TVxEHR, scalable_table_name: str, use_float16: bool):
        if len(getattr(tvx_ehr.dataset.tables, scalable_table_name)) == 0:
            raise pytest.skip(f"No {scalable_table_name} table found in dataset.")
        subjects = tvx_ehr.dataset.tables.static.index.tolist()
        tvx_ehr = eqx.tree_at(lambda x: x.splits, tvx_ehr, (tuple(subjects),), is_leaf=lambda x: x is None)

        tvx_ehr = eqx.tree_at(
            lambda x: getattr(x.config.numerical_processors.scalers, scalable_table_name),
            tvx_ehr,
            rx.ScalerConfig(use_float16=use_float16),
            is_leaf=lambda x: x is None,
        )
        return tvx_ehr

    @pytest.fixture(scope="class")
    def scaled_ehr(self, large_scalable_split_ehr, scaler_class: type[rx.TrainableTransformation]):
        return scaler_class.apply(large_scalable_split_ehr, DATASET_SCHEME_MANAGER, rx.TVxReport())[0]

    def test_trainable_transformer(
        self, large_scalable_split_ehr: rx.TVxEHR, scaled_ehr: rx.TVxEHR, scalable_table_name: str, use_float16: bool
    ):
        assert getattr(large_scalable_split_ehr.numerical_processors.scalers, scalable_table_name) is None
        scaler = getattr(scaled_ehr.numerical_processors.scalers, scalable_table_name)
        assert isinstance(scaler, rx.CodedValueScaler)
        assert scaler.table_getter(scaled_ehr.dataset) is getattr(scaled_ehr.dataset.tables, scalable_table_name)
        assert scaler.table_getter(large_scalable_split_ehr.dataset) is getattr(
            large_scalable_split_ehr.dataset.tables, scalable_table_name
        )
        assert scaler.config.use_float16 == use_float16

        table0 = scaler.table_getter(large_scalable_split_ehr.dataset)
        table1 = scaler.table_getter(scaled_ehr.dataset)
        assert scaler.value_column in table1.columns
        assert scaler.code_column in table1.columns
        assert table1 is not table0
        if use_float16:
            assert table1[scaler.value_column].dtype == np.float16
        else:
            assert table1[scaler.value_column].dtype == table0[scaler.value_column].dtype

    @pytest.fixture(scope="class")
    def processed_ehr(
        self, large_scalable_split_ehr: rx.TVxEHR, numerical_processor_config: rx.DatasetNumericalProcessorsConfig
    ) -> rx.TVxEHR:
        large_scalable_split_ehr = eqx.tree_at(
            lambda x: x.config.numerical_processors, large_scalable_split_ehr, numerical_processor_config
        )
        return large_scalable_split_ehr._execute_pipeline(
            [rx.ObsIQROutlierRemover(), rx.InputScaler(), rx.ObsAdaptiveScaler()], DATASET_SCHEME_MANAGER
        )

    @pytest.fixture(scope="class")
    def fitted_numerical_processors(self, processed_ehr: rx.TVxEHR) -> rx.DatasetNumericalProcessors:
        return processed_ehr.numerical_processors

    def test_numerical_processors_serialization(
        self, fitted_numerical_processors: rx.DatasetNumericalProcessors, tmpdir: str
    ):
        path = f"{tmpdir}/numerical_processors.h5"
        with tb.open_file(path, "w") as f:
            fitted_numerical_processors.save(f.create_group("/", "numerical_processors"))

        with tb.open_file(path, "r") as f:
            loaded_numerical_processors = rx.DatasetNumericalProcessors.load(f.root.numerical_processors)
        assert fitted_numerical_processors.equals(loaded_numerical_processors)

    def test_ehr_serialization(
        self, tvx_ehr: rx.TVxEHR, large_scalable_split_ehr: rx.TVxEHR, processed_ehr: rx.TVxEHR, tmpdir: str
    ):
        assert not tvx_ehr.equals(large_scalable_split_ehr)
        assert not tvx_ehr.equals(processed_ehr)
        assert not large_scalable_split_ehr.equals(processed_ehr)

        split_path = f"{tmpdir}/split_ehr"
        processed_path = f"{tmpdir}/processed_ehr"
        large_scalable_split_ehr.save(split_path)
        loaded_split_ehr = rx.TVxEHR.load(split_path)
        assert large_scalable_split_ehr.equals(loaded_split_ehr)

        processed_ehr.save(processed_path)
        loaded_processed_ehr = rx.TVxEHR.load(processed_path)
        assert processed_ehr.equals(loaded_processed_ehr)


# def test_obs_minmax_scaler(int_dataset: Dataset):
#     assert False
#
#
# def test_obs_adaptive_scaler(int_dataset: Dataset):
#     assert False
#
#
# def test_obs_iqr_outlier_remover(dataset: Dataset):
#     assert False


# @pytest.mark.parametrize('splits', [[0.5], [0.2, 0.5, 0.7], [0.1, 0.2, 0.3, 0.4, 0.5]])
# def test_random_splits(dataset: Dataset, splits: list[float]):
# The logic of splits already tested in test.ehr.dataset.test_dataset.
# Maybe assert that functions are called with the correct arguments.
# pass


class TestTVxConcepts:
    @pytest.fixture(params=["gender", "age", "ethnicity"], scope="class")
    def tvx_ehr_demographic_config(self, request) -> rx.DemographicVectorConfig:
        config = rx.DemographicVectorConfig(False, False, False)
        return eqx.tree_at(lambda x: getattr(x, request.param), config, True)

    @pytest.fixture(scope="class")
    def tvx_ehr_configured_demographic(
        self, tvx_ehr: rx.TVxEHR, tvx_ehr_demographic_config: rx.DemographicVectorConfig
    ) -> rx.TVxEHR:
        return eqx.tree_at(lambda x: x.config.demographic, tvx_ehr, tvx_ehr_demographic_config)

    @pytest.fixture(scope="class")
    def tvx_concepts_static(self, tvx_ehr_configured_demographic: rx.TVxEHR) -> dict[str, rx.StaticInfo]:
        return rx.TVxConcepts._static_info(tvx_ehr_configured_demographic, DATASET_SCHEME_MANAGER, rx.TVxReport())[0]

    def test_tvx_concepts_static(
        self,
        tvx_ehr: rx.TVxEHR,
        tvx_concepts_static: dict[str, rx.StaticInfo],
        tvx_ehr_demographic_config: rx.DemographicVectorConfig,
    ):
        assert len(tvx_concepts_static) == len(tvx_ehr.dataset.tables.static)
        for subject_id, static_info in tvx_concepts_static.items():
            if tvx_ehr_demographic_config.gender:
                assert static_info.gender is not None
                assert len(static_info.gender.vec) == len(tvx_ehr.scheme_proxy(DATASET_SCHEME_MANAGER).gender)
                assert static_info.gender.vec.sum() == 1
                assert static_info.gender.scheme == tvx_ehr.scheme_proxy(DATASET_SCHEME_MANAGER).gender.name
            else:
                assert static_info.gender is None

            if tvx_ehr_demographic_config.age:
                assert static_info.date_of_birth is not None
                assert static_info.age(static_info.date_of_birth) == 0.0
            else:
                assert static_info.date_of_birth is None

            if tvx_ehr_demographic_config.ethnicity:
                assert static_info.ethnicity is not None
                assert len(static_info.ethnicity.vec) == len(tvx_ehr.scheme_proxy(DATASET_SCHEME_MANAGER).ethnicity)
                assert static_info.ethnicity.vec.sum() == 1
                assert static_info.ethnicity.scheme == tvx_ehr.scheme_proxy(DATASET_SCHEME_MANAGER).ethnicity.name

    @pytest.fixture(scope="class")
    def tvx_ehr_with_dx(self, tvx_ehr: rx.TVxEHR) -> rx.TVxEHR:
        if len(tvx_ehr.dataset.tables.dx_discharge) == 0:
            raise pytest.skip("No diagnoses table found in dataset.")
        n = len(tvx_ehr.dataset.tables.admissions)
        c_admission_id = tvx_ehr.dataset.config.columns.admissions.admission_id
        random_admission_id = tvx_ehr.dataset.tables.admissions.index[n // 2]
        dx_discharge = tvx_ehr.dataset.tables.dx_discharge
        dx_discharge = dx_discharge[dx_discharge[c_admission_id] != random_admission_id]
        return eqx.tree_at(lambda x: x.dataset.tables.dx_discharge, tvx_ehr, dx_discharge)

    @pytest.fixture(scope="class")
    def admission_dx_codes(self, tvx_ehr_with_dx: rx.TVxEHR) -> dict[str, rx.CodesVector]:
        return rx.TVxConcepts._dx_discharge(tvx_ehr_with_dx, DATASET_SCHEME_MANAGER)[0]

    @pytest.fixture(scope="class")
    def admission_dx_codeset(self, tvx_ehr_with_dx: rx.TVxEHR) -> dict[str, set[str]]:
        return rx.TVxConcepts._dx_discharge(tvx_ehr_with_dx, DATASET_SCHEME_MANAGER)[1]

    def test_admission_dx_codes(self, tvx_ehr_with_dx: rx.TVxEHR, admission_dx_codes: dict[str, rx.CodesVector]):
        assert set(admission_dx_codes.keys()) == set(tvx_ehr_with_dx.admission_ids)
        for admission_id, codes in admission_dx_codes.items():
            assert codes.vec.dtype == bool
            assert codes.scheme == tvx_ehr_with_dx.scheme_proxy(DATASET_SCHEME_MANAGER).dx_discharge.name
            assert len(codes.vec) == len(tvx_ehr_with_dx.scheme_proxy(DATASET_SCHEME_MANAGER).dx_discharge)

    @pytest.fixture(scope="class")
    def admission_dx_history_codes(
        self, tvx_ehr_with_dx: rx.TVxEHR, admission_dx_codes: dict[str, rx.CodesVector]
    ) -> dict[str, rx.CodesVector]:
        return rx.TVxConcepts._dx_discharge_history(tvx_ehr_with_dx, DATASET_SCHEME_MANAGER, admission_dx_codes)

    def test_admission_dx_history_codes(
        self,
        tvx_ehr_with_dx: rx.TVxEHR,
        admission_dx_codes: dict[str, rx.CodesVector],
        admission_dx_history_codes: dict[str, rx.CodesVector],
    ):
        assert set(admission_dx_history_codes.keys()) == set(tvx_ehr_with_dx.admission_ids)
        assert set(admission_dx_history_codes.keys()) == set(admission_dx_codes.keys())
        for subject_id, admission_ids in tvx_ehr_with_dx.subjects_sorted_admission_ids.items():
            assert admission_dx_history_codes[admission_ids[0]].vec.sum() == 0

            accumulation = np.zeros(len(tvx_ehr_with_dx.scheme_proxy(DATASET_SCHEME_MANAGER).dx_discharge), dtype=bool)
            for i, admission_id in enumerate(admission_ids):
                assert admission_id in admission_dx_history_codes
                history = admission_dx_history_codes[admission_id]
                assert history.vec.dtype == bool
                assert history.scheme == tvx_ehr_with_dx.scheme_proxy(DATASET_SCHEME_MANAGER).dx_discharge.name
                assert len(history.vec) == len(tvx_ehr_with_dx.scheme_proxy(DATASET_SCHEME_MANAGER).dx_discharge)
                assert (history.vec == accumulation).all()
                if admission_id in admission_dx_codes:
                    accumulation |= admission_dx_codes[admission_id].vec

    @pytest.fixture(scope="class")
    def admission_outcome(
        self, tvx_ehr_with_dx: rx.TVxEHR, admission_dx_codeset: dict[str, set[str]]
    ) -> dict[str, rx.CodesVector]:
        return rx.TVxConcepts._outcome(tvx_ehr_with_dx, DATASET_SCHEME_MANAGER, admission_dx_codeset)

    def test_admission_outcome(
        self,
        tvx_ehr_with_dx: rx.TVxEHR,
        admission_dx_codes: dict[str, rx.CodesVector],
        admission_outcome: dict[str, rx.CodesVector],
    ):
        assert set(admission_outcome.keys()) == set(admission_dx_codes.keys())
        outcome_scheme = tvx_ehr_with_dx.scheme_proxy(DATASET_SCHEME_MANAGER).outcome
        for admission_id, outcome in admission_outcome.items():
            assert outcome.scheme == outcome_scheme.name
            assert len(outcome.vec) == len(outcome_scheme)

    @pytest.fixture(scope="class")
    def tvx_ehr_with_icu_inputs(self, tvx_ehr: rx.TVxEHR) -> rx.TVxEHR:
        if len(tvx_ehr.dataset.tables.icu_inputs) == 0:
            raise pytest.skip("No icu_inputs table found in dataset.")
        return tvx_ehr

    @pytest.fixture(scope="class")
    def admission_icu_inputs(self, tvx_ehr_with_icu_inputs: rx.TVxEHR) -> dict[str, rx.InpatientInput]:
        return rx.TVxConcepts._icu_inputs(tvx_ehr_with_icu_inputs, DATASET_SCHEME_MANAGER)

    def test_admission_icu_inputs(
        self, tvx_ehr_with_icu_inputs: rx.TVxEHR, admission_icu_inputs: dict[str, rx.InpatientInput]
    ):
        icu_inputs = tvx_ehr_with_icu_inputs.dataset.tables.icu_inputs
        c_admission_id = tvx_ehr_with_icu_inputs.dataset.config.columns.admissions.admission_id
        assert set(admission_icu_inputs.keys()).issubset(set(tvx_ehr_with_icu_inputs.admission_ids))
        assert sum(len(inputs.starttime) for inputs in admission_icu_inputs.values()) == len(icu_inputs)

        for admission_id, admission_inputs_df in icu_inputs.groupby(c_admission_id):
            tvx_inputs = admission_icu_inputs[admission_id]
            assert len(tvx_inputs.starttime) == len(admission_inputs_df)
            assert all(
                tvx_inputs.code_index
                < len(tvx_ehr_with_icu_inputs.dataset.scheme_proxy(DATASET_SCHEME_MANAGER).icu_inputs)
            )

    @pytest.fixture(scope="class")
    def tvx_ehr_with_obs(self, tvx_ehr: rx.TVxEHR) -> rx.TVxEHR:
        if len(tvx_ehr.dataset.tables.obs) == 0:
            raise pytest.skip("No observations table found in dataset.")
        return tvx_ehr

    @pytest.fixture(scope="class")
    def admission_obs(self, tvx_ehr_with_obs: rx.TVxEHR) -> dict[str, rx.InpatientObservables]:
        return rx.TVxConcepts._observables(tvx_ehr_with_obs, DATASET_SCHEME_MANAGER, rx.TVxReport())[0]

    def test_admission_obs(self, tvx_ehr_with_obs: rx.TVxEHR, admission_obs: dict[str, rx.InpatientObservables]):
        obs_df = tvx_ehr_with_obs.dataset.tables.obs
        c_admission_id = tvx_ehr_with_obs.dataset.config.columns.admissions.admission_id
        assert set(admission_obs.keys()) == set(tvx_ehr_with_obs.admission_ids)
        assert sum(obs.mask.sum() for obs in admission_obs.values()) == len(obs_df)

        for admission_id, admission_obs_df in obs_df.groupby(c_admission_id):
            tvx_obs = admission_obs[admission_id]
            vals = tvx_obs.value.flatten()[tvx_obs.mask.flatten()]
            tvx_time_vals = pd.Series(tvx_obs.time)
            tbl_time_vals = pd.Series(np.unique(admission_obs_df[rx.COLUMN.time]))

            tvx_vals_count = pd.Series(vals, name=rx.COLUMN.measurement).value_counts().sort_index()
            tbl_vals_count = admission_obs_df[rx.COLUMN.measurement].astype(vals.dtype).value_counts().sort_index()

            assert tvx_vals_count.equals(tbl_vals_count)
            assert tvx_time_vals.equals(tbl_time_vals)
            assert tvx_obs.mask.sum() == len(admission_obs_df)
            assert tvx_obs.value.shape[1] == len(tvx_ehr_with_obs.dataset.scheme_proxy(DATASET_SCHEME_MANAGER).obs)

    @pytest.fixture(scope="class")
    def tvx_ehr_with_hosp_procedures(self, tvx_ehr: rx.TVxEHR) -> rx.TVxEHR:
        if len(tvx_ehr.dataset.tables.hosp_procedures) == 0:
            raise pytest.skip("No hospital procedures table found in dataset.")
        return tvx_ehr

    @pytest.fixture(scope="class")
    def admission_hosp_procedures(self, tvx_ehr_with_hosp_procedures: rx.TVxEHR) -> dict[str, rx.InpatientInput]:
        return rx.TVxConcepts._hosp_procedures(tvx_ehr_with_hosp_procedures, DATASET_SCHEME_MANAGER)

    def test_admission_hosp_procedures(
        self, tvx_ehr_with_hosp_procedures: rx.TVxEHR, admission_hosp_procedures: dict[str, rx.InpatientInput]
    ):
        hosp_procedures = tvx_ehr_with_hosp_procedures.dataset.tables.hosp_procedures
        c_admission_id = tvx_ehr_with_hosp_procedures.dataset.config.columns.admissions.admission_id
        assert set(admission_hosp_procedures.keys()).issubset(set(tvx_ehr_with_hosp_procedures.admission_ids))
        assert sum(len(proc.starttime) for proc in admission_hosp_procedures.values() if proc is not None) == len(
            hosp_procedures
        )

        for admission_id, admission_hosp_procedures_df in hosp_procedures.groupby(c_admission_id):
            tvx_hosp_proc = admission_hosp_procedures[admission_id]
            assert len(tvx_hosp_proc.starttime) == len(admission_hosp_procedures_df)
            assert all(
                tvx_hosp_proc.code_index
                < len(tvx_ehr_with_hosp_procedures.scheme_proxy(DATASET_SCHEME_MANAGER).hosp_procedures)
            )

    @pytest.fixture(scope="class")
    def tvx_ehr_with_icu_procedures(self, tvx_ehr: rx.TVxEHR) -> rx.TVxEHR:
        if len(tvx_ehr.dataset.tables.icu_procedures) == 0:
            raise pytest.skip("No icu procedures table found in dataset.")
        return tvx_ehr

    @pytest.fixture(scope="class")
    def admission_icu_procedures(self, tvx_ehr_with_icu_procedures: rx.TVxEHR) -> dict[str, rx.InpatientInput]:
        return rx.TVxConcepts._icu_procedures(tvx_ehr_with_icu_procedures, DATASET_SCHEME_MANAGER)

    def test_admission_icu_procedures(
        self, tvx_ehr_with_icu_procedures: rx.TVxEHR, admission_icu_procedures: dict[str, rx.InpatientInput]
    ):
        icu_procedures = tvx_ehr_with_icu_procedures.dataset.tables.icu_procedures
        c_admission_id = tvx_ehr_with_icu_procedures.dataset.config.columns.admissions.admission_id
        assert set(admission_icu_procedures.keys()).issubset(set(tvx_ehr_with_icu_procedures.admission_ids))
        assert sum(len(proc.starttime) for proc in admission_icu_procedures.values() if proc is not None) == len(
            icu_procedures
        )

        for admission_id, admission_icu_procedures_df in icu_procedures.groupby(c_admission_id):
            tvx_icu_proc = admission_icu_procedures[str(admission_id)]
            assert len(tvx_icu_proc.starttime) == len(admission_icu_procedures_df)
            assert all(
                tvx_icu_proc.code_index
                < len(tvx_ehr_with_icu_procedures.scheme_proxy(DATASET_SCHEME_MANAGER).icu_procedures)
            )

    @pytest.fixture(scope="class")
    def tvx_ehr_with_all_interventions(self, tvx_ehr: rx.TVxEHR) -> rx.TVxEHR:
        if tvx_ehr.dataset.tables.icu_procedures is None or len(tvx_ehr.dataset.tables.icu_procedures) == 0:
            raise pytest.skip("No icu procedures table found in dataset.")
        if tvx_ehr.dataset.tables.hosp_procedures is None or len(tvx_ehr.dataset.tables.hosp_procedures) == 0:
            raise pytest.skip("No hospital procedures table found in dataset.")
        if tvx_ehr.dataset.tables.icu_inputs is None or len(tvx_ehr.dataset.tables.icu_inputs) == 0:
            raise pytest.skip("No icu inputs table found in dataset.")
        return tvx_ehr

    @pytest.fixture(scope="class")
    def admission_interventions(
        self, tvx_ehr_with_all_interventions: rx.TVxEHR
    ) -> dict[str, rx.InpatientInterventions]:
        return rx.TVxConcepts._interventions(tvx_ehr_with_all_interventions, DATASET_SCHEME_MANAGER, rx.TVxReport())[0]

    def test_admission_interventions(
        self, tvx_ehr_with_all_interventions: rx.TVxEHR, admission_interventions: dict[str, rx.InpatientInterventions]
    ):
        assert set(admission_interventions.keys()) == set(tvx_ehr_with_all_interventions.admission_ids)
        for attr in ("icu_procedures", "hosp_procedures", "icu_inputs"):
            table = getattr(tvx_ehr_with_all_interventions.dataset.tables, attr)
            interventions = {admission_id: getattr(v, attr) for admission_id, v in admission_interventions.items()}
            assert sum(len(v.starttime) for v in interventions.values() if v is not None) == len(table)

    @pytest.fixture(params=["interventions", "observables"], scope="class")
    def tvx_ehr_conf_concept(self, tvx_ehr: rx.TVxEHR, request) -> rx.TVxEHR:
        concept_name = request.param
        conf = tvx_ehr.config
        for cname in ("interventions", "observables"):
            conf = eqx.tree_at(lambda x: getattr(x, cname), conf, concept_name == cname)
        return eqx.tree_at(lambda x: x.config, tvx_ehr, conf)

    @pytest.fixture(scope="class")
    def tvx_ehr_concept(self, tvx_ehr_conf_concept: rx.TVxEHR):
        return tvx_ehr_conf_concept._execute_pipeline([rx.TVxConcepts()], DATASET_SCHEME_MANAGER)

    def test_tvx_ehr_concept(self, tvx_ehr_concept: rx.TVxEHR):
        assert set(tvx_ehr_concept.subjects.keys()) == set(tvx_ehr_concept.dataset.tables.static.index)
        if tvx_ehr_concept.config.interventions:
            assert all(
                adm.interventions is not None
                for patient in tvx_ehr_concept.subjects.values()
                for adm in patient.admissions
            )
        else:
            assert all(
                adm.interventions is None for patient in tvx_ehr_concept.subjects.values() for adm in patient.admissions
            )
        if tvx_ehr_concept.config.observables:
            assert all(
                adm.observables is not None
                for patient in tvx_ehr_concept.subjects.values()
                for adm in patient.admissions
            )
        else:
            assert all(
                adm.observables is None for patient in tvx_ehr_concept.subjects.values() for adm in patient.admissions
            )

        for subject_id, sorted_admission_ids in tvx_ehr_concept.subjects_sorted_admission_ids.items():
            assert [adm.admission_id for adm in tvx_ehr_concept.subjects[subject_id].admissions] == sorted_admission_ids

    @pytest.fixture(scope="class")
    def mutated_ehr_concept(self, tvx_ehr_concept: rx.TVxEHR, tvx_ehr_conf_concept: rx.TVxEHR):
        if tvx_ehr_conf_concept.config.interventions:
            s0 = tvx_ehr_concept.subject_ids[0]
            a0 = tvx_ehr_concept.subjects[s0].admissions[0]
            a1 = eqx.tree_at(lambda x: x.interventions.icu_inputs.rate, a0, a0.interventions.icu_inputs.rate + 0.1)
            return eqx.tree_at(lambda x: x.subjects[s0].admissions[0], tvx_ehr_concept, a1)
        else:
            s0 = tvx_ehr_concept.subject_ids[0]
            a0 = tvx_ehr_concept.subjects[s0].admissions[0]
            a1 = eqx.tree_at(lambda x: x.observables.value, a0, a0.observables.value + 0.1)
            return eqx.tree_at(lambda x: x.subjects[s0].admissions[0], tvx_ehr_concept, a1)

    def test_ehr_serialization(
        self, tvx_ehr_conf_concept: rx.TVxEHR, tvx_ehr_concept: rx.TVxEHR, mutated_ehr_concept: rx.TVxEHR, tmpdir: str
    ):
        ehr_list = [tvx_ehr_conf_concept, tvx_ehr_concept, mutated_ehr_concept]
        for i, ehr_i in enumerate(ehr_list):
            path = f"{tmpdir}/tvx_ehr_{i}"
            ehr_i.save(path)
            loaded_ehr_i = rx.TVxEHR.load(path)
            for j in range(i + 1, len(ehr_list)):
                ehr_j = ehr_list[j]
                assert loaded_ehr_i.equals(ehr_j) == (i == j)


class TestInterventionSegmentation:
    @pytest.fixture(scope="class")
    def tvx_ehr_concept(self, tvx_ehr: rx.TVxEHR):
        tvx_ehr = eqx.tree_at(lambda x: x.config.interventions, tvx_ehr, True)
        tvx_ehr = eqx.tree_at(lambda x: x.config.observables, tvx_ehr, True)
        return tvx_ehr._execute_pipeline([rx.TVxConcepts()], DATASET_SCHEME_MANAGER)

    @pytest.fixture(scope="class")
    def tvx_ehr_segmented(self, tvx_ehr_concept: rx.TVxEHR) -> rx.TVxEHR:
        tvx_ehr_concept = eqx.tree_at(lambda x: x.config.interventions_segmentation, tvx_ehr_concept, True)
        return tvx_ehr_concept._execute_pipeline([rx.InterventionSegmentation()], DATASET_SCHEME_MANAGER)

    def test_segmentation(self, tvx_ehr_concept: rx.TVxEHR, tvx_ehr_segmented: rx.TVxEHR):
        first_patient = next(iter(tvx_ehr_concept.subjects.values()))
        first_segmented_patient = next(iter(tvx_ehr_segmented.subjects.values()))
        assert isinstance(first_patient, rx.Patient)
        assert isinstance(first_segmented_patient, rx.SegmentedPatient)
        assert isinstance(first_patient.admissions[0], rx.Admission)
        assert isinstance(first_segmented_patient.admissions[0], rx.SegmentedAdmission)
        assert isinstance(first_patient.admissions[0].observables, rx.InpatientObservables)
        assert isinstance(first_segmented_patient.admissions[0].observables, rx.SegmentedInpatientObservables)
        assert isinstance(first_segmented_patient.admissions[0].observables[0], rx.InpatientObservables)
        assert isinstance(first_patient.admissions[0].interventions, rx.InpatientInterventions)
        assert isinstance(first_segmented_patient.admissions[0].interventions, rx.SegmentedInpatientInterventions)

    @pytest.fixture(
        params=["mutate_obs", "intervention_time", "mutate_icu_proc", "mutate_hosp_proc", "mutate_icu_input"],
        scope="class",
    )
    def mutated_ehr_concept(self, tvx_ehr_segmented: rx.TVxEHR, request):
        s0 = tvx_ehr_segmented.subject_ids[0]
        a0 = tvx_ehr_segmented.subjects[s0].admissions[0]
        if request.param == "mutate_obs":
            a1 = eqx.tree_at(lambda x: x.observables.time, a0, a0.observables.time + 0.001)
        elif request.param == "intervention_time":
            a1 = eqx.tree_at(lambda x: x.interventions.time, a0, a0.interventions.time + 1e-6)
        elif request.param == "mutate_icu_proc":
            if a0.interventions.icu_procedures is None:
                raise pytest.skip("No icu procedures in admission.")
            a1 = eqx.tree_at(lambda x: x.interventions.icu_procedures, a0, ~a0.interventions.icu_procedures)

        elif request.param == "mutate_hosp_proc":
            if a0.interventions.hosp_procedures is None:
                raise pytest.skip("No hospital procedures in admission.")
            a1 = eqx.tree_at(lambda x: x.interventions.hosp_procedures, a0, ~a0.interventions.hosp_procedures)
        elif request.param == "mutate_icu_input":
            if a0.interventions.icu_inputs is None:
                raise pytest.skip("No icu inputs in admission.")
            assert a0.interventions.icu_inputs is not None
            a1 = eqx.tree_at(lambda x: x.interventions.icu_inputs, a0, a0.interventions.icu_inputs + 0.1)
        else:
            raise ValueError(f"Invalid param: {request.param}")

        return eqx.tree_at(lambda x: x.subjects[s0].admissions[0], tvx_ehr_segmented, a1)

    def test_ehr_serialization(
        self,
        tvx_ehr_concept: rx.TVxEHR,
        tvx_ehr_segmented: rx.TVxEHR,
        mutated_ehr_concept: rx.TVxEHR,
        tmpdir: str,
    ):
        ehr_list = [tvx_ehr_concept, tvx_ehr_segmented, mutated_ehr_concept]
        for i, ehr_i in enumerate(ehr_list):
            path = f"{tmpdir}/tvx_ehr_{i}"
            ehr_i.save(path)
            loaded_ehr_i = rx.TVxEHR.load(path)
            for j in range(i + 1, len(ehr_list)):
                ehr_j = ehr_list[j]
                assert loaded_ehr_i.equals(ehr_j) == (i == j)


class TestObsTimeBinning:
    @pytest.fixture(scope="class")
    def tvx_ehr_concept(self, tvx_ehr: rx.TVxEHR):
        tvx_ehr = eqx.tree_at(lambda x: x.config.interventions, tvx_ehr, False)
        tvx_ehr = eqx.tree_at(lambda x: x.config.observables, tvx_ehr, True)
        return tvx_ehr._execute_pipeline([rx.TVxConcepts()], DATASET_SCHEME_MANAGER)

    @pytest.fixture(scope="class")
    def tvx_ehr_binned(self, tvx_ehr_concept: rx.TVxEHR) -> rx.TVxEHR:
        tvx_ehr_concept = eqx.tree_at(
            lambda x: x.config.time_binning, tvx_ehr_concept, 12.0, is_leaf=lambda x: x is None
        )
        return tvx_ehr_concept._execute_pipeline([rx.ObsTimeBinning()], DATASET_SCHEME_MANAGER)

    def test_binning(self, tvx_ehr_concept: rx.TVxEHR, tvx_ehr_binned: rx.TVxEHR):
        assert all(
            (np.diff(o.time) == tvx_ehr_binned.config.time_binning).all()
            for o in tvx_ehr_binned.iter_obs()
            if len(o) > 1
        )


class TestLeadExtraction:
    @pytest.fixture(scope="class")
    def tvx_ehr_concept(self, tvx_ehr: rx.TVxEHR):
        tvx_ehr = eqx.tree_at(lambda x: x.config.interventions, tvx_ehr, False)
        tvx_ehr = eqx.tree_at(lambda x: x.config.observables, tvx_ehr, True)
        obs_scheme = tvx_ehr.scheme_proxy(DATASET_SCHEME_MANAGER).obs
        lead_config = rx.LeadingObservableExtractorConfig(
            leading_hours=[1.0, 2.0],
            scheme=obs_scheme.name,
            entry_neglect_window=0.0,
            recovery_window=0.0,
            minimum_acquisitions=0,
            observable_code=obs_scheme.index2code[BINARY_OBSERVATION_CODE_INDEX],
        )
        tvx_ehr = eqx.tree_at(lambda x: x.config.leading_observable, tvx_ehr, lead_config, is_leaf=lambda x: x is None)
        return tvx_ehr._execute_pipeline([rx.TVxConcepts()], DATASET_SCHEME_MANAGER)

    @pytest.fixture(scope="class")
    def tvx_ehr_lead(self, tvx_ehr_concept: rx.TVxEHR) -> rx.TVxEHR:
        return tvx_ehr_concept._execute_pipeline([rx.LeadingObservableExtraction()], DATASET_SCHEME_MANAGER)

    def test_lead(self, tvx_ehr_concept: rx.TVxEHR, tvx_ehr_lead: rx.TVxEHR):
        first_patient0 = next(iter(tvx_ehr_concept.subjects.values()))
        first_patient1 = next(iter(tvx_ehr_lead.subjects.values()))
        assert isinstance(first_patient0, rx.Patient)
        assert isinstance(first_patient1, rx.Patient)
        assert isinstance(first_patient0.admissions[0], rx.Admission)
        assert isinstance(first_patient1.admissions[0], rx.Admission)
        assert isinstance(first_patient0.admissions[0].observables, rx.InpatientObservables)
        assert isinstance(first_patient1.admissions[0].observables, rx.InpatientObservables)
        assert first_patient0.admissions[0].leading_observable is None
        assert isinstance(first_patient1.admissions[0].leading_observable, rx.InpatientObservables)

        for subject_id, patient in tvx_ehr_lead.subjects.items():
            for admission in patient.admissions:
                assert admission.leading_observable is not None
                assert admission.observables is not None
                assert len(admission.leading_observable) == len(admission.observables)
                assert admission.leading_observable.value.shape[1] == len(
                    tvx_ehr_lead.config.leading_observable.leading_hours
                )

    def test_ehr_serialization(self, tvx_ehr_concept: rx.TVxEHR, tvx_ehr_lead: rx.TVxEHR, tmpdir: str):
        ehr_list = [tvx_ehr_concept, tvx_ehr_lead]
        for i, ehr_i in enumerate(ehr_list):
            path = f"{tmpdir}/tvx_ehr_{i}"
            ehr_i.save(path)
            loaded_ehr_i = rx.TVxEHR.load(path)
            for j in range(i + 1, len(ehr_list)):
                ehr_j = ehr_list[j]
                assert loaded_ehr_i.equals(ehr_j) == (i == j)

    @pytest.fixture(params=[(lambda tvx: tvx.dataset.tables, rx.DatasetTables), (lambda tvx: tvx.subjects, dict)])
    def tvx_ehr_defer_getters(self, request) -> tuple[Callable[[rx.TVxEHR], Any], type]:
        return request.param

    @pytest.fixture
    def tvx_ehr_lazy_loaded(
        self, tvx_ehr_lead, tmpdir: str, tvx_ehr_defer_getters: tuple[Callable[[rx.TVxEHR], Any], type]
    ) -> rx.TVxEHR:
        getter, _ = tvx_ehr_defer_getters
        tvx_ehr_lead.save(f"{tmpdir}/tvx_ehr_lead.h5", complevel=0)
        return tvx_ehr_lead.load(f"{tmpdir}/tvx_ehr_lead.h5", defer=(getter,), levels=(1,))

    def test_lazy_loading(
        self,
        tvx_ehr_lead: rx.TVxEHR,
        tvx_ehr_lazy_loaded: rx.TVxEHR,
        tvx_ehr_defer_getters: tuple[Callable[[rx.TVxEHR], Any], type],
    ):
        getter, node_type = tvx_ehr_defer_getters
        assert isinstance(getter(tvx_ehr_lazy_loaded), node_type)
        assert all(
            isinstance(child, rx.HDFVirtualNode) for child in eqx.tree_flatten_one_level(getter(tvx_ehr_lazy_loaded))[0]
        )
        loaded = rx.fetch_all(tvx_ehr_lazy_loaded)
        assert loaded.equals(tvx_ehr_lead)

    def test_lazy_loading_subjects(self, tvx_ehr_lead, tmpdir: str):
        tvx_ehr_lead.save(f"{tmpdir}/tvx_ehr_lead.h5", complevel=0)
        lazy_loaded = tvx_ehr_lead.load(f"{tmpdir}/tvx_ehr_lead.h5", defer=(lambda x: x.subjects,), levels=(1,))

        n = len(tvx_ehr_lead.subject_ids)
        splits = tvx_ehr_lead.subject_ids[: n // 2], tvx_ehr_lead.subject_ids[n // 2 :]
        for split in splits:
            lazy_loaded = lazy_loaded.fetch_subjects(split)
            for subject_id in split:
                assert lazy_loaded.subjects[subject_id].equals(tvx_ehr_lead.subjects[subject_id])


#
# class TestWideTVxEHR:
#     BAD_PERFORMANCE_N_ATTRS = 5000
#     BAD_PERFORMANCE_N_CHILDREN = 17000
#
#     @pytest.fixture(scope='class')
#     def wide_tvx_ehr(self, segmented_patient: rx.SegmentedPatient):
#         def array_thinner(x: Any):
#             if isinstance(x, rx.Array):
#                 return np.random.normal(size=())
#             else:
#                 return x
#
#         def obs_thinner(x: rx.InpatientObservables):
#             return rx.InpatientObservables.empty(5)
#
#         segmented_patient = jtu.tree_map(obs_thinner, segmented_patient,
#                                          is_leaf=lambda x: isinstance(x, rx.InpatientObservables))
#         subjects = {f'k_{i}': jtu.tree_map(array_thinner, segmented_patient)
#                     for i in range(self.BAD_PERFORMANCE_N_CHILDREN)}
#         return rx.TVxEHR(dataset=None, config=None, subjects=subjects)
#
#     @pytest.mark.filterwarnings('error:.*maximum number of.*')
#     def test_wide_tvx_ehr_serialization_performance_warnings(self, wide_tvx_ehr: rx.TVxEHR, tmpdir: str):
#         wide_tvx_ehr.save(f'{tmpdir}/wide_tvx_ehr.h5', complevel=3, complib='lzo')
#         loaded = wide_tvx_ehr.load(f'{tmpdir}/wide_tvx_ehr.h5')
#         assert wide_tvx_ehr.equals(loaded)
