from .mimic_in_memory import InMemoryStaticTableResource, InMemoryTableResource, MIMICTablesResources, \
    InMemoryMixedICDTableResource
from ..dataset import COLUMN, AdmissionsTableColumns
from ..freezer import FrozenDict11

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

MIMICIV_STATIC_RESOURCES = InMemoryStaticTableResource(MIMICIV_STATIC_COLMAP, MIMICIV_ADMISSIONS_COLMAP)

MIMICIV_ADMISSIONS_RESOURCES = InMemoryTableResource(AdmissionsTableColumns(), 'admissions',
                                                     MIMICIV_ADMISSIONS_COLMAP)
MIMICIV_DX_DISCHARGE_RESOURCES = InMemoryMixedICDTableResource('diagnoses_icd', MIMICIV_DIAGNOSES_ICD_COLMAP,
                                                               'd_icd_diagnoses', MIMICIV_D_ICD_DIAGNOSES_COLMAP)
MIMICIV_TABLES_RESOURCES = MIMICTablesResources(static=MIMICIV_STATIC_RESOURCES,
                                                admissions=MIMICIV_ADMISSIONS_RESOURCES,
                                                dx_discharge=MIMICIV_DX_DISCHARGE_RESOURCES)
