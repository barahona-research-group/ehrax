from .mimic_in_memory import InMemoryStaticTableResource, InMemoryTableResource, MIMICTablesResources, \
    InMemoryMixedICDTableResource
from ..dataset import COLUMN, AdmissionsTableColumns
from ..freezer import FrozenDict11

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

MIMICIII_STATIC_RESOURCES = InMemoryStaticTableResource(MIMICIII_STATIC_COLMAP, MIMICIII_ADMISSIONS_COLMAP)

MIMICIII_ADMISSIONS_RESOURCES = InMemoryTableResource(AdmissionsTableColumns(), 'admissions',
                                                      MIMICIII_ADMISSIONS_COLMAP)
MIMICIII_DX_DISCHARGE_RESOURCES = InMemoryMixedICDTableResource('diagnoses_icd', MIMICIII_DIAGNOSES_ICD_COLMAP,
                                                                'd_icd_diagnoses', MIMICIII_D_ICD_DIAGNOSES_COLMAP)
MIMICIII_TABLES_RESOURCES = MIMICTablesResources(static=MIMICIII_STATIC_RESOURCES,
                                                 admissions=MIMICIII_ADMISSIONS_RESOURCES,
                                                 dx_discharge=MIMICIII_DX_DISCHARGE_RESOURCES)
