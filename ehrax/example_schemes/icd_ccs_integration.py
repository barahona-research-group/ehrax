from dataclasses import fields
from typing import Self

from .ccs import MultiLevelCCSICD9MapOps, FlatCCS2ICD9MapOps, CCS2ICD10MapOps
from .icd import ICDMapOps
from .icd10 import ICD10CM, ICD10PCS
from .icd9 import ICD9
from ..base import AbstractConfig
from ..coding_scheme import (CodingSchemesManager, ExcludingOutcomeExtractor)


class Flags(AbstractConfig):

    @classmethod
    def all(cls) -> Self:
        return cls(**{f.name: True for f in fields(cls)})  # type: ignore

    @property
    def flag_set(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self) if isinstance(getattr(self, f.name), bool) and getattr(self, f.name))


class ICDSchemeSelection(Flags):
    dx_icd9: bool
    pr_icd9: bool
    dx_icd10: bool
    dx_flat_icd10: bool
    pr_flat_icd10: bool

    def __init__(self, dx_icd9: bool = False, dx_icd10: bool = False, pr_icd9: bool = False,
                 dx_flat_icd10: bool = False, pr_flat_icd10: bool = False):
        self.dx_icd9 = dx_icd9
        self.dx_icd10 = dx_icd10
        self.pr_icd9 = pr_icd9
        self.dx_flat_icd10 = dx_flat_icd10
        self.pr_flat_icd10 = pr_flat_icd10


class CCSSchemeSelection(Flags):
    dx_ccs: bool
    pr_ccs: bool
    dx_flat_ccs: bool
    pr_flat_ccs: bool

    def __init__(self, dx_ccs: bool = False,
                 pr_ccs: bool = False, dx_flat_ccs: bool = False,
                 pr_flat_ccs: bool = False):
        self.dx_ccs = dx_ccs
        self.pr_ccs = pr_ccs
        self.dx_flat_ccs = dx_flat_ccs
        self.pr_flat_ccs = pr_flat_ccs


class OutcomeSelection(Flags):
    dx_icd9_v1: bool
    dx_icd9_v2_groups: bool
    dx_icd9_v3_groups: bool
    dx_flat_ccs_mlhc_groups: bool
    dx_flat_ccs_v1: bool

    def __init__(self, dx_icd9_v1: bool = False, dx_icd9_v2_groups: bool = False,
                 dx_icd9_v3_groups: bool = False,
                 dx_flat_ccs_mlhc_groups: bool = False,
                 dx_flat_ccs_v1: bool = False):
        self.dx_icd9_v1 = dx_icd9_v1
        self.dx_icd9_v2_groups = dx_icd9_v2_groups
        self.dx_icd9_v3_groups = dx_icd9_v3_groups
        self.dx_flat_ccs_mlhc_groups = dx_flat_ccs_mlhc_groups
        self.dx_flat_ccs_v1 = dx_flat_ccs_v1


def setup_icd_schemes(icd_selection: ICDSchemeSelection) -> CodingSchemesManager:
    manager = ICD9.create_schemes(dx=icd_selection.dx_icd9, pr=icd_selection.pr_icd9)
    manager += ICD10CM.create_schemes(flat=icd_selection.dx_flat_icd10, hierarchical=icd_selection.dx_icd10)
    if icd_selection.pr_flat_icd10:
        manager = manager.add_scheme(ICD10PCS.create_scheme())
    return manager


def setup_ccs_schemes(manager: CodingSchemesManager, ccs_selection: CCSSchemeSelection) -> CodingSchemesManager:
    if ccs_selection.dx_ccs:
        manager = manager.add_scheme(MultiLevelCCSICD9MapOps.create_dx_ccs())
    if ccs_selection.pr_ccs:
        manager = manager.add_scheme(MultiLevelCCSICD9MapOps.create_pr_ccs())
    if ccs_selection.dx_flat_ccs:
        manager = manager.add_scheme(FlatCCS2ICD9MapOps.create_dx_flat_ccs())
    if ccs_selection.pr_flat_ccs:
        manager = manager.add_scheme(FlatCCS2ICD9MapOps.create_pr_flat_ccs())
    return manager


def setup_outcomes(manager: CodingSchemesManager,
                   outcome_selection: OutcomeSelection) -> CodingSchemesManager:
    for outcome_name in outcome_selection.flag_set:
        manager = manager.add_outcome(ExcludingOutcomeExtractor.from_spec_json(manager.scheme, f'{outcome_name}.json'))
    return manager


def setup_icd_icd_maps(manager: CodingSchemesManager, scheme_selection: ICDSchemeSelection) -> CodingSchemesManager:
    # ICD9 <-> ICD10s
    if scheme_selection.dx_icd9 and scheme_selection.dx_icd10:
        manager = ICDMapOps.register_mappings(manager, 'dx_icd10', 'dx_icd9', '2018_gem_cm_I10I9.txt.gz')
        manager = ICDMapOps.register_mappings(manager, 'dx_icd9', 'dx_icd10', '2018_gem_cm_I9I10.txt.gz')

    if scheme_selection.dx_flat_icd10 and scheme_selection.dx_icd9:
        manager = ICDMapOps.register_mappings(manager, 'dx_icd9', 'dx_flat_icd10',
                                              '2018_gem_cm_I9I10.txt.gz')
        manager = ICDMapOps.register_mappings(manager, 'dx_flat_icd10', 'dx_icd9',
                                              '2018_gem_cm_I10I9.txt.gz')
    if scheme_selection.pr_icd9 and scheme_selection.pr_flat_icd10:
        manager = ICDMapOps.register_mappings(manager, 'pr_flat_icd10', 'pr_icd9',
                                              '2018_gem_pcs_I10I9.txt.gz')
        manager = ICDMapOps.register_mappings(manager, 'pr_icd9', 'pr_flat_icd10',
                                              '2018_gem_pcs_I9I10.txt.gz')
    return manager


def setup_icd_ccs_maps(manager: CodingSchemesManager, icd_selection: ICDSchemeSelection,
                       ccs_selection: CCSSchemeSelection) -> CodingSchemesManager:
    # ICD9 <-> CCS
    if icd_selection.dx_icd9 and ccs_selection.dx_ccs:
        manager = MultiLevelCCSICD9MapOps.register_dx_ccs_maps(manager, 'dx_icd9')
    if icd_selection.pr_icd9 and ccs_selection.pr_ccs:
        manager = MultiLevelCCSICD9MapOps.register_pr_ccs_maps(manager, 'pr_icd9')

    if ccs_selection.dx_flat_ccs and icd_selection.dx_icd9:
        manager = FlatCCS2ICD9MapOps.register_dx_flat_ccs_maps(manager, 'dx_icd9')
    if ccs_selection.pr_flat_ccs and icd_selection.pr_icd9:
        manager = FlatCCS2ICD9MapOps.register_pr_flat_ccs_maps(manager, 'pr_icd9')

    if ccs_selection.dx_flat_ccs:
        if icd_selection.dx_icd10:
            manager = CCS2ICD10MapOps.register_dx_flat_ccs_maps(manager, 'dx_icd10')
        if icd_selection.dx_flat_icd10:
            manager = CCS2ICD10MapOps.register_dx_flat_ccs_maps(manager, 'dx_flat_icd10')

    if ccs_selection.pr_flat_ccs and icd_selection.pr_flat_icd10:
        manager = CCS2ICD10MapOps.register_pr_flat_ccs_maps(manager, 'pr_flat_icd10')

    # cross-maps
    if ccs_selection.dx_ccs and ccs_selection.dx_flat_ccs and icd_selection.dx_icd9:
        manager = manager.add_chained_map('dx_ccs', 'dx_icd9', 'dx_flat_ccs')
        manager = manager.add_chained_map('dx_flat_ccs', 'dx_icd9', 'dx_ccs')

    if ccs_selection.pr_ccs and ccs_selection.pr_flat_ccs and icd_selection.pr_icd9:
        manager = manager.add_chained_map('pr_ccs', 'pr_icd9', 'pr_flat_ccs')
        manager = manager.add_chained_map('pr_flat_ccs', 'pr_icd9', 'pr_ccs')

    return manager


def setup_standard_icd_ccs(icd_selection: ICDSchemeSelection = ICDSchemeSelection.all(),
                           ccs_selection: CCSSchemeSelection = CCSSchemeSelection.all(),
                           outcome_selection: OutcomeSelection = OutcomeSelection.all()) -> CodingSchemesManager:
    manager = setup_icd_schemes(icd_selection)
    manager = setup_icd_icd_maps(manager, icd_selection)
    manager = setup_icd_ccs_maps(manager, icd_selection, ccs_selection)
    manager = setup_outcomes(manager, outcome_selection)
    return manager
