from dataclasses import fields
from typing import Self

from ..base import AbstractConfig
from ..coding_scheme import CodingSchemesManager, FilterOutcomeMapData
from .ccs import CCSMapRegistration, FlatCCS2ICD9MapOps, MultiLevelCCSICD9MapOps
from .icd import ICDMapOps
from .icd9 import ICD9CMFactory, ICD9PCSFactory
from .icd10 import ICD10CMFactory, ICD10PCSFactory


class Flags(AbstractConfig):
    @classmethod
    def all(cls) -> Self:
        return cls(**{f.name: True for f in fields(cls)})  # type: ignore

    @property
    def flag_set(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self) if isinstance(getattr(self, f.name), bool) and getattr(self, f.name))


class ICDSchemeSelection(Flags):
    icd9cm: bool
    icd9pcs: bool
    icd10cm: bool
    icd10pcs: bool

    def __init__(self, icd9cm: bool = False, icd9pcs: bool = False, icd10cm: bool = False, icd10pcs: bool = False):
        self.icd9cm = icd9cm
        self.icd10cm = icd10cm
        self.icd9pcs = icd9pcs
        self.icd10pcs = icd10pcs


class CCSSchemeSelection(Flags):
    dx_ccs: bool
    pr_ccs: bool
    dx_flat_ccs: bool
    pr_flat_ccs: bool

    def __init__(
        self, dx_ccs: bool = False, pr_ccs: bool = False, dx_flat_ccs: bool = False, pr_flat_ccs: bool = False
    ):
        self.dx_ccs = dx_ccs
        self.pr_ccs = pr_ccs
        self.dx_flat_ccs = dx_flat_ccs
        self.pr_flat_ccs = pr_flat_ccs


class OutcomeSelection(Flags):
    icd9cm_v1: bool
    icd9cm_v2_groups: bool
    icd9cm_v3_groups: bool
    dx_flat_ccs_mlhc_groups: bool
    dx_flat_ccs_v1: bool

    def __init__(
        self,
        icd9cm_v1: bool = False,
        icd9cm_v2_groups: bool = False,
        icd9cm_v3_groups: bool = False,
        dx_flat_ccs_mlhc_groups: bool = False,
        dx_flat_ccs_v1: bool = False,
    ):
        self.icd9cm_v1 = icd9cm_v1
        self.icd9cm_v2_groups = icd9cm_v2_groups
        self.icd9cm_v3_groups = icd9cm_v3_groups
        self.dx_flat_ccs_mlhc_groups = dx_flat_ccs_mlhc_groups
        self.dx_flat_ccs_v1 = dx_flat_ccs_v1


def setup_icd_schemes(icd_selection: ICDSchemeSelection) -> CodingSchemesManager:
    manager = CodingSchemesManager()
    if icd_selection.icd9cm:
        manager = manager.add_scheme(ICD9CMFactory.create_scheme())
    if icd_selection.icd9pcs:
        manager = manager.add_scheme(ICD9PCSFactory.create_scheme())
    if icd_selection.icd10cm:
        manager = manager.add_scheme(ICD10CMFactory.create_scheme())
    if icd_selection.icd10pcs:
        manager = manager.add_scheme(ICD10PCSFactory.create_scheme())
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


def setup_outcomes(manager: CodingSchemesManager, outcome_selection: OutcomeSelection) -> CodingSchemesManager:
    for outcome_name in outcome_selection.flag_set:
        manager = manager.add_outcome(FilterOutcomeMapData.from_spec_json(manager.scheme, f"{outcome_name}.json"))
    return manager


def setup_icd_icd_maps(manager: CodingSchemesManager, scheme_selection: ICDSchemeSelection) -> CodingSchemesManager:
    # ICD9 <-> ICD10s
    if scheme_selection.icd9cm and scheme_selection.icd10cm:
        manager = ICDMapOps.register_mappings(manager, "icd10cm", "icd9cm", "2018_gem_cm_I10I9.txt.gz")
        manager = ICDMapOps.register_mappings(manager, "icd9cm", "icd10cm", "2018_gem_cm_I9I10.txt.gz")

    if scheme_selection.icd10pcs and scheme_selection.icd9pcs:
        manager = ICDMapOps.register_mappings(manager, "icd10pcs", "icd9pcs", "2018_gem_pcs_I10I9.txt.gz")
        manager = ICDMapOps.register_mappings(manager, "icd9pcs", "icd10pcs", "2018_gem_pcs_I9I10.txt.gz")
    return manager


def setup_icd_ccs_maps(
    manager: CodingSchemesManager, icd_selection: ICDSchemeSelection, ccs_selection: CCSSchemeSelection
) -> CodingSchemesManager:
    """
    Enables the following two-way mappings (if all selected):

    |                   | icd9cm   | icd9pcs   | icd10cm  | dx_flat_icd10 | icd9pcs |
    |-------------------|-----------|-----------|-----------|---------------|---------------|
    | dx_ccs            |   X       |           |   X       |      X        |               |
    | dx_flat_ccs       |   X       |           |   X       |      X        |               |
    | pr_ccs            |           |   X       |           |               |      X        |
    | pr_flat_ccs       |           |   X       |           |               |      X        |

    In addition to dx_ccs <-> dx_flat_ccs and pr_ccs <-> pr_flat_ccs.
    """

    if icd_selection.icd9cm:
        manager = CCSMapRegistration.icd9cm_maps(
            manager, "icd9cm", dx_ccs=ccs_selection.dx_ccs, dx_flat_ccs=ccs_selection.dx_flat_ccs
        )
    if icd_selection.icd9pcs:
        manager = CCSMapRegistration.icd9pcs_maps(
            manager, "icd9pcs", pr_ccs=ccs_selection.pr_ccs, pr_flat_ccs=ccs_selection.pr_flat_ccs
        )
    if icd_selection.icd10cm:
        manager = CCSMapRegistration.icd10cm_maps(
            manager, "icd10cm", dx_ccs=ccs_selection.dx_ccs, dx_flat_ccs=ccs_selection.dx_flat_ccs
        )
    if icd_selection.icd10pcs:
        manager = CCSMapRegistration.icd10pcs_maps(
            manager, "icd10pcs", pr_ccs=ccs_selection.pr_ccs, pr_flat_ccs=ccs_selection.pr_flat_ccs
        )

    # cross-maps
    dx_bridge, pr_bridge = None, None
    if icd_selection.icd9cm and ccs_selection.dx_ccs and ccs_selection.dx_flat_ccs:
        dx_bridge = "icd9cm"
    if icd_selection.icd9pcs and ccs_selection.pr_ccs and ccs_selection.pr_flat_ccs:
        pr_bridge = "icd9pcs"
    manager = CCSMapRegistration.ccs_flat_to_multi_maps(manager, dx_bridge=dx_bridge, pr_bridge=pr_bridge)

    return manager


def setup_standard_icd_ccs(
    icd_selection: ICDSchemeSelection = ICDSchemeSelection.all(),
    ccs_selection: CCSSchemeSelection = CCSSchemeSelection.all(),
    outcome_selection: OutcomeSelection = OutcomeSelection.all(),
) -> CodingSchemesManager:
    manager = setup_icd_schemes(icd_selection)
    manager = setup_icd_icd_maps(manager, icd_selection)
    manager = setup_ccs_schemes(manager, ccs_selection)
    manager = setup_icd_ccs_maps(manager, icd_selection, ccs_selection)
    manager = setup_outcomes(manager, outcome_selection)
    return manager
