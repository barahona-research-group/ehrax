import logging
from collections import defaultdict
from typing import Final

import pandas as pd

from ..coding_scheme import CodeMap, CodingScheme, CodingSchemesManager, FrozenDict1N, FrozenDict11, HierarchicalScheme
from ..utils import dataframe_log, resources_path
from .icd import ICDScheme
from .icd9 import ICD9CM, ICD9PCS
from .icd10 import ICD10CM, ICD10PCS
from .literals import (
    DxFlatCCSName,
    DxMultiCCSName,
    ICD9CMName,
    ICD9PCSName,
    ICD10CMName,
    ICD10PCSName,
    PrFlatCCSName,
    PrMultiCCSName,
)


class CommonPreprocess:  # mixin
    @staticmethod
    def raw_table(filetitle: str, skiprows: tuple[int, ...] = ()) -> pd.DataFrame:
        return pd.read_csv(resources_path("CCS", filetitle), skiprows=skiprows, dtype=str)

    @staticmethod
    def process_ccs_table(raw_ccs_table: pd.DataFrame, colmap: dict[str, str]) -> pd.DataFrame:
        df = raw_ccs_table.iloc[:, :]
        df = df[list(colmap.keys())].rename(columns=colmap)
        for c in df.columns:
            df.loc[:, c] = df.loc[:, c].str.strip("'").str.strip()
        return df

    @staticmethod
    def process_ccs_icd_table(icd_scheme: ICDScheme, ccs_table: pd.DataFrame) -> pd.DataFrame:
        ccs_table["ICD"] = ccs_table.loc[:, "ICD"].map(icd_scheme.format)
        valid_icd = ccs_table["ICD"].isin(icd_scheme.codes)
        unsupported_icd = ccs_table.loc[~valid_icd, :]
        dataframe_log.info(
            f"In processing CCS multi-level table mapping to {icd_scheme.name} "
            f"{(~valid_icd).sum()} (of {len(valid_icd)}) ICD codes were unsupported.",
            dataframe=unsupported_icd,
            tag=f"unsupported_icd_codes_by_{icd_scheme.name}",
        )
        return ccs_table.loc[valid_icd, :]


class CommonMultiLevelCCS(CommonPreprocess):  # mixin
    @staticmethod
    def c_level_code(level: int) -> str:
        return f"'CCS LVL {level}'"

    @staticmethod
    def c_level_code_description(level: int) -> str:
        return f"'CCS LVL {level} LABEL'"

    @staticmethod
    def register_multi_level_mappings(
        manager: CodingSchemesManager, ccs_scheme: str, icd_scheme: str, processed_ccs_table: pd.DataFrame
    ) -> CodingSchemesManager:
        ccs_scheme = manager.scheme[ccs_scheme]
        icd_scheme = manager.scheme[icd_scheme]
        # TODO: Do some hard coded tests for the correction of mapping.
        icd2ccs = defaultdict(set)
        ccs2icd = defaultdict(set)
        code_cols = [c for c in processed_ccs_table.columns if c.startswith("C")]
        for icd, levels in zip(processed_ccs_table["ICD"], zip(*[processed_ccs_table.loc[:, c] for c in code_cols])):
            levels = tuple(l for l in levels if l != "")
            if len(levels) > 0:
                icd2ccs[icd].add(levels[-1])
                ccs2icd[levels[-1]].add(icd)

        manager = manager.add_map(
            CodeMap(source_name=icd_scheme.name, target_name=ccs_scheme.name, data=FrozenDict1N(dict(icd2ccs)))
        )
        manager = manager.add_map(
            CodeMap(source_name=ccs_scheme.name, target_name=icd_scheme.name, data=FrozenDict1N(dict(ccs2icd)))
        )
        return manager

    @staticmethod
    def parent_child_mappings(ccs_processed_table: pd.DataFrame) -> dict[str, set[str]]:
        """Make a dictionary for parent-child connections."""
        pt2ch = {"root": set(ccs_processed_table["C1"])}
        levels = [c for c in ccs_processed_table.columns if c.startswith("C")]
        for pt_col, ch_col in zip(levels[:-1], levels[1:]):
            df_ = ccs_processed_table[(ccs_processed_table[pt_col] != "") & (ccs_processed_table[ch_col] != "")]
            df_ = df_[[pt_col, ch_col]].drop_duplicates()
            for parent_code, ch_df in df_.groupby(pt_col):
                pt2ch[parent_code] = set(ch_df[ch_col])
        return pt2ch

    @staticmethod
    def multi_level_desc_mappings(ccs_processed_table: pd.DataFrame) -> dict[str, str]:
        """Make a dictionary for CCS labels."""
        desc = {"root": "root"}
        levels = [c for c in ccs_processed_table.columns if c.startswith("C")]
        descriptions = [c for c in ccs_processed_table.columns if c.startswith("D")]
        for code_col, desc_col in zip(levels, descriptions):
            df_ = ccs_processed_table[ccs_processed_table[code_col] != ""]
            df_ = df_[[code_col, desc_col]].drop_duplicates()
            code_desc = dict(zip(df_[code_col], df_[desc_col]))
            desc.update(code_desc)
        return desc


class MultiLevelCCSICD9MapOps(CommonMultiLevelCCS):
    C_ICD_CODE: str = "'ICD-9-CM CODE'"
    DX_FILE: str = "ccs_multi_dx_tool_2015.csv.gz"
    PR_FILE: str = "ccs_multi_pr_tool_2015.csv.gz"
    DX_N_LEVELS: int = 4
    PR_N_LEVELS: int = 3

    @classmethod
    def colmap(cls, n_levels: int):
        return (
            {cls.c_level_code(l): f"C{l}" for l in range(1, n_levels + 1)}
            | {cls.c_level_code_description(l): f"D{l}" for l in range(1, n_levels + 1)}
            | {cls.C_ICD_CODE: "ICD"}
        )

    @classmethod
    def process_dx_ccs_icd_table(cls, icd_scheme: ICDScheme):
        ccs_table = cls.process_ccs_table(raw_ccs_table=cls.raw_table(cls.DX_FILE), colmap=cls.colmap(cls.DX_N_LEVELS))
        return cls.process_ccs_icd_table(icd_scheme, ccs_table)

    @classmethod
    def process_pr_ccs_icd_table(cls, icd_scheme: ICDScheme):
        ccs_table = cls.process_ccs_table(raw_ccs_table=cls.raw_table(cls.PR_FILE), colmap=cls.colmap(cls.PR_N_LEVELS))
        return cls.process_ccs_icd_table(icd_scheme, ccs_table)

    @classmethod
    def create_scheme(
        cls, name: PrMultiCCSName | DxMultiCCSName, raw_ccs_table: pd.DataFrame, n_levels: int
    ) -> HierarchicalScheme:
        df = CommonPreprocess.process_ccs_table(raw_ccs_table, colmap=cls.colmap(n_levels))
        ch2pt = HierarchicalScheme.reverse_connection(FrozenDict1N(cls.parent_child_mappings(df)))
        desc = FrozenDict11(cls.multi_level_desc_mappings(df))
        codes = tuple(sorted(desc.keys()))
        return HierarchicalScheme(name=name, ch2pt=ch2pt, codes=codes, desc=desc)

    @classmethod
    def create_dx_ccs(cls) -> HierarchicalScheme:  # expose
        return cls.create_scheme("dx_ccs", cls.raw_table(cls.DX_FILE), cls.DX_N_LEVELS)

    @classmethod
    def create_pr_ccs(cls) -> HierarchicalScheme:  # expose
        return cls.create_scheme("pr_ccs", cls.raw_table(cls.PR_FILE), cls.PR_N_LEVELS)

    @classmethod
    def register_dx_ccs_maps(
        cls, manager: CodingSchemesManager, icd_scheme: ICD9CMName
    ) -> CodingSchemesManager:  # expose
        scheme = manager.scheme[icd_scheme]
        assert isinstance(scheme, ICD9CM), f"Expected ICD9CM, got {type(scheme)}."
        return cls.register_multi_level_mappings(manager, "dx_ccs", icd_scheme, cls.process_dx_ccs_icd_table(scheme))

    @classmethod
    def register_pr_ccs_maps(
        cls, manager: CodingSchemesManager, icd_scheme: ICD9PCSName
    ) -> CodingSchemesManager:  # expose
        scheme = manager.scheme[icd_scheme]
        assert isinstance(scheme, ICD9PCS), f"Expected ICD9PCS, got {type(scheme)}."
        return cls.register_multi_level_mappings(manager, "pr_ccs", icd_scheme, cls.process_pr_ccs_icd_table(scheme))


class CommonFlatCCS(CommonPreprocess):
    @staticmethod
    def desc_flat_mappings(processed_ccs_table: pd.DataFrame) -> dict[str, str]:
        return processed_ccs_table.set_index("C")["D"].to_dict()

    @staticmethod
    def register_flat_mappings(
        manager: CodingSchemesManager,
        ccs_scheme: DxFlatCCSName | PrFlatCCSName,
        icd_scheme: str,
        processed_ccs_table: pd.DataFrame,
    ) -> CodingSchemesManager:
        ccs_scheme = manager.scheme[ccs_scheme]
        icd_scheme = manager.scheme[icd_scheme]
        # TODO: Do some hard coded tests for the correction of mapping.
        icd2ccs = processed_ccs_table.groupby("ICD")["C"].apply(set).to_dict()
        ccs2icd = processed_ccs_table.groupby("C")["ICD"].apply(set).to_dict()
        manager = manager.add_map(
            CodeMap(source_name=icd_scheme.name, target_name=ccs_scheme.name, data=FrozenDict1N(dict(icd2ccs)))
        )
        manager = manager.add_map(
            CodeMap(source_name=ccs_scheme.name, target_name=icd_scheme.name, data=FrozenDict1N(dict(ccs2icd)))
        )
        return manager


class FlatCCS2ICD9MapOps(CommonFlatCCS):
    C_ICD_CODE: Final[str] = "'ICD-9-CM CODE'"
    C_CCS_CODE: Final[str] = "'CCS CATEGORY'"
    C_CCS_DESC: Final[str] = "'CCS CATEGORY DESCRIPTION'"
    DX_FILE: Final[str] = "$dxref 2015.csv.gz"
    PR_FILE: Final[str] = "$prref 2015.csv.gz"
    skiprows: Final[tuple[int, int]] = (0, 2)

    @classmethod
    def colmap(cls):
        return {cls.C_ICD_CODE: "ICD", cls.C_CCS_CODE: "C", cls.C_CCS_DESC: "D"}

    @classmethod
    def process_dx_ccs_icd_table(cls, icd9_scheme: ICDScheme):
        return cls.process_ccs_icd_table(
            icd9_scheme, cls.process_ccs_table(cls.raw_table(cls.DX_FILE, cls.skiprows), cls.colmap())
        )

    @classmethod
    def process_pr_ccs_icd_table(cls, icd_scheme: ICDScheme):
        return cls.process_ccs_icd_table(
            icd_scheme, cls.process_ccs_table(cls.raw_table(cls.PR_FILE, cls.skiprows), cls.colmap())
        )

    @classmethod
    def create_scheme(cls, name: DxFlatCCSName | PrFlatCCSName, raw_ccs_table: pd.DataFrame) -> CodingScheme:
        desc = FrozenDict11(cls.desc_flat_mappings(cls.process_ccs_table(raw_ccs_table, cls.colmap())))
        codes = tuple(sorted(desc.keys()))
        return CodingScheme(name=name, codes=codes, desc=desc)

    @classmethod
    def create_dx_flat_ccs(cls) -> CodingScheme:  # expose
        return cls.create_scheme("dx_flat_ccs", cls.raw_table(cls.DX_FILE, cls.skiprows))

    @classmethod
    def create_pr_flat_ccs(cls) -> CodingScheme:  # expose
        return cls.create_scheme("pr_flat_ccs", cls.raw_table(cls.PR_FILE, cls.skiprows))

    @classmethod
    def register_dx_flat_ccs_maps(
        cls, manager: CodingSchemesManager, icd_scheme: ICD9CMName
    ) -> CodingSchemesManager:  # expose
        scheme = manager.scheme[icd_scheme]
        assert isinstance(scheme, ICD9CM), f"Expected ICD9CM. got {type(scheme)}."
        return cls.register_flat_mappings(manager, "dx_flat_ccs", icd_scheme, cls.process_dx_ccs_icd_table(scheme))

    @classmethod
    def register_pr_flat_ccs_maps(
        cls, manager: CodingSchemesManager, icd_scheme: ICD9PCSName
    ) -> CodingSchemesManager:  # expose
        scheme = manager.scheme[icd_scheme]
        assert isinstance(scheme, ICD9PCS), f"Expected PrHierarchicalICD9, got {type(scheme)}."
        return cls.register_flat_mappings(manager, "pr_flat_ccs", icd_scheme, cls.process_pr_ccs_icd_table(scheme))


class CCS2ICD10MapOps(CommonMultiLevelCCS):
    C_ICD_CM_CODE: Final[str] = "'ICD-10-CM CODE'"
    C_ICD_PCS_CODE: Final[str] = "'ICD-10-PCS CODE'"
    C_CCS_CODE: Final[str] = "'CCS CATEGORY'"
    C_CCS_DESC: Final[str] = "'CCS CATEGORY DESCRIPTION'"
    DX_FILE: Final[str] = "ccs_dx_icd10cm_2019_1.csv.gz"
    PR_FILE: Final[str] = "ccs_pr_icd10pcs_2020_1.csv.gz"
    DX_N_LEVELS: Final[int] = 2
    PR_N_LEVELS: Final[int] = 2

    @staticmethod
    def c_level_code(level: int) -> str:
        return f"'MULTI CCS LVL {level}'"

    @staticmethod
    def c_level_code_description(level: int) -> str:
        return f"'MULTI CCS LVL {level} LABEL'"

    @classmethod
    def dx_colmap(cls, n_levels: int) -> dict[str, str]:
        return (
            {cls.c_level_code(l): f"C{l}" for l in range(1, n_levels + 1)}
            | {cls.c_level_code_description(l): f"D{l}" for l in range(1, n_levels + 1)}
            | {cls.C_ICD_CM_CODE: "ICD", cls.C_CCS_CODE: "C", cls.C_CCS_DESC: "D"}
        )

    @classmethod
    def pr_colmap(cls, n_levels: int) -> dict[str, str]:
        colmap = cls.dx_colmap(n_levels)
        del colmap[cls.C_ICD_CM_CODE]
        colmap[cls.C_ICD_PCS_CODE] = "ICD"
        return colmap

    @classmethod
    def process_dx_ccs_icd_table(cls, icd10_scheme: ICD10CM):
        return cls.process_ccs_icd_table(
            icd10_scheme, cls.process_ccs_table(cls.raw_table(cls.DX_FILE), cls.dx_colmap(cls.DX_N_LEVELS))
        )

    @classmethod
    def process_pr_ccs_icd_table(cls, icd10_scheme: ICD10PCS):
        return cls.process_ccs_icd_table(
            icd10_scheme, cls.process_ccs_table(cls.raw_table(cls.PR_FILE), cls.pr_colmap(cls.PR_N_LEVELS))
        )

    @classmethod
    def register_dx_flat_ccs_maps(
        cls, manager: CodingSchemesManager, icd_scheme: ICD10CMName
    ) -> CodingSchemesManager:  # expose
        scheme = manager.scheme[icd_scheme]
        assert isinstance(scheme, ICD10CM), f"Expected ICD10CM got {type(scheme)}."

        return CommonFlatCCS.register_flat_mappings(
            manager, "dx_flat_ccs", icd_scheme, cls.process_dx_ccs_icd_table(scheme)
        )

    @classmethod
    def register_pr_flat_ccs_maps(
        cls, manager: CodingSchemesManager, icd_scheme: ICD10PCSName
    ) -> CodingSchemesManager:  # expose
        scheme = manager.scheme[icd_scheme]
        assert isinstance(scheme, ICD10PCS), f"Expected PrFlatICD10, got {type(scheme)}."
        return CommonFlatCCS.register_flat_mappings(
            manager, "pr_flat_ccs", icd_scheme, cls.process_pr_ccs_icd_table(scheme)
        )

    @classmethod
    def register_dx_ccs_maps(
        cls, manager: CodingSchemesManager, icd_scheme: ICD10CMName
    ) -> CodingSchemesManager:  # expose
        scheme = manager.scheme[icd_scheme]
        assert isinstance(scheme, ICD10CM), f"Expected ICD10CM got {type(scheme)}."
        return cls.register_multi_level_mappings(manager, "dx_ccs", icd_scheme, cls.process_dx_ccs_icd_table(scheme))

    @classmethod
    def register_pr_ccs_maps(
        cls, manager: CodingSchemesManager, icd_scheme: ICD10PCSName
    ) -> CodingSchemesManager:  # expose
        scheme = manager.scheme[icd_scheme]
        assert isinstance(scheme, ICD10PCS), f"Expected PrFlatICD10, got {type(scheme)}."
        return cls.register_multi_level_mappings(manager, "pr_ccs", icd_scheme, cls.process_pr_ccs_icd_table(scheme))


class CCSMapRegistration:
    @staticmethod
    def icd9cm_maps(
        manager: CodingSchemesManager, icd9_scheme: ICD9CMName, dx_ccs: bool, dx_flat_ccs: bool
    ) -> CodingSchemesManager:
        if not any((dx_ccs, dx_flat_ccs)):
            return manager

        if dx_ccs:
            logging.debug(f"[BEGIN] mapping from {icd9_scheme} to dx_ccs")
            manager = MultiLevelCCSICD9MapOps.register_dx_ccs_maps(manager, icd9_scheme)
            logging.debug(f"[DONE] mapping from {icd9_scheme} to dx_ccs")
        if dx_flat_ccs:
            logging.debug(f"[BEGIN] mapping from {icd9_scheme} to dx_flat_ccs")
            manager = FlatCCS2ICD9MapOps.register_dx_flat_ccs_maps(manager, icd9_scheme)
            logging.debug(f"[DONE] mapping from {icd9_scheme} to dx_flat_ccs")
        return manager

    @staticmethod
    def icd9pcs_maps(
        manager: CodingSchemesManager, icd9_scheme: ICD9PCSName, pr_ccs: bool, pr_flat_ccs: bool
    ) -> CodingSchemesManager:
        if pr_ccs:
            logging.debug(f"[BEGIN] mapping from {icd9_scheme} to pr_ccs")
            manager = MultiLevelCCSICD9MapOps.register_pr_ccs_maps(manager, icd9_scheme)
            logging.debug(f"[DONE] mapping from {icd9_scheme} to pr_ccs")
        if pr_flat_ccs:
            logging.debug(f"[BEGIN] mapping from {icd9_scheme} to pr_flat_ccs")
            manager = FlatCCS2ICD9MapOps.register_pr_flat_ccs_maps(manager, icd9_scheme)
            logging.debug(f"[DONE] mapping from {icd9_scheme} to pr_flat_ccs")
        return manager

    @staticmethod
    def icd10cm_maps(
        manager: CodingSchemesManager, icd10_scheme: ICD10CMName, dx_ccs: bool, dx_flat_ccs: bool
    ) -> CodingSchemesManager:
        if dx_ccs:
            logging.debug(f"[BEGIN] mapping from {icd10_scheme} to dx_ccs")
            manager = CCS2ICD10MapOps.register_dx_ccs_maps(manager, icd10_scheme)
            logging.debug(f"[DONE] mapping from {icd10_scheme} to dx_ccs")
        if dx_flat_ccs:
            logging.debug(f"[BEGIN] mapping from {icd10_scheme} to dx_flat_ccs")
            manager = CCS2ICD10MapOps.register_dx_flat_ccs_maps(manager, icd10_scheme)
            logging.debug(f"[DONE] mapping from {icd10_scheme} to dx_flat_ccs")
        return manager

    @staticmethod
    def icd10pcs_maps(
        manager: CodingSchemesManager, icd10_scheme: ICD10PCSName, pr_ccs: bool, pr_flat_ccs: bool
    ) -> CodingSchemesManager:
        if pr_ccs:
            logging.debug(f"[BEGIN] mapping from {icd10_scheme} to pr_ccs")
            manager = CCS2ICD10MapOps.register_pr_ccs_maps(manager, icd10_scheme)
            logging.debug(f"[DONE] mapping from {icd10_scheme} to pr_ccs")
        if pr_flat_ccs:
            logging.debug(f"[BEGIN] mapping from {icd10_scheme} to pr_flat_ccs")
            manager = CCS2ICD10MapOps.register_pr_flat_ccs_maps(manager, icd10_scheme)
            logging.debug(f"[DONE] mapping from {icd10_scheme} to pr_flat_ccs")
        return manager

    @staticmethod
    def ccs_flat_to_multi_maps(
        manager: CodingSchemesManager, dx_bridge: ICD9CMName | None = None, pr_bridge: ICD9PCSName | None = None
    ):
        if dx_bridge is not None:
            logging.debug(f"[BEGIN] mapping from dx_flat_ccs to dx_ccs via {dx_bridge}")
            manager = manager.add_chained_map("dx_ccs", dx_bridge, "dx_flat_ccs")
            manager = manager.add_chained_map("dx_flat_ccs", dx_bridge, "dx_ccs")
            logging.debug(f"[DONE] mapping from dx_flat_ccs to dx_ccs via {dx_bridge}")
        if pr_bridge is not None:
            logging.debug(f"[BEGIN] mapping from pr_flat_ccs to pr_ccs via {pr_bridge}")
            manager = manager.add_chained_map("pr_ccs", pr_bridge, "pr_flat_ccs")
            manager = manager.add_chained_map("pr_flat_ccs", pr_bridge, "pr_ccs")
            logging.debug(f"[DONE] mapping from pr_flat_ccs to pr_ccs via {pr_bridge}")
        return manager
