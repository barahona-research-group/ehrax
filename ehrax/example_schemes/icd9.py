from typing import Any, Final

import pandas as pd

from ..coding_scheme import FrozenDict11, HierarchicalScheme
from ..utils import resources_path
from .icd import ICDHierarchicalScheme


class ICD9CM(ICDHierarchicalScheme):
    @staticmethod
    def format(code: str) -> str:
        if "-" in code:
            return "-".join(map(ICD9CM.format, code.split("-")))
        if "." in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        n = len(code)
        if code[0] == "E":
            if n > 4:
                return code[:4] + "." + code[4:]
            else:
                return code
        if n > 3:
            return code[:3] + "." + code[3:]
        else:
            return code


class ICD9PCS(ICDHierarchicalScheme):
    @staticmethod
    def format(code: str) -> str:
        if "-" in code:
            return "-".join(map(ICD9PCS.format, code.split("-")))
        if "." in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        if len(code) > 2:
            return code[:2] + "." + code[2:]
        else:
            return code


class ICD9PCSFactory:
    ICD9_FILE: str = resources_path("ICD", "HOM-ICD9.csv.gz")
    DUMMY_ROOT_CLASS_ID: str = "owl#Thing"
    PCS_ROOT_CLASS_ID: str = "MM_CLASS_2"
    CM_ROOT_CLASS_ID: str = "MM_CLASS_21"

    @classmethod
    def create_scheme_data(
        cls,
        processed_icd_table: pd.DataFrame,
        all_parent_to_children_map: dict[str, frozenset[str]],
        select_tree: str | None = None,
        deselect_tree: str | None = None,
    ) -> dict[str, Any]:
        assert (select_tree is None) != (deselect_tree is None), (
            "Must specify exactly one of select_tree and deselect_tree."
        )
        if select_tree is not None:
            # Select a subset. Select the procedure tree if you want a procedure scheme, to discard everything else.
            pt2ch = cls.select_subtree(all_parent_to_children_map, select_tree)  # cls.PR_ROOT_CLASS_ID)
        else:
            # Remove a subset. Remove the procedure tree if you want the diagnostic scheme.
            pt2ch = cls.deselect_subtree(all_parent_to_children_map, deselect_tree)

        # Combining remaining node indices in one set.
        nodes = set(list(pt2ch.keys()) + [n for ns in pt2ch.values() for n in ns])

        # Filter out the procedure code from the df.
        df = processed_icd_table[processed_icd_table["NODE_IDX"].isin(nodes)]
        return cls.generate_dictionaries(df) | {"ch2pt": HierarchicalScheme.reverse_connection(pt2ch)}

    @classmethod
    def create_cm_scheme_data(
        cls, processed_icd_table: pd.DataFrame, all_parent_to_children_map: dict[str, frozenset[str]]
    ) -> dict[str, Any]:
        return cls.create_scheme_data(
            processed_icd_table, all_parent_to_children_map, deselect_tree=cls.PCS_ROOT_CLASS_ID
        )

    @classmethod
    def create_pcs_scheme_data(
        cls, processed_icd_table: pd.DataFrame, all_parent_to_children_map: dict[str, frozenset[str]]
    ) -> dict[str, Any]:
        return cls.create_scheme_data(
            processed_icd_table, all_parent_to_children_map, select_tree=cls.PCS_ROOT_CLASS_ID
        )

    @staticmethod
    def deselect_subtree(pt2ch: dict[str, frozenset[str]], sub_root: str) -> dict[str, frozenset[str]]:
        to_del = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        to_use = set(pt2ch.keys()) - set(to_del)
        return {pt: pt2ch[pt] for pt in to_use}

    @staticmethod
    def select_subtree(pt2ch: dict[str, frozenset[str]], sub_root: str) -> dict[str, frozenset[str]]:
        to_use = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        to_use = set(to_use) & set(pt2ch.keys())
        return {pt: pt2ch[pt] for pt in to_use}

    @classmethod
    def load_raw_table(cls) -> pd.DataFrame:
        # https://bioportal.bioontology.org/ontologies/HOM-ICD9
        return pd.read_csv(cls.ICD9_FILE, dtype=str)

    @classmethod
    def extract_icd9_codes(cls, df: pd.DataFrame) -> list[str]:
        return list(df["C_BASECODE"].apply(lambda c: c.split(":")[-1]))

    @classmethod
    def process_icd_table(cls, table: pd.DataFrame) -> pd.DataFrame:
        df = table.fillna("")

        def retain_suffix(cell):
            if "http" in cell:
                return cell.split("/")[-1]
            else:
                return cell

        df = df.map(retain_suffix)
        df.columns = list(map(retain_suffix, df.columns))

        return pd.DataFrame(
            {
                "ICD9": cls.extract_icd9_codes(df),
                "NODE_IDX": list(df["Class ID"]),
                "PARENT_IDX": list(df["Parents"]),
                "LABEL": list(df["Preferred Label"]),
            }
        )

    @classmethod
    def parent_child_mappings(cls, df: pd.DataFrame) -> dict[str, frozenset[str]]:
        pt2ch = df.groupby("PARENT_IDX")["NODE_IDX"].apply(frozenset).to_dict()
        # Remove dummy parent of diagnoses.
        del pt2ch[cls.DUMMY_ROOT_CLASS_ID]
        return pt2ch

    @staticmethod
    def generate_dictionaries(df: pd.DataFrame) -> dict[str, Any]:
        # df version for leaf nodes only (with non-empty ICD9 codes)
        df_leaves = df[df["ICD9"] != ""]

        icd2dag = dict(zip(df_leaves["ICD9"], df_leaves["NODE_IDX"]))

        # df version for internal nodes only (with empty ICD9 codes)
        df_internal = df[(df["ICD9"] == "") | df["ICD9"].isnull()]

        icd_codes = sorted(df_leaves["ICD9"])
        icd_desc = dict(zip(df_leaves["ICD9"], df_leaves["LABEL"]))

        dag_codes = list(map(icd2dag.get, icd_codes)) + sorted(df_internal["NODE_IDX"])
        dag_desc = dict(zip(df["NODE_IDX"], df["LABEL"]))

        return {
            "codes": tuple(sorted(icd_codes)),
            "desc": FrozenDict11(icd_desc),
            "code2dag": FrozenDict11(icd2dag),
            "dag_codes": tuple(sorted(dag_codes)),
            "dag_desc": FrozenDict11(dag_desc),
        }

    @classmethod
    def create_scheme(cls) -> ICD9PCS:
        # to reduce time of redundant processing.
        processed_icd_table = cls.process_icd_table(cls.load_raw_table())
        all_parent_to_children_map = cls.parent_child_mappings(processed_icd_table)
        return ICD9PCS(name="icd9pcs", **cls.create_pcs_scheme_data(processed_icd_table, all_parent_to_children_map))


class ICD9CMFactory(ICD9PCSFactory):
    ICD9_FILE: Final[str] = resources_path("ICD", "ICD9CM.csv.gz")
    DUMMY_ROOT_CLASS_ID: Final[str] = "owl#Thing"
    PCS_ROOT_CLASS_ID: Final[str] = "00-99.99"
    CM_ROOT_CLASS_ID: Final[str] = "001-999.99"

    @classmethod
    def extract_icd9_codes(cls, df: pd.DataFrame) -> list[str]:
        return df["Class ID"].tolist()

    @classmethod
    def create_scheme(cls) -> ICD9CM:
        # to reduce time of redundant processing.
        processed_icd_table = cls.process_icd_table(cls.load_raw_table())
        all_parent_to_children_map = cls.parent_child_mappings(processed_icd_table)
        return ICD9CM(name="icd9cm", **cls.create_cm_scheme_data(processed_icd_table, all_parent_to_children_map))
