import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any

import pandas as pd

from ..coding_scheme import FrozenDict11
from ..freezer import FrozenDict1N
from ..utils import resources_path
from .icd import ICDHierarchicalScheme, ICDScheme


class ICD10PCS(ICDScheme):
    @staticmethod
    def format(code: str) -> str:
        # No decimal point in ICD10-PCS
        return code


class ICD10CM(ICDHierarchicalScheme):
    """
    NOTE: for prediction targets, remember to exclude the following chapters:
        - 'chapter:19': 'Injury, poisoning and certain \
            other consequences of external causes (S00-T88)',
        - 'chapter:20': 'External causes of morbidity (V00-Y99)',
        - 'chapter:21': 'Factors influencing health status and \
            contact with health services (Z00-Z99)',
        - 'chapter:22': 'Codes for special purposes (U00-U85)'
    """

    @staticmethod
    def format(code: str) -> str:
        code = code.upper()  # ICD10 is case insensitive
        if "." in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        if len(code) > 3:
            return code[:3] + "." + code[3:]
        else:
            return code


class ICD10CMFactory:
    FILE_TABLE_TITLE: str = "icd-10-cm-tabular-2025.xml.gz"
    FILE_LIST_TITLE: str = "icd10cm-codes-2025.txt.gz"
    FILE_LEGACY_MAP_TITLE: str = "icd-10-cm-conversion-table-FY2025.csv.gz"

    C_LIST_CODE: str = "C"
    C_LIST_DESCRIPTION: str = "D"

    C_LEGACY_CURRENT: str = "C"
    C_LEGACY_RETIRED: str = "R"
    C_LEGACY_EFFECTIVE: str = "E"

    @classmethod
    def expand_ranges1(cls, first: pd.Series, second: pd.Series) -> pd.Series:
        # with no period.
        pass

    @classmethod
    def expand_ranges2(cls, first: pd.Series, second: pd.Series) -> pd.Series:
        # with period.
        pass

    @classmethod
    def expand_ranges(cls, r: pd.Series):
        r = r.str.split("-")
        first = r.map(lambda x: x[0])
        second = r.map(lambda x: x[1])
        # either both sides contain a dot or both do not.
        assert all(first.str.contains(".") == second.str.contains("."))
        if any(first.str.contains(".")):
            return cls.expand_ranges2(first, second)
        return cls.expand_ranges1(first, second)

    @classmethod
    def load_legacy_table(cls) -> pd.DataFrame:
        df = pd.read_csv(resources_path("ICD", cls.FILE_LEGACY_MAP_TITLE), dtype=str, sep="\t", skiprows=1)
        df.columns = [cls.C_LEGACY_CURRENT, cls.C_LEGACY_EFFECTIVE, cls.C_LEGACY_RETIRED]
        for ci in (0, 1):
            df.iloc[:, ci] = df.iloc[:, ci].str.strip()
        df = df.loc[(df.loc[:, cls.C_LEGACY_CURRENT] != "None") & (df.loc[:, cls.C_LEGACY_RETIRED].notnull()), :]
        return df[df.iloc[:, 0].notnull()].reset_index(drop=True)

    @classmethod
    def parse_legacy_map_file(cls) -> dict[str, str]:
        pass

    @classmethod
    def load_codes_list_file(cls) -> pd.DataFrame:
        lines = []
        with gzip.open(resources_path("ICD", cls.FILE_LIST_TITLE), "rb") as f:
            for line in f.readlines():
                code = ICD10CM.format(line[:7].strip().decode("utf-8"))
                desc = line[7:].strip().decode("utf-8")
                lines.append((code, desc))
        return pd.DataFrame(lines, columns=[cls.C_LIST_CODE, cls.C_LIST_DESCRIPTION])

    @classmethod
    def augment_hierarchy(cls, icd10cm_tabular_traversal: dict[str, Any]) -> dict[str, Any]:
        codes_list = cls.load_codes_list_file()
        codes = icd10cm_tabular_traversal["codes"]
        desc = icd10cm_tabular_traversal["desc"]
        ch2pt = icd10cm_tabular_traversal["ch2pt"]

        # update description.
        desc.update(codes_list.set_index("C")["D"].to_dict())

        def update_hierarchy(c, pt):
            ch2pt[c] = frozenset({pt})

        codes_list = codes_list[~codes_list.C.isin(codes)]
        # what remains undetected are full codes (8 characters: 7 stem + 1 period).
        assert (codes_list.C.map(len) == 8).all()
        # the code 'ABC.DEFG' is one level higher than 'ABC.DEF'
        # the code 'AB' is only four levels higher than 'ABC.DEF'. (period is just a place holder).
        for height in (1, 2, 3, 5):  #
            potential_parents = codes_list.C.map(lambda c: c[:-height])
            detected_codes = potential_parents.isin(codes)
            for ch, pt in zip(codes_list.loc[detected_codes, "C"], potential_parents):
                update_hierarchy(ch, pt)
            codes_list = codes_list.loc[~detected_codes, :]
        assert len(codes_list) == 0, "All should be populated."
        return {
            "codes": tuple(sorted(desc.keys())),
            "desc": FrozenDict11(desc),
            "ch2pt": FrozenDict1N(ch2pt),
        }

    @classmethod
    def traverse_icd10_xml(cls) -> dict[str, Any]:
        # https://www.cdc.gov/nchs/icd/Comprehensive-listing-of-ICD-10-CM-Files.htm
        with gzip.open(resources_path("ICD", cls.FILE_TABLE_TITLE), "r") as f:
            tree = ET.parse(f)

        code_level = {}
        root = tree.getroot()
        ch2pt = defaultdict(list)
        root_node = f"root:{root.tag}"
        desc = {root_node: "root"}
        chapters = [ch for ch in root if ch.tag == "chapter"]

        def _traverse_diag_dfs(parent_name, dx_element):
            dx_name = next(e for e in dx_element if e.tag == "name").text
            dx_desc = next(e for e in dx_element if e.tag == "desc").text
            desc[dx_name] = dx_desc
            ch2pt[dx_name].append(parent_name)
            children = tuple(dx for dx in dx_element if dx.tag == "diag")
            code_level[dx_name] = "node" if len(children) > 0 else "leaf"
            for ch in children:
                _traverse_diag_dfs(dx_name, ch)

        for chapter in chapters:
            ch_name = next(e for e in chapter if e.tag == "name").text
            ch_desc = next(e for e in chapter if e.tag == "desc").text
            code_level[ch_name] = "chapter"
            ch2pt[ch_name].append(root_node)
            desc[ch_name] = ch_desc

            sections = [sec for sec in chapter if sec.tag == "section"]
            for section in sections:
                sec_name = section.attrib["id"]
                sec_desc = next(e for e in section if e.tag == "desc").text
                code_level[sec_name] = "section"
                ch2pt[sec_name].append(ch_name)
                desc[sec_name] = sec_desc

                for dx in (dx for dx in section if dx.tag == "diag"):
                    _traverse_diag_dfs(sec_name, dx)

        # valid (complete codes) and invalid icd10 codes (e.g. chapters, sections, and internal nodes).
        return {
            "codes": tuple(sorted(desc.keys())),
            "desc": desc,
            "ch2pt": {ch: frozenset(pts) for ch, pts in ch2pt.items()},
        }

    @classmethod
    def create_scheme(cls) -> ICD10CM:  # expose
        return ICD10CM(name="icd10cm", **cls.augment_hierarchy(cls.traverse_icd10_xml()))


class ICD10PCSFactory:
    FILE_TITLE: str = "icd10pcs_codes_2024.txt.gz"

    @classmethod
    def distill_icd10_xml(cls) -> dict[str, Any]:
        with gzip.open(resources_path("ICD", cls.FILE_TITLE), "rt") as f:
            desc = {code: desc for code, desc in map(lambda line: line.strip().split(" ", 1), f.readlines())}
            return {"codes": tuple(sorted(desc)), "desc": FrozenDict11(desc)}

    @classmethod
    def create_scheme(cls) -> ICD10PCS:  # expose
        return ICD10PCS(name="icd10pcs", **cls.distill_icd10_xml())
