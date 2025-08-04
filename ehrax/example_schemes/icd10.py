import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, cast

from .icd import ICDHierarchicalScheme, ICDScheme
from ..coding_scheme import (FrozenDict11, HierarchicalScheme, CodingSchemesManager)
from ..utils import resources_path


class PrFlatICD10(ICDScheme):

    @staticmethod
    def format(code: str) -> str:
        # No decimal point in ICD10-PCS
        return code


class DxHierarchicalICD10(ICDHierarchicalScheme):
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
        if '.' in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        if len(code) > 3:
            return code[:3] + '.' + code[3:]
        else:
            return code


class DxFlatICD10(ICDScheme):

    @staticmethod
    def format(code: str) -> str:
        return DxHierarchicalICD10.format(code)


class ICD10CM:
    FILE_TITLE: str = 'icd10cm_tabular_2023.xml.gz'

    @classmethod
    def traverse_icd10_xml(cls) -> dict[str, Any]:
        # https://www.cdc.gov/nchs/icd/Comprehensive-listing-of-ICD-10-CM-Files.htm
        with gzip.open(resources_path("ICD", cls.FILE_TITLE), 'r') as f:
            tree = ET.parse(f)
        root = tree.getroot()
        pt2ch = defaultdict(list)
        root_node = f'root:{root.tag}'
        desc = {root_node: 'root'}
        chapters = [ch for ch in root if ch.tag == 'chapter']

        def _traverse_diag_dfs(parent_name, dx_element):
            dx_name = next(e for e in dx_element if e.tag == 'name').text
            dx_desc = next(e for e in dx_element if e.tag == 'desc').text
            dx_name = f'dx_icd10:{dx_name}'
            desc[dx_name] = dx_desc
            pt2ch[parent_name].append(dx_name)
            for dx in (dx for dx in dx_element if dx.tag == 'diag'):
                _traverse_diag_dfs(dx_name, dx)

        for chapter in chapters:
            ch_name = next(e for e in chapter if e.tag == 'name').text
            ch_desc = next(e for e in chapter if e.tag == 'desc').text
            ch_name = f'chapter:{ch_name}'
            pt2ch[root_node].append(ch_name)
            desc[ch_name] = ch_desc

            sections = [sec for sec in chapter if sec.tag == 'section']
            for section in sections:
                sec_name = section.attrib['id']
                sec_desc = next(e for e in section if e.tag == 'desc').text
                sec_name = f'section:{sec_name}'
                pt2ch[ch_name].append(sec_name)
                desc[sec_name] = sec_desc

                for dx in (dx for dx in section if dx.tag == 'diag'):
                    _traverse_diag_dfs(sec_name, dx)

        icd_codes = tuple(sorted(c.split(':')[1] for c in desc if 'dx_icd10:' in c))
        icd_desc = {c: desc[f'dx_icd10:{c}'] for c in icd_codes}
        return {'icd_codes': icd_codes, 'icd_desc': icd_desc, 'dag_desc': desc,
                'pt2ch': {pt: frozenset(ch) for pt, ch in pt2ch.items()}}

    @staticmethod
    def flat_scheme_data(data: dict[str, Any]) -> dict[str, Any]:
        return {
            'codes': tuple(sorted(data['icd_codes'])),
            'desc': FrozenDict11(data['icd_desc'])
        }

    @staticmethod
    def hierarchical_scheme_data(data: dict[str, Any]) -> dict[str, Any]:
        def leaf_code(c):
            return 'dx_icd10_leaf:' + c

        def node_code(c):
            return 'dx_icd10_node:' + c

        desc = data['dag_desc']
        icd_codes = data['icd_codes']
        icd_desc = data['icd_desc']
        pt2ch = data['pt2ch']
        raw_nodes = set(c for c in desc if 'dx_icd10:' not in c)

        dag_codes_nodes = tuple(sorted(node_code(c) for c in raw_nodes))
        dag_desc_nodes = {node_code(c): desc[c] for c in raw_nodes}
        pt2ch = {node_code(k): set(node_code(v) for v in vs) for k, vs in pt2ch.items()}
        return {
            'codes': icd_codes,
            'desc': FrozenDict11(icd_desc),
            'code2dag': FrozenDict11({c: leaf_code(c) for c in icd_codes}),
            'dag_codes': tuple(sorted(leaf_code(c) for c in icd_codes)) + dag_codes_nodes,
            'dag_desc': FrozenDict11({leaf_code(c): icd_desc[c] for c in icd_codes} | dag_desc_nodes),
            'ch2pt': HierarchicalScheme.reverse_connection(cast(dict[str, frozenset[str]], pt2ch))
        }

    @classmethod
    def create_schemes(cls, flat: bool, hierarchical: bool) -> CodingSchemesManager:  # expose
        manager = CodingSchemesManager()
        if not any((flat, hierarchical)):
            return manager
        data = cls.traverse_icd10_xml()
        if hierarchical:
            manager = manager.add_scheme(DxHierarchicalICD10(name='dx_icd10', **cls.hierarchical_scheme_data(data)))
        if flat:
            manager = manager.add_scheme(DxFlatICD10(name='dx_flat_icd10', **cls.flat_scheme_data(data)))
        return manager


class ICD10PCS:
    FILE_TITLE: str = 'icd10pcs_codes_2023.txt.gz'

    @classmethod
    def distill_icd10_xml(cls) -> dict[str, Any]:
        with gzip.open(resources_path("ICD", cls.FILE_TITLE), 'rt') as f:
            desc = {
                code: desc
                for code, desc in map(lambda line: line.strip().split(' ', 1),
                                      f.readlines())
            }
            return {
                'codes': tuple(sorted(desc)),
                'desc': FrozenDict11(desc)
            }

    @classmethod
    def create_scheme(cls) -> PrFlatICD10:  # expose
        return PrFlatICD10(name='pr_flat_icd10', **cls.distill_icd10_xml())
