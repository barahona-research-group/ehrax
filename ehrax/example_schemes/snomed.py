import json
from typing import Optional, Self

import networkx as nx
import pandas as pd

from ..coding_scheme import (FrozenDict11, FrozenDict1N, HierarchicalScheme)
from ..utils import tqdm_constructor


class SNOMEDCT(HierarchicalScheme):
    cdb_df: pd.DataFrame
    cdb_inactive_df: pd.DataFrame
    active_terms: set[str]

    def __init__(self, name: str, codes: tuple[str, ...], desc: FrozenDict11, cdb_df: pd.DataFrame,
                 cdb_inactive_df: pd.DataFrame,
                 active_terms: set[str], ch2pt: FrozenDict1N) -> None:
        super().__init__(name=name, codes=codes, desc=desc, ch2pt=ch2pt)
        self.cdb_df = cdb_df
        self.cdb_inactive_df = cdb_inactive_df
        self.active_terms = active_terms

    @classmethod
    def from_files(cls, name: str, cdb_active_path: str, cdb_inactive_path: str, ch2pt_json_path: str) -> Self:
        def cdb_table(filename: str) -> tuple[pd.DataFrame, set[str], pd.DataFrame]:
            df = pd.read_csv(filename, index_col=0)
            terms = set(df.cui.unique())
            desc_table = df[df.tty == 1].reset_index(drop=True)
            desc_table = desc_table.groupby('cui').agg(name=('str', lambda x: x.values[0]))
            return df, terms, desc_table

        cdb_df, active_terms, active_desc = cdb_table(cdb_active_path)
        cdb_inactive_df, inactive_terms, inactive_desc = cdb_table(cdb_inactive_path)
        # the active replaces inactive for any overlap
        desc = inactive_desc['name'].to_dict() | active_desc['name'].to_dict()

        with open(ch2pt_json_path) as json_file:
            ch2pt = {ch: set(pts) for ch, pts in json.load(json_file).items()}

        return cls(
            name=name,
            codes=tuple(sorted(active_terms | inactive_terms)),
            desc=FrozenDict11(desc),
            cdb_df=cdb_df, cdb_inactive_df=cdb_inactive_df, active_terms=active_terms,
            ch2pt=FrozenDict1N(ch2pt))

    def to_networkx(self,
                    codes: tuple[str, ...] = None,
                    discard_set: Optional[set[str]] = None,
                    node_attrs: Optional[dict[str, dict[str, str]]] = None) -> nx.DiGraph:
        """
        Generate a networkx.DiGraph (Directed Graph) from a table of SNOMED-CT codes.

        Args:
            codes (tuple[str, ...]): The table of codes, must have a column `core_code` for the SNOMED-CT codes.
            discard_set (Optional[set[str]]): A set of codes, which, if provided, they are excluded from
                the Graph object.
            node_attrs: A dictionary of node attributes, which, if provided, used to annotate nodes with additional
                information, such as the frequency of the corresponding SNOMED-CT code in a particular dataset.
        """

        if codes is None:
            codes = set(self.codes) & set(self.ch2pt.keys())

        def parents_traversal(x):
            ch2pt_edges = set()

            def parents_traversal_(node):
                for pt in self.ch2pt.get(node, set()):
                    ch2pt_edges.add((node, pt))
                    parents_traversal_(pt)

            parents_traversal_(x)
            return ch2pt_edges

        if discard_set:
            ch2pt_edges = [parents_traversal(c) for c in tqdm_constructor((c for c in codes if c not in discard_set))]

        else:
            ch2pt_edges = [parents_traversal(c) for c in tqdm_constructor(codes)]

        dag = nx.DiGraph()

        for ch, pt in set().union(*ch2pt_edges):
            dag.add_edge(ch, pt)

        if node_attrs is not None:
            for node in tqdm_constructor(dag.nodes):
                for attr_name, attr_dict in node_attrs.items():
                    dag.nodes[node][attr_name] = attr_dict.get(node, '')
        return dag
