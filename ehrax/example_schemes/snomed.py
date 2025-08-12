import json
import os
from collections import defaultdict
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


TERMS_DICT = {
    "T-00000": "SNOMED RT+CTV3",
    "T-01000": "body structure",
    "T-01100": "morphologic abnormality",
    "T-01200": "cell structure",
    "T-01210": "cell",
    "T-02000": "finding",
    "T-02100": "disorder",
    "T-03000": "environment / location",
    "T-03100": "environment",
    "T-03200": "geographic location",
    "T-04000": "event",
    "T-05000": "observable entity",
    "T-06000": "organism",
    "T-07000": "product",
    "T-07100": "medicinal product",
    "T-07110": "medicinal product form",
    "T-07111": "clinical drug",
    "T-08000": "physical force",
    "T-09000": "physical object",
    "T-10000": "procedure",
    "T-10100": "regime/therapy",
    "T-11000": "qualifier value",
    "T-11100": "administration method",
    "T-11200": "disposition",
    "T-11300": "intended site",
    "T-11800": "supplier",
    "T-11900": "product name",
    "T-11400": "release characteristic",
    "T-11500": "transformation",
    "T-11020": "basic dose form",
    "T-11030": "dose form",
    "T-11600": "role",
    "T-11700": "state of matter",
    "T-11040": "unit of presentation",
    "T-12000": "record artifact",
    "T-13000": "situation",
    "T-14000": "metadata",
    "T-14100": "core metadata concept",
    "T-14200": "foundation metadata concept",
    "T-14300": "linkage concept",
    "T-14310": "attribute",
    "T-14320": "link assertion",
    "T-14400": "namespace concept",
    "T-14500": "OWL metadata concept",
    "T-15000": "social concept",
    "T-15100": "life style",
    "T-15010": "racial group",
    "T-15020": "ethnic group",
    "T-15200": "occupation",
    "T-15300": "person",
    "T-15400": "religion/philosophy",
    "T-16000": "special concept",
    "T-16100": "inactive concept",
    "T-16200": "navigational concept",
    "T-17000": "specimen",
    "T-18000": "staging scale",
    "T-18100": "assessment scale",
    "T-18200": "tumor staging",
    "T-19000": "substance",
}

DESCRIPTION_RELATION = '900000000000003001'
SYNONYM_RELATION = '900000000000013009'


def link_with_desc(terms: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    # Create a MedCAT concept database including all synonyms
    with_desc = pd.merge(terms, desc[desc['typeId'] == DESCRIPTION_RELATION], left_on=['id'],
                         right_on=['conceptId'], how='inner')
    # drop duplicates
    _with_desc = with_desc.drop_duplicates(['id_x'], keep='first')
    assert len(with_desc) == len(terms)
    with_desc = with_desc.assign(tui=with_desc['term'].str.extract("\((\w+\s?.?\s?\w+.?\w+.?\w+.?)\)$"))
    _ = pd.merge(terms, with_desc, left_on=['id'], right_on=['conceptId'], how='inner')
    with_primary_desc = _[_['typeId'] == DESCRIPTION_RELATION]
    with_primary_desc = with_primary_desc.drop_duplicates(['id_x'], keep='first')
    with_synonym_desc = _[_['typeId'] == SYNONYM_RELATION]
    with_desc = pd.concat([with_primary_desc, with_synonym_desc])
    # Check if there are the same amount of active concepts
    assert len(with_desc[with_desc['typeId'] == DESCRIPTION_RELATION]) == len(terms)
    snomed_cdb = pd.merge(with_desc,  left_on=['id_x'], right_on=['conceptId'], how='inner')
    # clean up the merge and rename the columns to fit the medcat Concept database criteria
    snomed_cdb = snomed_cdb.loc[:, ['id_x_x', 'term_x', 'typeId_x', 'tui']]
    snomed_cdb.columns = ['cui', 'str', 'tty', 'sty']
    snomed_cdb['onto'] = 'SNOMED-CT'
    snomed_cdb['tty'] = snomed_cdb['tty'].replace([DESCRIPTION_RELATION, SYNONYM_RELATION],
                                                  [1, 0])
    snomed_cdb['cui'] = 'S-' + snomed_cdb['cui'].astype(str)

    # Check if all Semantic Tags are assigned a term unique identifier (TUI)
    # Add tui codes
    dict2 = {v: k for k, v in TERMS_DICT.items()}
    snomed_cdb["tui"] = snomed_cdb["sty"].map(dict2)
    return snomed_cdb


def load_snomed_ct_uk_monolith(monolith_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """
    # TODO: cleanup + factorise.
    To understand the SNOMED-CT organisation/philosophy: https://confluence.ihtsdotools.org/display/DOCRELFMT
    ## SNOMED CT Design

        ### SNOMED CT Components
        SNOMED CT is a clinical terminology containing concepts with unique meanings and formal logic based definitions organised into hierarchies.
        For further information please see: https://confluence.ihtsdotools.org/display/DOCSTART/4.+SNOMED+CT+Basics

        SNOMED CT content is represented into 3 main types of components:
        - __Concepts__ representing clinical meanings that are organised into hierarchies.
        - __Descriptions__ which link appropriate human-readable terms to concepts
        - __Relationships__ which link each concept to other related concepts
    
    """

    def parse_file(filename, first_row_header=True, columns=None) -> pd.DataFrame:
        with open(filename, encoding='utf-8') as f:
            entities = [[n.strip() for n in line.split('\t')] for line in f]
            return pd.DataFrame(entities[1:], columns=entities[0] if first_row_header else columns)

    def filename(l: list[str], prefix: str) -> str:
        match = [f for f in l if f.lower().startswith(prefix)]
        assert len(match) == 1
        return match[0]

    term_dir = f'{monolith_dir}/Snapshot/Terminology'
    term_dir_files = os.listdir(term_dir)
    concept_file = os.path.join(term_dir, filename(term_dir_files, 'sct2_concept'))
    description_file = os.path.join(term_dir, filename(term_dir_files, 'sct2_description'))
    terms = parse_file(concept_file)
    desc = parse_file(description_file)

    active_terms = terms[terms.active == '1']  # active concepts are represented with 1
    inactive_terms = terms[terms.active != '1']
    active_descs = desc[desc.active == '1']
    inactive_descs = desc[desc.active != '1']

    # Write the clinical terms to csv
    snomed_cdb_active_df = link_with_desc(active_terms, active_descs)
    snomed_cdb_inactive_df = link_with_desc(inactive_terms, inactive_descs)
    snomed_cdb_active_df.to_csv(f'snomed_cdb_active.csv.gz', compression='gzip')
    snomed_cdb_inactive_df.to_csv(f'snomed_cdb_inactive.csv.gz', compression='gzip')

    ###################
    ### Relations
    ###################
    relations_file = os.path.join(term_dir, filename(term_dir_files, 'sct2_relationship'))
    relations = parse_file(relations_file)
    active_relat = relations[relations.active == '1']
    active_relat[['sourceId', 'destinationId', 'typeId']] = 'S-' + active_relat[
        ['sourceId', 'destinationId', 'typeId']].astype(str)

    ch2pt = defaultdict(list)
    for index, v in active_relat[active_relat.typeId == "S-116680003"].iterrows():
        # Children to Parent dictionary ("Is a" relationships)
        ch2pt[v['sourceId']].append(v['destinationId'])

    # Write to 'isa' relationships to file
    with open(f'isa_active_rela_ch2pt.json', 'w') as outfile:
        json.dump(dict(ch2pt), outfile)

    return snomed_cdb_active_df, snomed_cdb_inactive_df, ch2pt
