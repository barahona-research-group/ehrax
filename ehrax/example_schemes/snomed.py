import json
import os
from collections.abc import Iterable, Mapping
from functools import cached_property
from types import MappingProxyType
from typing import Self

import networkx as nx
import pandas as pd

from ..coding_scheme import FrozenDict1N, FrozenDict11, HierarchicalScheme
from ..utils import tqdm_constructor


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

DESCRIPTION_RELATION = "900000000000003001"
SYNONYM_RELATION = "900000000000013009"
IS_A_RELATION = "116680003"

OTHER_RELATIONS = {
    "scale_type": "S-370132008",  # Scale type
    "property": "S-370130000",  # Property
    "inheres_in": "S-704319004",  # Inheres-in
    "inherent_location": "S-718497002",  # Inherent location
    "characterizes": "S-704321009",  # Characterises
    "direct_site": "S-704327008",  # Direct-Site
    "process_output": "S-704324001",  # Process-output
    "units": "S-246514001",  # Units
}


class SNOMEDCTGBMonolith:
    @staticmethod
    def link_with_desc(terms: pd.DataFrame, desc: pd.DataFrame, active: bool = True) -> pd.DataFrame:
        """
        terms is the basic table of concepts. The table consists of the following columns:

        |Field              |Data type  |Purpose    |Mutable|Part of Primary Key|
        |-------------------|-----------|-----------|-------|-------------------|
        |id                 |SCTID      |Uniquely Idenfies the concept|NO|YES (Full/Snapshot)|
        |effectiveTime      |Time       |Specifies the inclusive date at which the component version's state became the
                then current valid state of the component.|YES|YES (Full)<br>Optional (Snapshot)|
        |active             |Boolean    |Specifies whether the concept was active or inactive from the nominal release
                date specified by the effectiveTime.|YES|NO|
        |moduleId           |SCTID      |Identifies the concept version's module. Set to a descendant of
                900000000000443000(Module) within the metadata hierarchy.|YES|NO|
        |definitionStatusId |SCTID      |Specifies if the concept version is primitive or defined. Set to a descendant
                of 900000000000444006(Definition status)in the metadata hierarchy.|YES|NO|

        desc is the table of descriptions linked to each concept. The table consists of the following columns:

        |Field              |Data type  |Purpose    |Mutable|Part of Primary Key|
        |-------------------|-----------|-----------|-------|-------------------|
        |id                 |SCTID      |Uniquely identifies the description.|NO|YES (Full/Snapshot)|
        |effectiveTime      |Time       |Specifies the inclusive date at which the component version's state became the
                then current valid state of the component|YES|YES (Full)<br>Optional |Snapshot||
        |active             |Boolean    |Specifies whether the state of the description was active or inactive from the
                nominal release date specified by the effectiveTime.|YES|NO|
        |moduleId           |SCTID      |Identifies the description version's module. Set to a child of
                900000000000443000|Module| within the metadata hierarchy.|YES|NO|
        |conceptId          |SCTID      |Identifies the concept to which this description applies. Set to the identifier
                of a concept in the 138875005 |SNOMED CT Concept| hierarchy within the Concept. Note that a specific
                version of a description is not directly bound to a specific version of the concept to which it applies.
                Which version of a description applies to a concept depends on its effectiveTime and the point in time
                at which it is accessed.|NO|NO|
        |languageCode       |String     |Specifies the language of the description text using the two character
                ISO-639-1 code. Note that this specifies a language level only, not a dialect or country code.|NO|NO|
        |typeId             |SCTID      |Identifies whether the description is fully specified name a synonym or other
                description type. This field is set to a child of 900000000000446008|Description type| in the
                Metadata hierarchy.|NO|NO|
        |term               |String     |The description version's text value, represented in UTF-8 encoding.|YES|NO|
        |caseSignificanceId |SCTID      |Identifies the concept enumeration value that represents the case
                significance of this description version. For example, the term may be completely case sensitive, case
                insensitive or initial letter case insensitive. This field will be set to a child of
                900000000000447004|Case significance| within the metadata hierarchy.|YES|NO|


        Taken from: https://confluence.ihtsdotools.org/display/DOCRELFMT
        """

        def link_all_desc(terms_df: pd.DataFrame, desc_df: pd.DataFrame) -> pd.DataFrame:
            _ = pd.merge(terms_df, desc_df, left_on=["id"], right_on=["conceptId"], how="inner")
            with_primary_desc = _[_["typeId"] == DESCRIPTION_RELATION]
            with_primary_desc = with_primary_desc.drop_duplicates(["id_x"], keep="first")
            with_synonym_desc = _[_["typeId"] == SYNONYM_RELATION]
            return pd.concat([with_primary_desc, with_synonym_desc])

        with_desc = pd.merge(
            terms, desc[desc["typeId"] == DESCRIPTION_RELATION], left_on=["id"], right_on=["conceptId"], how="inner"
        )
        with_desc = with_desc.drop_duplicates(["id_x"], keep="first")
        assert not active or len(with_desc) == len(terms)
        with_desc = with_desc.assign(tui=with_desc["term"].str.extract(r"\((\w+\s?.?\s?\w+.?\w+.?\w+.?)\)$"))
        with_all_desc = link_all_desc(terms, desc)
        # Check if there are the same amount of active concepts
        assert not active or len(with_all_desc[with_all_desc["typeId"] == DESCRIPTION_RELATION]) == len(terms)
        snomed_cdb_df = pd.merge(with_all_desc, with_desc, left_on=["id_x"], right_on=["conceptId"], how="inner")
        # clean up the merge and rename the columns to fit the medcat Concept database criteria
        rename_cols = {"id_x_x": "cui", "term_x": "str", "typeId_x": "tty", "tui": "sty"}
        snomed_cdb_df = snomed_cdb_df.loc[:, list(rename_cols.keys())].rename(columns=rename_cols)

        # Add tui codes
        dict2 = {v: k for k, v in TERMS_DICT.items()}
        return snomed_cdb_df.assign(
            onto="SNOMED-CT",
            tty=snomed_cdb_df["tty"].str.replace(DESCRIPTION_RELATION, "1").str.replace(SYNONYM_RELATION, "0"),
            cui="S-" + snomed_cdb_df["cui"].astype(str),
            tui=snomed_cdb_df["sty"].map(dict2),
        )

    @staticmethod
    def parse_file(fname, first_row_header=True, columns=None) -> pd.DataFrame:
        with open(fname, encoding="utf-8") as f:
            entities = [[n.strip() for n in line.split("\t")] for line in f]
            return pd.DataFrame(entities[1:], columns=entities[0] if first_row_header else columns)

    @classmethod
    def load(
        cls, monolith_dir: str, write_disk: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], Mapping[str, Mapping[str, str]]]:
        """
        # TODO: cleanup + factorise.
        To understand the SNOMED-CT organisation/philosophy: https://confluence.ihtsdotools.org/display/DOCRELFMT
        ## SNOMED CT Design

            ### SNOMED CT Components
            SNOMED CT is a clinical terminology containing concepts with unique meanings and formal logic based
                definitions organised into hierarchies.
            For further information please see: https://confluence.ihtsdotools.org/display/DOCSTART/4.+SNOMED+CT+Basics

            SNOMED CT content is represented into 3 main types of components:
            - __Concepts__ representing clinical meanings that are organised into hierarchies.
            - __Descriptions__ which link appropriate human-readable terms to concepts
            - __Relationships__ which link each concept to other related concepts

        """

        def filename(l: list[str], prefix: str) -> str:
            match = [f for f in l if f.lower().startswith(prefix)]
            assert len(match) == 1
            return match[0]

        term_dir = f"{monolith_dir}/Snapshot/Terminology"
        term_dir_files = os.listdir(term_dir)
        concept_file = os.path.join(term_dir, filename(term_dir_files, "sct2_concept_"))
        description_file = os.path.join(term_dir, filename(term_dir_files, "sct2_description_"))
        terms = cls.parse_file(concept_file)
        desc = cls.parse_file(description_file)

        active_terms = terms[terms.active == "1"]  # active concepts are represented with 1
        inactive_terms = terms[terms.active != "1"]
        active_descs = desc[desc.active == "1"]
        inactive_descs = desc[desc.active != "1"]

        snomed_cdb_active_df = cls.link_with_desc(active_terms, active_descs, active=True)
        snomed_cdb_inactive_df = cls.link_with_desc(inactive_terms, inactive_descs, active=False)

        ###################
        ### Relations
        ###################
        # |Field                |Data type  |Purpose|Mutable|Part of Primary Key|
        # |---------------------|-----------|-------|-------|-----|
        # |id                   |SCTID      |Uniquely identifies the relationship.|NO|YES(Full/Snapshot)|
        # |effectiveTime        |Time       |Specifies the inclusive date at which the component version's state
        #           became the then current valid state of the component.|YES|YES(Full) Optional(Snapshot)|
        # |active               |Boolean    |Specifies whether the state of the relationship was active or inactive
        #       from the nominal release date specified by the effectiveTime field.|YES|NO|
        # |moduleId             |SCTID      |Identifies the relationship version's module. Set to a child of
        #       900000000000443000|Module| within the metadata hierarchy.|YES|NO|
        # |sourceId             |SCTID      |Identifies the source concept of the relationship version. That is the
        #       concept defined by this relationship. Set to the identifier of a concept.|NO|NO|
        # |destinationId |SCTID      |Identifies the concept that is the destination of the relationship version.
        #       <br>That is the concept representing the value of the attribute represented by the typeId column.
        #       <br>Set to the identifier of a concept.<br>Note that the values that can be applied to particular
        #       attributes are formally defined by the SNOMED CT Machine Readable Concept Model.|NO|NO|
        # |relationshipGroup    |Integer    |Groups together relationship versions that are part of a
        #       logically associated relationshipGroup. All active Relationship records with the same relationship Group
        #       number and sourceId are grouped in this way.|YES|NO|
        # |typeId               |SCTID      |Identifies the concept that represent the defining attribute (or
        #       relationship type) represented by this relationship version.<br><br>That is the concept representing
        #       the value of the attribute represented by the typeId column. <br><br>Set to the identifier of a concept.
        #       The concept identified must be either 116680003|Is a| or a subtype of
        #       410662002|Concept model attribute|. The concepts that can be used as in the typeId column are formally
        #       defined as follows:<br>116680003|is a| OR < 410662002|concept model attribute|<br><br>__Note__ that the
        #       attributes that can be applied to particular concepts are formally defined by the SNOMED CT Machine
        #       Readable Concept Model.|NO|NO|
        # |characteristicTypeId |SCTID      |A concept enumeration value that identifies the characteristic type of the
        #       relationship version (i.e. whether the relationship version is defining, qualifying, etc.) This field is
        #       set to a descendant of 900000000000449001|Characteristic type|in the metadata hierarchy.|YES|NO|
        # |modifierId           |SCTID      |A concept enumeration value that identifies the type of Description
        #       Logic(DL) restriction (some, all, etc.). Set to a child of 900000000000450001|Modifier| in the metadata
        #       hierarchy.<br> __Note__ Currently the only value used in this column is 900000000000451002|Some| and
        #       thus in practical terms this column can be ignored.|YES|NO|
        relations_file = os.path.join(term_dir, filename(term_dir_files, "sct2_relationship_"))
        rel = cls.parse_file(relations_file)
        is_a = rel.loc[(rel.active == "1") & (rel.typeId == IS_A_RELATION), ["sourceId", "destinationId"]].astype(str)
        is_a = "S-" + is_a
        ch2pt = is_a.groupby("sourceId")["destinationId"].apply(list).to_dict()

        if write_disk:
            # Write the clinical terms to csv
            snomed_cdb_active_df.to_csv("snomed_cdb_active.csv.gz", compression="gzip")
            snomed_cdb_inactive_df.to_csv("snomed_cdb_inactive.csv.gz", compression="gzip")

            # Write to 'isa' relationships to file
            with open("isa_active_rela_ch2pt.json", "w") as outfile:
                json.dump(dict(ch2pt), outfile)

        # Other relations.
        other_rel = rel.loc[rel.active == "1", ["sourceId", "destinationId", "typeId"]]
        other_rel = "S-" + other_rel
        other_rel = other_rel.loc[other_rel["typeId"].isin(OTHER_RELATIONS.values()), :]
        relation_label = {v: k for k, v in OTHER_RELATIONS.items()}
        other_relations = {
            relation_label[relation_type_id]: relations.set_index("sourceId")["destinationId"].to_dict()
            for relation_type_id, relations in other_rel.groupby("typeId")
        }

        return snomed_cdb_active_df, snomed_cdb_inactive_df, ch2pt, other_relations

    @classmethod
    def process_refset(cls, filename: str) -> pd.DataFrame:
        df = cls.parse_file(filename)
        df = df[df.active == "1"]
        df = df.rename(columns={"referencedComponentId": "member", "refsetId": "refset"})
        df = df[["member", "refset"]]
        for c in df.columns:
            df.loc[:, c] = df.loc[:, c].str.strip()
        return "S-" + df


class SNOMEDCT(HierarchicalScheme):
    cdb_df: pd.DataFrame
    cdb_inactive_df: pd.DataFrame
    active_terms: frozenset[str]
    other_relations: MappingProxyType[str, FrozenDict11[str]] | None

    def __init__(
        self,
        name: str,
        codes: tuple[str, ...],
        desc: FrozenDict11,
        cdb_df: pd.DataFrame,
        cdb_inactive_df: pd.DataFrame,
        active_terms: set[str],
        ch2pt: FrozenDict1N,
        other_relations: Mapping[str, Mapping[str, str]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, codes=codes, desc=desc, ch2pt=ch2pt, **kwargs)
        self.cdb_df = cdb_df
        self.cdb_inactive_df = cdb_inactive_df
        self.active_terms = frozenset(active_terms)
        if other_relations is None:
            other_relations = MappingProxyType({k: MappingProxyType({}) for k in OTHER_RELATIONS.keys()})

        self.other_relations = other_relations

    def to_desc(self, m: Mapping[str, str]) -> Mapping[str, str]:
        return {k: self.desc.get(v) for k, v in m.items()}

    @cached_property
    def code_inheres_in(self) -> Mapping[str, str]:
        return self.to_desc(self.other_relations["inheres_in"])

    @cached_property
    def code_property(self) -> Mapping[str, str]:
        return self.to_desc(self.other_relations["property"])

    @cached_property
    def code_inherent_location(self) -> Mapping[str, str]:
        return self.to_desc(self.other_relations["inherent_location"])

    @cached_property
    def code_characterizes(self) -> Mapping[str, str]:
        return self.to_desc(self.other_relations["characterizes"])

    @cached_property
    def code_direct_site(self) -> Mapping[str, str]:
        return self.to_desc(self.other_relations["direct_site"])

    @cached_property
    def code_units(self) -> Mapping[str, str]:
        return self.to_desc(self.other_relations["units"])

    @classmethod
    def from_tables(
        cls,
        name: str,
        cdb_active: pd.DataFrame,
        cdb_inactive: pd.DataFrame,
        ch2pt: dict[str, list[str]],
        other_relations: Mapping[str, Mapping[str, str]],
    ) -> Self:
        cdb_df, active_terms, active_desc = cls.cdb_table(cdb_active)
        cdb_inactive_df, inactive_terms, inactive_desc = cls.cdb_table(cdb_inactive)
        # the active replaces inactive for any overlap
        desc = inactive_desc["name"].to_dict() | active_desc["name"].to_dict()
        return cls(
            name=name,
            codes=tuple(sorted(active_terms | inactive_terms)),
            desc=FrozenDict11(desc),
            cdb_df=cdb_df,
            cdb_inactive_df=cdb_inactive_df,
            active_terms=active_terms,
            ch2pt=FrozenDict1N({k: frozenset(v) for k, v in ch2pt.items()}),
            other_relations=MappingProxyType({k: MappingProxyType(v) for k, v in other_relations.items()}),
        )

    @classmethod
    def cdb_table(cls, cdb: pd.DataFrame) -> tuple[pd.DataFrame, set[str], pd.DataFrame]:
        terms = set(cdb.cui.unique())
        desc_table = cdb[cdb.tty == "1"].reset_index(drop=True)
        desc_table = desc_table.groupby("cui").agg(name=("str", lambda x: x.values[0]))
        return cdb, terms, desc_table

    @classmethod
    def from_files(cls, name: str, cdb_active_path: str, cdb_inactive_path: str, ch2pt_json_path: str) -> Self:
        cdb_active_table = pd.read_csv(cdb_active_path, index_col=0)
        cdb_inactive_table = pd.read_csv(cdb_inactive_path, index_col=0)
        with open(ch2pt_json_path) as json_file:
            return cls.from_tables(name, cdb_active_table, cdb_inactive_table, json.load(json_file))

    @classmethod
    def from_gb_monolith_dir(cls, name: str, gb_monolith_dir: str) -> Self:
        cdb_active, cdb_inactive, ch2pt, other_relations = SNOMEDCTGBMonolith.load(gb_monolith_dir, write_disk=False)
        return cls.from_tables(name, cdb_active, cdb_inactive, ch2pt, other_relations=other_relations)

    def to_networkx(
        self,
        codes: tuple[str, ...] = None,
        discard_set: set[str] | None = None,
        node_attrs: dict[str, dict[str, str]] | None = None,
    ) -> nx.DiGraph:
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

        def parents_traversal(x) -> frozenset[str]:
            ch2pt_edges = set()

            def parents_traversal_(node):
                for pt in self.ch2pt.get(node, ()):
                    ch2pt_edges.add((node, pt))
                    parents_traversal_(pt)

            parents_traversal_(x)
            return frozenset(ch2pt_edges)

        if discard_set:
            ch2pt_edges = [parents_traversal(c) for c in tqdm_constructor(c for c in codes if c not in discard_set)]
        else:
            ch2pt_edges = [parents_traversal(c) for c in tqdm_constructor(codes)]

        dag = nx.DiGraph()

        for ch, pt in frozenset().union(*ch2pt_edges):
            dag.add_edge(ch, pt)

        if node_attrs is not None:
            for node in tqdm_constructor(dag.nodes):
                for attr_name, attr_dict in node_attrs.items():
                    dag.nodes[node][attr_name] = attr_dict.get(node, "")
        return dag

    def as_dataframe(self, codes: Iterable[str] | None = None) -> pd.DataFrame:
        """
        Returns the scheme as a Pandas DataFrame.
        The DataFrame contains the following columns:
            - code: the code string
            - desc: the code description
        """
        if codes is None:
            codes = self.codes
        index = codes
        return pd.DataFrame(
            {
                "code": codes,
                "desc": list(map(self.desc.get, codes)),
                "inheres_in": self.code_inheres_in,
                "property": self.code_property,
                "units": self.code_units,
                "characetrizes": self.code_characterizes,
                "direct_site": self.code_direct_site,
                "inherent_location": self.code_inherent_location,
            },
            index=index,
        )
