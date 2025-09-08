"""Extract diagnostic/procedure information of CCS files into new
data structures to support conversion between CCS and ICD9."""

import logging
import math
import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
from collections.abc import Collection, Iterable, Mapping, Sized
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import cast, Self

import numpy as np
import pandas as pd

from ._literals import AggregationLiteral, NumericalTypeHint
from .base import AbstractVxData
from .freezer import FrozenDict1N, FrozenDict1NM, FrozenDict11
from .utils import Array, dataframe_log, load_config, resources_path, tqdm_constructor


class CodesVector(AbstractVxData):
    vec: Array
    scheme: str

    def __init__(self, vec: Array, scheme: str):
        self.vec = vec
        self.scheme = scheme

    def __hash__(self) -> int:
        return hash(self.scheme + str(self.vec))

    @classmethod
    def empty_like(cls, other: Self) -> Self:
        """
        Creates an empty CodesVector with the same shape as the given CodesVector.

        Args:
            other (CodesVector): the CodesVector to mimic.

        Returns:
            CodesVector: the empty CodesVector.
        """
        return cls(np.zeros_like(other.vec), other.scheme)

    def union(self, other: Self) -> Self:
        """
        Returns the union of the current CodesVector with another CodesVector.

        Args:
            other (CodesVector): the other CodesVector to union with.

        Returns:
            CodesVector: the union of the two CodesVectors.
        """
        assert self.scheme == other.scheme, "Schemes should be the same."
        assert self.vec.dtype == bool and other.vec.dtype == bool, "Vector types should be the same."
        return type(self)(self.vec | other.vec, self.scheme)


class Formatter(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def format(code: str) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def deformat(code: str) -> str:
        raise NotImplementedError

    @classmethod
    def reformat(cls, code: str) -> str:
        return cls.format(cls.deformat(code))


@dataclass
class LegacyUpdater(metaclass=ABCMeta):
    legacy_map: FrozenDict11[str]

    def __init__(self, legacy_map: FrozenDict11[str]):
        self.legacy_map = legacy_map

    def update(self, code: str) -> str:
        if code not in self.legacy_map:
            return code
        return self.legacy_map[code]


class CodingScheme(AbstractVxData):
    name: str
    codes: tuple[str, ...]
    desc: FrozenDict11[str]

    def __init__(self, name: str, codes: tuple[str, ...], desc: FrozenDict11[str] | None = None):
        self.name = name
        self.desc = desc or FrozenDict11({c: c for c in codes})
        self.codes = codes

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, codes({len(self.codes)}), desc({len(self.desc)}))"

    def __check_init__(self):
        self._check_types()

        # Check types.
        assert isinstance(self.name, str), "Scheme name must be a string."
        assert isinstance(self.codes, tuple), "Scheme codes must be a tuple."
        assert all(type(c) is str for c in self.codes), "Scheme codes must be str."
        assert isinstance(self.desc, FrozenDict11), "Scheme description must be a FrozenDict11."

        assert tuple(sorted(self.codes)) == self.codes, "Scheme codes must be sorted."

        # Check for uniqueness of codes.
        assert len(self.codes) == len(set(self.codes)), f"{self}: Codes should be unique."

        # Check sizes.
        assert len(self.codes) == len(self.desc), f"{self}: Codes and descriptions should have the same size."

    def _check_types(self):
        for collection in [self.codes, self.desc]:
            assert all(isinstance(c, str) for c in collection), f"{self}: All name types should be str."

        assert all(isinstance(desc, str) for desc in self.desc.values()), f"{self}: All desc types should be str."

    @cached_property
    def index(self) -> dict[str, int]:
        return {code: idx for idx, code in enumerate(self.codes)}

    @cached_property
    def index2code(self) -> dict[int, str]:
        return {idx: code for code, idx in self.index.items()}

    @cached_property
    def index2desc(self) -> dict[int, str]:
        return {self.index[code]: _desc for code, _desc in self.desc.items()}

    def __len__(self) -> int:
        """
        Returns the number of codes in the current scheme.
        """
        return len(self.codes)

    def __bool__(self) -> bool:
        """
        Returns True if the current scheme is not empty.
        """
        return len(self.codes) > 0

    def __str__(self) -> str:
        """
        Returns the name of the current scheme.
        """
        return self.name

    def __contains__(self, code: str) -> bool:
        """Returns True if `code` is contained in the current scheme."""
        return code in self.codes

    def search_regex(self, query: str, regex_flags: int = re.I) -> set[str]:
        """
        A regex-supported search of codes by a `query` string. the search is applied on the code description.
        For example, you can use it to return all codes related to cancer by setting the
        `query = 'cancer'` and `regex_flags = re.i` (for case-insensitive search).

        Args:
            query (str): the query string.
            regex_flags (int): the regex flags.

        Returns:
            set[str]: the set of codes matching the query.
        """
        return set(filter(lambda c: re.findall(query, self.desc[c], flags=regex_flags), self.codes))

    def codeset2vec(self, codeset: Iterable[str]) -> CodesVector:
        """
        Convert a codeset to a vector representation.
        Args:
            codeset (set[str]): the codeset to convert.
        Returns:
            CodesVector: a vector representation of the current scheme.
        """
        vec = np.zeros(len(self), dtype=bool)
        try:
            for c in codeset:
                vec[self.index[c]] = True
        except KeyError as missing:
            logging.error(f"Code {missing} is missing.Accepted keys: {self.index.keys()}")

        return CodesVector(vec, self.name)

    def vec2codeset(self, vec: Array) -> set[str]:
        assert vec.ndim == 1, "Vector should be 1-dimensional."
        assert len(vec) == len(self), f"Vector length should be {len(self)}."
        return set(map(lambda idx: self.index2code[int(idx)], np.where(vec)[0]))

    def as_dataframe(self, codes: Collection[str] | None = None) -> pd.DataFrame:
        """
        Returns the scheme as a Pandas DataFrame.
        The DataFrame contains the following columns:
            - code: the code string
            - desc: the code description
        """
        if codes is None:
            codes = self.codes
        index = pd.Series(list(map(self.index.get, codes)))
        return pd.DataFrame(
            {
                "code": self.index2code,
                "desc": self.index2desc,
            },
            index=index,
        )

    @classmethod
    def _init_args_from_table(
        cls,
        name: str,
        table: pd.DataFrame,
        c_code: str,
        c_desc: str | None,
        code_selection: pd.DataFrame | None,
        **kwargs,
    ) -> tuple[str, tuple[str, ...], FrozenDict11]:
        # TODO: test this method.
        # drop=False in case we use c_code as c_desc.
        df = table.astype({c_code: str}).drop_duplicates(c_code).set_index(c_code, drop=False)
        if code_selection is not None:
            df = df.loc[code_selection[c_code].drop_duplicates().astype(str).tolist()]
        return name, tuple(sorted(df.index.tolist())), FrozenDict11(df[(c_desc or c_code)].to_dict())

    @classmethod
    def from_table(
        cls,
        name: str,
        table: pd.DataFrame,
        c_code: str,
        c_desc: str | None = None,
        code_selection: pd.DataFrame | None = None,
        *args,
        **kwargs,
    ) -> Self:
        return cls(
            *cls._init_args_from_table(
                name=name, table=table, c_code=c_code, c_desc=c_desc, code_selection=code_selection
            )
        )


class NumericScheme(CodingScheme):
    """
    NumericScheme is a subclass of FlatScheme that represents a numerical coding scheme.
    Additional to `FlatScheme` attributes, it contains the following attributes to represent the coding scheme:
    - type_hint: dict mapping codes to their type hint (B: binary, N: numerical, O: ordinal, C: categorical)
    """

    type_hint: FrozenDict11[NumericalTypeHint]
    default_type_hint: NumericalTypeHint
    group: FrozenDict11[str]

    def __init__(
        self,
        name: str,
        codes: tuple[str, ...],
        desc: FrozenDict11[str] | None = None,
        group: FrozenDict11[str] | None = None,
        type_hint: FrozenDict11[NumericalTypeHint] | None = None,
        default_type_hint: NumericalTypeHint = "N",
    ):
        super().__init__(name=name, codes=codes, desc=desc)
        self.group = group or FrozenDict11({code: code for code in codes})
        self.type_hint = type_hint or FrozenDict11({code: default_type_hint for code in codes})
        self.default_type_hint = default_type_hint

    def __check_init__(self):
        assert set(self.codes) == set(self.type_hint.keys()), (
            f"The set of codes ({self.codes}) does not match the set of type hints ({self.type_hint.keys()})."
        )
        assert set(self.type_hint.values()) <= {"B", "N", "O", "C"}, (
            f"The set of type hints ({self.type_hint.values()}) contains invalid values."
        )

    @cached_property
    def types(self) -> tuple[NumericalTypeHint, ...]:
        """
        Returns the type hint of the codes in the scheme as a numpy array.
        """
        assert set(self.index[c] for c in self.codes) == set(range(len(self))), (
            f"The order of codes ({self.codes}) does not match the order of type hints ({self.type_hint.keys()})."
        )
        return tuple(self.type_hint[code] for code in self.codes)

    @cached_property
    def index2group(self) -> dict[int, str]:
        return {i: self.group[code] for i, code in enumerate(self.codes)}

    def as_dataframe(self, codes: Collection[str] | None = None) -> pd.DataFrame:
        if codes is None:
            codes = self.codes
        index = pd.Series(list(map(self.index.get, codes)))
        return pd.DataFrame(
            {"code": self.index2code, "desc": self.index2desc, "type": self.types, "group": self.index2group},
            index=index,
        )


class CodingSchemeWithUOM(CodingScheme):
    uom_normalization_factor: FrozenDict1NM  # dict[code, dict[unit, conversion_factor]]
    universal_unit: FrozenDict11

    def __init__(
        self,
        name: str,
        codes: tuple[str, ...],
        desc: FrozenDict11,
        uom_normalization_factor: FrozenDict1NM,
        universal_unit: FrozenDict11,
    ):
        super().__init__(name=name, codes=codes, desc=desc)
        self.uom_normalization_factor = uom_normalization_factor
        self.universal_unit = universal_unit

    @classmethod
    def from_table(
        cls,
        name: str,
        table: pd.DataFrame,
        c_code: str,
        c_desc: str | None = None,
        code_selection: pd.DataFrame | None = None,
        *,
        c_normalization_factor: str,
        c_unit: str,
        c_universal_unit: str | None = None,
    ) -> Self:
        name, codes, desc = cls._init_args_from_table(
            name=name, table=table, c_code=c_code, c_desc=c_desc, code_selection=code_selection
        )
        # TODO: test this method.
        df = table.astype({c_code: str, c_normalization_factor: float})
        df = df.assign(**{c_unit: df[c_unit].str.lower()})
        df = df.drop_duplicates([c_code, c_unit]).set_index(c_code)
        df = df[df.index.isin(codes)]
        assert all(c in df.columns for c in (c_unit, c_normalization_factor)), "Some columns are missing."
        if c_universal_unit is not None and c_universal_unit in df.columns:
            uom_universal = df.loc[:, c_universal_unit].to_dict()
        else:
            _u = df.loc[:, c_unit].to_dict()
            # 1. Choose one of the units with 1.0 as a normalization factor.
            uom_universal_a = df.loc[df[c_normalization_factor] == 1.0, c_unit].to_dict()
            # 2. Or if there is only one unit per code.
            uom_universal_b = {c: _u[c] for c, count in df.index.value_counts().to_dict().items() if count == 1}
            # 3. If any item remains, choose a universal unit without any specific rule, e.g. what remains in _u.
            uom_universal_c = {
                c: u for c, u in _u.items() if c not in (set(uom_universal_a.keys()) | set(uom_universal_b.keys()))
            }

            uom_universal = uom_universal_a | uom_universal_b | uom_universal_c

        # Narrow down the codes to those who have at least one universal unit (the target unit to which all
        # units are converted).
        df = df.loc[df.index.isin(uom_universal.keys()), :]
        uom_data = {
            str(code): code_df.set_index(c_unit)[c_normalization_factor].to_dict()
            for code, code_df in df.groupby(df.index)
        }
        return cls(
            name=name,
            codes=codes,
            desc=desc,
            uom_normalization_factor=FrozenDict1NM(uom_data),
            universal_unit=FrozenDict11(uom_universal),
        )


class HierarchicalScheme(CodingScheme):
    """
    A class representing a hierarchical coding scheme.

    This class extends the functionality of the FlatScheme class and provides
    additional methods for working with hierarchical coding schemes.
    """

    ch2pt: FrozenDict1N[str]
    dag_codes: tuple[str, ...]
    dag_desc: FrozenDict11[str]
    code2dag: FrozenDict11[str]

    def __init__(
        self,
        name: str,
        codes: tuple[str, ...],
        desc: FrozenDict11[str] | None = None,
        dag_codes: Iterable[str] | None = None,
        dag_desc: FrozenDict11[str] | None = None,
        code2dag: FrozenDict11[str] | None = None,
        *,
        ch2pt: FrozenDict1N[str],
    ):
        CodingScheme.__init__(self, name, codes, desc)
        self.ch2pt = ch2pt
        self.dag_codes = tuple(dag_codes or self.codes)
        self.dag_desc = dag_desc or self.desc
        self.code2dag = code2dag or FrozenDict11({c: c for c in self.codes})

    def __check_init__(self):
        # Check types
        assert isinstance(self.dag_codes, tuple), f"{self.dag_codes}: codes should be a list."
        assert isinstance(self.dag_desc, FrozenDict11), f"{self.dag_desc}: desc should be a dict."
        assert isinstance(self.code2dag, FrozenDict11), f"{self.code2dag}: code2dag should be a dict."
        assert isinstance(self.ch2pt, FrozenDict1N), f"{self.ch2pt}: ch2pt should be a dict."
        for collection in [
            self.dag_codes,
            self.dag_desc.values(),
            self.dag_desc.keys(),
            self.code2dag.keys(),
            self.dag_desc.values(),
            self.code2dag.values(),
            self.ch2pt.keys(),
            frozenset(v for vs in self.ch2pt.values() for v in vs),
        ]:
            assert all(isinstance(c, str) for c in collection), f"{self}: All name types should be str."

        # Check sizes
        # TODO: note in the documentation that dag2code size can be less than the dag_codes since some dag_codes are
        #  internal nodes that themselves are not are not complete clinical concepts.
        for collection in [self.dag_codes, self.dag_desc]:
            assert len(collection) == len(self.dag_codes), f"{self}: All collections should have the same size."

    def make_ancestors_tuples(self, include_itself: bool = True) -> tuple[tuple[int, ...], ...]:
        """
        Creates a list of ancestors for each code.

        Args:
            include_itself (bool): whether to include the code itself as its own ancestor. Defaults to True.

        Returns:
            Array: a boolean matrix where each element (i, j) is True if code i is an ancestor of code j, and
                False otherwise.
        """
        parents_indices = [[] for _ in range(len(self.dag_index))]
        for code_i, i in self.dag_index.items():
            for ancestor_j in self.code_ancestors_bfs(code_i, include_itself):
                parents_indices[i].append(self.dag_index[ancestor_j])
        return tuple(tuple(e) for e in parents_indices)

    @cached_property
    def dag_index(self) -> dict[str, int]:
        """
        dict[str, int]: a dictionary mapping codes to their indices in the hierarchy.
        """
        return {c: i for i, c in enumerate(self.dag_codes)}

    @cached_property
    def dag2code(self) -> dict[str, str]:
        """
        dict[str, str]: a dictionary mapping codes in the hierarchy to their corresponding codes.
        """
        return {d: c for c, d in self.code2dag.items()}

    @cached_property
    def pt2ch(self) -> FrozenDict1N[str]:
        return self.reverse_connection(self.ch2pt.data)

    def __contains__(self, code: str) -> bool:
        """
        Checks if a code is contained in the current hierarchy.

        Args:
            code (str): the code to check.

        Returns:
            bool: true if the code is contained in the hierarchy, False otherwise.
        """
        return code in self.dag_codes or code in self.codes

    @staticmethod
    def reverse_connection(connection: Mapping[str, frozenset[str]]) -> FrozenDict1N[str]:
        """
        Reverses a connection dictionary.

        Args:
            connection (dict[str, set[str]]): the connection dictionary to reverse.

        Returns:
            dict[str, set[str]]: the reversed connection dictionary.
        """
        rev_connection: dict[str, set[str]] = defaultdict(set)
        for node, conns in connection.items():
            for conn in conns:
                rev_connection[conn].add(node)
        return FrozenDict1N(rev_connection)

    @staticmethod
    def _bfs_traversal(connection: Mapping[str, frozenset[str]], code: str, include_itself: bool) -> list[str]:
        """
        Performs a breadth-first traversal of the hierarchy.

        Args:
            connection (dict[str, set[str]]): the connection dictionary representing the hierarchy.
            code (str): the starting code for the traversal.
            include_itself (bool): whether to include the starting code in the traversal.

        Returns:
            list[str]: a list of codes visited during the traversal.
        """
        result = OrderedDict()
        q = [code]

        while len(q) != 0:
            # remove the first element from the stack
            current_code = q.pop(0)
            current_connections = connection.get(current_code) or ()
            q.extend([c for c in current_connections if c not in result])
            if current_code not in result:
                result[current_code] = 1

        if not include_itself:
            del result[code]
        return list(result.keys())

    @staticmethod
    def _dfs_traversal(connection: FrozenDict1N, code: str, include_itself: bool) -> list[str]:
        """
        Performs a depth-first traversal of the hierarchy.

        Args:
            connection (dict[str, set[str]]): the connection dictionary representing the hierarchy.
            code (str): the starting code for the traversal.
            include_itself (bool): whether to include the starting code in the traversal.

        Returns:
            list[str]: A list of codes visited during the traversal.
        """
        result = [code] if include_itself else []

        def _traversal(_node):
            for conn in connection.get(_node) or ():
                result.append(conn)
                _traversal(conn)

        _traversal(code)

        return list(set(result))

    @staticmethod
    def _dfs_edges(connection: FrozenDict1N, code: str) -> set[tuple[str, str]]:
        """
        Returns the edges of the hierarchy obtained through a depth-first traversal.

        Args:
            connection (dict[str, set[str]]): the connection dictionary representing the hierarchy.
            code (str): the starting code for the traversal.

        Returns:
            set[tuple[str, str]]: a set of edges in the hierarchy.
        """
        result = []

        def _edges(_node):
            connections = connection.get(_node) or ()
            for conn in connections:
                result.append((_node, conn))
                _edges(conn)

        _edges(code)
        return set(result)

    def code_ancestors_bfs(self, code: str, include_itself: bool) -> list[str]:
        """
        Returns the ancestors of a code in the hierarchy using breadth-first traversal.

        Args:
            code (str): the code for which to find the ancestors.
            include_itself (bool): whether to include the code itself as its own ancestor. Defaults to True.

        Returns:
            list[str]: A list of ancestor codes.
        """
        return self._bfs_traversal(self.ch2pt, code, include_itself)

    def code_ancestors_dfs(self, code: str, include_itself: bool) -> list[str]:
        """
        Returns the ancestors of a code in the hierarchy using depth-first traversal.

        Args:
            code (str): the code for which to find the ancestors.
            include_itself (bool): whether to include the code itself as its own ancestor. Defaults to True.

        Returns:
            list[str]: a list of ancestor codes.
        """
        return self._dfs_traversal(self.ch2pt, code, include_itself)

    def code_successors_bfs(self, code: str, include_itself: bool) -> list[str]:
        """
        Returns the successors of a code in the hierarchy using breadth-first traversal.

        Args:
            code (str): the code for which to find the successors.
            include_itself (bool): whether to include the code itself as its own successor. Defaults to True.

        Returns:
            list[str]: A list of successor codes.
        """
        return self._bfs_traversal(self.pt2ch, code, include_itself)

    def code_successors_dfs(self, code: str, include_itself: bool) -> list[str]:
        """
        Returns the successors of a code in the hierarchy using depth-first traversal.

        Args:
            code (str): the code for which to find the successors.
            include_itself (bool): whether to include the code itself as its own successor. Defaults to True.

        Returns:
            list[str]: a list of successor codes.
        """
        return self._dfs_traversal(self.pt2ch, code, include_itself)

    def ancestors_edges_dfs(self, code: str) -> set[tuple[str, str]]:
        """
        Returns the edges of the hierarchy obtained through a depth-first traversal of ancestors.

        Args:
            code (str): the code for which to find the ancestor edges.

        Returns:
            set[tuple[str, str]]: a set of edges in the hierarchy.
        """
        return self._dfs_edges(self.ch2pt, code)

    def successors_edges_dfs(self, code: str) -> set[tuple[str, str]]:
        """
        Returns the edges of the hierarchy obtained through a depth-first traversal of successors.

        Args:
            code (str): the code for which to find the successor edges.

        Returns:
            set[tuple[str, str]]: a set of edges in the hierarchy.
        """
        return self._dfs_edges(self.pt2ch, code)

    def least_common_ancestor(self, codes: list[str]) -> str:
        """
        Finds the least common ancestor of a list of codes in the hierarchy.

        Args:
            codes (list[str]): the list of codes for which to find the least common ancestor.

        Returns:
            str: the least common ancestor code.

        Raises:
            RuntimeError: if a common ancestor is not found.
        """
        while len(codes) > 1:
            a, b = codes[:2]
            a_ancestors = self.code_ancestors_bfs(a, True)
            b_ancestors = self.code_ancestors_bfs(b, True)
            last_len = len(codes)
            for ancestor in a_ancestors:
                if ancestor in b_ancestors:
                    codes = [ancestor] + codes[2:]
            if len(codes) == last_len:
                raise RuntimeError("Common Ancestor not Found!")
        return codes[0]

    def search_regex(self, query: str, regex_flags: int = re.I) -> set[str]:
        """
        A regex-based search of codes by a `query` string. the search is \
            applied on the code descriptions. for example, you can use it \
            to return all codes related to cancer by setting the \
            `query = 'cancer'` and `regex_flags = re.i` \
            (for case-insensitive search). For all found codes, \
            their successor codes are also returned in the resutls.

        Args:
            query (str): The regex query string.
            regex_flags (int): The flags to use for the regex search. Defaults to re.I (case-insensitive).

        Returns:
            set[str]: A set of codes that match the regex query, including their successor codes.
        """
        codes = [
            len(re.findall(query, self.desc[c], flags=regex_flags)) > 0
            for c in tqdm_constructor(self.codes, leave=False)
        ]
        codes = [c for c, b in zip(self.codes, codes) if b]

        dag_codes = [
            len(re.findall(query, self.dag_desc[c], flags=regex_flags)) > 0
            for c in tqdm_constructor(self.dag_codes, leave=False)
        ]
        dag_codes = [c for c, b in zip(self.dag_codes, dag_codes) if b]

        all_codes = set(map(lambda c: self.code2dag[c], codes)) | set(dag_codes)

        for c in tqdm_constructor(list(all_codes), leave=False):
            all_codes.update(self.code_successors_dfs(c, include_itself=False))

        return all_codes


class Ethnicity(CodingScheme):
    pass


class UnsupportedMapping(ValueError):
    pass


class CodeMap(AbstractVxData):
    source_name: str
    target_name: str
    data: FrozenDict1N

    def __init__(self, source_name: str, target_name: str, data: FrozenDict1N):
        self.source_name = source_name
        self.target_name = target_name
        self.data = data

    def __check_init__(self):
        assert isinstance(self.data, FrozenDict1N), "Data should be a FrozenDict1N."

    def mapped_to_dag_space(self, target_scheme: CodingScheme) -> bool:
        """
        Returns True if the CodeMap is mapped to DAG space, False otherwise.

        Returns:
            bool: True if the CodeMap is mapped to DAG space, False otherwise.
        """
        assert target_scheme.name == self.target_name, "The target scheme must be the same as the target name."
        if not isinstance(target_scheme, HierarchicalScheme) or target_scheme.dag_codes is target_scheme.codes:
            return False
        map_target_codes = frozenset([v for vs in self.data.values() for v in vs])
        target_codes = set(target_scheme.codes)
        target_dag_codes = set(target_scheme.dag_codes)
        is_code_subset = map_target_codes.issubset(target_codes)
        is_dag_subset = map_target_codes.issubset(target_dag_codes)
        assert is_code_subset or is_dag_subset, (
            "The target codes are not a subset of the target codes or the DAG codes."
        )
        return is_dag_subset

    def support_ratio(self, source_scheme: CodingScheme) -> float:
        """
        Returns the ratio between the source scheme codes covered by the mapping and the total source scheme codes.

        Returns:
            float: the support ratio of the CodeMap.
        """
        assert self.source_name == source_scheme.name, "The source scheme must be the same as the source name."
        assert self.domain.issubset(source_scheme.codes)
        return len(self.domain) / len(source_scheme.codes)

    def range_ratio(self, target_scheme: CodingScheme) -> float:
        """
        Returns the ratio between the target scheme codes covered by the mapping and the total target scheme codes.

        Returns:
            float: the range ratio of the CodeMap.
        """
        assert self.target_name == target_scheme.name, "The target scheme must be the same as the target name."
        assert self.range.issubset(target_scheme.codes)
        return len(self.range) / len(target_scheme.codes)

    def log_ratios(self, source_scheme: CodingScheme, target_scheme: CodingScheme) -> float:
        """
        Returns the score of mapping between schemes as the logarithm of support ratio multiplied by range ratio.
        """
        return math.log(self.support_ratio(source_scheme)) + math.log(self.range_ratio(target_scheme))

    def __str__(self):
        """
        Returns a string representation of the CodeMap.

        Returns:
            str: the string representation of the CodeMap.
        """
        return f"{self.source_name}->{self.target_name}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.source_name}->{self.target_name}, {len(self.data)} mappings)"

    def __hash__(self):
        """
        Returns the hash value of the CodeMap.

        Returns:
            int: the hash value of the CodeMap.
        """
        return hash(str(self))

    def __bool__(self):
        """
        Returns True if the CodeMap is not empty, False otherwise.

        Returns:
            bool: true if the CodeMap is not empty, False otherwise.
        """
        return len(self) > 0

    def __len__(self):
        """
        Returns the number of supported codes in the CodeMap.

        Returns:
            int: the number of supported codes in the CodeMap.
        """
        return len(self.data)

    def target_index(self, target_scheme: CodingScheme | HierarchicalScheme) -> dict[str, int]:
        """
        Returns the target coding scheme index.

        Returns:
            dict: the target coding scheme index.
        """
        assert self.target_name == target_scheme.name, "The target scheme must be the same as the target name."
        if (
            isinstance(target_scheme, HierarchicalScheme)
            and self.mapped_to_dag_space(target_scheme)
            and self.source_name != self.target_name
            and hasattr(target_scheme, "dag_index")
        ):
            return target_scheme.dag_index
        return target_scheme.index

    def target_desc(self, target_scheme: CodingScheme | HierarchicalScheme):
        """
        Returns the target coding scheme description.

        Returns:
            dict: the target coding scheme description.
        """
        assert self.target_name == target_scheme.name, "The target scheme must be the same as the target name."
        if (
            isinstance(target_scheme, HierarchicalScheme)
            and self.mapped_to_dag_space(target_scheme)
            and self.source_name != self.target_name
            and hasattr(target_scheme, "dag_desc")
        ):
            return target_scheme.dag_desc
        return target_scheme.desc

    def __getitem__(self, item):
        """
        Returns the mapped codes for the given item.

        Args:
            item: the item to retrieve the mapped codes for.

        Returns:
            set[str]: the mapped codes for the given item.
        """
        return self.data[item]

    def __contains__(self, item):
        """
        Checks if an item is in the CodeMap.

        Args:
            item: the item to check.

        Returns:
            bool: True if the item is in the CodeMap, False otherwise.
        """
        return item in self.data

    @cached_property
    def domain(self) -> frozenset[str]:
        return frozenset(self.data.keys())

    @cached_property
    def range(self) -> frozenset[str]:
        return frozenset(v for vs in self.data.values() for v in vs)

    def map_codeset(self, codeset: Iterable[str]) -> frozenset:
        """
        Maps a codeset to the target coding scheme.

        Args:
            codeset (set[str]): the codeset to map.

        Returns:
            set[str]: The mapped codeset.
        """
        supported_set = self.domain.intersection(codeset)
        if len(supported_set) == 0 and len(cast(Sized, codeset)) > 0:
            logging.debug(f"No code in ({codeset}) maps to the target coding scheme.")
            return frozenset()
        return frozenset(t for c in supported_set for t in self[c])

    def map_dataframe(self, df: pd.DataFrame, code_column: str) -> pd.DataFrame:
        df = df.iloc[:, :]
        code2list = {k: list(v) if len(v) > 1 else next(iter(v)) for k, v in self.data.items()}
        df[code_column] = df.loc[:, code_column].map(code2list)
        invalid_rows = df[code_column].isna()
        if invalid_rows.sum() > 0:
            unique_codes = df.loc[invalid_rows, code_column].unique()
            total_unique_codes = df[code_column].unique()
            dataframe_log.info(
                (
                    f"Some codes are not mapped to the target scheme: "
                    f"Total rows removed {invalid_rows.sum()} / {len(invalid_rows)} = {invalid_rows.mean(): .3f}. "
                    f"Unique codes dropped: {len(unique_codes)} / {len(total_unique_codes)} = "
                    f"{len(unique_codes) / len(total_unique_codes): .3f}.",
                    pd.DataFrame(unique_codes, columns=pd.Series([code_column])),
                    "unique_columns_missed",
                )
            )
            df = df.loc[~df[code_column].isna(), :]
        return df.explode(code_column)

    def target_code_ancestors(self, target_scheme: HierarchicalScheme, t_code: str, include_itself=True) -> list[str]:
        assert self.target_name == target_scheme.name, "The target scheme must be the same as the target name."
        if not self.mapped_to_dag_space(target_scheme):
            t_code = target_scheme.code2dag[t_code]
        return target_scheme.code_ancestors_bfs(t_code, include_itself=include_itself)

    @classmethod
    def _init_args_from_table(
        cls,
        source_scheme: CodingScheme,
        target_scheme: CodingScheme,
        map_table: pd.DataFrame,
        c_source_code: str,
        c_target_code: str,
        *args,
        **kwargs,
    ) -> tuple[str, str, FrozenDict1N]:
        """
        # TODO: test me.
        """
        map_table = map_table.loc[:, [c_source_code, c_target_code]].astype(str)
        map_table = map_table.loc[
            map_table[c_source_code].isin(source_scheme.codes) & map_table[c_target_code].isin(target_scheme.codes), :
        ]
        mapping = map_table.groupby(c_source_code)[c_target_code].apply(set).to_dict()
        return source_scheme.name, target_scheme.name, FrozenDict1N(mapping)

    @classmethod
    def from_table(
        cls,
        source_scheme: CodingScheme,
        target_scheme: CodingScheme,
        c_source_code: str,
        c_target_code: str,
        table: pd.DataFrame,
        **kwargs,
    ):
        return cls(
            *cls._init_args_from_table(
                source_scheme=source_scheme,
                target_scheme=target_scheme,
                c_source_code=c_source_code,
                c_target_code=c_target_code,
                map_table=table,
            )
        )


class GroupingData(AbstractVxData):
    permute: tuple[int, ...]
    split: tuple[int, ...]
    size: tuple[int, ...]
    aggregation: tuple[AggregationLiteral, ...]

    def __init__(
        self,
        permute: tuple[int, ...],
        split: tuple[int, ...],
        size: tuple[int, ...],
        aggregation: tuple[AggregationLiteral, ...],
    ):
        self.permute = permute
        self.split = split
        self.size = size
        self.aggregation = aggregation

    @property
    def scheme_size(self) -> tuple[int, int]:
        return sum(self.size), len(self.size)


class ReducedCodeMapN1(CodeMap):
    set_aggregation: FrozenDict11
    reduced_groups: FrozenDict1N

    def __init__(
        self,
        source_name: str,
        target_name: str,
        data: FrozenDict1N,
        set_aggregation: FrozenDict11,
        reduced_groups: FrozenDict1N,
    ):
        super().__init__(source_name=source_name, target_name=target_name, data=data)
        self.set_aggregation = set_aggregation
        self.reduced_groups = reduced_groups

    @classmethod
    def from_data(
        cls, source_name: str, target_name: str, map_data: FrozenDict1N, set_aggregation: FrozenDict11
    ) -> Self:
        assert all(len(t) == 1 for t in map_data.values()), "A code should have one target."

        new_map: dict[str, set[str]] = defaultdict(set)
        for code, (target,) in map_data.items():
            new_map[target].add(code)

        return cls(
            source_name=source_name,
            target_name=target_name,
            data=map_data,
            set_aggregation=set_aggregation,
            reduced_groups=FrozenDict1N(new_map),
        )

    @classmethod
    def from_table(
        cls,
        source_scheme: CodingScheme,
        target_scheme: CodingScheme,
        c_source_code: str,
        c_target_code: str,
        table: pd.DataFrame,
        **kwargs,
    ) -> Self:
        c_target_agg = kwargs["c_target_agg"]
        source_name, target_name, map_data = cls._init_args_from_table(
            source_scheme=source_scheme,
            target_scheme=target_scheme,
            c_source_code=c_source_code,
            c_target_code=c_target_code,
            map_table=table,
        )
        return cls.from_data(
            source_name=source_name,
            target_name=target_name,
            map_data=map_data,
            set_aggregation=FrozenDict11(table.set_index(c_target_code)[c_target_agg].to_dict()),
        )

    def groups(self, source_index: dict[str, int]) -> tuple[tuple[str, ...], ...]:
        target_codes = tuple(sorted(self.reduced_groups.keys()))
        return tuple(tuple(sorted(self.reduced_groups[t], key=lambda c: source_index[c])) for t in target_codes)

    @staticmethod
    def _validate_aggregation(a) -> AggregationLiteral:
        if a in ("sum", "or", "w_sum"):
            return a
        else:
            raise ValueError(f"Unrecognised aggregation: {a}")

    @cached_property
    def groups_aggregation(self) -> tuple[AggregationLiteral, ...]:
        aggregation = tuple(self.set_aggregation[g] for g in sorted(self.reduced_groups.keys()))
        return tuple(self._validate_aggregation(a) for a in aggregation)

    def groups_size(self, source_index: dict[str, int]) -> tuple[int, ...]:
        return tuple(len(g) for g in self.groups(source_index))

    def groups_split(self, source_index: dict[str, int]) -> tuple[int, ...]:
        return tuple(np.cumsum(self.groups_size(source_index)).tolist())

    def groups_permute(self, source_index: dict[str, int]) -> tuple[int, ...]:
        permutes: tuple[int, ...] = sum(
            (tuple(map(source_index.__getitem__, g)) for g in self.groups(source_index)), tuple()
        )
        if len(permutes) == len(source_index):
            return permutes
        else:
            return permutes + tuple(set(source_index.values()) - set(permutes))

    def grouping_data(self, source_index: dict[str, int]) -> GroupingData:
        return GroupingData(
            permute=self.groups_permute(source_index),
            split=self.groups_split(source_index),
            size=self.groups_size(source_index),
            aggregation=self.groups_aggregation,
        )


class FilterOutcomeMapData(AbstractVxData):
    name: str
    base_name: str
    exclude_codes: tuple[str, ...]

    def __init__(self, name: str, exclude_codes: tuple[str, ...], base_name: str):
        self.name = name
        self.exclude_codes = exclude_codes
        self.base_name = base_name

    @classmethod
    def from_spec_json(cls, available_schemes: Mapping[str, CodingScheme], json_file: str) -> Self:
        conf = load_config(json_file, relative_to=resources_path("outcomes"))
        exclude_codes = []
        if "exclude_branches" in conf:
            pass
        if "select_branches" in conf:
            pass
        if "selected_codes" in conf:
            base_scheme = available_schemes[conf["code_scheme"]]
            exclude_codes.extend(c for c in base_scheme.codes if c not in conf["selected_codes"])
        if "exclude_codes" in conf:
            exclude_codes.extend(conf["exclude_codes"])

        name = conf.get("name", json_file.split(".")[0])
        return cls(name=name, base_name=conf["code_scheme"], exclude_codes=tuple(exclude_codes))

    def as_coding_scheme(self, base_scheme: CodingScheme) -> CodingScheme:
        assert base_scheme.name == self.base_name, "Base scheme mismatch."
        codes = tuple(c for c in base_scheme.codes if c not in self.exclude_codes)
        desc = FrozenDict11({c: base_scheme.desc[c] for c in codes})
        return CodingScheme(name=self.name, codes=codes, desc=desc)

    def as_code_map(
        self, outcome_scheme: CodingScheme, source_scheme: CodingScheme, base_scheme: CodingScheme, source2base: CodeMap
    ) -> CodeMap:
        assert source2base.source_name == source_scheme.name
        assert source2base.target_name == base_scheme.name == self.base_name, "Base scheme mismatch."
        # as_scheme = self.as_coding_scheme(base_scheme)
        map_data = {s_code: t_codes.intersection(outcome_scheme.codes) for s_code, t_codes in source2base.data.items()}
        return CodeMap(source_name=source_scheme.name, target_name=self.name, data=FrozenDict1N(map_data))


@dataclass
class FilterOutcomeMap:
    scheme: CodingScheme
    codemap: CodeMap

    def __init__(self, scheme: CodingScheme, codemap: CodeMap):
        self.scheme = scheme
        self.codemap = codemap

    @property
    def name(self) -> str:
        return self.scheme.name

    @property
    def index(self):
        return self.scheme.index

    @property
    def desc(self):
        return self.scheme.desc

    def __len__(self):
        return len(self.index)

    def map_codeset(self, codeset: Iterable[str]):
        return self.codemap.map_codeset(codeset)

    def __call__(self, codeset: Iterable[str]) -> CodesVector:
        codeset = self.map_codeset(codeset)
        vec = np.zeros(len(self), dtype=bool)
        for c in codeset:
            vec[self.index[c]] = True
        return CodesVector(vec, self.name)


class CodingSchemesManager(AbstractVxData):
    schemes: tuple[CodingScheme, ...]
    maps: tuple[CodeMap, ...]
    outcomes: tuple[FilterOutcomeMapData, ...]

    def __init__(
        self,
        schemes: tuple[CodingScheme, ...] = (),
        maps: tuple[CodeMap, ...] = (),
        outcomes: tuple[FilterOutcomeMapData, ...] = (),
    ):
        self.schemes = tuple(sorted(schemes, key=lambda s: s.name))
        self.maps = tuple(sorted(maps, key=lambda m: m.source_name + m.target_name))
        self.outcomes = tuple(sorted(outcomes, key=lambda o: o.name))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.schemes}, {self.maps}, {self.outcomes})"

    def __len__(self):
        return len(self.schemes) + len(self.maps) + len(self.outcomes)

    def add_scheme(self, scheme: CodingScheme) -> Self:
        assert isinstance(scheme, CodingScheme), f"{scheme} is not a CodingScheme."
        if scheme.name in self.scheme:
            logging.warning(f"Scheme {scheme.name} already exists")
            return self
        return type(self)(schemes=self.schemes + (scheme,), maps=self.maps, outcomes=self.outcomes)

    def add_map(self, map: CodeMap, overwrite: bool = False) -> Self:
        assert isinstance(map, CodeMap), f"{map} is not a CodeMap."
        if (map.source_name, map.target_name) in self.map:
            if not overwrite:
                logging.warning(
                    f"Map {map.source_name}->{map.target_name} already exists. "
                    f"If you want to replace it, use overwrite=True."
                )
                return self
            logging.info(f"Map {map.source_name}->{map.target_name} already exists and will be overwritten")

        return type(self)(schemes=self.schemes, maps=self.maps + (map,), outcomes=self.outcomes)

    def add_outcome(self, outcome: FilterOutcomeMapData) -> Self:
        assert isinstance(outcome, FilterOutcomeMapData), f"{outcome} is not an OutcomeExtractor."
        if outcome.name in self.outcome_data:
            logging.warning(f"Outcome {outcome.name} already exists")
            return self
        return type(self)(schemes=self.schemes, maps=self.maps, outcomes=self.outcomes + (outcome,))

    def supported_outcome(self, outcome_name: str, source_scheme: str) -> bool:
        assert outcome_name in self.outcome_data, (
            f"This outcome ({outcome_name}) doesn't exist. Current outcomes: {list(self.outcome_data.keys())}"
        )

        return (source_scheme, outcome_name) in self.outcome

    def supported_outcomes(self, source_scheme: str) -> tuple[str, ...]:
        return tuple(o.name for o in self.outcomes if self.supported_outcome(o.name, source_scheme))

    def union(self, other: Self) -> Self:
        updated = self
        for s in (s for s in other.schemes if s.name not in updated.scheme):
            updated = updated.add_scheme(s)
        for m in (m for m in other.maps if (m.source_name, m.target_name) not in updated.map):
            updated = updated.add_map(m)
        for o in (o for o in other.outcomes if o.name not in updated.outcome_data):
            updated = updated.add_outcome(o)
        return updated

    def __add__(self, other: Self) -> Self:
        return self.union(other)

    @cached_property
    def scheme(self) -> Mapping[str, CodingScheme]:
        return MappingProxyType({s.name: s for s in self.schemes})

    @cached_property
    def identity_maps(self) -> dict[tuple[str, str], CodeMap]:
        return {
            (s, s): CodeMap(source_name=s, target_name=s, data=FrozenDict1N({c: {c} for c in self.scheme[s].codes}))
            for s in self.scheme.keys()
        }

    @cached_property
    def chainable_maps(self) -> dict[tuple[str, str, str], CodeMap]:
        # TODO: to automatically generate chained maps, we need to define a graph algorithm traversing
        # the paths from a source node to target node.
        # Notes:
        # 1. Maybe restrict the chained maps to only two edges. Or allow a hyperparameter (max_edges).
        # 2. There will be a need to define quality metrics of chained maps, because there are potentially
        #   multiple chained maps from node i to node j.
        #   Quality metric would consider the following factors:
        #   1. Lost codes in the mapping. If we have multiple paths from node i to node j, favor the ones that
        #       loses least number of codes of node i in the mapping.
        #   2. Spread of codes in the target. If we have multiple paths from node i to node j, favor the ones
        #       that does distributes more evenly over the codes of node j. Perhaps this can be
        #       quantified with entropy. Let
        #       p_k = (number of mapped codes from node i to code k in node j) / |normalizer|,
        #       then $entropy = \sum p_k \log{p_k}$
        # How to combine these two objectives?
        raise NotImplementedError

    @cached_property
    def map(self) -> Mapping[tuple[str, str], CodeMap]:
        return MappingProxyType({(m.source_name, m.target_name): m for m in self.maps} | self.identity_maps)

    @cached_property
    def outcome_data(self) -> Mapping[str, FilterOutcomeMapData]:
        return MappingProxyType({o.name: o for o in self.outcomes})

    @cached_property
    def outcome_scheme(self) -> Mapping[str, CodingScheme]:
        return MappingProxyType({k: o.scheme for (_, k), o in self.outcome.items()})

    @cached_property
    def outcome(self) -> Mapping[tuple[str, str], FilterOutcomeMap]:
        # get all outcome mappings possible.
        # basically includes any codemap that has as a target base_scheme. for all base_schemes.
        results = {}
        for o in self.outcomes:
            t_scheme = o.base_name
            o_scheme = o.as_coding_scheme(self.scheme[t_scheme])
            feasible_maps = {k: m for k, m in self.map.items() if m.target_name == t_scheme}
            for (source_name, target_name), m in feasible_maps.items():
                o_map = o.as_code_map(o_scheme, self.scheme[source_name], self.scheme[target_name], m)
                results[(source_name, o.name)] = FilterOutcomeMap(o_scheme, o_map)
        return MappingProxyType(results)

    def make_chained_map(self, chain: tuple[str, ...]) -> CodeMap:
        """
        Registers a chained CodeMap. The source and target coding schemes are chained together if there is an
        intermediate scheme that can act as a bridge between the two. There must be registered two CodeMaps, one that
        maps between the source and intermediate coding schemes and one that maps between the intermediate and
        target coding schemes.
        Args:
            s_scheme (str): the source coding scheme.
            inter_scheme (str): the intermediate coding scheme.
            t_scheme (str): the target coding scheme.
        """
        assert len(chain) > 2
        scheme = tuple(map(lambda n: self.scheme[n], chain))
        maps = tuple(self.map[si, sj] for si, sj in zip(chain[:-1], chain[1:]))
        assert all(len(m.range.intersection(sj.codes)) > 0 for m, sj in zip(maps, scheme[1:]))
        assert all(len(m.domain.intersection(si.codes)) > 0 for m, si in zip(maps, scheme[:-1]))

        def bridge(c: str) -> frozenset[str]:
            codeset = frozenset((c,))
            for m in maps:
                codeset = m.map_codeset(codeset)
                if len(codeset) == 0:
                    return frozenset()
            return codeset

        data = {c: bridge(c) for c in scheme[0].codes}
        data = FrozenDict1N({c: target for c, target in data.items() if len(target) > 0})
        return CodeMap(source_name=chain[0], target_name=chain[-1], data=data)

    def add_chained_map(self, s_scheme: str, inter_scheme: str, t_scheme: str, overwrite: bool = False) -> Self:
        """
        Registers a chained CodeMap. The source and target coding schemes are chained together if there is an
        intermediate scheme that can act as a bridge between the two. There must be registered two CodeMaps, one that
        maps between the source and intermediate coding schemes and one that maps between the intermediate and target
        coding schemes.
        Args:
            s_scheme (str): the source coding scheme.
            inter_scheme (str): the intermediate coding scheme.
            t_scheme (str): the target coding scheme.
        """
        return self.add_map(self.make_chained_map((s_scheme, inter_scheme, t_scheme)), overwrite=overwrite)

    def scheme_supported_targets(self, scheme: CodingScheme) -> tuple[str, ...]:
        return tuple(t for s, t in self.map.keys() if s == scheme.name)

    def add_match_map(self, scheme_a: str, scheme_b: str) -> Self:
        scheme_a_ = self.scheme[scheme_a]
        scheme_b_ = self.scheme[scheme_b]
        normalize = lambda c: c.strip().lower()
        normalized_a = tuple(sorted(map(normalize, scheme_a_.codes)))
        normalized_b = tuple(sorted(map(normalize, scheme_b_.codes)))
        assert normalized_a == normalized_b, (
            f"The codes of {scheme_a_.name} and {scheme_b_.name} mismatch. "
            f"(Normalized) codes of {scheme_a_.name}: {normalized_a}. "
            f"(Normalized) codes of {scheme_b_.name}: {normalized_b}."
            f"Difference a - b: {set(normalized_a) - set(normalized_b)}. "
            f"Difference b - a: {set(normalized_b) - set(normalized_a)}."
        )
        codes_a = tuple(sorted(scheme_a_.codes, key=normalize))
        codes_b = tuple(sorted(scheme_b_.codes, key=normalize))
        m_ab = CodeMap(scheme_a_.name, scheme_b_.name, data=FrozenDict1N({a: {b} for a, b in zip(codes_a, codes_b)}))
        m_ba = CodeMap(scheme_b_.name, scheme_a_.name, data=FrozenDict1N({b: {a} for a, b in zip(codes_a, codes_b)}))
        return self.add_map(m_ab).add_map(m_ba)
