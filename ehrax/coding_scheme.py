"""Extract diagnostic/procedure information of CCS files into new
data structures to support conversion between CCS and ICD9."""

import logging
import math
import os
import re
from abc import abstractmethod, ABCMeta
from collections import defaultdict, OrderedDict
from functools import cached_property
from types import MappingProxyType
from typing import Optional, ClassVar, Iterable, Mapping, Callable, Self

import numpy as np
import pandas as pd
import tables as tbl  # type: ignore

from .base import AbstractVxData
from .freezer import FrozenDict11, FrozenDict1N, FrozenDict1NM
from .literals import AggregationLiteral, NumericalTypeHint
from .utils import load_config, tqdm_constructor, Array


def resources_dir(*subdir: str) -> str:
    return os.path.join(os.path.dirname(__file__), "resources", *subdir)


class CodesVector(AbstractVxData):
    vec: Array
    scheme: str

    def __init__(self, vec: Array, scheme: str):
        self.vec = vec
        self.scheme = scheme

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


class CodingScheme(AbstractVxData):
    name: str
    codes: tuple[str, ...]
    desc: FrozenDict11[str]
    vector_cls: ClassVar[type[CodesVector]] = CodesVector

    def __init__(self, name: str, codes: tuple[str, ...], desc: Optional[FrozenDict11[str]] = None):
        self.name = name
        self.codes = codes
        self.desc = desc or FrozenDict11({c: c for c in codes})

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, codes({len(self.codes)}), desc({len(self.desc)}))"

    def __check_init__(self):
        self._check_types()

        # Check types.
        assert isinstance(self.name, str), "Scheme name must be a string."
        assert isinstance(self.codes, tuple), "Scheme codes must be a tuple."
        assert isinstance(self.desc, FrozenDict11), "Scheme description must be a FrozenDict11."

        assert tuple(sorted(self.codes)) == self.codes, "Scheme codes must be sorted."

        # Check for uniqueness of codes.
        assert len(self.codes) == len(set(self.codes)), f"{self}: Codes should be unique."

        # Check sizes.
        assert len(self.codes) == len(self.desc), f"{self}: Codes and descriptions should have the same size."

    def _check_types(self):
        for collection in [self.codes, self.desc]:
            assert all(
                isinstance(c, str) for c in collection
            ), f"{self}: All name types should be str."

        assert all(
            isinstance(desc, str)
            for desc in self.desc.values()

        ), f"{self}: All desc types should be str."

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

    def wrap_vector(self, vec: Array) -> CodesVector:
        """
        Wrap a numpy array as a vector representation of the current scheme.
        Args:
            vec (Array): the numpy array to wrap.
        Returns:
            CodingScheme.vector_cls: a vector representation of the current scheme.
        """
        assert len(vec) == len(self), f"Vector length should be {len(self)}."
        assert vec.ndim == 1, f"Vector should be 1-dimensional."

        return CodingScheme.vector_cls(vec, self.name)

    def codeset2vec(self, codeset: set[str]) -> CodesVector:
        """
        Convert a codeset to a vector representation.
        Args:
            codeset (set[str]): the codeset to convert.
        Returns:
            CodingScheme.vector_cls: a vector representation of the current scheme.
        """
        vec = np.zeros(len(self), dtype=bool)
        try:
            for c in codeset:
                vec[self.index[c]] = True
        except KeyError as missing:
            logging.error(
                f"Code {missing} is missing." f"Accepted keys: {self.index.keys()}"
            )

        return CodesVector(vec, self.name)

    def vec2codeset(self, vec: Array) -> set[str]:
        assert vec.ndim == 1, f"Vector should be 1-dimensional."
        assert len(vec) == len(self), f"Vector length should be {len(self)}."
        return set(map(lambda idx: self.index2code[int(idx)], np.where(vec)[0]))

    def as_dataframe(self) -> pd.DataFrame:
        """
        Returns the scheme as a Pandas DataFrame.
        The DataFrame contains the following columns:
            - code: the code string
            - desc: the code description
        """

        index = list(range(len(self)))
        return pd.DataFrame(
            {
                "code": self.index2code,
                "desc": self.index2desc,
            },
            index=index,
        )

    @classmethod
    def _init_args_from_table(cls, name: str,
                              table: pd.DataFrame,
                              c_code: str, c_desc: Optional[str],
                              code_selection: Optional[pd.DataFrame], **kwargs) -> tuple[
        str, tuple[str, ...], FrozenDict11]:
        # TODO: test this method.
        # drop=False in case we use c_code as c_desc.
        df = table.astype({c_code: str}).drop_duplicates(c_code).set_index(c_code, drop=False)
        if code_selection is not None:
            df = df.loc[code_selection[c_code].drop_duplicates().astype(str).tolist()]
        return name, tuple(sorted(df.index.tolist())), FrozenDict11(df[(c_desc or c_code)].to_dict())

    @classmethod
    def from_table(cls, name: str, table: pd.DataFrame, c_code: str, c_desc: Optional[str] = None,
                   code_selection: Optional[pd.DataFrame] = None, *args, **kwargs) -> Self:
        return cls(*cls._init_args_from_table(name=name, table=table, c_code=c_code, c_desc=c_desc,
                                              code_selection=code_selection))


class NumericScheme(CodingScheme):
    """
    NumericScheme is a subclass of FlatScheme that represents a numerical coding scheme.
    Additional to `FlatScheme` attributes, it contains the following attributes to represent the coding scheme:
    - type_hint: dict mapping codes to their type hint (B: binary, N: numerical, O: ordinal, C: categorical)
    """
    type_hint: FrozenDict11[NumericalTypeHint]
    default_type_hint: NumericalTypeHint

    def __init__(self, name: str, codes: tuple[str, ...], desc: Optional[FrozenDict11[str]] = None,
                 group: Optional[FrozenDict11[str]] = None,
                 type_hint: Optional[FrozenDict11[NumericalTypeHint]] = None,
                 default_type_hint: NumericalTypeHint = 'N'):
        super().__init__(name=name, codes=codes, desc=desc)
        self.group = group or FrozenDict11({code: code for code in codes})
        self.type_hint = type_hint or FrozenDict11({code: default_type_hint for code in codes})
        self.default_type_hint = default_type_hint

    def __check_init__(self):
        assert set(self.codes) == set(self.type_hint.keys()), \
            f"The set of codes ({self.codes}) does not match the set of type hints ({self.type_hint.keys()})."
        assert set(self.type_hint.values()) <= {'B', 'N', 'O', 'C'}, \
            f"The set of type hints ({self.type_hint.values()}) contains invalid values."

    @cached_property
    def type_array(self) -> Array:
        """
        Returns the type hint of the codes in the scheme as a numpy array.
        """
        assert set(self.index[c] for c in self.codes) == set(range(len(self))), \
            f"The order of codes ({self.codes}) does not match the order of type hints ({self.type_hint.keys()})."
        return np.array([self.type_hint[code] for code in self.codes])

    @cached_property
    def index2group(self) -> dict[int, str]:
        return {i: self.group[code] for i, code in enumerate(self.codes)}

    def as_dataframe(self) -> pd.DataFrame:
        index = list(range(len(self)))
        return pd.DataFrame(
            {
                "code": self.index2code,
                "desc": self.index2desc,
                "type": self.type_array,
                "group": self.index2group
            },
            index=index,
        )


class CodingSchemeWithUOM(CodingScheme):
    uom_normalization_factor: FrozenDict1NM  # dict[code, dict[unit, conversion_factor]]
    universal_unit: FrozenDict11

    def __init__(self, name: str, codes: tuple[str, ...],
                 desc: FrozenDict11, uom_normalization_factor: FrozenDict1NM,
                 universal_unit: FrozenDict11):
        super().__init__(name=name, codes=codes, desc=desc)
        self.uom_normalization_factor = uom_normalization_factor
        self.universal_unit = universal_unit

    @classmethod
    def from_table(cls, name: str,
                   table: pd.DataFrame,
                   c_code: str,
                   c_desc: Optional[str] = None,
                   code_selection: Optional[pd.DataFrame] = None, *,
                   c_normalization_factor: str,
                   c_unit: str,
                   c_universal_unit: Optional[str] = None) -> Self:
        name, codes, desc = cls._init_args_from_table(name=name, table=table, c_code=c_code, c_desc=c_desc,
                                                      code_selection=code_selection)
        # TODO: test this method.
        df = table.astype({c_code: str, c_normalization_factor: float}).drop_duplicates(c_code).set_index(c_code)
        df = df[df.index.isin(codes)]
        assert all(c in df.columns for c in (c_unit, c_normalization_factor)), "Some columns are missing."
        if c_universal_unit is not None and c_universal_unit in df.columns:
            uom_universal = df[c_universal_unit].to_dict()
        else:
            # Choose one of the units with 1.0 as a normalization factor.
            uom_universal = df[df[c_normalization_factor] == 1.0][c_unit].to_dict()
        # Narrow down the codes to those who have at least one universal unit (the target unit to which all units are converted).
        df = df[df.index.isin(uom_universal.keys())]
        uom_data = {code: code_df.set_index(c_unit)[c_normalization_factor].to_dict() for code, code_df in
                    df.groupby(df.index)}
        return cls(name=name, codes=codes, desc=desc,
                   uom_normalization_factor=FrozenDict1NM(uom_data), universal_unit=FrozenDict11(uom_universal))


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

    def __init__(self, name: str, codes: tuple[str, ...], desc: Optional[FrozenDict11[str]] = None,
                 dag_codes: Optional[Iterable[str]] = None, dag_desc: Optional[FrozenDict11[str]] = None,
                 code2dag: Optional[FrozenDict11[str]] = None, *, ch2pt: FrozenDict1N[str]):
        super().__init__(name, codes, desc)
        self.ch2pt = ch2pt
        self.dag_codes = tuple(dag_codes or self.codes)
        self.dag_desc = dag_desc or self.desc
        self.code2dag = code2dag or FrozenDict11({c: c for c in self.codes})

    def __check_init__(self):
        # Check types
        assert isinstance(self.dag_codes, tuple), f"{self}: codes should be a list."
        assert isinstance(self.dag_desc, FrozenDict11), f"{self}: desc should be a dict."
        assert isinstance(self.code2dag, FrozenDict11), f"{self}: code2dag should be a dict."
        assert isinstance(self.ch2pt, FrozenDict1N), f"{self}: ch2pt should be a dict."
        for collection in [self.dag_codes, self.dag_desc.values(), self.dag_desc.keys(), self.code2dag.keys(),
                           self.dag_desc.values(), self.code2dag.values(), self.ch2pt.keys(),
                           set.union(*self.ch2pt.values())]:
            assert all(
                isinstance(c, str) for c in collection
            ), f"{self}: All name types should be str."

        # Check sizes
        # TODO: note in the documentation that dag2code size can be less than the dag_codes since some dag_codes are internal nodes that themselves are not are not complete clinical concepts.
        for collection in [self.dag_codes, self.dag_desc]:
            assert len(collection) == len(self.dag_codes), f"{self}: All collections should have the same size."

    def make_ancestors_tuples(self, include_itself: bool = True) -> tuple[tuple[int, ...], ...]:
        """
        Creates a list of ancestors for each code.

        Args:
            include_itself (bool): whether to include the code itself as its own ancestor. Defaults to True.

        Returns:
            Array: a boolean matrix where each element (i, j) is True if code i is an ancestor of code j, and False otherwise.
        """
        parents_indices = [[]] * len(self.dag_index)
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
    def reverse_connection(connection: Mapping[str, set[str]]) -> FrozenDict1N[str]:
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
    def _bfs_traversal(connection: FrozenDict1N, code: str, include_itself: bool) -> list[str]:
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
        result = {code} if include_itself else set()

        def _traversal(_node):
            for conn in connection.get(_node) or ():
                result.add(conn)
                _traversal(conn)

        _traversal(code)

        return list(result)

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
        result = set()

        def _edges(_node):
            connections = connection.get(_node) or ()
            for conn in connections:
                result.add((_node, conn))
                _edges(conn)

        _edges(code)
        return result

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
                raise RuntimeError('Common Ancestor not Found!')
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
        codes = [len(re.findall(query, self.desc[c], flags=regex_flags)) > 0 for c in
                 tqdm_constructor(self.codes, leave=False)]
        codes = [c for c, b in zip(self.codes, codes) if b]

        dag_codes = [len(re.findall(query, self.dag_desc[c], flags=regex_flags)) > 0 for c in
                     tqdm_constructor(self.dag_codes, leave=False)]
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
        map_target_codes = set.union(*self.data.values())
        target_codes = set(target_scheme.codes)
        target_dag_codes = set(target_scheme.dag_codes)
        is_code_subset = map_target_codes.issubset(target_codes)
        is_dag_subset = map_target_codes.issubset(target_dag_codes)
        assert is_code_subset != is_dag_subset, "The target codes are not a subset " \
                                                "of the target codes or the DAG codes."
        return is_dag_subset

    def support_ratio(self, source_scheme: CodingScheme) -> float:
        """
        Returns the ratio between the source scheme codes covered by the mapping and the total source scheme codes.

        Returns:
            float: the support ratio of the CodeMap.
        """
        assert self.source_name == source_scheme.name, "The source scheme must be the same as the source name."
        return len(set(self.data.keys()) & set(source_scheme.codes)) / len(source_scheme.codes)

    def range_ratio(self, target_scheme: CodingScheme) -> float:
        """
        Returns the ratio between the target scheme codes covered by the mapping and the total target scheme codes.

        Returns:
            float: the range ratio of the CodeMap.
        """
        assert self.target_name == target_scheme.name, "The target scheme must be the same as the target name."
        return len(set.union(*self.data.values()) & set(target_scheme.codes)) / len(target_scheme.codes)

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
        return f'{self.source_name}->{self.target_name}'

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
        if isinstance(target_scheme, HierarchicalScheme) and self.mapped_to_dag_space(
                target_scheme) and self.source_name != self.target_name and hasattr(target_scheme,
                                                                                    'dag_index'):
            return target_scheme.dag_index
        return target_scheme.index

    def target_desc(self, target_scheme: CodingScheme | HierarchicalScheme):
        """
        Returns the target coding scheme description.

        Returns:
            dict: the target coding scheme description.
        """
        assert self.target_name == target_scheme.name, "The target scheme must be the same as the target name."
        if isinstance(target_scheme, HierarchicalScheme) and self.mapped_to_dag_space(
                target_scheme) and self.source_name != self.target_name and hasattr(target_scheme,
                                                                                    'dag_desc'):
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

    def map_codeset(self, codeset: set[str]):
        """
        Maps a codeset to the target coding scheme.

        Args:
            codeset (set[str]): the codeset to map.

        Returns:
            set[str]: The mapped codeset.
        """
        return set().union(*[self[c] for c in codeset])

    def map_dataframe(self, df: pd.DataFrame, code_column: str) -> pd.DataFrame:
        df = df.iloc[:, :]
        code2list = {k: list(v) if len(v) > 1 else next(iter(v)) for k, v in self.data.items()}
        df[code_column] = df[code_column].map(code2list)

        if df[code_column].isna().sum() > 0:
            logging.warning(
                f"Some codes are not mapped to the target scheme: {df[code_column].isna().sum()} / {len(df)}")
            # TODO: Add to the report the conversion misses.
            df = df[~df[code_column].isna()]
        return df.explode(code_column)

    def target_code_ancestors(self, target_scheme: HierarchicalScheme, t_code: str, include_itself=True) -> list[str]:
        assert self.target_name == target_scheme.name, "The target scheme must be the same as the target name."
        if not self.mapped_to_dag_space(target_scheme):
            t_code = target_scheme.code2dag[t_code]
        return target_scheme.code_ancestors_bfs(t_code, include_itself=include_itself)

    @classmethod
    def _init_args_from_table(cls, source_scheme: CodingScheme, target_scheme: CodingScheme,
                              map_table: pd.DataFrame, c_source_code: str, c_target_code: str, *args, **kwargs) -> \
            tuple[str, str, FrozenDict1N]:
        """
        # TODO: test me.
        """
        map_table = map_table[[c_source_code, c_target_code]].astype(str)
        map_table = map_table[
            map_table[c_source_code].isin(source_scheme.codes) & map_table[c_target_code].isin(target_scheme.codes)]
        mapping = map_table.groupby(c_source_code)[c_target_code].apply(set).to_dict()
        return source_scheme.name, target_scheme.name, FrozenDict1N(mapping)

    @classmethod
    def from_table(cls, *args, **kwargs):
        return cls(*cls._init_args_from_table(*args, **kwargs))


class GroupingData(AbstractVxData):
    permute: tuple[int, ...]
    split: tuple[int, ...]
    size: tuple[int, ...]
    aggregation: tuple[AggregationLiteral, ...]

    def __init__(self, permute: tuple[int, ...], split: tuple[int, ...], size: tuple[int, ...],
                 aggregation: tuple[AggregationLiteral, ...]):
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

    def __init__(self, source_name: str, target_name: str, data: FrozenDict1N,
                 set_aggregation: FrozenDict11, reduced_groups: FrozenDict1N):
        super().__init__(source_name=source_name, target_name=target_name, data=data)
        self.set_aggregation = set_aggregation
        self.reduced_groups = reduced_groups

    @classmethod
    def from_data(cls, source_name: str, target_name: str, map_data: FrozenDict1N,
                  set_aggregation: FrozenDict11) -> Self:
        assert all(len(t) == 1 for t in map_data.values()), "A code should have one target."

        new_map: dict[str, set[str]] = defaultdict(set)
        for code, (target,) in map_data.items():
            new_map[target].add(code)

        return cls(source_name=source_name, target_name=target_name, data=map_data,
                   set_aggregation=set_aggregation,
                   reduced_groups=FrozenDict1N(new_map))

    @classmethod
    def from_table(cls,
                   source_scheme: CodingScheme,
                   target_scheme: CodingScheme,
                   c_source_code: str,
                   c_target_code: str,
                   c_target_agg: str,
                   table: pd.DataFrame) -> Self:
        source_name, target_name, map_data = cls._init_args_from_table(source_scheme=source_scheme,
                                                                       target_scheme=target_scheme,
                                                                       c_source_code=c_source_code,
                                                                       c_target_code=c_target_code,
                                                                       map_table=table)
        return cls.from_data(source_name=source_name,
                             target_name=target_name,
                             map_data=map_data,
                             set_aggregation=FrozenDict11(table.set_index(c_target_code)[c_target_agg].to_dict()))

    def groups(self, source_index: dict[str, int]) -> tuple[tuple[str, ...], ...]:
        target_codes = tuple(sorted(self.reduced_groups.keys()))
        return tuple(tuple(sorted(self.reduced_groups[t], key=lambda c: source_index[c])) for t in target_codes)

    @staticmethod
    def _validate_aggregation(a) -> AggregationLiteral:
        if a in ('sum', 'or', 'w_sum'):
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
        permutes: tuple[int, ...] = sum((tuple(map(source_index.__getitem__, g)) for g in self.groups(source_index)),
                                        tuple())
        if len(permutes) == len(source_index):
            return permutes
        else:
            return permutes + tuple(set(source_index.values()) - set(permutes))

    def grouping_data(self, source_index: dict[str, int]) -> GroupingData:
        return GroupingData(permute=self.groups_permute(source_index),
                            split=self.groups_split(source_index),
                            size=self.groups_size(source_index),
                            aggregation=self.groups_aggregation)


class OutcomeExtractor(AbstractVxData, metaclass=ABCMeta):
    name: str
    base_name: str

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @abstractmethod
    def codes(self, base_scheme: CodingScheme) -> tuple[str, ...]:
        ...

    def desc(self, base_scheme: CodingScheme) -> FrozenDict11:
        return FrozenDict11({c: base_scheme.desc[c] for c in self.codes(base_scheme)})

    def index(self, base_scheme: CodingScheme) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.codes(base_scheme))}

    def outcome_dim(self, base_scheme: CodingScheme) -> int:
        """
        Gets the dimension of the outcome.

        Returns:
            int: the dimension of the outcome.

        """

        return len(self.index(base_scheme))

    def codeset2vec_extractor(self, base_scheme: CodingScheme, codemap: Optional[CodeMap]) -> Callable[
        [set[str]], CodesVector]:
        assert codemap is None or codemap.target_name == base_scheme.name, "Code map mismatch."
        assert base_scheme.name == self.base_name, "Base scheme name mismatch."

        codes = set(self.codes(base_scheme))
        index = self.index(base_scheme)

        def _apply(codeset: set[str]):
            codeset = codemap.map_codeset(codeset) & codes
            vec = np.zeros(len(codes), dtype=bool)
            for c in codeset:
                vec[index[c]] = True
            return CodesVector(vec, self.name)

        def _apply_without_conversion(codeset: set[str]):
            vec = np.zeros(len(codes), dtype=bool)
            for c in codeset & codes:
                vec[index[c]] = True
            return CodesVector(vec, self.name)

        return _apply if codemap is not None else _apply_without_conversion


class ExcludingOutcomeExtractor(OutcomeExtractor):
    name: str
    base_name: str
    exclude_codes: tuple[str, ...]

    def __init__(self, name: str, exclude_codes: tuple[str, ...], base_name: str):
        self.name = name
        self.exclude_codes = exclude_codes
        self.base_name = base_name

    def codes(self, base_scheme: CodingScheme) -> tuple[str, ...]:
        assert base_scheme.name == self.base_name, "Base scheme mismatch."
        return tuple(c for c in base_scheme.codes if c not in self.exclude_codes)

    @classmethod
    def from_spec_json(cls, available_schemes: dict[str, CodingScheme], json_file: str) -> Self:
        conf = load_config(json_file, relative_to=resources_dir('outcome_filters'))
        exclude_codes = []
        if 'exclude_branches' in conf:
            # TODO
            pass
        if 'select_branches' in conf:
            # TODO
            pass
        if 'selected_codes' in conf:
            base_scheme = available_schemes[conf['code_scheme']]
            exclude_codes.extend(c for c in base_scheme.codes if c not in conf['selected_codes'])
        if 'exclude_codes' in conf:
            exclude_codes.extend(conf['exclude_codes'])

        name = conf.get('name', json_file.split('.')[0])

        return cls(name=name, base_name=conf['code_scheme'], exclude_codes=tuple(exclude_codes))


class CodingSchemesManager(AbstractVxData):
    schemes: tuple[CodingScheme, ...]
    maps: tuple[CodeMap, ...]
    outcomes: tuple[OutcomeExtractor, ...]

    def __init__(self, schemes: tuple[CodingScheme, ...] = (), maps: tuple[CodeMap, ...] = (),
                 outcomes: tuple[OutcomeExtractor, ...] = ()):
        self.schemes = schemes
        self.maps = maps
        self.outcomes = outcomes

    def __repr__(self):
        return f"{self.__class__.__name__}({self.schemes}, {self.maps}, {self.outcomes})"

    def __len__(self):
        return len(self.schemes) + len(self.maps) + len(self.outcomes)

    def add_scheme(self, scheme: CodingScheme) -> Self:
        assert isinstance(scheme, CodingScheme), f"{scheme} is not a CodingScheme."
        if scheme.name in self.scheme:
            logging.warning(f'Scheme {scheme.name} already exists')
            return self
        return type(self)(schemes=self.schemes + (scheme,), maps=self.maps, outcomes=self.outcomes)

    def add_map(self, map: CodeMap) -> Self:
        assert isinstance(map, CodeMap), f"{map} is not a CodeMap."
        if (map.source_name, map.target_name) in self.map:
            logging.warning(f'Map {map.source_name}->{map.target_name} already exists')
            return self
        return type(self)(schemes=self.schemes, maps=self.maps + (map,), outcomes=self.outcomes)

    def add_outcome(self, outcome: OutcomeExtractor) -> Self:
        assert isinstance(outcome, OutcomeExtractor), f"{outcome} is not an OutcomeExtractor."
        if outcome.name in self.outcome:
            logging.warning(f'Outcome {outcome.name} already exists')
            return self
        return type(self)(schemes=self.schemes, maps=self.maps, outcomes=self.outcomes + (outcome,))

    def supported_outcome(self, outcome_name: str, supporting_scheme: str) -> bool:
        return outcome_name in self.outcome and (supporting_scheme, self.outcome[outcome_name].base_name) in self.map

    def supported_outcomes(self, supporting_scheme: str) -> tuple[str, ...]:
        return tuple(o for o in self.outcome if self.supported_outcome(o, supporting_scheme))

    def union(self, other: Self) -> Self:
        updated = self
        for s in (s for s in other.schemes if s.name not in updated.scheme):
            updated = updated.add_scheme(s)
        for m in (m for m in other.maps if (m.source_name, m.target_name) not in updated.map):
            updated = updated.add_map(m)
        for o in (o for o in other.outcomes if o.name not in updated.outcome):
            updated = updated.add_outcome(o)
        for u in (u for u in other.uom_normalizers if u.name not in updated.uom_normalizers):
            updated = updated.add_uom_normalizer(u)
        return updated

    def __add__(self, other: Self) -> Self:
        return self.union(other)

    @cached_property
    def scheme(self) -> Mapping[str, CodingScheme]:
        return MappingProxyType({s.name: s for s in self.schemes})

    @cached_property
    def identity_maps(self) -> dict[tuple[str, str], CodeMap]:
        return {(s, s): CodeMap(source_name=s, target_name=s,
                                data=FrozenDict1N({c: {c} for c in self.scheme[s].codes})) for s in
                self.scheme.keys()}

    @cached_property
    def chainable_maps(self) -> dict[tuple[str, str, str], CodeMap]:
        raise NotImplementedError

    @cached_property
    def map(self) -> Mapping[tuple[str, str], CodeMap]:
        return MappingProxyType({(m.source_name, m.target_name): m for m in self.maps} | self.identity_maps)

    @cached_property
    def outcome(self) -> Mapping[str, OutcomeExtractor]:
        return MappingProxyType({o.name: o for o in self.outcomes})

    def register_chained_map(self, s_scheme: str, inter_scheme: str, t_scheme: str) -> Self:
        """
        Registers a chained CodeMap. The source and target coding schemes are chained together if there is an intermediate scheme that can act as a bridge between the two.
        There must be registered two CodeMaps, one that maps between the source and intermediate coding schemes and one that maps between the intermediate and target coding schemes.
        Args:
            s_scheme (str): the source coding scheme.
            inter_scheme (str): the intermediate coding scheme.
            t_scheme (str): the target coding scheme.
        """
        assert len({s_scheme, inter_scheme, t_scheme}) == 3, "The schemes should be different."

        s_scheme_object = self.scheme[s_scheme]
        i_scheme_object = self.scheme[inter_scheme]
        t_scheme_object = self.scheme[t_scheme]
        map1 = self.map[(s_scheme, inter_scheme)]
        map2 = self.map[(inter_scheme, t_scheme)]
        assert not map1.mapped_to_dag_space(i_scheme_object)
        assert not map2.mapped_to_dag_space(t_scheme_object)

        bridge = lambda x: set.union(*[map2[c] for c in map1[x]])

        # Supported codes in the new map are the intersection of the source codes and the source codes of the first map
        new_source_codes = set(s_scheme_object.codes) & set(map1.data.keys())
        data = FrozenDict1N({c: bridge(c) for c in new_source_codes})
        return self.add_map(CodeMap(source_name=s_scheme, target_name=t_scheme, data=data))

    def scheme_supported_targets(self, scheme) -> tuple[str, ...]:
        return tuple(t for s, t in self.map.keys() if s == scheme.name)
