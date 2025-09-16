
import dataclasses
import enum
import json
import logging
import re
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sized
from pathlib import Path
from types import MappingProxyType, NoneType
from typing import Any, cast, Self, TYPE_CHECKING, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pandas as pd
import tables as tb

from ._literals import CompressionLibLiteral
from .utils import (
    ArrayTypes,
    equal_arrays,
    load_config,
    NumpyEncoder,
    path_from_getter,
    path_from_jax_keypath,
    write_config,
)


_factory_registry: dict[str, type[eqx.Module]] = {}


class _ModuleMeta(type(eqx.Module)):
    """Meta-class that auto-registers subclasses for factory-based loading.

    Each subclass gains ``__class_key__`` and ``__get_factory__`` for
    round-trippable, type-aware deserialization.
    """
    # This method is called whenever you definite a module: `class Foo(eqx.Module): ...`
    def __new__(
        mcs,
        name,
        bases,
        dict_,
        /,
        strict: bool = False,
        **kwargs,
    ):
        cls = super().__new__(mcs, name, bases, dict_, strict=strict, **kwargs)

        # We need to collect all constructors of the subclasses in one dictionary
        # to use them in deserialization of objects that, themselves, composed of
        # other native types or objects that has the same MetaClass.

        def __class_key__(cls):
            return f"{cls.__module__}.{cls.__qualname__}"

        def __get_factory__(type_str: str):
            try:
                return _factory_registry[type_str]
            except KeyError:
                raise KeyError(
                    f"{type_str} is not a registered type. "
                    f"You may need to import the module that was used during the construction of "
                    f"{type_str} object(s). Registered types ({len(_factory_registry)}): "
                    f"...{list(_factory_registry.keys())[-10:]}. "
                )

        cls.__class_key__ = classmethod(__class_key__)
        cls.__get_factory__ = __get_factory__
        _factory_registry[cls.__class_key__()] = cls  # type: ignore

        return cls


if TYPE_CHECKING:

    class AbstractModule(eqx.Module, metaclass=_ModuleMeta):
        @classmethod
        def __class_key__(cls) -> str:
            return "I am just a placeholder for type checking purposes.."

        @classmethod
        def __get_factory__(cls, type_str: str) -> type[Self]:
            return cls
else:

    class AbstractModule(eqx.Module, metaclass=_ModuleMeta):
        """Base class for all registered modules.

        Subclasses are automatically discoverable during deserialization
        through the registry maintained by ``_ModuleMeta``.
        """


class AbstractHDFSerializable(AbstractModule):
    """Interface for objects that can be serialized into an HDF5 group."""
    @abstractmethod
    def to_hdf_group(self, group: tb.Group) -> None:
        """Write this object into the given HDF5 group."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_hdf_group(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...], levels: int | None) -> Self:
        """Create an instance from an HDF5 group.

        The ``defer`` argument specifies relative paths (as tuples of lambda getter functions) that should
        not be materialized immediately and may be represented by virtual nodes.
        """
        raise NotImplementedError

    @abstractmethod
    def equals(self, other: Self) -> bool:
        """Return True if both objects are equal under the class-specific rules."""
        raise NotImplementedError


class HDFVirtualNode(AbstractHDFSerializable):
    """Placeholder for a value that has not yet been read from HDF5.

    Instances point to the on-disk location and type of a child attribute but
    do not hold the actual data until fetched using ``fetch_at`` or
    ``fetch_all``. Any attempt to access undefined attributes raises with a
    guidance message to fetch first.
    """

    filename: str
    parent_path: str
    type_enum: str
    key: str

    def __init__(self, filename: str, parent_path: str, type_enum: str, key: str = "") -> None:
        self.filename = filename
        self.parent_path = parent_path
        self.type_enum = type_enum
        self.key = key

    def __check_init__(self):
        """Validate types of dataclass fields based on annotations."""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            assert isinstance(value, cast(type, field.type))

    @property
    def _v_parent_path_seq(self) -> list[str]:  # to a series of directories with the root directory represented by ''
        """Return the split HDF5 path of the parent group as a list of node names."""
        if len(self.parent_path) == 0:
            return []
        if len(self.parent_path) == 1:
            return [""]
        return self.parent_path.split("/")

    def __getattribute__(self, attr: str) -> NoneType:
        """Disallow attribute access on virtual nodes until fetched."""
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            raise AttributeError(
                "You are trying to access an attribute in a lazy-loaded node. Please call `fetch_at(..,..)` or "
                "`fetch_all()` on any of the node ancestors first."
            )

    def to_hdf_group(self, group: tb.Group) -> None:
        """Virtual nodes cannot be serialized; instruct user to fetch first."""
        raise ValueError(
            "You are trying to serialize an unfetched node in a PyTree/AbstractHDFSerializable. "
            "Please call `fetch_at(..,..)` or "
            "`fetch_all()` on any of the node ancestors first."
        )

    @classmethod
    def from_hdf_group(
        cls, group: tb.Group, defer: tuple[tuple[str, ...], ...] = (), levels: int | None = None
    ) -> Self:
        """Virtual nodes cannot be directly deserialized."""
        raise ValueError("You are trying to deserialize a VirtualNode.")

    def equals(self, other: AbstractHDFSerializable) -> bool:
        """Virtual nodes cannot participate in equality until fetched."""
        raise ValueError(
            "You are trying to test equality with a virtual unfetched node. Please call `fetch_at(..,..)` or "
            "`fetch_all()` on any of the node ancestors first."
        )


class AbstractConfig(AbstractHDFSerializable):
    """Config objects with JSON-friendly dict views and HDF5 support."""
    @classmethod
    def _map_hierarchical_config(
        cls, unit_config_map: Callable[[Self], dict[str, Any]], x: Any, levels: int | None = None
    ) -> Any:
        if levels is not None and levels <= 0:
            return x
        next_level = levels if levels is None else levels - 1
        if isinstance(x, cls):
            x = unit_config_map(x)
        if isinstance(x, dict):
            return {k: cls._map_hierarchical_config(unit_config_map, v, next_level) for k, v in x.items()}
        elif isinstance(x, list):
            return [cls._map_hierarchical_config(unit_config_map, v, next_level) for v in x]
        elif isinstance(x, tuple):
            return tuple(cls._map_hierarchical_config(unit_config_map, v, next_level) for v in x)
        else:
            return x

    @classmethod
    def _as_normal_dict(cls, x: Self) -> dict[str, Any]:
        return {field.name: getattr(x, field.name) for field in dataclasses.fields(x) if not field.name.startswith("_")}

    @staticmethod
    def _as_typed_dict(x: "AbstractConfig") -> dict[str, Any]:
        return AbstractConfig._as_normal_dict(x) | {"_type": x.__class_key__()}

    @staticmethod
    def _is_typed_dict(x) -> bool:
        return isinstance(x, dict) and "_type" in x

    @staticmethod
    def _map_config_to_dict(unit_map: Callable[[Any], dict[str, Any]], x, levels: int | None = None) -> dict[str, Any]:
        dictionary = AbstractConfig._map_hierarchical_config(unit_map, x, levels)
        if levels is None:
            return json.loads(json.dumps(dictionary, cls=NumpyEncoder))
        return dictionary

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of fields without type information."""
        return AbstractConfig._map_config_to_dict(AbstractConfig._as_normal_dict, self)

    def as_one_level_dict(self) -> dict[str, Any]:
        """Return a shallow one-level dict view of this config."""
        return AbstractConfig._map_config_to_dict(AbstractConfig._as_normal_dict, self, levels=1)

    def to_dict(self) -> dict[str, Any]:
        """Return a dict that includes ``_type`` for round-trip deserialization."""
        return AbstractConfig._map_config_to_dict(AbstractConfig._as_typed_dict, self)

    def equals(self, other: AbstractHDFSerializable) -> bool:
        assert isinstance(other, AbstractConfig), "Can only compare equality with another AbstractConfig."
        return self.to_dict() == other.to_dict()

    def to_hdf_group(self, group: tb.Group) -> None:
        """Store the typed JSON representation as a raw byte array in HDF5."""
        data = json.dumps(self.to_dict(), cls=NumpyEncoder).encode("utf-8")
        group._v_file.create_array(group, "data", obj=data)

    @classmethod
    def from_hdf_group(
        cls, group: tb.Group, defer: tuple[tuple[str, ...], ...] = (), levels: int | None = None
    ) -> Self:
        assert len(defer) == 0, "Unexpected."
        return cls.from_dict(json.loads(group["data"].read().decode("utf-8")))  # pyright: ignore[reportAttributeAccessIssue] #

    def log_json(self, path: str | Path, key: str):
        """Append this config under ``key`` to a JSON file next to the HDF5 file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        json_path = path.with_suffix(".json")  # config goes here.
        config = load_config(str(json_path)) if json_path.exists() else {}
        config[key] = self.as_dict()  # logging does not aim to produce deserializable config json.
        write_config(config, str(json_path))

    def update(self, other: Self | dict[str, Any]) -> Self:
        """Return a new config by overlaying ``other`` on top of this one.

        !!! Example
        TODO: add example.
        
        """
        if isinstance(other, AbstractConfig):
            other = other.to_dict()
            other["_type"] = self.__class_key__()
        return self.from_dict(self.to_dict() | other)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        """Construct a config from a nested dict containing ``_type`` markers."""
        def _map_dict_to_config(x):
            if cls._is_typed_dict(x):
                config_class = cls.__get_factory__(x.pop("_type"))
                config_kwargs = {k: _map_dict_to_config(v) for k, v in x.items()}
                return config_class(**config_kwargs)
            elif isinstance(x, dict):
                return {k: _map_dict_to_config(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [_map_dict_to_config(v) for v in x]
            elif isinstance(x, tuple):
                return tuple(_map_dict_to_config(v) for v in x)
            else:
                return x

        return _map_dict_to_config(config)  # pyright: ignore[reportReturnType]

    def path_update(self, path: str, value: Any) -> Self:
        """Create a new config with the attribute at ``path`` replaced by ``value``.

        ``path`` is a dot-separated attribute path; ``value=None`` clears the value.

        !!! Example
        TODO: add example.
        """
        nesting = path.split(".")

        def _get(x):
            for n in nesting:
                x = getattr(x, n)
            return x

        if value is not None:
            return eqx.tree_at(_get, self, type(_get(self))(value))  # pyright: ignore[reportCallIssue]
        return eqx.tree_at(_get, self, None)


class AbstractWithPandasEquivalent(AbstractHDFSerializable):
    """Mixin for objects that can be converted to/from pandas structures."""
    @staticmethod
    def empty_pandas_meta(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Return a one-row metadata table describing an empty DataFrame/Series."""
        meta = {"index_dtype": str(df.index.dtype), "index_name": str(df.index.name)}
        if isinstance(df, pd.Series):
            meta = meta | {"dtype": str(df.dtype), "name": df.name}
        else:
            meta = meta | {f"column_{i}": col for i, col in enumerate(df.columns)}
            meta = meta | {f"column_dtype_{df.columns[i]}": str(dtype) for i, dtype in enumerate(df.dtypes)}
        return pd.DataFrame(meta, index=pd.Index([0]))

    @staticmethod
    def empty_pandas_from_metadata(meta_table: pd.DataFrame) -> pd.DataFrame | pd.Series:
        """Reconstruct an empty DataFrame/Series, preserving all types, columns names, and index name from a metadata table."""
        meta = meta_table.iloc[0].to_dict()
        index_name = meta.pop("index_name")
        index = pd.Index([], dtype=meta.pop("index_dtype"), name=None if index_name == "None" else index_name)
        if "dtype" in meta:
            dtype = meta.pop("dtype")
            name = meta.pop("name")
            name = name if name == name else None  # when None serialized it ended up as a nan
            return pd.Series(dtype=dtype, name=name, index=index)

        cols = []
        column_types = {}
        while len(meta) > 0:
            k, v = meta.popitem()
            if k.startswith("column_dtype_"):
                column_types[k.split("column_dtype_")[1]] = v
            elif k.startswith("column_"):
                order = int(k.split("column_")[1])
                cols.append((order, v))
        cols = pd.Series([col for _, col in sorted(cols, key=lambda x: x[0])])
        return pd.DataFrame(columns=cols, index=index).astype(column_types)

    @classmethod
    def serialize_pandas(cls, data: pd.DataFrame | pd.Series, group: tb.Group):
        """Persist pandas data to HDF5, preserving empty-shape metadata when needed."""
        # Empty dataframes/series are not saved in hdf file, https://github.com/pandas-dev/pandas/issues/13016,
        # https://github.com/PyTables/PyTables/issues/592
        # We still need to preserve the column names, dtypes, index name, index dtype, in a metadata object, to
        # fulfill perfect serialisation/deserialisation and have stricter unit-testing.
        hdf = group._v_file
        if data.empty:
            key = hdf.create_group(group, "empty_metadata")._v_pathname
            cls.empty_pandas_meta(data).to_hdf(hdf.filename, key=key, format="table")
        else:
            data.to_hdf(hdf.filename, key=group._v_pathname, format="table")

    @classmethod
    def deserialize_pandas(cls, store: tb.Group) -> pd.DataFrame | pd.Series:
        """Load pandas data from HDF5, handling the empty-data case via metadata."""
        hdf = store._v_file
        if "empty_metadata" in store:
            meta_table = pd.read_hdf(hdf.filename, key=store.empty_metadata._v_pathname)
            assert isinstance(meta_table, pd.DataFrame)
            return cls.empty_pandas_from_metadata(meta_table)
        else:
            table_or_series = pd.read_hdf(hdf.filename, key=store._v_pathname)
            assert isinstance(table_or_series, (pd.DataFrame, pd.Series))
            return table_or_series

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame | pd.Series:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pandas(cls, pandas: pd.DataFrame | pd.Series) -> Self:
        raise NotImplementedError

    def to_hdf_group(self, group: tb.Group) -> None:
        """Store the pandas-equivalent representation under a ``data`` subgroup."""
        group._v_attrs.classname = self.__class_key__()
        hdf_file = group._v_file
        self.serialize_pandas(self.to_pandas(), hdf_file.create_group(group, "data"))

    @classmethod
    def from_hdf_group(
        cls, group: tb.Group, defer: tuple[tuple[str, ...], ...] = (), levels: int | None = None
    ) -> Self:
        assert len(defer) == 0, "Unexpected."
        classname = group._v_attrs.classname.item()
        hdf_file = group._v_file
        return cls.__get_factory__(classname).from_pandas(cls.deserialize_pandas(hdf_file.get_node(group, "data")))

    def equals(self, other: AbstractHDFSerializable) -> bool:
        """Equality based on pandas ``.equals`` for the underlying data."""
        assert isinstance(other, AbstractWithPandasEquivalent), "Can only compare with a compatible pandas-equivalent."
        return type(self) is type(other) and self.to_pandas().equals(other.to_pandas())


class AbstractWithDataframeEquivalent(AbstractWithPandasEquivalent):
    """Mixin specializing the pandas equivalent to a DataFrame."""
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> Self:
        raise NotImplementedError

    def to_pandas(self) -> pd.DataFrame:
        return self.to_dataframe()

    @classmethod
    def from_pandas(cls, pandas: pd.DataFrame | pd.Series) -> Self:
        assert isinstance(pandas, pd.DataFrame), "Expected a pandas DataFrame."
        return cls.from_dataframe(pandas)


class AbstractWithSeriesEquivalent(AbstractWithPandasEquivalent):
    """Mixin specializing the pandas equivalent to a Series."""
    @abstractmethod
    def to_series(self) -> pd.Series:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_series(cls, dataframe: pd.Series) -> Self:
        raise NotImplementedError

    def to_pandas(self) -> pd.Series:
        return self.to_series()

    @classmethod
    def from_pandas(cls, pandas: pd.DataFrame | pd.Series) -> Self:
        assert isinstance(pandas, pd.Series), "Expected a pandas Series."
        return cls.from_series(pandas)


class SERIALIZABLE_FIELD(enum.Enum):
    """Enumeration of supported field categories for HDF5 serialization."""
    none = (type(None),)
    numpy_array = tuple(ArrayTypes)
    pandas_dataframe = (pd.DataFrame,)
    pandas_series = (pd.Series,)
    pandas_equivalent = (AbstractWithPandasEquivalent,)
    hdf_serializable = (AbstractHDFSerializable,)
    config = (AbstractConfig,)
    string = (str,)
    integer = (int,)
    float = (float,)
    boolean = (bool,)
    timestamp = (pd.Timestamp,)
    list = (list,)
    tuple = (tuple,)
    dict = (dict,)
    set = (set,)
    frozenset = (frozenset,)
    mapping_proxy = (MappingProxyType,)


COMPARISON_PRIORITY = MappingProxyType(
    {e.name: 100 for e in SERIALIZABLE_FIELD}
    | {  # default: 100
        e.name: 0
        for e in (SERIALIZABLE_FIELD.none,)  # begin with None
    }
    | {
        e.name: 1
        for e in (
            SERIALIZABLE_FIELD.boolean,
            SERIALIZABLE_FIELD.integer,
            SERIALIZABLE_FIELD.float,
            SERIALIZABLE_FIELD.timestamp,
        )  # then scalers
    }
    | {
        e.name: 2
        for e in (SERIALIZABLE_FIELD.string,)  # then variable-length strings
    }
    | {
        e.name: 3
        for e in (SERIALIZABLE_FIELD.config,)  # then config
    }
    | {
        e.name: 4
        for e in (  # then intensive data.
            SERIALIZABLE_FIELD.numpy_array,
            SERIALIZABLE_FIELD.pandas_dataframe,
            SERIALIZABLE_FIELD.pandas_series,
            SERIALIZABLE_FIELD.pandas_equivalent,
        )
    }
)  # collections, mappings, and HDFSerializable subclasses are left because they can be nested with
# other HDFSerializables
assert all(isinstance(e.value, tuple) and all(isinstance(t, type) for t in e.value) for e in SERIALIZABLE_FIELD), (
    "Expected a tuple of types."
)

_TYPE_ENUM_DICT: MappingProxyType[type, str] = MappingProxyType(
    {t: e.name for e in SERIALIZABLE_FIELD for t in e.value}
)
SERIALIZABLE_FIELD_TYPES = tuple(_TYPE_ENUM_DICT.keys())
SERIALIZABLE_FLAT_COLLECTION = (
    SERIALIZABLE_FIELD.set,
    SERIALIZABLE_FIELD.list,
    SERIALIZABLE_FIELD.tuple,
    SERIALIZABLE_FIELD.frozenset,
)
SERIALIZABLE_FLAT_DICT = (SERIALIZABLE_FIELD.dict, SERIALIZABLE_FIELD.mapping_proxy)
SERIALIZABLE_FLAT_DICT_KEY = (SERIALIZABLE_FIELD.integer, SERIALIZABLE_FIELD.string)

SERIALIZABLE_FLAT_COLLECTION_TYPES: tuple[type, ...] = sum((e.value for e in SERIALIZABLE_FLAT_COLLECTION), ())
SERIALIZABLE_FLAT_DICT_TYPES: tuple[type, ...] = sum((e.value for e in SERIALIZABLE_FLAT_DICT), ())
SERIALIZABLE_FLAT_DICT_KEY_TYPES: tuple[type, ...] = sum((e.value for e in SERIALIZABLE_FLAT_DICT_KEY), ())

# Serialised elements within homogeneous- and flat-list, -tuple, -dict.
SERIALIZABLE_ELEMENT = (
    SERIALIZABLE_FIELD.numpy_array,
    SERIALIZABLE_FIELD.pandas_dataframe,
    SERIALIZABLE_FIELD.pandas_series,
    SERIALIZABLE_FIELD.pandas_equivalent,
    SERIALIZABLE_FIELD.hdf_serializable,
    SERIALIZABLE_FIELD.config,
    SERIALIZABLE_FIELD.string,
    SERIALIZABLE_FIELD.integer,
    SERIALIZABLE_FIELD.float,
    SERIALIZABLE_FIELD.boolean,
    SERIALIZABLE_FIELD.timestamp,
)
SERIALIZABLE_ELEMENT_TYPES = sum((e.value for e in SERIALIZABLE_ELEMENT), ())

# These element types contained within a homogeneous container, can be converted to a pd.Series first.
SERIES_GROUPED_ELEMENT = (
    SERIALIZABLE_FIELD.float,
    SERIALIZABLE_FIELD.integer,
    SERIALIZABLE_FIELD.boolean,
    SERIALIZABLE_FIELD.string,
    SERIALIZABLE_FIELD.none,
)
SERIES_GROUPED_ELEMENT_TYPES = sum((e.value for e in SERIES_GROUPED_ELEMENT), ())


class _MyCustomList_set(list):
    # a custom list used to deal with replacing sets with a list, since sets are not considered a pytree
    # like lists/dicts/tuples, so we replace all sets with this type of list, so we can reverse the operation
    # (i.e. map back to a set) by identifying this type in the pytree. See `fetch_all` for more details.
    pass


class _MyCustomList_frozenset(list):
    # same but for frozenset.
    pass


jtu.register_pytree_node(_MyCustomList_set, lambda x: (list(x), None), lambda _, x: _MyCustomList_set(x))
jtu.register_pytree_node(_MyCustomList_frozenset, lambda x: (list(x), None), lambda _, x: _MyCustomList_frozenset(x))

MAX_SEGMENT_SIZE = 500


class AbstractVxData(AbstractHDFSerializable):
    """Base class providing field-wise HDF5 serialization and lazy-loading.

    Subclasses must be constructible from their dataclass-like fields so that
    deserialization can materialize instances via ``__init__(**fields)``.
    """

    def __check_init__(self):
        for f in (k for k in self.fields):
            obj = getattr(self, f)
            if obj is None:
                continue
            try:
                _ = self.object_type_enum_name(obj)
            except (KeyError, StopIteration):
                assert False, (
                    f"Unsupported field type {type(obj)} for attribute {f}. It must be a type in "
                    f"{tuple(t.__qualname__ for t in SERIALIZABLE_FIELD_TYPES)} or a subclass of "
                    f"AbstractHDFSerializable."
                )

    @property
    def fields(self) -> tuple[str, ...]:
        """Return the names of the class fields."""
        return tuple(k.name for k in dataclasses.fields(self))

    @property
    def comparison_prioritized_fields(self) -> tuple[str, ...]:
        """Return field names ordered by comparison priority for equality checks."""
        return tuple(
            sorted(self.fields, key=lambda f: COMPARISON_PRIORITY[self.object_type_enum_name(getattr(self, f))])
        )

    @classmethod
    def object_type_enum_name(cls, obj: Any) -> str:
        """Map a value to its ``SERIALIZABLE_FIELD`` enum name."""
        if type(obj) in _TYPE_ENUM_DICT:
            return _TYPE_ENUM_DICT[type(obj)]
        if type(obj) in (_MyCustomList_set, _MyCustomList_frozenset):
            return _TYPE_ENUM_DICT[type(list())]
        elif isinstance(obj, SERIALIZABLE_FIELD.config.value):
            return SERIALIZABLE_FIELD.config.name
        elif isinstance(obj, SERIALIZABLE_FIELD.pandas_equivalent.value):
            return SERIALIZABLE_FIELD.pandas_equivalent.name
        ## Potentially a subclass of AbstractHDFSerializable
        elif isinstance(obj, SERIALIZABLE_FIELD.hdf_serializable.value):
            return SERIALIZABLE_FIELD.hdf_serializable.name
        ## This is for PipelineReportTable.
        elif isinstance(obj, SERIALIZABLE_FIELD.pandas_dataframe.value):
            return SERIALIZABLE_FIELD.pandas_dataframe.name
        ## This is for jax Array subclasses (e.g. jax.ArrayImpl)
        elif isinstance(obj, SERIALIZABLE_FIELD.numpy_array.value):
            return SERIALIZABLE_FIELD.numpy_array.name
        else:
            raise ValueError(f"Unsupported type {type(obj)}.")

    def equals(self, other: AbstractHDFSerializable) -> bool:
        # Need stricter than equinox's `equal_trees(... ,typematch=True)`; For example, ensures pandas.DataFrame
        # objects are compared with `equals` instead of `__eq__`.
        assert isinstance(other, AbstractVxData), "Can only compare equality with another AbstractVxData."
        return type(self) is type(other) and self.fields == other.fields and self.equal_attributes(self, other)

    @staticmethod
    def equal_attributes(self_obj: AbstractHDFSerializable, other_obj: AbstractHDFSerializable) -> bool:
        def _equal_attributes(a: Any, b: Any):
            if type(a) is not type(b):
                return False
            type_enum_name = self_obj.object_type_enum_name(a)
            match SERIALIZABLE_FIELD[type_enum_name]:
                case (
                    SERIALIZABLE_FIELD.boolean
                    | SERIALIZABLE_FIELD.integer
                    | SERIALIZABLE_FIELD.float
                    | SERIALIZABLE_FIELD.timestamp
                    | SERIALIZABLE_FIELD.none
                    | SERIALIZABLE_FIELD.string
                ):
                    if a != b:
                        return False
                case SERIALIZABLE_FIELD.numpy_array:
                    if not equal_arrays(a, b):
                        return False
                case SERIALIZABLE_FIELD.pandas_dataframe | SERIALIZABLE_FIELD.pandas_series:
                    if a.index.name != b.index.name:
                        return False
                    if isinstance(a, pd.Series) and (a.name != b.name):
                        return False
                    if not a.equals(b):
                        return False
                case (
                    SERIALIZABLE_FIELD.config
                    | SERIALIZABLE_FIELD.hdf_serializable
                    | SERIALIZABLE_FIELD.pandas_equivalent
                ):
                    if not a.equals(b):
                        return False

                case (
                    SERIALIZABLE_FIELD.list
                    | SERIALIZABLE_FIELD.tuple
                    | SERIALIZABLE_FIELD.set
                    | SERIALIZABLE_FIELD.frozenset
                ):
                    if len(a) != len(b):
                        return False
                    if isinstance(a, set):  # order is not guaranteed!.
                        a, b = sorted(a, key=hash), sorted(b, key=hash)
                    for a_item, b_item in zip(a, b):
                        if not _equal_attributes(a_item, b_item):
                            return False
                case SERIALIZABLE_FIELD.dict | SERIALIZABLE_FIELD.mapping_proxy:
                    if len(a) != len(b):
                        return False

                    a_keys, a_values = a.keys(), a.values()
                    b_keys, b_values = b.keys(), b.values()
                    if a_keys != b_keys:
                        return False
                    for a_item, b_item in zip(a_values, b_values):
                        if not _equal_attributes(a_item, b_item):
                            return False
                case _:
                    raise ValueError(f"Unhandled type {type_enum_name} for attribute {attribute}.")
            return True

        for attribute in self_obj.comparison_prioritized_fields:
            self_value, other_value = getattr(self_obj, attribute), getattr(other_obj, attribute)
            if not _equal_attributes(self_value, other_value):
                return False

        return True

    @classmethod
    def serialize_object(cls, parent_group: tb.Group, obj: Any, hdf_key: str):
        """Serialize a single object under ``hdf_key`` attached to ``parent_group``."""
        # Don't create a group for native-type attributes (float, int, boolean, string).
        # Attach their values as attributes to the parent group.
        hdf = parent_group._v_file
        group = lambda: hdf.create_group(parent_group, hdf_key)
        match SERIALIZABLE_FIELD[cls.object_type_enum_name(obj)]:
            case SERIALIZABLE_FIELD.numpy_array:
                hdf.create_array(parent_group, hdf_key, obj=obj)
            case SERIALIZABLE_FIELD.pandas_dataframe | SERIALIZABLE_FIELD.pandas_series:
                AbstractWithPandasEquivalent.serialize_pandas(obj, group())
            case SERIALIZABLE_FIELD.hdf_serializable | SERIALIZABLE_FIELD.config | SERIALIZABLE_FIELD.pandas_equivalent:
                obj.to_hdf_group(group())
            case (
                SERIALIZABLE_FIELD.integer
                | SERIALIZABLE_FIELD.float
                | SERIALIZABLE_FIELD.boolean
                | SERIALIZABLE_FIELD.string
                | SERIALIZABLE_FIELD.none
            ):
                parent_group._v_attrs[hdf_key] = obj
            case SERIALIZABLE_FIELD.timestamp:
                hdf.create_array(parent_group, hdf_key, obj=obj.value)
            case (
                SERIALIZABLE_FIELD.list
                | SERIALIZABLE_FIELD.tuple
                | SERIALIZABLE_FIELD.set
                | SERIALIZABLE_FIELD.frozenset
            ):
                cls.serialize_sequence(group(), list(obj))
            case SERIALIZABLE_FIELD.dict | SERIALIZABLE_FIELD.mapping_proxy:
                cls.serialize_dict(group(), obj)
            case _:
                raise ValueError(f"Unknown type {type(obj)} for attribute {hdf_key}")

    @classmethod
    def deserialize_object(
        cls,
        parent_group: tb.Group,
        hdf_key: str,
        attr_type_enum_name: str,
        defer: tuple[tuple[str, ...], ...],
        levels: int | None,
    ):
        """Inverse of ``serialize_object`` supporting deferral and virtual nodes."""
        hd_file = parent_group._v_file
        node = lambda: hd_file.get_node(parent_group, hdf_key)
        defer_current = any(len(d) == 1 and d[0] == hdf_key for d in defer)
        defer_next = tuple(d[1:] for d in defer if len(d) > 1 and d[0] == hdf_key)

        if defer_current or levels == 0:
            return HDFVirtualNode(hd_file.filename, parent_group._v_pathname, attr_type_enum_name, key=hdf_key)

        next_level = levels - 1 if levels is not None else None

        match SERIALIZABLE_FIELD[attr_type_enum_name]:
            case SERIALIZABLE_FIELD.numpy_array:
                return node().read()
            case SERIALIZABLE_FIELD.pandas_dataframe | SERIALIZABLE_FIELD.pandas_series:
                return AbstractWithPandasEquivalent.deserialize_pandas(node())
            case SERIALIZABLE_FIELD.pandas_equivalent:
                return AbstractWithPandasEquivalent.from_hdf_group(node(), defer_next, next_level)
            case SERIALIZABLE_FIELD.config:
                return AbstractConfig.from_hdf_group(node(), defer_next, next_level)
            case SERIALIZABLE_FIELD.hdf_serializable:
                return cls.from_hdf_group(node(), defer_next, next_level)
            case SERIALIZABLE_FIELD.none:
                return None
            case (
                SERIALIZABLE_FIELD.integer
                | SERIALIZABLE_FIELD.float
                | SERIALIZABLE_FIELD.boolean
                | SERIALIZABLE_FIELD.string
            ):
                value = parent_group._v_attrs[hdf_key]
                return value.item() if value is not None else None
            case SERIALIZABLE_FIELD.timestamp:
                return pd.Timestamp(node().read())
            case (
                SERIALIZABLE_FIELD.list
                | SERIALIZABLE_FIELD.tuple
                | SERIALIZABLE_FIELD.set
                | SERIALIZABLE_FIELD.frozenset
            ):
                (sequence_type,) = SERIALIZABLE_FIELD[attr_type_enum_name].value
                assert sequence_type in (list, tuple, set, frozenset)
                seq = cls.deserialize_sequence(node(), defer_next, next_level)
                return sequence_type(seq)
            case SERIALIZABLE_FIELD.dict | SERIALIZABLE_FIELD.mapping_proxy:
                (collection_type,) = SERIALIZABLE_FIELD[attr_type_enum_name].value
                c = cls.deserialize_dict(node(), defer_next, next_level)
                assert collection_type in (dict, MappingProxyType)
                return collection_type(c)
            case _:
                raise ValueError(f"Unhandled type {attr_type_enum_name} for attribute {hdf_key}.")

    @classmethod
    def make_hdf_key(cls, item: str | int):
        """Create a stable HDF5 key for a collection element index/key."""
        return f"key_{item}"

    @classmethod
    def make_hdf_segment_key(cls, item: str | int) -> str:
        """Key used to group large collections into segments for performance."""
        return f"_x_segment_{item}"

    @classmethod
    def create_bookkeeping_segments(
        cls, parent_group: tb.Group, entries: list[str | int], objects: list[Any]
    ) -> tuple[dict[str, tb.Group], pd.DataFrame]:
        """Create per-segment groups and a metadata table for heterogeneous collections."""
        # pytables raises a performance warning when the number of children exceeds 16
        assert len(entries) != 0 and len(entries) == len(objects)
        hdf_keys = list(map(cls.make_hdf_key, entries))
        types = list(map(cls.object_type_enum_name, objects))
        metadata = pd.DataFrame({"hdf_key": hdf_keys, "type": types}, index=pd.Index(entries))
        metadata["segment"] = [cls.make_hdf_segment_key(i // MAX_SEGMENT_SIZE) for i in range(len(entries))]
        if metadata["segment"].nunique() > 1:
            h5file = parent_group._v_file
            segmented_groups = {k: h5file.create_group(parent_group, k) for k in metadata["segment"].unique()}
        else:
            segmented_groups = {cls.make_hdf_segment_key(0): parent_group}
        return segmented_groups, metadata

    @classmethod
    def get_bookkeeping_segments(cls, parent_group: tb.Group, metadata: pd.DataFrame) -> dict[str, tb.Group]:
        """Resolve or create segment groups referenced by the metadata table."""
        if metadata["segment"].nunique() == 1:
            return {cls.make_hdf_segment_key(0): parent_group}
        else:
            hf5_file = parent_group._v_file
            return {k: hf5_file.get_node(parent_group, k, "Group") for k in metadata["segment"].unique()}

    @classmethod
    def serialize_sequence(cls, group: tb.Group, sequence: list[Any]):
        """Serialize a list-like sequence, grouping scalar-like items as a Series when possible."""
        if len(sequence) == 0:
            return
        if set(map(type, sequence)).issubset(SERIES_GROUPED_ELEMENT_TYPES):
            cls.serialize_object(group, pd.Series(sequence), "data")
        else:
            cls.serialize_heterogeneous_collection(group, dict(zip(range(len(sequence)), sequence)))

    @classmethod
    def serialize_dict(cls, group: tb.Group, d: Mapping[str | int, Any]):
        """Serialize a mapping, grouping scalar-like values as a Series when possible."""
        if len(d) == 0:
            return
        if set(map(type, d.values())).issubset(SERIES_GROUPED_ELEMENT_TYPES):
            cls.serialize_object(group, pd.Series(d), "data")
        else:
            cls.serialize_heterogeneous_collection(group, d)

    @classmethod
    def serialize_heterogeneous_collection(cls, group: tb.Group, data: Mapping[str | int, Any]):
        """Serialize a mapping of heterogeneously-typed items with segmentation metadata."""
        segment_group, metadata = cls.create_bookkeeping_segments(group, list(data.keys()), list(data.values()))
        cls.serialize_object(group, metadata, "metadata")
        for segment, hdf_key, key in zip(metadata["segment"], metadata["hdf_key"], metadata.index):
            cls.serialize_object(segment_group[segment], data[key], hdf_key)

    @classmethod
    def deserialize_sequence(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...], levels: int | None) -> list[Any]:
        """Deserialize a sequence previously stored by ``serialize_sequence``."""
        if group._v_nchildren == 0:
            return []
        if "data" in group:
            series = pd.read_hdf(group._v_file.filename, key=group.data._v_pathname)
            assert isinstance(series, pd.Series)
            return series.values.tolist()
        return list(cls.deserialize_heterogeneous_collection(group, defer, levels).values())

    @classmethod
    def deserialize_dict(
        cls, group: tb.Group, defer: tuple[tuple[str, ...], ...], levels: int | None
    ) -> dict[str | int, Any]:
        """Deserialize a mapping previously stored by ``serialize_dict``."""
        if group._v_nchildren == 0:
            return {}
        elif "data" in group:
            series = pd.read_hdf(group._v_file.filename, key=group.data._v_pathname)
            assert isinstance(series, pd.Series)
            return series.to_dict()

        return cls.deserialize_heterogeneous_collection(group, defer, levels)

    @classmethod
    def deserialize_heterogeneous_collection(
        cls, group: tb.Group, defer: tuple[tuple[str, ...], ...], levels: int | None
    ) -> dict[str | int, Any]:
        """Deserialize a heterogeneously-typed mapping using its segmentation metadata."""
        meta = pd.read_hdf(group._v_file.filename, key=group.metadata._v_pathname)
        assert isinstance(meta, pd.DataFrame)
        segment_group = cls.get_bookkeeping_segments(group, meta)
        return {
            k: cls.deserialize_object(segment_group[s], hdf_k, t, defer, levels)
            for k, hdf_k, s, t in zip(meta.index, meta["hdf_key"], meta["segment"], meta["type"])
        }

    def to_hdf_group(self, group: tb.Group) -> None:
        """Serialize all fields into the given group."""
        group._v_attrs.classname = self.__class_key__()
        # Store the types enum for each attribute as a pd.Series (directly equivalent to a dictionary).
        fields = self.fields
        values = [getattr(self, attribute) for attribute in fields]
        group._v_attrs.type_enum = json.dumps(dict(zip(fields, map(self.object_type_enum_name, values))))
        for attribute, obj in zip(fields, values):
            self.serialize_object(group, obj, attribute)

    @classmethod
    def _from_hdf_group(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...], levels: int | None) -> Self:
        """Instantiate this class by reading all fields serialized by ``to_hdf_group``."""
        type_enum = json.loads(group._v_attrs.type_enum)
        data = {
            attr: cls.deserialize_object(group, attr, attr_type_enum, defer, levels)
            for attr, attr_type_enum in type_enum.items()
        }
        return cls(**data)

    @classmethod
    def from_hdf_group(
        cls, group: tb.Group, defer: tuple[tuple[str, ...], ...] = (), levels: int | None = None
    ) -> Self:
        """Factory-based loader that dispatches based on stored classname."""
        classname = group._v_attrs.classname.item()
        return cls.__get_factory__(classname)._from_hdf_group(group, defer, levels=levels)

    def save(
        self,
        store: str | Path | tb.Group,
        complib: CompressionLibLiteral = "blosc",
        complevel: int = 9,
        log_config_json: bool = True,
    ):
        """Persist the object to an HDF5 file or group.

        If ``store`` is a path, the file ``<store>.h5`` is created. When
        ``log_config_json`` is True, any ``AbstractConfig`` fields are also
        written next to the HDF5 as a JSON file for convenience.

        Parameters:
            store: Path to the HDF5 file or a PyTables group. If a path, the file ``<store>.h5`` is created.
            complib: Compression library to use. Default is ``blosc``.
            complevel: Compression level to use. Default is ``9``.
            log_config_json: Whether to log the config as a JSON file. Default is ``True``.

        Returns:
            None


        !!! Example
        TODO: add example.

        !!! Example
        TODO: add another example.

        !!! Example
        TODO: add another example.

        """
        # complib: Literal['blosc', 'zlib', 'lzo', 'bzip2'] = 'blosc', complevel: int = 9
        # lzo lvl1 is a good compromise between speed and compression ratio.
        # https://www.pytables.org/usersguide/optimization.html (Figure 15).
        if not isinstance(store, tb.Group):
            filters = tb.Filters(complib=complib, complevel=complevel)
            with tb.open_file(
                str(Path(store).with_suffix(".h5")),
                mode="w",
                filters=filters,
                max_numexpr_threads=None,
                max_blosc_threads=None,
            ) as store:
                return self.save(store.root, log_config_json=log_config_json)
        if log_config_json:
            config_fields = [c for c in self.fields if isinstance(getattr(self, c), AbstractConfig)]
            filename = store._v_file.filename
            for key, config in zip(config_fields, map(lambda c: getattr(self, c), config_fields)):
                config.log_json(filename, key=key)

        self.to_hdf_group(store)

    @classmethod
    def load(
        cls,
        hf5_filename_or_group: str | Path | tb.Group,
        defer: tuple[Callable[[Self], Any], ...] = (),
        levels: tuple[int, ...] | None = None,
    ) -> Self:
        """Load an instance from HDF5 with optional deferral and subtree materialization limits.

        Parameters:
            hf5_filename_or_group: Path to the HDF5 file or a PyTables group.
            defer: Tuple of getters selecting attributes to remain virtual.
            levels: If given with ``defer``, fetch only this many levels deep.

        Returns:
            An instance of the class.

        !!! Example
        TODO: add example.

        !!! Example
        TODO: add another example.

        !!! Example
        TODO: add another example.
        """
        if not isinstance(hf5_filename_or_group, tb.Group):
            with tb.open_file(str(Path(hf5_filename_or_group).with_suffix(".h5")), "r") as hf5_file:
                return cls.load(hf5_file.root, defer, levels=levels)

        defer_paths = tuple(tuple(path_from_getter(d, getitem_transform=cls.make_hdf_key)) for d in defer)
        loaded = cls.from_hdf_group(hf5_filename_or_group, defer_paths)
        if levels is not None:
            return fetch_at(defer, loaded, levels=levels)
        return loaded

    def to_numpy_arrays(self):
        """Convert all array leaves to NumPy arrays and return a new instance."""
        arrs, others = eqx.partition(self, eqx.is_array)
        arrs = jtu.tree_map(lambda a: np.array(a), arrs)
        return eqx.combine(arrs, others)

    def to_jax_arrays(self):
        """Convert all array leaves to JAX arrays and return a new instance."""
        arrs, others = eqx.partition(self, eqx.is_array)
        arrs = jtu.tree_map(lambda a: jnp.array(a), arrs)
        return eqx.combine(arrs, others)


def _match_child_parent_paths(ch: list[str], pt: list[str]):
    """Return True if child and parent path segments are consistent for fetches."""
    # For example:
    # - if we hold a node representing the _patients_, and we want to fetch
    #   a v_node representing the observables of the third admission of the patient at index 6, and
    # - if the _patients_ itself has a sequence of ancestors
    #   (a.b.....y.z), such that the observables accessed with the expression:
    #       a.b.....y.z.patients[key].admissions[index].observables, then:
    # - child (relative to _patients_) = ["6", "admissions", "2", "observables"]
    # - parent (relative to the child) = ["a", "b", ..., "x", "y", "z", "patients", "6", "admissions", "2"]
    if len(ch) == 1:
        return True
    # Remove any segment node from parent.
    segment_pattern = re.compile(AbstractVxData.make_hdf_segment_key(r"\d+"))
    pt = [pti for pti in pt if re.match(segment_pattern, pti) is None]

    child_overlap = ch[:-1]
    parent_overlap = pt[-(len(ch) - 1) :]
    return child_overlap == parent_overlap


T = TypeVar("T", bound=AbstractVxData)
type HDFVirtualNodeGet[T] = Callable[[T], HDFVirtualNode]


def fetch_at[T](
    where: HDFVirtualNodeGet[T] | tuple[HDFVirtualNodeGet[T], ...],
    tree: T,
    levels: int | None | tuple[int | None, ...] = None,
) -> T:
    """Fetch one or more virtual nodes from disk into ``tree``.

    Parameters:
        where: Getter or tuple of getters selecting the virtual node(s).
        tree: Root object containing the virtual node(s).
        levels: If provided, controls depth of subtree materialization for each
            getter (single int applies to all; tuple applies per getter).

    Returns:
        The root object with the virtual node(s) fetched.

    !!! Example
    TODO: add example.

    !!! Example
    TODO: add another example.

    !!! Example
    TODO: add another example.
    """
    # deal with a sequence to avoid opening a file for each v_node fetch.
    if callable(where):
        where = (where,)
    if isinstance(levels, Sized):
        levels = tuple(levels)
    else:
        return fetch_at(where, tree, (levels,) * len(where))

    if len(where) == 0:
        return tree

    assert len(where) == len(levels), (
        f"Passed a tuple of getters and a tuple of levels of different sizes: {len(where)} and {len(levels)}."
    )

    if any(not isinstance(w(tree), HDFVirtualNode) for w in where):
        logging.warning("You are trying to fetch a non-virtual node, which might be already fetched.")

    v_nodes = [w(tree) for w in where]
    v_nodes_paths = [path_from_getter(w, getitem_transform=AbstractVxData.make_hdf_key) for w in where]
    parent_paths = [v_node._v_parent_path_seq for v_node in v_nodes]

    assert all(map(_match_child_parent_paths, v_nodes_paths, parent_paths)), (
        "Incompatible parent path and v_node path and parent path."
    )
    assert all(v_node.filename == v_nodes[0].filename for v_node in v_nodes), "Unexpected Filename mismatch!"
    with tb.open_file(v_nodes[0].filename, "r") as hf5_file:
        for w, v_node, hdf_key, levels_i in zip(where, v_nodes, [p[-1] for p in v_nodes_paths], levels):
            parent_group = hf5_file.get_node(v_node.parent_path)
            assert isinstance(parent_group, tb.Group)
            fetched = AbstractVxData.deserialize_object(
                parent_group, hdf_key=hdf_key, attr_type_enum_name=v_node.type_enum, defer=(), levels=levels_i
            )
            tree = eqx.tree_at(w, tree, fetched)
        return tree
    raise RuntimeError(f"Failed to open the HDFFile: {v_nodes[0].filename}.")


def fetch_one_level_at[T](where: HDFVirtualNodeGet[T] | tuple[HDFVirtualNodeGet[T], ...], tree: T) -> T:
    """Convenience wrapper to fetch exactly one level at selected locations.
    
    Parameters:
        where: Getter or tuple of getters selecting the virtual node(s).
        tree: Root object containing the virtual node(s).

    Returns:
        The root object with the virtual node(s) fetched.

    !!! Example
    TODO: add example.

    !!! Example
    TODO: add another example.

    !!! Example
    TODO: add another example.
    """
    # Useful to fetch dictionary keys with virtual nodes for values.
    # Or a sequence of vitruals, or object with virtual nodes for attributes.
    return fetch_at(where, tree=tree, levels=1)


def fetch_all[T](tree: T) -> T:
    """Fetch all virtual nodes anywhere in the tree, preserving sets/frozensets.
    
    Parameters:
        tree: Root object containing the virtual node(s).

    Returns:
        The root object with the virtual node(s) fetched.

    !!! Example
    TODO: add example.

    !!! Example
    TODO: add another example.

    !!! Example
    TODO: add another example.
    """
    # Note 1:
    # Preprocessing to catch any set in the pytree. JAX pytree does not
    # navigate into sets as it does with list/dicts/tuples.
    # We will transform any such set into a list, then we return them to sets after fetching.
    # Maybe an alternative is to enforce a design where lazy loading is not allowed with sets involved.

    is_set = jtu.tree_map(lambda a: isinstance(a, set), tree)
    is_frozenset = jtu.tree_map(lambda a: isinstance(a, frozenset), tree)
    apply = (
        lambda a, isset, isfrozenset: _MyCustomList_set(a)
        if isset
        else (_MyCustomList_frozenset(a) if isfrozenset else a)
    )
    tree = jtu.tree_map(apply, tree, is_set, is_frozenset)

    _is_vnode = lambda x: isinstance(x, HDFVirtualNode)
    _is_leaf = lambda x: isinstance(x, HDFVirtualNode)

    with_v_nodes, without_v_nodes = eqx.partition(tree, _is_vnode, is_leaf=_is_leaf)
    leaves_with_path, struct = jtu.tree_flatten_with_path(with_v_nodes, is_leaf=_is_leaf)
    if len(leaves_with_path) == 0:
        logging.warning("No virtual nodes found.")
        return tree
    logging.info(f"Fetching {len(leaves_with_path)} leaf nodes from {type(tree).__name__}")

    assert all(isinstance(vn, HDFVirtualNode) for (_, vn) in leaves_with_path)

    filename = leaves_with_path[0][1].filename
    assert all(vn.filename == filename for (_, vn) in leaves_with_path), (
        f"Filenames of virtual nodes {tuple(vn.filename for (_, vn) in leaves_with_path)} do not match."
    )
    v_nodes_paths = [
        path_from_jax_keypath(p, getitem_transform=AbstractVxData.make_hdf_key) for (p, _) in leaves_with_path
    ]
    parents_paths = [vn._v_parent_path_seq for (_, vn) in leaves_with_path]
    attrs = [vn.key for (_, vn) in leaves_with_path]
    assert all(map(_match_child_parent_paths, v_nodes_paths, parents_paths)), (
        "Incompatible parent path and v_node path and parent path."
    )
    with tb.open_file(filename, mode="r") as hf5_file:
        fetched_nodes = []
        for (_, v_node), attr in zip(leaves_with_path, attrs):
            parent_group = hf5_file.get_node(v_node.parent_path)
            assert isinstance(parent_group, tb.Group)
            fetched = AbstractVxData.deserialize_object(
                parent_group, hdf_key=attr, attr_type_enum_name=v_node.type_enum, defer=(), levels=None
            )
            fetched_nodes.append(fetched)

        with_v_nodes = jtu.tree_unflatten(struct, fetched_nodes)
        tree = eqx.combine(without_v_nodes, with_v_nodes, is_leaf=_is_leaf)
        # return all transformed sets to lists back to sets (read Note 1 above).
        apply = (
            lambda a: set(a)
            if isinstance(a, _MyCustomList_set)
            else (frozenset(a) if isinstance(a, _MyCustomList_frozenset) else a)
        )
        return jtu.tree_map(apply, tree, is_leaf=lambda a: isinstance(a, (_MyCustomList_set, _MyCustomList_frozenset)))

    raise RuntimeError(f"Failed to open the HDFFile: {filename}.")
