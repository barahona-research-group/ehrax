import ast
import dataclasses
import enum
import json
import logging
from abc import abstractmethod
from pathlib import Path
from types import MappingProxyType, NoneType
from typing import Any, Callable, Self, TYPE_CHECKING, Collection, Mapping, Literal, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pandas as pd
import tables as tb

from ehrax.utils import tree_hasnan, NumpyEncoder, ArrayTypes, np_module, load_config, write_config, equal_arrays, \
    path_from_getter
from utils import path_from_jax_keypath

_factory_registry: dict[str, type[eqx.Module]] = {}


class _ModuleMeta(type(eqx.Module)):
    # This method is called whenever you definite a module: `class Foo(eqx.Module): ...`
    def __new__(
            mcs,
            name,
            bases,
            dict_,
            /,
            strict: bool | eqx.StrictConfig = False,
            **kwargs,
    ):
        cls = super().__new__(mcs, name, bases, dict_, strict=strict, **kwargs)

        # We need to collect all constructors of the subclasses in one dictionary
        # to use them in deserialization of objects that, themselves, composed of
        # other native types or objects that has the same MetaClass.

        def __class_key__(cls):
            return f"{cls.__module__}.{cls.__qualname__}"

        def __get_factory__(type_str: str) -> type[Self]:
            return _factory_registry[type_str]

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
        pass


class AbstractHDFSerializable(AbstractModule):

    @abstractmethod
    def to_hdf_group(self, group: tb.Group) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_hdf_group(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...], levels: Optional[int]) -> Self:
        raise NotImplementedError

    @abstractmethod
    def equals(self, other: Self) -> bool:
        raise NotImplementedError


class HDFVirtualNode(AbstractHDFSerializable):
    """
    This class represents an unfetched node in a PyTree/AbstractHDFSerializable.
    This is similar to the notion of lazy-loading, but the library explicitly requires calling `fetch_at(..,..)`
    or `fetch_all()` on any of the node ancestors, a
    """
    filename: str
    parent_path: str
    type_enum: str

    def __init__(self, filename: str, parent_path: str, type_enum: str) -> None:
        self.filename = filename
        self.parent_path = parent_path
        self.type_enum = type_enum

    def __check_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            assert isinstance(value, field.type)

    @property
    def _v_parent_path_seq(self) -> list[str]:  # to a series of directories with root directory represented by ''
        if len(self.parent_path) == 0: return []
        if len(self.parent_path) == 1: return ['']
        return self.parent_path.split('/')

    def __getattribute__(self, attr: str) -> NoneType:
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            raise AttributeError(
                f"You are trying to access an attribute in a lazy-loaded node. Please call `fetch_at(..,..)` or "
                f"`fetch_all()` on any of the node ancestors first."
            )

    def to_hdf_group(self, group: tb.Group) -> None:
        raise ValueError(
            f"You are trying to serialize an unfetched node in a PyTree/AbstractHDFSerializable. Please call `fetch_at(..,..)` or "
            f"`fetch_all()` on any of the node ancestors first."
        )

    @classmethod
    def from_hdf_group(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...] = (),
                       levels: Optional[int] = None) -> Self:
        raise ValueError(
            f"You are trying to deserialize a VirtualNode."
        )

    def equals(self, other: Self) -> bool:
        raise ValueError(
            f"You are trying to test equality with a virtual unfetched node. Please call `fetch_at(..,..)` or "
            f"`fetch_all()` on any of the node ancestors first."
        )


class AbstractConfig(AbstractHDFSerializable):

    @classmethod
    def _map_hierarchical_config(cls, unit_config_map: Callable[[Self], dict[str, Any]], x: Any) -> Any:
        if isinstance(x, AbstractConfig):
            x = unit_config_map(x)
        if isinstance(x, dict):
            return {k: AbstractConfig._map_hierarchical_config(unit_config_map, v) for k, v in x.items()}
        elif isinstance(x, list):
            return [AbstractConfig._map_hierarchical_config(unit_config_map, v) for v in x]
        elif isinstance(x, tuple):
            return tuple(AbstractConfig._map_hierarchical_config(unit_config_map, v) for v in x)
        else:
            return x

    @classmethod
    def _as_normal_dict(cls, x: Self) -> dict[str, Any]:
        return {field.name: getattr(x, field.name) for field in dataclasses.fields(x) if not field.name.startswith("_")}

    @staticmethod
    def _as_typed_dict(x: 'AbstractConfig') -> dict[str, Any]:
        return AbstractConfig._as_normal_dict(x) | {"_type": x.__class_key__()}

    @staticmethod
    def _is_typed_dict(x) -> bool:
        return isinstance(x, dict) and "_type" in x

    @staticmethod
    def _map_config_to_dict(unit_map: Callable[[Any], dict[str, Any]], x) -> dict[str, Any]:
        return json.loads(json.dumps(AbstractConfig._map_hierarchical_config(unit_map, x), cls=NumpyEncoder))

    def as_dict(self) -> dict[str, Any]:
        return AbstractConfig._map_config_to_dict(AbstractConfig._as_normal_dict, self)

    def to_dict(self) -> dict[str, Any]:
        # Fully deserializable to a Config object.
        return AbstractConfig._map_config_to_dict(AbstractConfig._as_typed_dict, self)

    def equals(self, other: Self) -> bool:
        return self.to_dict() == other.to_dict()

    def to_hdf_group(self, group: tb.Group) -> None:
        data = json.dumps(self.to_dict(), cls=NumpyEncoder).encode('utf-8')
        group._v_file.create_array(group, 'data', obj=data)

    @classmethod
    def from_hdf_group(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...] = (),
                       levels: Optional[int] = None) -> Self:
        assert len(defer) == 0, "Unexpected."
        return cls.from_dict(json.loads(group["data"].read().decode('utf-8')))

    def log_json(self, path: str | Path, key: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        json_path = path.with_suffix('.json')  # config goes here.
        config = load_config(str(json_path)) if json_path.exists() else {}
        config[key] = self.as_dict()  # logging does not aim to produce deserializable config json.
        write_config(config, str(json_path))

    def update(self, other: Self | dict[str, Any]) -> Self:
        if isinstance(other, AbstractConfig):
            other = other.to_dict()
            other["_type"] = self.__class_key__()
        return AbstractConfig.from_dict(self.to_dict() | other)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
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

        return _map_dict_to_config(config)

    def path_update(self, path, value) -> Self:
        nesting = path.split(".")

        def _get(x):
            for n in nesting:
                x = getattr(x, n)
            return x

        _constructor = type(_get(self)) if value is not None else lambda x: None
        return eqx.tree_at(_get, self, _constructor(value))


class AbstractWithPandasEquivalent(AbstractHDFSerializable):

    @staticmethod
    def empty_pandas_meta(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
        meta = {
            f'index_dtype': str(df.index.dtype),
            f'index_name': str(df.index.name)}
        if isinstance(df, pd.Series):
            meta.update({'dtype': str(df.dtype), 'name': df.name})
        else:
            meta.update({
                f'column_{i}': col for i, col in enumerate(df.columns)})
            meta.update({
                f'column_dtype_{df.columns[i]}': str(dtype) for i, dtype in enumerate(df.dtypes)})
        return pd.DataFrame(meta, index=[0])

    @staticmethod
    def empty_pandas_from_metadata(meta: pd.DataFrame) -> pd.DataFrame | pd.Series:
        meta = meta.iloc[0].to_dict()
        index_name = meta.pop('index_name')
        index = pd.Index([], dtype=meta.pop('index_dtype'), name=None if index_name == 'None' else index_name)
        if 'dtype' in meta:
            dtype = meta.pop('dtype')
            name = meta.pop('name')
            name = name if name == name else None  # when None serialized it ended up as a nan
            return pd.Series(dtype=dtype, name=name, index=index)

        cols = []
        column_types = {}
        while len(meta) > 0:
            k, v = meta.popitem()
            if k.startswith('column_dtype_'):
                column_types[k.split('column_dtype_')[1]] = v
            elif k.startswith('column_'):
                order = int(k.split('column_')[1])
                cols.append((order, v))
        cols = [col for _, col in sorted(cols, key=lambda x: x[0])]
        return pd.DataFrame(columns=cols, index=index).astype(column_types)

    @classmethod
    def serialize_pandas(cls, data: pd.DataFrame | pd.Series, group: tb.Group):
        # Empty dataframes/series are not saved in hdf file, https://github.com/pandas-dev/pandas/issues/13016,
        # https://github.com/PyTables/PyTables/issues/592
        # We still need to preserve the column names, dtypes, index name, index dtype, in a metadata object, to
        # fulfill perfect serialization/deserialization and have stricter unit-testing.
        hdf = group._v_file
        if data.empty:
            key = hdf.create_group(group, 'empty_metadata')._v_pathname
            cls.empty_pandas_meta(data).to_hdf(hdf.filename, key=key, format='table')
        else:
            data.to_hdf(hdf.filename, key=group._v_pathname, format='table')

    @classmethod
    def deserialize_pandas(cls, store: tb.Group) -> pd.DataFrame | pd.Series:
        hdf = store._v_file
        if 'empty_metadata' in store:
            return cls.empty_pandas_from_metadata(pd.read_hdf(hdf.filename, key=store.empty_metadata._v_pathname))
        else:
            return pd.read_hdf(hdf.filename, key=store._v_pathname)

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame | pd.Series:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pandas(cls, pandas: pd.DataFrame | pd.Series) -> Self:
        raise NotImplementedError

    def to_hdf_group(self, group: tb.Group) -> None:
        self.serialize_pandas(self.to_pandas(), group)

    @classmethod
    def from_hdf_group(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...] = (),
                       levels: Optional[int] = None) -> Self:
        assert len(defer) == 0, "Unexpected."
        return cls.from_pandas(cls.deserialize_pandas(group))

    def equals(self, other: Self) -> bool:
        return type(self) == type(other) and self.to_pandas().equals(other.to_pandas())


class AbstractWithDataframeEquivalent(AbstractWithPandasEquivalent):
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
    def from_pandas(cls, pandas: pd.DataFrame) -> Self:
        return cls.from_dataframe(pandas)


class AbstractWithSeriesEquivalent(AbstractWithPandasEquivalent):
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
    def from_pandas(cls, pandas: pd.Series) -> Self:
        return cls.from_series(pandas)


class SERIALIZABLE_FIELD(enum.Enum):
    none = (type(None),)
    numpy_array = tuple(ArrayTypes)
    pandas_dataframe = (pd.DataFrame,)
    pandas_series = (pd.Series,)
    hdf_serializable = (AbstractHDFSerializable,)
    config = (AbstractConfig,)
    string = (str,)
    integer = (int,)
    float = (float,)
    boolean = (bool,)
    timestamp = (pd.Timestamp,)
    homogeneous_list = (list,)
    homogeneous_tuple = (tuple,)
    homogeneous_dict = (dict,)
    homogeneous_set = (set,)
    homogeneous_mapping_proxy = (MappingProxyType,)


COMPARISON_PRIORITY = MappingProxyType(
    {e.name: 100 for e in SERIALIZABLE_FIELD} |
    {  # default: 100
        e.name: 0 for e in (SERIALIZABLE_FIELD.none,)  # begin with None
    } | {
        e.name: 1 for e in (SERIALIZABLE_FIELD.boolean, SERIALIZABLE_FIELD.integer, SERIALIZABLE_FIELD.float,
                            SERIALIZABLE_FIELD.timestamp)  # then scalers
    } | {
        e.name: 2 for e in (SERIALIZABLE_FIELD.string,)  # then variable-length strings
    } | {
        e.name: 3 for e in (SERIALIZABLE_FIELD.config,)  # then config
    } | {
        e.name: 4 for e in  # then intensive data.
        (SERIALIZABLE_FIELD.numpy_array, SERIALIZABLE_FIELD.pandas_dataframe, SERIALIZABLE_FIELD.pandas_series)
    })  # collections, mappings, and HDFSerializable subclasses are left because they can be nested with other HDFSerializables
assert all(isinstance(e.value, tuple) and all(isinstance(t, type) for t in e.value) for e in
           SERIALIZABLE_FIELD), "Expected a tuple of types."

_TYPE_ENUM_DICT: MappingProxyType[type, str] = MappingProxyType({
    t: e.name for e in SERIALIZABLE_FIELD for t in e.value
})
SERIALIZABLE_FIELD_TYPES = tuple(_TYPE_ENUM_DICT.keys())
SERIALIZABLE_FLAT_COLLECTION = (SERIALIZABLE_FIELD.homogeneous_set, SERIALIZABLE_FIELD.homogeneous_list,
                                SERIALIZABLE_FIELD.homogeneous_tuple)
SERIALIZABLE_FLAT_DICT = (SERIALIZABLE_FIELD.homogeneous_dict, SERIALIZABLE_FIELD.homogeneous_mapping_proxy)
SERIALIZABLE_FLAT_DICT_KEY = (SERIALIZABLE_FIELD.integer, SERIALIZABLE_FIELD.string)

SERIALIZABLE_FLAT_COLLECTION_TYPES: tuple[type, ...] = sum((e.value for e in SERIALIZABLE_FLAT_COLLECTION), ())
SERIALIZABLE_FLAT_DICT_TYPES: tuple[type, ...] = sum((e.value for e in SERIALIZABLE_FLAT_DICT), ())
SERIALIZABLE_FLAT_DICT_KEY_TYPES: tuple[type, ...] = sum((e.value for e in SERIALIZABLE_FLAT_DICT_KEY), ())

# Serialized elements within homogeneous- and flat-list, -tuple, -dict.
SERIALIZABLE_ELEMENT = (SERIALIZABLE_FIELD.numpy_array, SERIALIZABLE_FIELD.pandas_dataframe,
                        SERIALIZABLE_FIELD.pandas_series, SERIALIZABLE_FIELD.hdf_serializable,
                        SERIALIZABLE_FIELD.config, SERIALIZABLE_FIELD.string, SERIALIZABLE_FIELD.integer,
                        SERIALIZABLE_FIELD.float, SERIALIZABLE_FIELD.boolean, SERIALIZABLE_FIELD.timestamp)
SERIALIZABLE_ELEMENT_TYPES = sum((e.value for e in SERIALIZABLE_ELEMENT), ())

# These element types contained within homogeneous container, can be converted to a pandas Series first.
SERIES_GROUPED_ELEMENT = (SERIALIZABLE_FIELD.float, SERIALIZABLE_FIELD.integer, SERIALIZABLE_FIELD.boolean,
                          SERIALIZABLE_FIELD.string, SERIALIZABLE_FIELD.none)
SERIES_GROUPED_ELEMENT_TYPES = sum((e.value for e in SERIES_GROUPED_ELEMENT), ())


class AbstractVxData(AbstractHDFSerializable, eqx.Module):
    """
    AbstractVxData class represents vectorized data object, which inherits from eqx.AbstractVxData.

    Methods:

        to_cpu() - Copy arrays in module to CPU.

        to_device() - Copy arrays in module to device.
    """

    @staticmethod
    def validate_flat_homogeneous_collection(collection: Collection[Any]):
        assert isinstance(collection, SERIALIZABLE_FLAT_COLLECTION_TYPES), "Expected homogeneous collection."
        if len(collection) > 0:
            element = next(iter(collection))
            element_types = set(map(type, collection)) - {type(None)}
            assert len(element_types) == 1, "All elements must be of the same type."
            (element_type,) = element_types
            assert element_type in SERIALIZABLE_ELEMENT_TYPES, f"Expected a serializable element, got {element_type}."

    @staticmethod
    def validate_flat_dict(d: Mapping[str | int, Any]):
        assert isinstance(d, SERIALIZABLE_FLAT_DICT_TYPES), f"Expected homogeneous flat dict type. Got {type(d)}."
        if len(d) > 0:
            v_types = set(map(type, d.values())) - {type(None)}
            k_types = set(map(type, d.keys()))
            assert len(v_types) == 1, f"All values must be of the same type. Got {tuple(v_types)}"
            assert len(k_types) == 1, f"All keys must be of the same type. Got {tuple(k_types)}"
            (v_type,), (k_type) = v_types, k_types
            assert v_type in SERIALIZABLE_ELEMENT_TYPES, f"Expected a serializable element, got {v_type}."
            assert k_type in SERIALIZABLE_FLAT_DICT_KEY_TYPES, f"Invalid key type {k_type}."

    def __check_init__(self):
        for f in (k for k in self.fields):
            obj = getattr(self, f)
            if obj is None:
                continue
            try:
                enum_type = self.object_type_enum_name(obj)
            except (KeyError, StopIteration):
                assert False, (f"Unsupported field type {type(obj)} for attribute {f}. It must be a type in "
                               f"{tuple(t.__qualname__ for t in SERIALIZABLE_FIELD_TYPES)} or a subclass of AbstractHDFSerializable.")

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(k.name for k in dataclasses.fields(self))

    @property
    def comparison_prioritized_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(self.fields, key=lambda f: COMPARISON_PRIORITY[self.object_type_enum_name(getattr(self, f))]))

    @classmethod
    def object_type_enum_name(cls, obj: Any) -> str:
        if type(obj) in _TYPE_ENUM_DICT:
            return _TYPE_ENUM_DICT[type(obj)]
        elif isinstance(obj,
                        SERIALIZABLE_FIELD.hdf_serializable.value):  ## Potentially a subclass of AbstractHDFSerializable
            return SERIALIZABLE_FIELD.hdf_serializable.name
        elif isinstance(obj, SERIALIZABLE_FIELD.config.value):
            return SERIALIZABLE_FIELD.config.name
        elif isinstance(obj, SERIALIZABLE_FIELD.pandas_dataframe.value):  ## This is for PipelineReportTable.
            return SERIALIZABLE_FIELD.pandas_dataframe.name
        else:
            raise ValueError(f"Unsupported type {type(obj)}.")

    def equals(self, other: Self) -> bool:
        # Need stricter than equinox's `equal_trees(... ,typematch=True)`; For example, ensures pandas.DataFrame
        # objects are compared with `equals` instead of `__eq__`.
        return type(self) == type(other) and self.fields == other.fields and self.equal_attributes(self, other)

    @classmethod
    def equal_attributes(cls, self_obj: Self, other_obj: Self) -> bool:
        def _equal_attributes(a: Any, b: Any):
            if type(a) != type(b):
                return False
            type_enum_name = self_obj.object_type_enum_name(a)
            match SERIALIZABLE_FIELD[type_enum_name]:
                case SERIALIZABLE_FIELD.boolean | SERIALIZABLE_FIELD.integer | SERIALIZABLE_FIELD.float | SERIALIZABLE_FIELD.timestamp | SERIALIZABLE_FIELD.none | SERIALIZABLE_FIELD.string:
                    if a != b: return False
                case SERIALIZABLE_FIELD.numpy_array:
                    if not equal_arrays(a, b): return False
                case SERIALIZABLE_FIELD.pandas_dataframe | SERIALIZABLE_FIELD.pandas_series:
                    if a.index.name != b.index.name: return False
                    if isinstance(a, pd.Series) and (a.name != b.name): return False
                    if not a.equals(b): return False
                case SERIALIZABLE_FIELD.config | SERIALIZABLE_FIELD.hdf_serializable:
                    if not a.equals(b): return False
                case SERIALIZABLE_FIELD.homogeneous_list | SERIALIZABLE_FIELD.homogeneous_tuple | SERIALIZABLE_FIELD.homogeneous_set:
                    if len(a) != len(b): return False
                    for a_item, b_item in zip(a, b):
                        if not _equal_attributes(a_item, b_item): return False
                case SERIALIZABLE_FIELD.homogeneous_dict | SERIALIZABLE_FIELD.homogeneous_mapping_proxy:
                    if len(a) != len(b): return False

                    a_keys, a_values = a.keys(), a.values()
                    b_keys, b_values = b.keys(), b.values()
                    if a_keys != b_keys: return False
                    for a_item, b_item in zip(a_values, b_values):
                        if not _equal_attributes(a_item, b_item): return False
                case _:
                    raise ValueError(f"Unhandled type {type_enum_name} for attribute {attribute}.")
            return True

        for attribute in self_obj.comparison_prioritized_fields:
            self_value, other_value = getattr(self_obj, attribute), getattr(other_obj, attribute)
            if not _equal_attributes(self_value, other_value): return False

        return True

    @classmethod
    def serialize_object(cls, parent_group: tb.Group, obj: Any, attribute: str):
        # Don't create a group for native-type attributes (float, int, boolean, string).
        # Attach their values as attributes to the parent group.
        hdf = parent_group._v_file
        group = lambda: hdf.create_group(parent_group, attribute)
        match SERIALIZABLE_FIELD[cls.object_type_enum_name(obj)]:
            case SERIALIZABLE_FIELD.numpy_array:
                hdf.create_array(parent_group, attribute, obj=obj)
            case SERIALIZABLE_FIELD.pandas_dataframe | SERIALIZABLE_FIELD.pandas_series:
                AbstractWithPandasEquivalent.serialize_pandas(obj, group())
            case SERIALIZABLE_FIELD.hdf_serializable | SERIALIZABLE_FIELD.config:
                parent_group._v_attrs[attribute] = obj.__class_key__()
                obj.to_hdf_group(group())
            case SERIALIZABLE_FIELD.integer | SERIALIZABLE_FIELD.float | SERIALIZABLE_FIELD.boolean | SERIALIZABLE_FIELD.string | SERIALIZABLE_FIELD.none:
                parent_group._v_attrs[attribute] = obj
            case SERIALIZABLE_FIELD.timestamp:
                hdf.create_array(parent_group, attribute, obj=obj.value)
            case SERIALIZABLE_FIELD.homogeneous_list | SERIALIZABLE_FIELD.homogeneous_tuple | SERIALIZABLE_FIELD.homogeneous_set:
                cls.serialize_collection(group(), list(obj))
            case SERIALIZABLE_FIELD.homogeneous_dict | SERIALIZABLE_FIELD.homogeneous_mapping_proxy:
                cls.serialize_dict(group(), obj)
            case _:
                raise ValueError(f"Unknown type {type(obj)} for attribute {attribute}")

    @classmethod
    def deserialize_object(cls, parent_group: tb.Group, attribute: str, attr_type_enum_name: str,
                           defer: tuple[tuple[str, ...], ...], levels: Optional[int]):
        hd_file = parent_group._v_file
        node = lambda: hd_file.get_node(parent_group, attribute)
        defer_current = any(len(d) == 1 and d[0] == attribute for d in defer)
        defer_next = tuple(d[1:] for d in defer if len(d) > 1 and d[0] == attribute)

        if defer_current or levels == 0:
            return HDFVirtualNode(hd_file.filename, parent_group._v_pathname, attr_type_enum_name)

        next_level = levels - 1 if levels is not None else None

        match SERIALIZABLE_FIELD[attr_type_enum_name]:
            case SERIALIZABLE_FIELD.numpy_array:
                return node().read()
            case SERIALIZABLE_FIELD.pandas_dataframe | SERIALIZABLE_FIELD.pandas_series:
                return AbstractWithPandasEquivalent.deserialize_pandas(node())
            case SERIALIZABLE_FIELD.hdf_serializable | SERIALIZABLE_FIELD.config:
                cls_key = parent_group._v_attrs[attribute].item()
                cls = cls.__get_factory__(cls_key)
                return cls.from_hdf_group(node(), defer_next, next_level)
            case SERIALIZABLE_FIELD.none:
                return None
            case SERIALIZABLE_FIELD.integer | SERIALIZABLE_FIELD.float | SERIALIZABLE_FIELD.boolean | SERIALIZABLE_FIELD.string:
                value = parent_group._v_attrs[attribute]
                return value.item() if value is not None else None
            case SERIALIZABLE_FIELD.timestamp:
                return pd.Timestamp(node().read())
            case SERIALIZABLE_FIELD.homogeneous_list | SERIALIZABLE_FIELD.homogeneous_tuple | SERIALIZABLE_FIELD.homogeneous_set:
                (collection_type,) = SERIALIZABLE_FIELD[attr_type_enum_name].value
                return collection_type(cls.deserialize_collection(node(), defer_next, next_level))
            case SERIALIZABLE_FIELD.homogeneous_dict | SERIALIZABLE_FIELD.homogeneous_mapping_proxy:
                (collection_type,) = SERIALIZABLE_FIELD[attr_type_enum_name].value
                return collection_type(cls.deserialize_dict(node(), defer_next, next_level))
            case _:
                raise ValueError(f"Unhandled type {attr_type_enum_name} for attribute {attribute}.")

    @classmethod
    def serialize_collection(cls, group: tb.Group, collection: list[Any]):
        if len(collection) == 0:
            return
        if set(map(type, collection)).issubset(SERIES_GROUPED_ELEMENT_TYPES):
            cls.serialize_object(group, pd.Series(collection), 'data')
        else:
            fields = list(map(str, range(len(collection))))
            group._v_attrs.type_enum = cls._dict_to_str(dict(zip(fields, map(cls.object_type_enum_name, collection))))
            for i, item in enumerate(collection):
                cls.serialize_object(group, item, str(i))

    @classmethod
    def deserialize_collection(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...], levels: Optional[int]) -> list[
        Any]:
        if group._v_nchildren == 0:
            return []
        if 'data' in group:
            return pd.read_hdf(group._v_file.filename, key=group.data._v_pathname).values.tolist()
        metadata = cls._str_to_dict(group._v_attrs.type_enum)
        return [cls.deserialize_object(group, str(k), element_type, defer, levels) for k, element_type in
                metadata.items()]

    @classmethod
    def serialize_dict(cls, group: tb.Group, d: Mapping[str | int, Any]):
        if len(d) == 0:
            return
        if set(map(type, d.values())).issubset(SERIES_GROUPED_ELEMENT_TYPES):
            cls.serialize_object(group, pd.Series(d), 'data')
        else:
            fields = list(d.keys())
            group._v_attrs.type_enum = cls._dict_to_str(dict(zip(fields, map(cls.object_type_enum_name, d.values()))))
            for k, v in d.items():
                cls.serialize_object(group, v, str(k))

    @classmethod
    def deserialize_dict(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...], levels: Optional[int]) -> dict[
        str | int, Any]:
        if group._v_nchildren == 0:
            return {}
        elif 'data' in group:
            return pd.read_hdf(group._v_file.filename, key=group.data._v_pathname).to_dict()
        type_enum = cls._str_to_dict(group._v_attrs.type_enum)
        return {k: cls.deserialize_object(group, str(k), value_type_enum, defer, levels) for k, value_type_enum in
                type_enum.items()}

    @staticmethod
    def _dict_to_str(x: dict) -> str:
        # we could have used pd.Series to store the dict, but could invoke pickle library
        # which is not safe.
        # Also could have used json.dumps, but it restricts key types to str type.
        # It is a fair restriction, but for the sake of completeness we can just stringigy
        # the dict with str(dict). Then later to restore it with ast.literal_eval(string)
        # which deemed relatively safe to unpickling (no code executions), but not immune
        # from DOS attacks: https://stackoverflow.com/a/7689085
        # TODO: add tests to ensure pickle function are never invoked by tables library.
        return str(x)

    @staticmethod
    def _str_to_dict(x: np.str_) -> dict:
        return ast.literal_eval(x.item())

    def to_hdf_group(self, group: tb.Group) -> None:
        h5file = group._v_file
        h5file.create_array(group, 'classname', obj=self.__class_key__().encode('utf-8'))
        # Store the types enum for each attribute as a pd.Series (directly equivalent to a dictionary).
        fields = self.fields
        values = [getattr(self, attribute) for attribute in fields]
        group._v_attrs.type_enum = self._dict_to_str(dict(zip(fields, map(self.object_type_enum_name, values))))
        for attribute, obj in zip(fields, values):
            self.serialize_object(group, obj, attribute)

    @classmethod
    def _from_hdf_group(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...], levels: Optional[int]) -> Self:
        type_enum = cls._str_to_dict(group._v_attrs.type_enum)
        data = {attr: cls.deserialize_object(group, attr, attr_type_enum, defer, levels) for attr, attr_type_enum in
                type_enum.items()}
        return cls(**data)

    @classmethod
    def from_hdf_group(cls, group: tb.Group, defer: tuple[tuple[str, ...], ...] = (),
                       levels: Optional[int] = None) -> Self:
        classname = group.classname.read().decode('utf-8')
        return cls.__get_factory__(classname)._from_hdf_group(group, defer, levels=levels)

    def save(self, store: str | Path | tb.Group, complib: Literal['blosc', 'zlib', 'lzo', 'bzip2'] = 'blosc',
             complevel: int = 9, log_config_json: bool = True):
        # TODO: set BLOSC_NTHREADS in test and deployment
        # complib: Literal['blosc', 'zlib', 'lzo', 'bzip2'] = 'blosc', complevel: int = 9
        # lzo lvl1 is a good compromise between speed and compression ratio.
        # https://www.pytables.org/usersguide/optimization.html (Figure 15).
        if not isinstance(store, tb.Group):
            filters = tb.Filters(complib=complib, complevel=complevel)
            with tb.open_file(str(Path(store).with_suffix('.h5')), mode='w', filters=filters,
                              max_numexpr_threads=None, max_blosc_threads=None) as store:
                return self.save(store.root, log_config_json=log_config_json)
        if log_config_json:
            config_fields = [c for c in self.fields if isinstance(getattr(self, c), AbstractConfig)]
            filename = store._v_file.filename
            for key, config in zip(config_fields, map(lambda c: getattr(self, c), config_fields)):
                config.log_json(filename, key=key)

        self.to_hdf_group(store)

    @classmethod
    def load(cls, hf5_filename_or_group: str | Path | tb.Group,
             defer: tuple[Callable[[AbstractHDFSerializable], Any], ...] = (),
             levels: Optional[tuple[int]] = None) -> Self:
        if not isinstance(hf5_filename_or_group, tb.Group):
            with tb.open_file(str(Path(hf5_filename_or_group).with_suffix('.h5')), 'r') as hf5_file:
                return cls.load(hf5_file.root, defer)

        defer_paths = tuple(map(tuple, map(path_from_getter, defer)))
        loaded = cls.from_hdf_group(hf5_filename_or_group, defer_paths)
        if levels is not None:
            return fetch_at(defer, loaded, levels=levels)
        return loaded

    def to_numpy_arrays(self):
        arrs, others = eqx.partition(self, eqx.is_array)
        arrs = jtu.tree_map(lambda a: np.array(a), arrs)
        return eqx.combine(arrs, others)

    def to_jax_arrays(self):
        arrs, others = eqx.partition(self, eqx.is_array)
        arrs = jtu.tree_map(lambda a: jnp.array(a), arrs)
        return eqx.combine(arrs, others)

    def replace_nans(self):
        arrs, others = eqx.partition(self, eqx.is_array)
        arrs = jtu.tree_map(lambda a: np_module(a).nan_to_num(a), arrs)
        return eqx.combine(arrs, others)

    def has_nans(self):
        return tree_hasnan(self)

    #
    # def __eq__(self, other: Self) -> bool:
    #     return self.equals(other)


HDFVirtualNodeGet = Callable[[AbstractHDFSerializable], HDFVirtualNode]


def _match_child_parent_paths(ch: list[str], pt: list[str]):
    # E.g., if we hold a node representing the *patients*, and we want to fetch
    # a v_node at the observables of the third admission of the patient at key/index 6, considering the nesting
    # a.b.....y.z.patients[key].admissions[index].observables, then:
    # child = ["6", "admissions", "2", "observables"]
    # parent = ["a", "b", ..., "x", "y", "z", "patients", "6", "admissions", "2"]
    if len(ch) == 1:
        return True
    return ch[:-1] == pt[-(len(ch) - 1):]


def fetch_at(where: HDFVirtualNodeGet | tuple[HDFVirtualNodeGet, ...], tree: AbstractVxData,
             levels: Optional[int] | tuple[int, ...] = None) -> AbstractVxData:
    # deal with a collection to avoid opening a file for each v_node fetch.
    if callable(where):
        where = (where,)
    if not isinstance(levels, (tuple, list)):
        levels = (levels,) * len(where)

    assert len(where) == len(levels), (
        f"Passed a tuple of getters and a tuple of levels of different sizes: {len(where)} and {len(levels)}.")

    if any(not isinstance(w(tree), HDFVirtualNode) for w in where):
        logging.warning(f"You are trying to fetch a non-virtual node, which might be already fetched.")

    v_nodes = [w(tree) for w in where]
    v_nodes_paths = [path_from_getter(w) for w in where]
    parent_paths = [v_node._v_parent_path_seq for v_node in v_nodes]

    assert all(map(_match_child_parent_paths, v_nodes_paths, parent_paths)), (
        f"Incompatible parent path and v_node path and parent path.")
    assert all(v_node.filename == v_nodes[0].filename for v_node in v_nodes), "Unexpected Filename mismatch!"
    with tb.open_file(v_nodes[0].filename, 'r') as hf5_file:
        for w, v_node, attr, levels_i in zip(where, v_nodes, map(lambda p: p[-1], v_nodes_paths), levels):
            parent_group = hf5_file.get_node(v_node.parent_path)
            assert isinstance(parent_group, tb.Group)
            fetched = AbstractVxData.deserialize_object(parent_group, attribute=attr,
                                                        attr_type_enum_name=v_node.type_enum, defer=(), levels=levels_i)
            tree = eqx.tree_at(w, tree, fetched)
        return tree
    assert 0, "Unreachable."


def fetch_one_level_at(where: HDFVirtualNodeGet | tuple[HDFVirtualNodeGet, ...],
                       tree: AbstractVxData) -> AbstractVxData:
    # Useful to fetch dictionary keys with virtual nodes for values.
    # Or a collection of vitruals, or object with virtual nodes for attributes.
    return fetch_at(where, tree=tree, levels=1)


def fetch_all(tree: AbstractVxData) -> AbstractVxData:
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
        f"Filenames of virtual nodes {tuple(vn.filename for (_, vn) in leaves_with_path)} do not match.")
    v_nodes_paths = [path_from_jax_keypath(p) for (p, _) in leaves_with_path]
    parents_paths = [vn._v_parent_path_seq for (_, vn) in leaves_with_path]
    attrs = [p[-1] for p in v_nodes_paths]
    assert all(map(_match_child_parent_paths, v_nodes_paths, parents_paths)), (
        f"Incompatible parent path and v_node path and parent path.")
    with tb.open_file(filename, mode='r') as hf5_file:
        fetched_nodes = []
        for (_, v_node), attr in zip(leaves_with_path, attrs):
            parent_group = hf5_file.get_node(v_node.parent_path)
            assert isinstance(parent_group, tb.Group)
            fetched = AbstractVxData.deserialize_object(parent_group, attribute=attr,
                                                        attr_type_enum_name=v_node.type_enum, defer=(), levels=None)
            fetched_nodes.append(fetched)

        with_v_nodes = jtu.tree_unflatten(struct, fetched_nodes)
        return eqx.combine(without_v_nodes, with_v_nodes, is_leaf=_is_leaf)
    assert 0, "Unreachable."
