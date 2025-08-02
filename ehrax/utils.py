"""Miscalleneous utility functions."""

import json
import os
from types import ModuleType
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, KeyEntry, SequenceKey
from jaxlib._jax import ArrayImpl
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

ArrayTypes = (np.ndarray, jnp.ndarray, jax.Array, ArrayImpl)
Array = np.ndarray | jnp.ndarray | jax.Array | ArrayImpl


def _tqdm_backend():
    """Return the appropriate constructor of tqdm based on the executor
    interpreter, i.e. if it is running on a notebook or not."""
    try:
        ipy_str = str(type(get_ipython()))  # type: ignore
        if 'zmqshell' in ipy_str:
            return tqdm_notebook
    except NameError:
        pass
    return tqdm


tqdm_constructor = _tqdm_backend()


def translate_path(path: str, relative_to: Optional[str] = None):
    """Translate a filesystem path by replacing environment
    variables with their values. If relative_to is specified,
    and path is a relative path to nonexistent file, then the
    path is interpreted as relative to this path.

    Parameters:
        path: Filesystem path to translate.
        relative_to: if specified, and path is a relative path to nonexistent file, then the path is
            interpreted as relative to this path.
    Returns:
        Absolute, expanded, translated path.
    """
    assert isinstance(path, str), "Path must be a string."

    path = os.path.expandvars(os.path.expanduser(path))

    if relative_to is not None:
        relative_to = translate_path(relative_to)
    if relative_to is not None and not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(relative_to, path)

    return os.path.abspath(path)


def resources_path(*subdir: str) -> str:
    return str(os.path.join(os.path.dirname(__file__), "resources", *subdir))


def load_config(config_file: str, relative_to: Optional[str] = None):
    """Load a JSON file from `config_file`.

    Parameters:
        config_file: Path to the JSON configuration file to load.
        relative_to: if specified, and config_file is a relative path to nonexistent file, then the path is
            interpreted as relative to this path.

    Returns:
        The deserialized JSON contents of the configuration file.
    """
    config_file = translate_path(config_file, relative_to=relative_to)
    with open(config_file) as json_file:
        return json.load(json_file)


def write_config(data, config_file):
    """Write the given data object to the specified config file as JSON.

    Parameters:
        data: The data object to serialize to JSON and write.
        config_file: The path to the config file to write.
    """
    with open(translate_path(config_file), "w") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True, cls=NumpyEncoder)


def path_from_getter(getter: Callable[[Any], Any],
                     getattr_transform: Callable[[str], str] = lambda x: x,
                     getitem_transform: Callable[[Any], str] = lambda x: x) -> list[str]:
    """
    Generate a sequence of attribute names or indices (converted to strings) recording the sequence of access steps
    applied by the function on its input.
    
    !!! Example
    
    ```python
    path_from_getter(lambda x: x.y.z.a.b)
    # returns ['y', 'z', 'a', 'b']
    path_from_getter(lambda x: x["y"][4].money)
    # returns ['y', '4', 'money']
    ```
    """

    class _M:
        _x_path: list[str]

        def __init__(self, _x_path: list[str]):
            self._x_path = _x_path

        def __getattribute__(self, item: str):
            try:
                return object.__getattribute__(self, item)
            except AttributeError:
                return _M(object.__getattribute__(self, '_x_path') + [getattr_transform(item)])

        def __getitem__(self, item: str):
            return _M(object.__getattribute__(self, '_x_path') + [str(getitem_transform(item))])

    return getter(_M([]))._x_path


def path_from_jax_keypath(path: tuple[KeyEntry, ...],
                          getattr_transform: Callable[[str], str] = lambda x: x,
                          getitem_transform: Callable[[Any], str] = lambda x: x) -> list[str]:
    def _extract(entry: KeyEntry):
        match entry:
            case GetAttrKey(name):
                return getattr_transform(name)
            case SequenceKey(idx):
                return str(getitem_transform(idx))
            case DictKey(key):
                return str(getitem_transform(key))
            case FlattenedIndexKey(key):
                return str(getitem_transform(key))
            case _:
                raise ValueError(f"Unexpected key {entry}")

    return list(map(_extract, path))


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj: object) -> object:  # type: ignore
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def np_module(a: Array) -> ModuleType:  # [np, jnp]:
    if isinstance(a, np.ndarray):
        return np
    elif isinstance(a, jnp.ndarray):  # type: ignore
        return jnp
    else:
        raise TypeError(f"Unsupported array type {type(a)}.")


def equal_arrays(a: Array, b: Array) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    _np = np if isinstance(a, np.ndarray) else jnp
    if _np.size(a) == 0:
        return True
    is_nan = _np.isnan(a) & _np.isnan(b)
    return _np.array_equal(a[~is_nan], b[~is_nan], equal_nan=False)  # type: ignore
