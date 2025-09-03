"""Miscalleneous utility functions."""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, MutableMapping

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
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
        if "zmqshell" in ipy_str:
            return tqdm_notebook
    except NameError:
        pass
    return tqdm


tqdm_constructor = _tqdm_backend()


def translate_path(path: str, relative_to: str | None = None):
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


def load_config(config_file: str, relative_to: str | None = None):
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


def path_from_getter(
        getter: Callable[[Any], Any],
        getattr_transform: Callable[[str], str] = lambda x: x,
        getitem_transform: Callable[[Any], str] = lambda x: x,
) -> list[str]:
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
                return _M(object.__getattribute__(self, "_x_path") + [getattr_transform(item)])

        def __getitem__(self, item: str):
            return _M(object.__getattribute__(self, "_x_path") + [str(getitem_transform(item))])

    return getter(_M([]))._x_path


def path_from_jax_keypath(
        path: tuple[KeyEntry, ...],
        getattr_transform: Callable[[str], str] = lambda x: x,
        getitem_transform: Callable[[Any], str] = lambda x: x,
) -> list[str]:
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
    """Custom encoder for numpy data types"""

    def default(self, obj: object) -> object:  # type: ignore
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.isreal(obj):
            return float(obj)
        elif np.iscomplex(obj):
            return {"real": obj.real, "imag": obj.imag}

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


class DataFrameLogger(logging.LoggerAdapter):
    @property
    def extract_file_handler_names(self) -> tuple[str, str, str] | None:
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                # baseFilename is actually the absolute path.
                # https://github.com/python/cpython/blob/801cf3fcdd27d8b6dd0fdd3c39e6c996e2b2f7fa/Lib/logging/__init__.py#L1200
                file_title = Path(handler.baseFilename).stem
                file_suffix = Path(handler.baseFilename).suffix
                file_parent = str(Path(handler.baseFilename).parent)
                return file_parent, file_title, file_suffix
        return None

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, dict[str, Any]]:
        df, tag = kwargs.pop("dataframe"), kwargs.pop("tag", "")
        assert tuple(map(type, (msg, df, tag))) == (str, pd.DataFrame, str)
        if len(df) == 0:
            return msg, kwargs

        # timestamp representative and incremental id.
        timestamp = pd.Timestamp.now().strftime("%Y_%m_%dT_%H_%M_%S")
        if self.extra is None:
            self.extra = {}
        incremental_id = int(self.extra.get("incremental_id", 0)) + 1
        self.extra = dict(self.extra) | {"incremental_id": incremental_id}
        filehandler_names = self.extract_file_handler_names
        if filehandler_names is None:
            return (
                f"{msg}. Appendix report will not be saved to disk because "
                f"no single file handler found in the logger. "
                f"To store the appendix report, configure the logger {self.logger.name} "
                f"by either adding a FileHandler manually or call logging.basicConfig "
                f"with setting the filename argument."
            ), kwargs
        parent_dir, main_log_file, main_log_file_suffix = filehandler_names
        tag = tag.replace(".", "_")
        file_title = "_".join((main_log_file, tag, timestamp, f"{incremental_id:03d}"))
        file_path = Path(parent_dir, file_title).with_suffix(f"{main_log_file_suffix}.csv")
        df.to_csv(file_path)
        return (
            f"{msg}. Find the appendix report stored as a table of "
            f"columns {df.columns.tolist()} and {len(df)} rows at ({file_path})."
        ), kwargs


def attached_dataframe_logger(logger: logging.Logger, extra: dict[str, Any] | None = None) -> DataFrameLogger:
    if extra is None:
        extra = {}
    return DataFrameLogger(logger, extra)


dataframe_log = attached_dataframe_logger(logging.getLogger())
