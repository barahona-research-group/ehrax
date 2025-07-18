"""Miscalleneous utility functions."""

import contextlib
import json
import os
import zipfile
from datetime import datetime
from types import ModuleType
from typing import Optional, Callable, TypeVar, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.tree_util import KeyEntry, GetAttrKey, SequenceKey, DictKey, FlattenedIndexKey
from jax.tree_util import tree_flatten, tree_map, tree_leaves
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

ArrayTypes = (np.ndarray, jnp.ndarray, jax.Array)
Array = np.ndarray | jnp.ndarray | jax.Array


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


def params_size(m: eqx.Module) -> int:
    """
    Parameters:
        m (PyTree object) - The PyTree object to get the parameter count of.

    Returns:
        int - The total number of parameters in the PyTree object m.
    """
    leaves, _ = tree_flatten(eqx.filter(m, filter_spec=lambda x: eqx.is_inexact_array(x)))  # type: ignore
    return sum(jnp.size(x) for x in leaves)


def tree_hasnan(m: eqx.Module) -> bool:
    """Check if a PyTree contains any NaN values.

    Parameters:
        m (PyTree) - The PyTree to check for NaN values.

    Returns:
        bool - True if m contains any NaN values, False otherwise.
    """
    t = eqx.filter(m, eqx.is_inexact_array)
    return any(map(lambda x: jnp.any(jnp.isnan(x)), tree_leaves(t)))


def tree_lognan(m: eqx.Module) -> eqx.Module:
    """Returns a PyTree object with the same structure of the
    input (PyTree object). For each leaf (i.e. jax.numpy object) in
    the input, assign True if that leaf has NaN value(s).
    """
    t = eqx.filter(m, eqx.is_inexact_array)
    return tree_map(lambda x: jnp.any(jnp.isnan(x)).item(), t)


def tree_add_scalar_mul(tree_x: eqx.Module, scalar: float, tree_y: eqx.Module) -> eqx.Module:
    """Compute tree_x + scalar * tree_y."""
    tree_x = eqx.filter(tree_x, eqx.is_inexact_array)
    tree_y = eqx.filter(tree_y, eqx.is_inexact_array)
    return tree_map(lambda x, y: x + scalar * y, tree_x, tree_y)


T = TypeVar('T')


def model_params_scaler(model: T, scaler: float, filter_spec: Callable[[eqx.Module], bool]) -> T:
    """Scale the parameters in a model by a given scaler.

    This scales the parameters (as selected by filter_spec) of the model
    by the given scaler value, while keeping the functional part of the
    model unchanged.

    Parameters:
        model: The model PyTree object.
        scaler: The scalar value to multiply the parameters by.
        filter_spec: The filter specification to select parameters.

    Returns:
        The model PyTree with parameters scaled.
    """
    func_model = eqx.filter(model, filter_spec, inverse=True)
    prms_model = eqx.filter(model, filter_spec, inverse=False)
    return eqx.combine(func_model, tree_scalar(prms_model, scaler))


def tree_scalar(tree: eqx.Module, scalar: float) -> eqx.Module:
    """Multiply all leaf nodes in a PyTree by a scalar value.

    Parameters:
        tree: PyTree object to multiply.
        scalar: Scalar value to multiply.

    Returns:
        PyTree object with all leaf nodes multiplied by the scalar.
    """

    return tree_map(lambda x: scalar * x, tree)  # type: ignore


def array_hasnan(arr: jnp.ndarray) -> bool:
    """Check if a jax numpy array contains any NaN or infinite values.

    Parameters:
        arr: jax numpy array to check.

    Returns: 
        bool: True if arr contains any NaN or infinite values.
    """
    return jnp.any(jnp.isnan(arr) | jnp.isinf(arr)).item()


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


def append_params_to_zip(model: eqx.Module, params_name: str, zipfile_fname: str):
    """Append model parameters to a zip file.

    Appends the parameters from `model` to the zip file at `zipfile_fname`
    under the name `params_name`, using `tree_serialise_leaves` to serialize
    the parameters.

    Parameters:
        model: Model whose parameters to serialize.
        params_name: Name to save parameters under in the zip file.
        zipfile_fname: Path to zip file to append parameters to.
    """
    with zipfile.ZipFile(
            translate_path(zipfile_fname), compression=zipfile.ZIP_STORED, mode="a"
    ) as archive:
        # create a ZipInfo model with date_time.
        file_info = zipfile.ZipInfo(
            filename=params_name,
            date_time=datetime.now().timetuple()[:6],
            # for demo purpose. may need carefully examine timezone in case of practice.
        )

        # important: explicitly set compress_type here to sync with ZipFile,
        # otherwise a bad default ZIP_STORED will be used.
        file_info.compress_type = archive.compression
        with archive.open(file_info, "w") as zip_member:
            eqx.tree_serialise_leaves(zip_member, model)  # type: ignore


def zip_members(zipfile_fname: str) -> list[str]:
    """Return a list of the names of members in the ZIP file.

    Args:
        zipfile_fname: Path to the ZIP file.

    Returns:
        A list of member names in the ZIP file.
    """
    with zipfile.ZipFile(translate_path(zipfile_fname)) as archive:
        return archive.namelist()


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


"""NumpyEncoder class to handle encoding numpy arrays to JSON."""


def write_config(data, config_file):
    """Write the given data object to the specified config file as JSON.

    Parameters:
        data: The data object to serialize to JSON and write.
        config_file: The path to the config file to write.
    """
    with open(translate_path(config_file), "w") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True, cls=NumpyEncoder)


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Copy from: https://stackoverflow.com/a/34333710
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


def path_from_getter(getter: Callable[[Any], Any]) -> list[str]:
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
                return _M(object.__getattribute__(self, '_x_path') + [item])

        def __getitem__(self, item: str):
            return _M(object.__getattribute__(self, '_x_path') + [str(item)])

    return getter(_M([]))._x_path


def path_from_jax_keypath(path: tuple[KeyEntry, ...]) -> list[str]:
    def _extract(entry: KeyEntry):
        match entry:
            case GetAttrKey(name):
                return name
            case SequenceKey(idx):
                return str(idx)
            case DictKey(key):
                return str(key)
            case FlattenedIndexKey(key):
                return str(key)
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

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
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
