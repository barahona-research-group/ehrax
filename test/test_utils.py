import ehrax as rx
import pytest
from jax._src.tree_util import DictKey, GetAttrKey, SequenceKey


def test_path_from_jax():
    kpath = (SequenceKey(4), DictKey(5), DictKey("6"), DictKey("x"), GetAttrKey("as"))
    assert rx.path_from_jax_keypath(kpath) == ["4", "5", "6", "x", "as"]


@pytest.mark.parametrize(
    "lambda_, expected",
    [
        (lambda x: x, []),
        (lambda x: x.y, ["y"]),
        (lambda x: x[100], ["100"]),
        (lambda x: x.s.m.n[34].at["d"], ["s", "m", "n", "34", "at", "d"]),
    ],
)
def test_path_from_getter(lambda_, expected):
    assert rx.path_from_getter(lambda_) == expected
