from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import ehrax as rx
import equinox as eqx
import numpy as np
import pandas as pd
import pytest
import tables as tb
from ehrax.base import MAX_SEGMENT_SIZE
from tables import PerformanceWarning


# (A) Test ModuleMeta and AbstractModule
class Module(rx.AbstractModule):
    att1: Any
    att2: Any

    def __init__(self, att1: Any, att2: Any):
        self.att1 = att1
        self.att2 = att2


class TestAbstractModule:
    def test_module_registration(self):
        assert Module in rx.base._factory_registry.values()
        assert Module.__class_key__() == f"{self.__module__}.Module"
        assert f"{self.__module__}.Module" in rx.base._factory_registry.keys()
        assert rx.base._factory_registry[Module.__class_key__()] == Module


# (B) Test Config
class Config(rx.AbstractConfig):
    att1: Any
    att2: Any

    def __init__(self, att1: Any, att2: Any):
        self.att1 = att1
        self.att2 = att2


class TestAbstractConfig:
    def test_equality(self):
        assert Config(3, 5) == Config(3, 5)
        assert Config(-1, 2) != Config(0, 2)
        assert Config(Config(10, 5), {"k": 45}) == Config(Config(10, 5), dict(k=45))
        assert Config(Config(10, 5), {"k": 45}) != Config(Config(10, 5), dict(j=45))
        assert Config(Config(10, 5), {"k": 45}) != Config(Config(10, 0), dict(k=45))
        assert Config(Config(0, 0), Config(1, 1)) != Config(Config(0, 0), 5)

    def test_as_dict(self):
        assert Config(3, 5).as_dict() == {"att1": 3, "att2": 5}
        assert Config(-1, 2).as_dict() != {"att1": 0, "att2": 2}
        assert Config(Config(10, 5), {"k": 45}).as_dict() == {"att1": {"att1": 10, "att2": 5}, "att2": {"k": 45}}
        assert Config(Config(10, 5), {"k": 45}).as_dict() != {"att1": {"att1": 10, "att2": 5}, "att2": {"j": 45}}
        assert Config(Config(10, 5), {"k": 45}).as_dict() != {"att1": {"att1": 10, "att2": 0}, "att2": {"j": 45}}
        assert Config(Config(0, 0), Config(1, 1)).as_dict() != {"att1": {"att1": 0}, "att2": 5}


# (C) Test AbstractWithPandasEquivalent


class WithDataframeEquivalent(rx.AbstractWithDataframeEquivalent):
    a: dict[str, tuple[int, str, float]]

    def __init__(self, a: dict[str, tuple[int, str, float]]):
        self.a = a

    def to_dataframe(self):
        if len(self.a) == 0:
            return pd.DataFrame(columns=["c1", "c2", "c3"])

        return pd.DataFrame(
            data=list(self.a.values()),
            columns=["c1", "c2", "c3"],
            index=pd.Index(list(self.a.keys()), dtype="str", name="theindex"),
        ).astype({"c1": "int", "c2": "str", "c3": "float"})

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        vals = zip(df.c1, df.c2, df.c3)
        return cls(dict(zip(df.index, vals)))


class WithSeriesEquivalent(rx.AbstractWithSeriesEquivalent):
    a: dict[str, float]

    def __init__(self, a: Any):
        self.a = a

    def to_series(self):
        return pd.Series(
            list(self.a.values()),
            index=pd.Index(list(self.a.keys()), name="indexooo", dtype="str"),
            dtype=float,
            name="7lw",
        )

    @classmethod
    def from_series(cls, sr: pd.Series):
        return cls(sr.to_dict())


class TestAbstractWithPandasEquivalent:
    @pytest.fixture(
        scope="class",
        params=[
            {"pressure": 1.0, "temperature": 2.0, "humidity": 3.0},
            {"a": 0},
            {},
        ],
    )
    def a_proper_sr_eq(self, request):
        return WithSeriesEquivalent(request.param)

    @pytest.fixture(
        scope="class",
        params=[
            {"mondy": (0, "work", 1.0), "teuesday": (1, "cook", 2.0), "wednesday": (3, "study", 0.0)},
            {"sunday": (25, "run", 8.0)},
            {},
        ],
    )
    def a_proper_df_eq(self, request):
        return WithDataframeEquivalent(request.param)

    def test_dataframe_equivalence(self, a_proper_df_eq: WithDataframeEquivalent):
        assert WithDataframeEquivalent.from_dataframe(a_proper_df_eq.to_dataframe()) == a_proper_df_eq

    def test_series_equivalence(self, a_proper_sr_eq: WithSeriesEquivalent):
        to_sr = a_proper_sr_eq.to_series()
        from_sr = WithSeriesEquivalent.from_series(to_sr)
        assert from_sr == a_proper_sr_eq
        assert to_sr.to_dict() == from_sr.to_series().to_dict()

    @pytest.fixture
    def deserialized_df_eq(self, a_proper_df_eq: WithDataframeEquivalent, hf5_group_writer: tb.Group):
        a_proper_df_eq.to_hdf_group(hf5_group_writer)
        hf5 = hf5_group_writer._v_file
        filename = hf5.filename
        group_path = hf5_group_writer._v_pathname
        hf5.close()
        with tb.open_file(filename, mode="r") as hf5:
            return WithDataframeEquivalent.from_hdf_group(hf5.get_node(group_path))

    @pytest.fixture
    def deserialized_sr_eq(self, a_proper_sr_eq: WithSeriesEquivalent, hf5_group_writer: tb.Group):
        a_proper_sr_eq.to_hdf_group(hf5_group_writer)
        hf5 = hf5_group_writer._v_file
        filename = hf5.filename
        group_path = hf5_group_writer._v_pathname
        hf5.close()
        with tb.open_file(filename, mode="r") as hf5:
            return WithSeriesEquivalent.from_hdf_group(hf5.get_node(group_path))

    def test_hdf_df_serialization(
        self, a_proper_df_eq: WithDataframeEquivalent, deserialized_df_eq: WithDataframeEquivalent
    ):
        assert a_proper_df_eq == deserialized_df_eq
        assert a_proper_df_eq.to_dataframe().equals(deserialized_df_eq.to_dataframe())

    def test_hdf_series_serialization(
        self, a_proper_sr_eq: WithSeriesEquivalent, deserialized_sr_eq: WithSeriesEquivalent
    ):
        assert a_proper_sr_eq == deserialized_sr_eq
        assert a_proper_sr_eq.to_series().equals(deserialized_sr_eq.to_series())


# (D) Test  rx.AbstractVxData


class VxData(rx.AbstractVxData):
    a: Any
    b: Any
    c: Any

    def __init__(self, a: Any, b: Any, c: Any):
        self.a = a
        self.b = b
        self.c = c


class TestVxData:
    @pytest.fixture(scope="class", params=[])
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        assert False, "Should be subclassed."

    @pytest.fixture
    def deserialized_pair(self, vxdata_pair: tuple[VxData, VxData], hf5_writer_file: tb.File) -> tuple[VxData, VxData]:
        a, b = vxdata_pair
        a.to_hdf_group(hf5_writer_file.create_group("/", "a"))
        b.to_hdf_group(hf5_writer_file.create_group("/", "b"))
        filename = hf5_writer_file.filename
        hf5_writer_file.close()
        with tb.open_file(filename, mode="r") as hf5_reader:
            _a = VxData.from_hdf_group(hf5_reader.get_node("/", "a"))
            _b = VxData.from_hdf_group(hf5_reader.get_node("/", "b"))
            return _a, _b

    def test_vxdata_equalities(self, vxdata_pair: tuple[VxData, VxData]):
        a, b = vxdata_pair
        assert not a.equals(b)

    def test_vxdata_serialization(self, vxdata_pair: tuple[VxData, VxData], deserialized_pair: tuple[VxData, VxData]):
        a, b = vxdata_pair
        a_, b_ = deserialized_pair
        assert a.equals(a_)
        assert b.equals(b_)
        assert not a_.equals(b_)


class TestVxDataWithPlainTypes(TestVxData):
    # (D.2) Test serialization/deserialization
    #       * with numpy
    #       * with pandas
    #       * with plain data types
    #       * with panda timestamp
    #       * with iterables
    #       * with maps
    #       * disk io
    @pytest.fixture(
        scope="class",
        params=[
            # Case 1
            [(None, None, None), (None, None, 0)],
            # Case 2
            [(1, 2, 3), (1, 2, 2)],
            # Case 3
            [(1.0, 2.0, 3.0), (1.0, 2.0, 3.01)],
            # Case 4
            [("a", "b", "c"), ("a", "b", "c_")],
            # Case 5
            [(True, False, True), (True, False, 1)],
            # Case 6
            [(1, 2.0, False), (1, 2.0, 0)],
            # Case 7
            [(1.0, "a", pd.Timestamp(5)), (1.0, "a", pd.Timestamp(4))],
        ],
    )
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        return VxData(*request.param[0]), VxData(*request.param[1])


class TestVxDataWithCollectionTypes(TestVxData):
    @pytest.fixture(
        scope="class",
        params=[
            [((4, 3), ["aa", "bb"], {3.0, 5.0}), ((4, 3), ["aa", "bb"], {3.0, 5.1})],
            [({5: {}}, {"a": ()}, {"z": []}), ({5: []}, {"a": ()}, {"z": []})],
        ],
    )
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        return VxData(*request.param[0]), VxData(*request.param[1])


class TestVxDataWithNumpy(TestVxData):
    @pytest.fixture(
        scope="class",
        params=[
            # Case 1: element values
            [(np.arange(10), np.array([]), np.array(-1)), (np.arange(10), np.array([]), np.array(0))],
            # Case 2: dtype
            [
                (np.arange(10), np.array([], dtype=float), np.array(0)),
                (np.arange(10), np.array([], dtype=int), np.array(0)),
            ],
            # Case 3: extra dim of 1
            [(np.arange(10), np.array([]), np.array([0])), (np.arange(10), np.array([]), np.array(0))],
            # Case 4: partly nan
            [
                (np.arange(10), np.array([float("nan"), 0.0]), np.array(0)),
                (np.arange(10), np.array([0.0, 0.0]), np.array(0)),
            ],
        ],
    )
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        return VxData(*request.param[0]), VxData(*request.param[1])


class TestVxDataWithPandas(TestVxData):
    @pytest.fixture(
        scope="class",
        params=[
            # Case 1: element values in dataframe
            [
                (pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series([32])),
                (pd.DataFrame([1, 3]), pd.DataFrame(), pd.Series([32])),
            ],
            # Case 2: element value in series
            [
                (pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series([32])),
                (pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series([34])),
            ],
            # Case 3: columns in empty dataframe
            [
                (pd.DataFrame([1, 2]), pd.DataFrame(columns=["x"]), pd.Series([32])),
                (pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series([32])),
            ],
            # Case 4: name in empty series
            [
                (pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series()),
                (pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series(name="x")),
            ],
        ],
    )
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        return VxData(*request.param[0]), VxData(*request.param[1])


class TestVxDataWithNesting(TestVxData):
    @pytest.fixture(
        scope="class",
        params=[
            # Case 1
            [
                (
                    {
                        "a": 1,
                        "b": 2,
                    },  # --------------------------------------_------------------(!)--- <- the only change is 5
                    {
                        "c": VxData(2.0, VxData(np.arange(100), pd.Timestamp(4), [pd.Series(np.arange(5))]), {34: "x"}),
                        "d": VxData(None, pd.Timestamp(10), [Config(Config(6, None), False)]),
                    },
                    {pd.Timestamp(0), pd.Timestamp(100)},
                ),
                (
                    {"a": 1, "b": 2},
                    # ---------------------------------------------------------(!)--- <- changed to 4
                    {
                        "c": VxData(2.0, VxData(np.arange(100), pd.Timestamp(4), [pd.Series(np.arange(4))]), {34: "x"}),
                        "d": VxData(None, pd.Timestamp(10), [Config(Config(6, None), False)]),
                    },
                    {pd.Timestamp(0), pd.Timestamp(100)},
                ),
            ],
            # Case 2
            # array, series, tuple[array, ...] ----------(!)-
            [
                (np.arange(29), pd.Series([4, 2]), (np.array([]), np.array([5.0]))),
                (np.arange(29), pd.Series([4, 2]), (np.array(5), np.array([5.0]))),
            ],
            # Case 3
            [
                (
                    pd.DataFrame(columns=["x", "y"]),
                    pd.DataFrame([5, 1]),
                    MappingProxyType(
                        # -------------------------------------------------------------------------------(!)
                        {
                            1: pd.Series(
                                [], pd.Index([], dtype="bool", name="indexooo"), dtype="bool", name="veryboolseries"
                            )
                        }
                    ),
                ),
                (
                    pd.DataFrame(columns=["x", "y"]),
                    pd.DataFrame([5, 1]),
                    MappingProxyType(
                        # --------------------------------------------------------------------------------(!)
                        {
                            1: pd.Series(
                                [], pd.Index([], dtype="bool", name="indexooo"), dtype="bool", name="--------------"
                            )
                        }
                    ),
                ),
            ],
        ],
    )
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        return VxData(*request.param[0]), VxData(*request.param[1])


# Test lazy-loading...


class TestHDFVirtualNode:
    @pytest.fixture(
        scope="class",
        params=[
            ("x", "y", None),  # Cannot contain None
            (VxData(None, None, None), "y", "z"),  # Cannot contain non-plain types
            (rx.HDFVirtualNode("x", "y", "z"), "y", "z"),  # Cannot contain a virtual node
        ],
    )
    def invalid_args(self, request) -> tuple[Any, Any, Any]:
        return request.param

    def test_invalid_args(self, invalid_args):
        with pytest.raises(AssertionError):
            rx.HDFVirtualNode(*invalid_args)

    @pytest.fixture(scope="class")
    def dummy_vnode(self):
        return rx.HDFVirtualNode("x", "y", "z")

    def test_attribute_access(self, dummy_vnode):
        assert dummy_vnode.filename == "x"
        assert dummy_vnode.parent_path == "y"
        assert dummy_vnode.type_enum == "z"
        with pytest.raises(AttributeError):
            _ = dummy_vnode.foo

    def test_equality(self, dummy_vnode):
        with pytest.raises(ValueError, match="You are trying to test equality with a virtual unfetched node"):
            _ = dummy_vnode.equals(dummy_vnode)

    def test_to_hdf(self, dummy_vnode):
        with pytest.raises(ValueError, match="You are trying to serialize an unfetched node"):
            dummy_vnode.to_hdf_group(None)

    def test_from_hdf(self):
        with pytest.raises(ValueError, match="You are trying to deserialize"):
            _ = rx.HDFVirtualNode.from_hdf_group(None)


COMPLETE_VX_DATA = VxData(
    a={"a": 1, "b": VxData(None, 1, "x")},
    b={
        "c": VxData(2.0, VxData(np.arange(100), pd.Timestamp(4), [pd.Series(np.arange(4))]), {34: "x"}),
        "d": VxData(None, pd.Timestamp(10), [Config(Config(6, None), False)]),
    },
    c={pd.Timestamp(0), pd.Timestamp(100), "yyy"},
)

GETTER_NODE_PAIR = (
    # a pair of getter lambda, and the expected node from `complete_vx_data` to be returned.
    (lambda x: x.a, dict(a=1, b=VxData(None, 1, "x"))),
    (lambda x: x.b["c"].b, VxData(np.arange(100), pd.Timestamp(4), [pd.Series(np.arange(4))])),
    (lambda x: x.b["d"], VxData(None, pd.Timestamp(10), [Config(Config(6, None), False)])),
    (lambda x: x.c, {pd.Timestamp(0), pd.Timestamp(100), "yyy"}),
)


def _cmp(a: Any, b: Any):
    if hasattr(a, "equals"):
        return a.equals(b)
    else:
        return a == b


class TestLazyLoading:
    @pytest.fixture(scope="class")
    def complete_vx_data(self) -> VxData:
        return COMPLETE_VX_DATA

    @pytest.fixture(scope="class", params=GETTER_NODE_PAIR)
    def getter_node_pair(self, request, complete_vx_data: VxData) -> tuple[Callable[[VxData], Any], Any]:
        # ensure we are writing the right getters.
        getter, node = request.param
        assert _cmp(node, getter(complete_vx_data))
        return getter, node

    @pytest.fixture(scope="class")
    def pruned_vx_data(self, complete_vx_data: VxData, getter_node_pair: tuple[Callable[[VxData], Any], Any]) -> VxData:
        getter, _ = getter_node_pair
        return eqx.tree_at(getter, complete_vx_data, rx.HDFVirtualNode("x", "y", "z"))

    @pytest.fixture
    def hdf_serialized_vxdata(self, complete_vx_data: VxData, tmp_path_factory) -> str:
        filename = tmp_path_factory.mktemp("vxdata").joinpath("vxdata.h5")
        complete_vx_data.save(filename, complevel=0)
        loaded = VxData.load(filename)
        assert complete_vx_data.equals(loaded)
        return str(filename)

    @pytest.fixture
    def hdf_deserialized_deferred_vxdata(
        self, hdf_serialized_vxdata: str, getter_node_pair: tuple[Callable[[VxData], Any], Any]
    ) -> VxData:
        getter, _ = getter_node_pair
        return VxData.load(hdf_serialized_vxdata, defer=(getter,))

    @pytest.fixture
    def hdf_deserialized_deferred_vxdata2(self, hdf_serialized_vxdata: str) -> VxData:
        getters, _ = zip(*GETTER_NODE_PAIR)
        # No all four nodes deferred at once.
        return VxData.load(hdf_serialized_vxdata, defer=getters)

    @pytest.fixture
    def hdf_deserialized_fetched_at_vxdata(
        self, hdf_deserialized_deferred_vxdata: VxData, getter_node_pair: tuple[Callable[[VxData], Any], Any]
    ) -> VxData:
        getter, _ = getter_node_pair
        return rx.fetch_at(getter, hdf_deserialized_deferred_vxdata)

    @pytest.fixture
    def hdf_deserialized_one_level_fetched_at_vxdata(
        self, hdf_deserialized_deferred_vxdata: VxData, getter_node_pair: tuple[Callable[[VxData], Any], Any]
    ) -> VxData:
        getter, _ = getter_node_pair
        return rx.fetch_one_level_at(getter, hdf_deserialized_deferred_vxdata)

    @pytest.fixture
    def hdf_deserialized_fetched_at_vxdata2(self, hdf_deserialized_deferred_vxdata2: VxData) -> VxData:
        getters, _ = zip(*GETTER_NODE_PAIR)
        return rx.fetch_at(getters, hdf_deserialized_deferred_vxdata2)

    @pytest.fixture
    def hdf_deserialized_fetched_all_vxdata(self, hdf_deserialized_deferred_vxdata: VxData) -> VxData:
        return rx.fetch_all(hdf_deserialized_deferred_vxdata)

    @pytest.fixture
    def hdf_deserialized_fetched_all_vxdata2(self, hdf_deserialized_deferred_vxdata2: VxData) -> VxData:
        return rx.fetch_all(hdf_deserialized_deferred_vxdata2)

    def test_invalid_to_hdf(self, pruned_vx_data: VxData, hf5_group_writer: tb.Group):
        with pytest.raises(ValueError, match="You are trying to serialize an unfetched node"):
            pruned_vx_data.to_hdf_group(hf5_group_writer)

    def test_invalid_equality(self, pruned_vx_data: VxData, complete_vx_data: VxData):
        assert not pruned_vx_data.equals(complete_vx_data)
        assert not complete_vx_data.equals(pruned_vx_data)
        with pytest.raises(ValueError, match="You are trying to test equality with a virtual unfetched node"):
            _ = pruned_vx_data.equals(pruned_vx_data)

    def test_defer(
        self, hdf_deserialized_deferred_vxdata: VxData, getter_node_pair: tuple[Callable[[VxData], Any], Any]
    ):
        getter, _ = getter_node_pair
        assert isinstance(getter(hdf_deserialized_deferred_vxdata), rx.HDFVirtualNode)

    def _aux_test_fetch(
        self,
        hdf_deserialized_fetched_vxdata: VxData,
        getter_node_pair: tuple[Callable[[VxData], Any], Any],
        complete_vx_data: VxData,
    ):
        getter, node = getter_node_pair
        assert _cmp(getter(hdf_deserialized_fetched_vxdata), node)
        assert _cmp(getter(complete_vx_data), getter(hdf_deserialized_fetched_vxdata))
        assert complete_vx_data.equals(hdf_deserialized_fetched_vxdata)

    def test_fetch_at(
        self,
        hdf_deserialized_fetched_at_vxdata: VxData,
        getter_node_pair: tuple[Callable[[VxData], Any], Any],
        complete_vx_data: VxData,
    ):
        self._aux_test_fetch(hdf_deserialized_fetched_at_vxdata, getter_node_pair, complete_vx_data)

    def test_fetch_all(
        self,
        hdf_deserialized_fetched_all_vxdata: VxData,
        getter_node_pair: tuple[Callable[[VxData], Any], Any],
        complete_vx_data: VxData,
    ):
        self._aux_test_fetch(hdf_deserialized_fetched_all_vxdata, getter_node_pair, complete_vx_data)

    def test_fetch_one_level_at(
        self,
        complete_vx_data: VxData,
        hdf_deserialized_one_level_fetched_at_vxdata: VxData,
        getter_node_pair: tuple[Callable[[VxData], Any], Any],
    ):
        getter, node = getter_node_pair
        # Type is equal but contents are not
        assert type(getter(hdf_deserialized_one_level_fetched_at_vxdata)) is type(node)
        assert not _cmp(getter(hdf_deserialized_one_level_fetched_at_vxdata), node)
        # Children themselves must be virtual nodes.

        # if the virtual node represents a sequence containing plain types, then it will be loaded!
        get_immediate_leaves = (
            (lambda x: eqx.tree_flatten_one_level(x)[0]) if type(node) is not set else (lambda x: list(x))
        )
        if set(map(type, get_immediate_leaves(node))).issubset(rx.base.SERIES_GROUPED_ELEMENT_TYPES):
            assert all(
                isinstance(child, rx.HDFVirtualNode)
                for child in get_immediate_leaves(getter(hdf_deserialized_one_level_fetched_at_vxdata))
            )
            assert hdf_deserialized_one_level_fetched_at_vxdata.equals(complete_vx_data)
        else:
            assert not hdf_deserialized_one_level_fetched_at_vxdata.equals(complete_vx_data)

    def test_fetch_all_after_fetch_one_level_at(
        self, complete_vx_data: VxData, hdf_deserialized_one_level_fetched_at_vxdata: VxData
    ):
        all_fetched = rx.fetch_all(hdf_deserialized_one_level_fetched_at_vxdata)
        assert all_fetched.equals(complete_vx_data)

    def test_fetch_at2(
        self,
        hdf_deserialized_fetched_at_vxdata2: VxData,
        hdf_deserialized_fetched_all_vxdata2: VxData,
        complete_vx_data: VxData,
    ):
        assert hdf_deserialized_fetched_at_vxdata2.equals(complete_vx_data)
        assert hdf_deserialized_fetched_all_vxdata2.equals(complete_vx_data)
        assert hdf_deserialized_fetched_at_vxdata2.equals(hdf_deserialized_fetched_all_vxdata2)


class TestLargeDataSegmentation:
    BAD_PERFORMANCE_N_ATTRS = 5000
    BAD_PERFORMANCE_N_CHILDREN = 17000

    @pytest.fixture(
        scope="class",
        params=[
            MAX_SEGMENT_SIZE - 1,
            MAX_SEGMENT_SIZE,
            MAX_SEGMENT_SIZE + 1,
            max(BAD_PERFORMANCE_N_ATTRS, BAD_PERFORMANCE_N_CHILDREN),
        ],
    )
    def wide_data(self, request):
        n = request.param
        return VxData({5: [VxData(0, {}, [])] * n}, {"a": ()}, {i: VxData(0, 12, i) for i in range(n)})

    def test_ensure_correctness_of_attributes_warning_capture(self, hf5_group_writer: tb.Group):
        with pytest.warns(PerformanceWarning, match="maximum number of attributes"):
            for i in range(self.BAD_PERFORMANCE_N_ATTRS):
                hf5_group_writer._v_attrs[f"x{i}"] = i

    def test_ensure_correctness_of_children_warning_capture(self, hf5_writer_file: tb.File):
        with pytest.warns(PerformanceWarning, match="maximum number of children"):
            for i in range(self.BAD_PERFORMANCE_N_CHILDREN):
                hf5_writer_file.create_group(f"/x{i}/y", "z", createparents=True)

    @pytest.mark.filterwarnings("error:.*maximum number of.*")  # children or attributes
    def test_no_serialization_performance_warning(self, wide_data: VxData, tmp_path_factory):
        filename = tmp_path_factory.mktemp("wide").joinpath("data.h5")
        wide_data.save(filename, complevel=0)
        loaded = VxData.load(filename)
        assert wide_data.equals(loaded)

    def test_lazy_loading(self, wide_data: VxData, tmp_path_factory):
        filename = tmp_path_factory.mktemp("wide").joinpath("data.h5")
        wide_data.save(filename, complevel=0)
        # node
        loaded = VxData.load(filename, defer=(lambda x: x.c,), levels=(1,))
        assert wide_data.equals(rx.fetch_all(loaded))
