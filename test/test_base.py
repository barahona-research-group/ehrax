from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import tables as tb

from ehrax.base import AbstractModule, _factory_registry, AbstractConfig, AbstractWithDataframeEquivalent, \
    AbstractWithSeriesEquivalent, AbstractVxData


# (A) Test ModuleMeta and AbstractModule
class Module(AbstractModule):
    att1: Any
    att2: Any

    def __init__(self, att1: Any, att2: Any):
        self.att1 = att1
        self.att2 = att2


class TestAbstractModule:
    def test_module_registration(self):
        assert Module in _factory_registry.values()
        assert Module.__class_key__() == f"{self.__module__}.Module"
        assert f"{self.__module__}.Module" in _factory_registry.keys()
        assert _factory_registry[Module.__class_key__()] == Module


# (B) Test Config
class Config(AbstractConfig):
    att1: Any
    att2: Any

    def __init__(self, att1: Any, att2: Any):
        self.att1 = att1
        self.att2 = att2


class TestAbstractConfig:
    def test_equality(self):
        assert Config(3, 5) == Config(3, 5)
        assert Config(-1, 2) != Config(0, 2)
        assert Config(Config(10, 5), {'k': 45}) == Config(Config(10, 5), dict(k=45))
        assert Config(Config(10, 5), {'k': 45}) != Config(Config(10, 5), dict(j=45))
        assert Config(Config(10, 5), {'k': 45}) != Config(Config(10, 0), dict(k=45))
        assert Config(Config(0, 0), Config(1, 1)) != Config(Config(0, 0), 5)

    def test_as_dict(self):
        assert Config(3, 5).as_dict() == {'att1': 3, 'att2': 5}
        assert Config(-1, 2).as_dict() != {'att1': 0, 'att2': 2}
        assert Config(Config(10, 5), {'k': 45}).as_dict() == {'att1': {'att1': 10, 'att2': 5},
                                                              'att2': {'k': 45}}
        assert Config(Config(10, 5), {'k': 45}).as_dict() != {'att1': {'att1': 10, 'att2': 5},
                                                              'att2': {'j': 45}}
        assert Config(Config(10, 5), {'k': 45}).as_dict() != {'att1': {'att1': 10, 'att2': 0},
                                                              'att2': {'j': 45}}
        assert Config(Config(0, 0), Config(1, 1)).as_dict() != {'att1': {'att1': 0}, 'att2': 5}


# (C) Test AbstractWithPandasEquivalent

class WithDataframeEquivalent(AbstractWithDataframeEquivalent):
    a: dict[str, tuple[int, str, float]]

    def __init__(self, a: dict[str, tuple[int, str, float]]):
        self.a = a

    def to_dataframe(self):
        if len(self.a) == 0:
            return pd.DataFrame(columns=['c1', 'c2', 'c3'])

        return pd.DataFrame(data=list(self.a.values()), columns=['c1', 'c2', 'c3'],
                            index=pd.Index(list(self.a.keys()), dtype='str', name='theindex')).astype(
            {'c1': 'int', 'c2': 'str', 'c3': 'float'})

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        vals = zip(df.c1, df.c2, df.c3)
        return cls(dict(zip(df.index, vals)))


class WithSeriesEquivalent(AbstractWithSeriesEquivalent):
    a: dict[str, float]

    def __init__(self, a: Any):
        self.a = a

    def to_series(self):
        return pd.Series(list(self.a.values()), index=pd.Index(list(self.a.keys()), name='indexooo', dtype='str'),
                         dtype=float, name='7lw')

    @classmethod
    def from_series(cls, sr: pd.Series):
        return cls(sr.to_dict())


class TestAbstractWithPandasEquivalent:

    @pytest.fixture(scope='class', params=[
        {'pressure': 1.0, 'temperature': 2.0, 'humidity': 3.0},
        {'a': 0},
        {},
    ])
    def a_proper_sr_eq(self, request):
        return WithSeriesEquivalent(request.param)

    @pytest.fixture(scope='class', params=[
        {'mondy': (0, 'work', 1.0), 'teuesday': (1, 'cook', 2.0), 'wednesday': (3, 'study', 0.0)},
        {'sunday': (25, 'run', 8.0)},
        {},
    ])
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
        with tb.open_file(filename, mode='r') as hf5:
            return WithDataframeEquivalent.from_hdf_group(hf5.get_node(group_path))

    @pytest.fixture
    def deserialized_sr_eq(self, a_proper_sr_eq: WithSeriesEquivalent, hf5_group_writer: tb.Group):
        a_proper_sr_eq.to_hdf_group(hf5_group_writer)
        hf5 = hf5_group_writer._v_file
        filename = hf5.filename
        group_path = hf5_group_writer._v_pathname
        hf5.close()
        with tb.open_file(filename, mode='r') as hf5:
            return WithSeriesEquivalent.from_hdf_group(hf5.get_node(group_path))

    def test_hdf_df_serialization(self, a_proper_df_eq: WithDataframeEquivalent,
                                  deserialized_df_eq: WithDataframeEquivalent):
        assert a_proper_df_eq == deserialized_df_eq
        assert a_proper_df_eq.to_dataframe().equals(deserialized_df_eq.to_dataframe())

    def test_hdf_series_serialization(self, a_proper_sr_eq: WithSeriesEquivalent,
                                      deserialized_sr_eq: WithSeriesEquivalent):
        assert a_proper_sr_eq == deserialized_sr_eq
        assert a_proper_sr_eq.to_series().equals(deserialized_sr_eq.to_series())


# (D) Test AbstractVxData

class VxData(AbstractVxData):
    a: Any
    b: Any
    c: Any

    def __init__(self, a: Any, b: Any, c: Any):
        self.a = a
        self.b = b
        self.c = c


class TestVxData:
    @pytest.fixture(scope='class', params=[])
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        assert False, "Should be subclassed."

    @pytest.fixture
    def deserialized_pair(self, vxdata_pair: tuple[VxData, VxData],
                          hf5_writer_file: tb.File) -> tuple[VxData, VxData]:
        a, b = vxdata_pair
        a.to_hdf_group(hf5_writer_file.create_group("/", 'a'))
        b.to_hdf_group(hf5_writer_file.create_group("/", 'b'))
        filename = hf5_writer_file.filename
        hf5_writer_file.close()
        with tb.open_file(filename, mode='r') as hf5_reader:
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
    @pytest.fixture(scope='class', params=[
        # Case 1
        [(None, None, None),
         (None, None, 0)],
        # Case 2
        [(1, 2, 3),
         (1, 2, 2)],
        # Case 3
        [(1.0, 2.0, 3.0),
         (1.0, 2.0, 3.01)],
        # Case 4
        [('a', 'b', 'c'),
         ('a', 'b', 'c_')],
        # Case 5
        [(True, False, True),
         (True, False, 1)],
        # Case 6
        [(1, 2.0, False),
         (1, 2.0, 0)],
        # Case 7
        [(1.0, 'a', pd.Timestamp(5)),
         (1.0, 'a', pd.Timestamp(4))],
    ])
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        return VxData(*request.param[0]), VxData(*request.param[1])


class TestVxDataWithCollectionTypes(TestVxData):
    @pytest.fixture(scope='class', params=[
        [((4, 3), ['aa', 'bb'], {3.0, 5.0}),
         ((4, 3), ['aa', 'bb'], {3.0, 5.1})],

        [({5: {}}, {'a': ()}, {'z': []}),
         ({5: []}, {'a': ()}, {'z': []})]
    ])
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        return VxData(*request.param[0]), VxData(*request.param[1])


class TestVxDataWithNumpy(TestVxData):
    @pytest.fixture(scope='class', params=[
        # Case 1: element values
        [(np.arange(10), np.array([]), np.array(-1)),
         (np.arange(10), np.array([]), np.array(0))],

        # Case 2: dtype
        [(np.arange(10), np.array([], dtype=float), np.array(0)),
         (np.arange(10), np.array([], dtype=int), np.array(0))],

        # Case 3: extra dim of 1
        [(np.arange(10), np.array([]), np.array([0])),
         (np.arange(10), np.array([]), np.array(0))],

        # Case 4: partly nan
        [(np.arange(10), np.array([float('nan'), 0.0]), np.array(0)),
         (np.arange(10), np.array([0.0, 0.0]), np.array(0))],
    ])
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        return VxData(*request.param[0]), VxData(*request.param[1])


class TestVxDataWithPandas(TestVxData):
    @pytest.fixture(scope='class', params=[
        # Case 1: element values in dataframe
        [(pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series([32])),
         (pd.DataFrame([1, 3]), pd.DataFrame(), pd.Series([32]))],

        # Case 2: element value in series
        [(pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series([32])),
         (pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series([34]))],

        # Case 3: columns in empty dataframe
        [(pd.DataFrame([1, 2]), pd.DataFrame(columns=['x']), pd.Series([32])),
         (pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series([32]))],

        # Case 4: name in empty series
        [(pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series()),
         (pd.DataFrame([1, 2]), pd.DataFrame(), pd.Series(name='x'))],
    ])
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        return VxData(*request.param[0]), VxData(*request.param[1])


class TestVxDataWithNesting(TestVxData):
    @pytest.fixture(scope='class', params=[
        # Case 1
        [({'a': 1, 'b': 2},  # --------------------------------------_------------------(!)--- <- the only change is 5
          {'c': VxData(2.0, VxData(np.arange(100), pd.Timestamp(4), [pd.Series(np.arange(5))]), {34: 'x'}),
           'd': VxData(None, pd.Timestamp(10), [Config(Config(6, None), False)])},
          {pd.Timestamp(0), pd.Timestamp(100)}),
         ({'a': 1, 'b': 2},  # ---------------------------------------------------------(!)--- <- changed to 4
          {'c': VxData(2.0, VxData(np.arange(100), pd.Timestamp(4), [pd.Series(np.arange(4))]), {34: 'x'}),
           'd': VxData(None, pd.Timestamp(10), [Config(Config(6, None), False)])},
          {pd.Timestamp(0), pd.Timestamp(100)})
         ],
        # Case 2
        # array, series, tuple[array, ...] ----------(!)-
        [(np.arange(29), pd.Series([4, 2]), (np.array([]), np.array([5.0]))),
         (np.arange(29), pd.Series([4, 2]), (np.array(5), np.array([5.0])))],

        # Case 3
        [(pd.DataFrame(columns=['x', 'y']), pd.DataFrame([5, 1]), MappingProxyType(
            # -----------------------------------------------------------------------------------------------(!)
            {1: pd.Series([], pd.Index([], dtype='bool', name='indexooo'), dtype='bool', name='veryboolseries')})),
         (pd.DataFrame(columns=['x', 'y']), pd.DataFrame([5, 1]), MappingProxyType(
             # ------------------------------------------------------------------------------------------------(!)
             {1: pd.Series([], pd.Index([], dtype='bool', name='indexooo'), dtype='bool', name='--------------')}))
         ]

    ])
    def vxdata_pair(self, request) -> tuple[VxData, VxData]:
        return VxData(*request.param[0]), VxData(*request.param[1])


# Test lazy-loading...