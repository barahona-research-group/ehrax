from unittest import mock

import ehrax as rx
import pytest
import tables as tb
from ehrax import CodeMap, CodingScheme, CodingSchemesManager, FrozenDict1N
from ehrax.example_schemes.icd_ccs_integration import (
    CCSSchemeSelection,
    ICDSchemeSelection,
    OutcomeSelection,
    setup_ccs_schemes,
    setup_icd_schemes,
    setup_outcomes,
    setup_standard_icd_ccs,
)


class TestFlatScheme:
    @pytest.fixture(
        scope="class",
        params=[
            dict(name="one", codes=["1"], desc={"1": "one"}),
            dict(name="zero", codes=[], desc=dict()),
            dict(name="100", codes=list(f"code_{i}" for i in range(100))),
        ],
    )
    def primitive_flat_scheme_kwarg(self, request):
        if "desc" in request.param:
            desc = request.param["desc"]
        elif len(request.param["codes"]) > 0:
            desc = dict(zip(request.param["codes"], request.param["codes"]))
        else:
            desc = dict()
        return dict(name=request.param["name"], codes=tuple(sorted(request.param["codes"])), desc=rx.FrozenDict11(desc))

    @pytest.fixture(scope="class")
    def primitive_flat_scheme(self, primitive_flat_scheme_kwarg) -> rx.CodingScheme:
        return rx.CodingScheme(**primitive_flat_scheme_kwarg)

    @pytest.fixture(scope="class")
    def scheme_manager(self, primitive_flat_scheme: rx.CodingScheme) -> rx.CodingSchemesManager:
        return rx.CodingSchemesManager().add_scheme(primitive_flat_scheme)

    def test_from_name(self, primitive_flat_scheme: rx.CodingScheme, scheme_manager: rx.CodingSchemesManager):
        assert scheme_manager.scheme[primitive_flat_scheme.name].equals(primitive_flat_scheme)

        with pytest.raises(KeyError):
            # Unregistered scheme
            _ = scheme_manager.scheme["42"]

    @pytest.mark.parametrize("codes", [("A", "B", "C", "C"), ("A", "B", "C", "B"), ("A", "A", "A", "A")])
    def test_codes_uniqueness(self, codes):
        with pytest.raises(AssertionError) as excinfo:
            _ = rx.CodingScheme(name="test", codes=tuple(sorted(codes)), desc=rx.FrozenDict11({c: c for c in codes}))
            assert "should be unique" in str(excinfo.value)

    def test_register_scheme(self, primitive_flat_scheme):
        """
        Test the register_scheme method.

        This method tests two scenarios:
        1. It tests that the register_scheme method works by registering a scheme and then
           asserting that the registered scheme can be retrieved using its name.
        2. It tests that the register_scheme method logs a warning when trying to
           register a scheme that is already registered with the same name and content.
        """
        # First, test that the register_scheme method works.
        manager = rx.CodingSchemesManager().add_scheme(primitive_flat_scheme)
        assert manager.scheme[primitive_flat_scheme.name].equals(primitive_flat_scheme)

        # Second, test that the register_scheme method raises an error when
        # the scheme is already registered.
        with mock.patch("logging.warning") as mocker:
            manager.add_scheme(primitive_flat_scheme)
            mocker.assert_called_once()

    def test_scheme_equality(self, primitive_flat_scheme):
        """
        Test the equality of schemes.

        This test asserts that a scheme equal to its deepcopy.
        It then mutates the description and index of one of the schemes and asserts that the two
        schemes are not equal.
        """
        assert primitive_flat_scheme.equals(primitive_flat_scheme)

        if len(primitive_flat_scheme) > 0:
            desc_mutated = rx.FrozenDict11({code: f"{desc} muted" for code, desc in primitive_flat_scheme.desc.items()})
            mutated_scheme = rx.CodingScheme(
                name=primitive_flat_scheme.name, codes=primitive_flat_scheme.codes, desc=desc_mutated
            )
            assert not primitive_flat_scheme.equals(mutated_scheme)

    @pytest.fixture(
        params=[
            (("icd10cm",), ()),
            (("icd9cm",), ()),
            (("icd10pcs",), ()),
            (("icd9pcs",), ()),
            (("icd9cm",), ("dx_ccs",)),
            (("icd9pcs",), ("pr_ccs",)),
            (("icd9cm",), ("dx_flat_ccs",)),
            (("icd9pcs",), ("pr_flat_ccs",)),
        ],
        ids=lambda x: "_".join(sum(x, ())),
        scope="class",
    )
    def scheme_selection(self, request) -> tuple[ICDSchemeSelection, CCSSchemeSelection]:
        kwargs = lambda p: {k: True for k in p}
        return ICDSchemeSelection(**kwargs(request.param[0])), CCSSchemeSelection(**kwargs(request.param[1]))

    @pytest.fixture(
        params=[
            (("icd10cm", "icd9cm"), ()),
            (("icd9cm", "icd10cm"), ()),
            (("icd10pcs", "icd9pcs"), ()),
            (("icd10pcs", "icd9pcs"), ()),
            (("icd9cm",), ("dx_ccs",)),
            (("icd9pcs",), ("pr_ccs",)),
            (("icd9cm",), ("dx_flat_ccs",)),
            (("icd9pcs",), ("pr_flat_ccs",)),
        ],
        scope="class",
        ids=lambda x: "_".join(sum(x, ())),
    )
    def scheme_pair_selection(self, request):
        kwargs = lambda p: {k: True for k in p}
        return ICDSchemeSelection(**kwargs(request.param[0])), CCSSchemeSelection(**kwargs(request.param[1]))

    @pytest.fixture(
        params=["icd9cm_v1", "icd9cm_v2_groups", "icd9cm_v3_groups", "dx_flat_ccs_mlhc_groups", "dx_flat_ccs_v1"],
        scope="class",
    )
    def outcome_selection(self, request):
        return OutcomeSelection(**{request.param: True})

    @pytest.fixture(scope="class")
    def outcome_selection_name(self, outcome_selection: OutcomeSelection) -> str:
        (name,) = outcome_selection.flag_set
        return name

    @pytest.fixture(scope="class")
    def selection_names(self, scheme_selection: tuple[ICDSchemeSelection, CCSSchemeSelection]) -> tuple[str, ...]:
        return tuple(scheme_selection[0].flag_set) + tuple(scheme_selection[1].flag_set)

    @pytest.fixture(scope="class")
    def icd_ccs_scheme_manager(
        self, scheme_selection: tuple[ICDSchemeSelection, CCSSchemeSelection]
    ) -> rx.CodingSchemesManager:
        icd_selection, ccs_selection = scheme_selection
        return setup_ccs_schemes(setup_icd_schemes(icd_selection), ccs_selection)

    @pytest.fixture(scope="class")
    def icd_ccs_outcome_manager_prerequisite(self, outcome_selection: OutcomeSelection) -> rx.CodingSchemesManager:
        return setup_ccs_schemes(
            setup_icd_schemes(ICDSchemeSelection(icd9cm=True)), CCSSchemeSelection(dx_flat_ccs=True)
        )

    @pytest.fixture(scope="class")
    def icd_ccs_map_manager(
        self, scheme_pair_selection: tuple[ICDSchemeSelection, CCSSchemeSelection]
    ) -> rx.CodingSchemesManager:
        icd_selection, ccs_selection = scheme_pair_selection
        return setup_standard_icd_ccs(
            icd_selection=icd_selection, ccs_selection=ccs_selection, outcome_selection=OutcomeSelection()
        )

    @pytest.fixture(scope="class")
    def icd_ccs_outcome_manager(
        self, icd_ccs_outcome_manager_prerequisite: rx.CodingSchemesManager, outcome_selection: OutcomeSelection
    ) -> rx.CodingSchemesManager:
        return setup_outcomes(icd_ccs_outcome_manager_prerequisite, outcome_selection)

    def test_icd_ccs_schemes(self, icd_ccs_scheme_manager, selection_names):
        assert len(icd_ccs_scheme_manager.schemes) == len(selection_names)
        assert len(icd_ccs_scheme_manager.scheme) == len(selection_names)
        for name in selection_names:
            assert name in icd_ccs_scheme_manager.scheme
            assert icd_ccs_scheme_manager.scheme[name].name == name
            assert isinstance(icd_ccs_scheme_manager.scheme[name], rx.CodingScheme)

    def test_icd_ccs_outcomes(self, icd_ccs_outcome_manager, outcome_selection_name):
        assert len(icd_ccs_outcome_manager.outcomes) == 1
        assert len(icd_ccs_outcome_manager.outcome_data) == 1

        assert outcome_selection_name in icd_ccs_outcome_manager.outcome_data
        assert icd_ccs_outcome_manager.outcome_data[outcome_selection_name].name == outcome_selection_name
        assert isinstance(icd_ccs_outcome_manager.outcome_data[outcome_selection_name], rx.FilterOutcomeMapData)

    def test_icd_ccs_maps(
        self,
        icd_ccs_map_manager: rx.CodingSchemesManager,
        scheme_pair_selection: tuple[ICDSchemeSelection, CCSSchemeSelection],
    ):
        icd_selection, ccs_selection = scheme_pair_selection
        assert len(icd_selection.flag_set) + len(ccs_selection.flag_set) == 2
        assert len(icd_ccs_map_manager.maps) == 2
        assert len(icd_ccs_map_manager.map) == 2 + 2  # the two identity maps
        (a, b) = icd_selection.flag_set + ccs_selection.flag_set
        s_a = icd_ccs_map_manager.scheme[a]
        s_b = icd_ccs_map_manager.scheme[b]
        assert (a, b) in icd_ccs_map_manager.map
        assert (b, a) in icd_ccs_map_manager.map
        m1 = icd_ccs_map_manager.map[(a, b)]
        m2 = icd_ccs_map_manager.map[(b, a)]
        for m in (m1, m2):
            assert set(m.data.keys()) == set(m.domain)
            assert frozenset().union(*tuple(m.data.values())) == frozenset(m.range)
        assert m1.source_name == a
        assert m1.target_name == b
        assert m2.source_name == b
        assert m2.target_name == a
        assert m1.source_name == a
        assert m1.target_name == b
        assert m2.source_name == b
        assert m2.target_name == a
        assert m1.support_ratio(s_a) > 0.7
        assert m2.support_ratio(s_b) > 0.7
        assert m1.range_ratio(s_b) > 0.15
        assert m2.range_ratio(s_a) > 0.15

    def test_primitive_scheme_serialization(self, primitive_flat_scheme: rx.CodingScheme, tmpdir: str):
        path = f"{tmpdir}/coding_scheme.h5"
        with tb.open_file(path, "w") as f:
            primitive_flat_scheme.to_hdf_group(f.create_group("/", "scheme_data"))

        with tb.open_file(path, "r") as f:
            reloaded = rx.CodingScheme.from_hdf_group(f.root.scheme_data)

        assert primitive_flat_scheme.equals(reloaded)

    def test_icd_ccs_scheme_serialization(self, icd_ccs_scheme_manager: rx.CodingSchemesManager, tmpdir: str):
        path = f"{tmpdir}/coding_schemes.h5"
        with tb.open_file(path, "w") as f:
            icd_ccs_scheme_manager.to_hdf_group(f.create_group("/", "context_view"))

        with tb.open_file(path, "r") as f:
            reloaded = rx.CodingSchemesManager.from_hdf_group(f.root.context_view)

        assert icd_ccs_scheme_manager.equals(reloaded)

    def test_icd_ccs_outcome_serialization(self, icd_ccs_outcome_manager: rx.CodingSchemesManager, tmpdir: str):
        path = f"{tmpdir}/coding_schemes.h5"
        with tb.open_file(path, "w") as f:
            icd_ccs_outcome_manager.to_hdf_group(f.create_group("/", "context_view"))

        with tb.open_file(path, "r") as f:
            reloaded = rx.CodingSchemesManager.from_hdf_group(f.root.context_view)

        assert icd_ccs_outcome_manager.equals(reloaded)

    def test_icd_ccs_map_serialization(
        self, icd_ccs_map_manager: rx.CodingSchemesManager, hf5_group_writer: tb.Group, tmpdir: str
    ):
        path = f"{tmpdir}/coding_schemes.h5"
        with tb.open_file(path, "w") as f:
            icd_ccs_map_manager.to_hdf_group(f.create_group("/", "context_view"))

        with tb.open_file(path, "r") as f:
            reloaded = rx.CodingSchemesManager.from_hdf_group(f.root.context_view)

        assert icd_ccs_map_manager.equals(reloaded)

    @pytest.mark.parametrize(
        "name, codes, desc",
        [
            ("problematic_codes", [1], {"1": "one"}),
            ("problematic_desc", ["1"], {1: "one"}),
            ("problematic_desc", ["1"], {"1": 5}),
        ],
    )
    def test_type_error(self, name: str, codes: list[str], desc: dict[str, str]):
        """
        Test for type error handling in the FlatScheme constructor.

        This test adds a problematic scheme to test the error handling of the constructor.
        The code and description types should be strings, not integers. The test expects an AssertionError
        or KeyError to be raised.
        """
        with pytest.raises((AssertionError, KeyError)):
            rx.CodingScheme(name=name, codes=tuple(sorted(codes)), desc=rx.FrozenDict11(desc))

    @pytest.mark.parametrize(
        "name, codes, desc",
        [
            ("problematic_desc", ["1", "3"], {"1": "one"}),
            ("problematic_desc", ["3"], {"3": "three", "1": "one"}),
            ("duplicate_codes", ["1", "2", "2"], {"1": "one", "2": "two"}),
        ],
    )
    def test_sizes(self, name: str, codes: list[str], desc: dict[str, str]):
        """
        Test the consistency between scheme components, in their size, and mapping correctness.

        This method adds a problematic scheme to test error handling.
        The codes, description, and index collections should all have the same sizes,
        codes should be unique, and mapping should be correct and 1-to-1.
        FlatScheme constructor raises either an AssertionError or KeyError when provided with invalid input.
        """
        with pytest.raises((AssertionError, KeyError)):
            rx.CodingScheme(name=name, codes=tuple(sorted(codes)), desc=rx.FrozenDict11(desc))

    def test_index2code(self, primitive_flat_scheme):
        """
        Test the index to code mapping in the coding scheme.

        It tests if the index to code mapping is correct and respects the codes order.
        """

        assert all(c == primitive_flat_scheme.codes[i] for i, c in primitive_flat_scheme.index2code.items())

    def test_index2desc(self, primitive_flat_scheme):
        """
        Test the mapping of index to description in the coding scheme.

        Iterates over the scheme_kwargs and creates a FlatScheme object using each set of keyword arguments.
        Then, it checks if the description for each code in the scheme matches the description obtained from the index.
        """

        assert all(
            desc == primitive_flat_scheme.desc[primitive_flat_scheme.codes[i]]
            for i, desc in primitive_flat_scheme.index2desc.items()
        )

    def test_search_regex(self):
        """
        Test the search_regex method of the FlatScheme class.

        This method tests the search_regex method of the FlatScheme class by creating a FlatScheme object
        with a specific coding scheme configuration and performing various search operations using the
        search_regex method. It asserts the expected results for each search operation.

        """
        # Arrange
        scheme = rx.CodingScheme(
            name="simple_searchable", codes=("1", "3"), desc=rx.FrozenDict11({"1": "one", "3": "pancreatic cAnCeR"})
        )
        # Act & Assert
        assert scheme.search_regex("cancer") == {"3"}
        assert scheme.search_regex("one") == {"1"}
        assert scheme.search_regex("pancreatic") == {"3"}
        assert scheme.search_regex("cAnCeR") == {"3"}
        assert scheme.search_regex("x") == set()

    # def test_mapper_to(self):
    #     self.fail()

    # def test_codeset2vec(self):
    #     self.fail()

    # def test_empty_vector(self):
    #     self.fail()

    # def test_supported_targets(self):
    #     self.fail()

    def test_as_dataframe(self, primitive_flat_scheme):
        """
        Test the `as_dataframe` method of the FlatScheme class.

        This method tests whether the `as_dataframe` method returns a DataFrame with the expected structure and values.
        It checks if the index of the DataFrame matches the index values of the scheme and if the columns of the
        DataFrame are 'code' and 'desc'. It also verifies if the scheme object is equal to a new FlatScheme object
        created with the same configuration, codes, descriptions, and index.
        """
        df = primitive_flat_scheme.as_dataframe()
        assert set(df.index) == set(primitive_flat_scheme.index.values())
        assert set(df.columns) == {"code", "desc"}
        codes = df.code.tolist()
        desc = df.set_index("code")["desc"].to_dict()
        assert primitive_flat_scheme.equals(
            rx.CodingScheme(name=primitive_flat_scheme.name, codes=tuple(sorted(codes)), desc=rx.FrozenDict11(desc))
        )


class TestReducedCodeMapN1:
    @pytest.fixture(scope="class")
    def codes_n1(self):
        return {
            "A1": ["B0", "B1", "B2"],
            "A2": ["B3", "B5", "B7"],
            "A3": ["B6"],
            "A4": ["B4", "B8"],
        }  # permute: 0, 1, 2, 3, 5, 7, 6, 4, 8

    @pytest.fixture(scope="class")
    def aggregation(self):
        return rx.FrozenDict11({"A1": "w_sum", "A2": "w_sum", "A3": "w_sum", "A4": "w_sum"})

    @pytest.fixture(scope="class")
    def source_code_scheme(self, codes_n1) -> rx.CodingScheme:
        codes = tuple(sorted(b for bs in codes_n1.values() for b in bs))
        desc = rx.FrozenDict11({b: b for b in codes})
        return rx.CodingScheme(name="source", codes=codes, desc=desc)

    @pytest.fixture(scope="class")
    def target_code_scheme(self, codes_n1) -> rx.CodingScheme:
        desc = rx.FrozenDict11({k: k for k in codes_n1.keys()})
        return rx.CodingScheme(name="target", codes=tuple(codes_n1.keys()), desc=desc)

    @pytest.fixture(scope="class")
    def mapping_data(self, codes_n1) -> rx.FrozenDict1N:
        return rx.FrozenDict1N({b: {a} for a, bs in codes_n1.items() for b in bs})

    @pytest.fixture(scope="class")
    def reduced_code_map(
        self, source_code_scheme, target_code_scheme, mapping_data, aggregation
    ) -> rx.ReducedCodeMapN1:
        return rx.ReducedCodeMapN1.from_data(
            source_code_scheme.name, target_code_scheme.name, mapping_data, aggregation
        )

    @pytest.fixture(scope="class")
    def scheme_manager(self, source_code_scheme, target_code_scheme, reduced_code_map):
        return (
            rx.CodingSchemesManager()
            .add_scheme(source_code_scheme)
            .add_scheme(target_code_scheme)
            .add_map(reduced_code_map)
        )

    @pytest.fixture(scope="class")
    def source_index(
        self, reduced_code_map: rx.ReducedCodeMapN1, scheme_manager: rx.CodingSchemesManager
    ) -> dict[str, int]:
        return scheme_manager.scheme[reduced_code_map.source_name].index

    def test_reduced_groups(self, reduced_code_map: rx.ReducedCodeMapN1):
        assert set(reduced_code_map.reduced_groups.data.keys()) == {"A1", "A2", "A3", "A4"}

    def test_groups_aggregation(self, reduced_code_map: rx.ReducedCodeMapN1):
        assert reduced_code_map.groups_aggregation == ("w_sum",) * 4

    def test_groups_size(self, reduced_code_map: rx.ReducedCodeMapN1, source_index: dict[str, int]):
        assert reduced_code_map.groups_size(source_index) == (3, 3, 1, 2)

    def test_groups_split(self, reduced_code_map: rx.ReducedCodeMapN1, source_index: dict[str, int]):
        assert reduced_code_map.groups_split(source_index) == (3, 6, 7, 9)

    def test_groups_permute(self, reduced_code_map: rx.ReducedCodeMapN1, source_index: dict[str, int]):
        assert reduced_code_map.groups_permute(source_index) == (0, 1, 2, 3, 5, 7, 6, 4, 8)


class TestCodingSchemesManager:
    @pytest.fixture(scope="class")
    def scheme1(self) -> CodingScheme:
        return CodingScheme(name="s", codes=("A", "B"))

    @pytest.fixture(scope="class")
    def scheme2(self) -> CodingScheme:
        return CodingScheme(name="t", codes=("C", "D"))

    @pytest.fixture(scope="class")  # to test chaining.
    def scheme3(self) -> CodingScheme:
        return CodingScheme(name="q", codes=("E", "F"))

    @pytest.fixture(scope="class")
    def map_a(self, scheme1: CodingScheme, scheme2: CodingScheme) -> CodeMap:
        return CodeMap(source_name=scheme1.name, target_name=scheme2.name, data=FrozenDict1N({"A": {"D"}}))

    @pytest.fixture(scope="class")  # to test chaining
    def map_b(self, scheme2: CodingScheme, scheme3: CodingScheme) -> CodeMap:
        return CodeMap(source_name=scheme2.name, target_name=scheme3.name, data=FrozenDict1N({"D": {"F"}}))

    @pytest.fixture(scope="class")
    def scheme_x(self) -> CodingScheme:  # to test match map
        return CodingScheme(name="x", codes=tuple(sorted(("xO", "hY", " zN"))))

    @pytest.fixture(scope="class")
    def scheme_y(self) -> CodingScheme:  # to test match map
        return CodingScheme(name="y", codes=tuple(sorted((" Xo", "Hy ", "ZN"))))

    @pytest.fixture(scope="class")
    def scheme_z(self) -> CodingScheme:  # to test match map
        return CodingScheme(name="z", codes=(" Xoo", "iHy ", "iiii"))

    @pytest.fixture(scope="class")
    def manager1(self, scheme1: CodingScheme) -> CodingSchemesManager:
        return CodingSchemesManager().add_scheme(scheme1)

    @pytest.fixture(scope="class")
    def manager2(self, scheme2: CodingScheme) -> CodingSchemesManager:
        return CodingSchemesManager().add_scheme(scheme2)

    @pytest.fixture(scope="class")
    def manager3(self, map_a: CodeMap) -> CodingSchemesManager:
        return CodingSchemesManager().add_map(map_a)

    @pytest.fixture(scope="class")
    def manager_union(
        self, manager1: CodingSchemesManager, manager2: CodingSchemesManager, manager3: CodingSchemesManager
    ) -> CodingSchemesManager:
        return manager1 + manager2 + manager3

    @pytest.fixture(scope="class")  # to test chaining
    def manager_all(
        self, manager_union: CodingSchemesManager, scheme3: CodingScheme, map_b: CodeMap
    ) -> CodingSchemesManager:
        return manager_union.add_scheme(scheme3).add_map(map_b)

    @pytest.fixture(scope="class")  # tot test chaining
    def manager_with_chained_map(self, manager_all: CodingSchemesManager):
        return manager_all.add_chained_map("s", "t", "q")

    @pytest.fixture(scope="class")
    def manager4matchmap(
        self, scheme_x: CodingScheme, scheme_y: CodingScheme, scheme_z: CodingScheme
    ) -> CodingSchemesManager:
        return CodingSchemesManager().add_scheme(scheme_x).add_scheme(scheme_y).add_scheme(scheme_z)

    @pytest.fixture(scope="class")
    def manager_with_match_map(self, manager4matchmap: CodingSchemesManager):
        return manager4matchmap.add_match_map("x", "y")

    def test_dims0(self):
        m = CodingSchemesManager()
        assert len(m.scheme) == 0
        assert len(m.map) == 0
        assert len(m.outcome_data) == 0
        assert len(m.identity_maps) == 0
        assert m.equals(CodingSchemesManager())

    def test_dims1(self, manager1: CodingSchemesManager, manager2: CodingSchemesManager):
        assert len(manager1.scheme) == len(manager2.scheme) == 1
        assert len(manager1.map) == len(manager2.scheme) == 1  # now it contains an identity map (automatically made)
        assert len(manager1.identity_maps) == len(manager2.identity_maps) == 1
        assert manager1.equals(manager1)
        assert not manager1.equals(manager2)
        assert not manager1.equals(CodingSchemesManager())

    def test_dims2(self, manager3: CodingSchemesManager):
        assert len(manager3.scheme) == 0
        assert len(manager3.map) == 1
        assert len(manager3.identity_maps) == 0
        assert not manager3.equals(CodingSchemesManager())
        assert manager3.equals(manager3)

    def test_dims3(
        self,
        manager1: CodingSchemesManager,
        manager2: CodingSchemesManager,
        manager3: CodingSchemesManager,
        manager_union: CodingSchemesManager,
    ):
        assert (manager1 + manager2).equals(manager2 + manager1)
        assert not (manager1 + manager2 + manager3).equals(manager1 + manager3)
        assert (manager1 + manager2 + manager3).equals(manager3 + manager2 + manager1 + CodingSchemesManager())
        assert (manager1 + manager1 + manager1).equals(manager1)
        assert len(manager_union.schemes) == 2
        assert len(manager_union.identity_maps) == 2
        assert len(manager_union.map) == 3

    def test_chained_map(self, manager_all: CodingSchemesManager, manager_with_chained_map: CodingSchemesManager):
        assert ("s", "t") in manager_all.map
        assert ("s", "t") in manager_with_chained_map.map
        assert ("s", "q") not in manager_all.map
        assert ("s", "q") in manager_with_chained_map.map
        s = manager_with_chained_map.scheme["s"]
        q = manager_with_chained_map.scheme["q"]
        s_q = manager_with_chained_map.map[("s", "q")]
        assert set(s.codes).issuperset(s_q.data.keys())
        assert s_q.map_codeset(s.codes).issubset(q.codes)

    def test_manager_with_match_map(self, manager_with_match_map: CodingSchemesManager):
        scheme_x = manager_with_match_map.scheme["x"]
        scheme_y = manager_with_match_map.scheme["y"]
        m_xy = manager_with_match_map.map[("x", "y")]
        m_yx = manager_with_match_map.map[("y", "x")]
        assert set(scheme_x.codes) == set(m_xy.data.keys())
        assert set(scheme_y.codes) == set(m_yx.data.keys())
        assert set(scheme_x.codes) == frozenset().union(*list(m_yx.data.values()))
        assert set(scheme_y.codes) == frozenset().union(*list(m_xy.data.values()))
        assert set(m_xy.map_codeset(scheme_x.codes)) == set(scheme_y.codes)
        assert set(m_yx.map_codeset(scheme_y.codes)) == set(scheme_x.codes)
        assert m_yx.map_codeset(m_xy.map_codeset(scheme_x.codes)) == set(scheme_x.codes)
        assert m_xy.map_codeset(m_yx.map_codeset(scheme_y.codes)) == set(scheme_y.codes)

    def test_invalid_match_map(self, manager4matchmap: CodingSchemesManager):
        with pytest.raises(AssertionError):
            _ = manager4matchmap.add_match_map("x", "z")
