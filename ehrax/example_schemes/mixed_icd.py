from typing import Self

import pandas as pd

from ..coding_scheme import CodeMap, CodingScheme, CodingSchemesManager, Formatter, FrozenDict1N, FrozenDict11
from ..dataset import COLUMN
from ..utils import dataframe_log


class MultiVersionScheme(CodingScheme):
    component_scheme_names: FrozenDict11
    sep: str

    def __init__(
        self, component_scheme_names: FrozenDict11, sep: str, *, name: str, codes: tuple[str, ...], desc: FrozenDict11
    ):
        super().__init__(name, codes, desc)
        self.component_scheme_names = component_scheme_names
        self.sep = sep

    def component_schemes(self, manager: CodingSchemesManager) -> dict[str, CodingScheme]:
        return {k: manager.scheme[v] for k, v in self.component_scheme_names.items()}

    @staticmethod
    def reformat(df: pd.DataFrame, component_schemes: dict[str, CodingScheme]) -> pd.DataFrame:
        if any(isinstance(s, Formatter) for s in component_schemes.values()):
            df = df.copy()
            id_fmt = lambda c: c
            reformat = {
                v: scheme.reformat if isinstance(scheme, Formatter) else id_fmt
                for v, scheme in component_schemes.items()
            }
            df[COLUMN.code] = list(map(lambda c, v: reformat[v](c), df[COLUMN.code], df[COLUMN.version]))
        return df

    @classmethod
    def from_selection(
        cls, name: str, multi_version_selection: pd.DataFrame | None, component_schemes: dict[str, CodingScheme]
    ) -> Self:
        if multi_version_selection is None:
            multi_version_selection = pd.concat(
                [
                    pd.DataFrame(
                        {
                            str(COLUMN.code): list(si.codes),
                            str(COLUMN.version): [v] * len(si),
                            str(COLUMN.description): list(map(si.desc.get, si.codes)),
                        }
                    )
                    for v, si in component_schemes.items()
                ]
            )
        selection = multi_version_selection.sort_values([str(COLUMN.version), str(COLUMN.code)])
        selection = selection.drop_duplicates([str(COLUMN.version), str(COLUMN.code)]).astype(str)
        assert selection[COLUMN.version].isin(component_schemes.keys()).all(), (
            f"Only versions {', '.join(f'({v}, {s.name})' for v, s in component_schemes.items())} are expected."
        )
        assert selection.groupby([str(COLUMN.version), str(COLUMN.code)]).size().max() == 1, (
            "Duplicate (version, code) pairs are not allowed."
        )

        df = cls.reformat(selection, component_schemes)
        valid_sep = lambda s: not (df[COLUMN.version].str.contains(s).any() or df[COLUMN.code].str.contains(s).any())
        sep = next(s for s in "@:-#$%!=_/+><?;|~," if valid_sep(s))
        df[COLUMN.code] = (df[COLUMN.version] + sep + df[COLUMN.code]).tolist()
        desc = df.set_index(str(COLUMN.code))[COLUMN.description].to_dict()
        return cls(
            name=name,
            codes=tuple(sorted(df[COLUMN.code].tolist())),
            desc=FrozenDict11(desc),
            component_scheme_names=FrozenDict11({k: s.name for k, s in component_schemes.items()}),
            sep=sep,
        )

    def mixed_code_format_table(self, manager: CodingSchemesManager, table: pd.DataFrame) -> pd.DataFrame:
        c_code = str(COLUMN.code)
        c_version = str(COLUMN.version)

        assert c_version in table.columns, f"Column {c_version} not found."
        assert c_code in table.columns, f"Column {c_code} not found."
        assert table[c_version].isin(self.component_scheme_names.keys()).all(), (
            f"Only ICD version {list(self.component_scheme_names.keys())} are expected."
        )

        table = self.reformat(table, self.component_schemes(manager))

        table.loc[:, c_code] = table[c_version] + self.sep + table[c_code]

        # filter out codes that are not in the scheme.
        use_rows = table[c_code].isin(self.codes)
        removed_rows = table[~use_rows]
        removed_rows = removed_rows.assign(component_scheme=removed_rows[c_version].map(self.component_scheme_names))
        dataframe_log.info(
            f"When transforming a table to mixed code format. {len(removed_rows)} codes "
            f"were not found in the corresponding component schemes.",
            dataframe=removed_rows,
            tag="del_rows_mixed_format",
        )
        return table[use_rows].reset_index(drop=True)

    def report_lost_codes(self, dataframe: pd.DataFrame, target_name: str, mixed2target: dict[str, str]):
        lost_codes_df = dataframe[~dataframe["code"].isin(mixed2target.keys())]
        dataframe_log.info(
            f"Lost {len(lost_codes_df)} codes when generating the mapping between the Mixed  "
            f"({self.name})) and the standard ({target_name}). ",
            dataframe=lost_codes_df,
            tag=f"mixed_{self.name}_to_{target_name}_lost_codes",
        )

    def report_map_stats(
        self,
        dataframe: pd.DataFrame,
        dataframe_groupby: tuple[tuple[str, pd.DataFrame], ...],
        target_name: str,
        mixed2target: dict[str, str],
    ):
        mapped_col = f"mapped_to_{target_name}"
        subset_index = lambda name: f"{name}_subset"
        stats = pd.DataFrame(
            columns=["count", mapped_col],
            index=[subset_index(name) for name in list(zip(*dataframe_groupby))[0]] + ["total"],
        )
        stats.loc["total", "count"] = len(dataframe)
        stats.loc["total", mapped_col] = dataframe["code"].isin(mixed2target).sum()
        for name, df in dataframe_groupby:
            stats.loc[subset_index(name), "count"] = len(df)
            stats.loc[subset_index(name), mapped_col] = df["code"].isin(mixed2target).sum()

        lost_stats = stats["count"].values.reshape(-1, 1) - stats.iloc[:, 1:].rename(columns=lambda c: f"missed: {c}")
        stats = pd.concat([stats, lost_stats], axis=1)
        norm_stats = stats.rename(index=lambda i: f"%{i}") / stats["count"].values.reshape(-1, 1)
        stats = pd.concat([stats, norm_stats], axis=0)
        dataframe_log.info(
            f"Statistics of the mapping between the mixed ({self.name}) and {target_name}.",
            dataframe=stats,
            tag=f"mixed_{self.name}_to_{target_name}_stats",
        )

    def register_infer_map(self, manager: CodingSchemesManager, target_name: str) -> CodingSchemesManager:
        required_maps = tuple((component_s, target_name) for component_s in self.component_scheme_names.values())
        assert all(m in manager.map for m in required_maps), (
            f"Mapping between the mixed scheme and {target_name} is not supported due to the absence of "
            f"the map(s): {','.join(str(m) for m in required_maps if m not in manager.map)}"
        )
        dataframe = self.as_dataframe()
        dataframe_groupby = tuple((name, df) for name, df in dataframe.groupby("component_scheme"))
        mixed2target = {}
        for scheme_name, scheme_subset in dataframe_groupby:
            fmt_union_to_std = scheme_subset.set_index("code")["component_code"].to_dict()
            m = manager.map[(scheme_name, target_name)]
            mixed2target.update({mixed_c: m[comp_c] for mixed_c, comp_c in fmt_union_to_std.items() if comp_c in m})
        self.report_lost_codes(dataframe, target_name, mixed2target)
        self.report_map_stats(dataframe, dataframe_groupby, target_name, mixed2target)
        return manager.add_map(CodeMap(source_name=self.name, target_name=target_name, data=FrozenDict1N(mixed2target)))

    def register_map(
        self,
        manager: CodingSchemesManager,
        target_name: str,
        mapping: pd.DataFrame,
        c_code: str,
        c_version: str,
        c_target_code: str,
        c_target_desc: str,
    ) -> CodingSchemesManager:
        """
        Register a mapping between the current Mixed ICD scheme and a target scheme.
        """
        mapping = self.reformat(mapping.astype(str), self.component_schemes(manager))
        mapping.loc[:, c_code] = (mapping[c_version] + self.sep + mapping[c_code]).tolist()
        mapping = mapping[mapping[c_code].isin(self.codes)]
        assert len(mapping) > 0, "No mapping between the Mixed ICD scheme and the target scheme was found."
        target_codes = tuple(sorted(mapping[c_target_code].drop_duplicates().tolist()))
        target_desc = FrozenDict11(mapping.set_index(c_target_code)[c_target_desc].to_dict())
        manager = manager.add_scheme(CodingScheme(name=target_name, codes=target_codes, desc=target_desc))

        mapping = mapping[[c_code, c_target_code]].astype(str)
        mapping = mapping[mapping[c_code].isin(self.codes) & mapping[c_target_code].isin(target_codes)]
        mapping = FrozenDict1N(mapping.groupby(c_code)[c_target_code].apply(set).to_dict())
        self.report_lost_codes(self.as_dataframe(), target_name, mapping.data)
        return manager.add_map(CodeMap(source_name=self.name, target_name=target_name, data=mapping))

    def as_dataframe(self, codes: tuple[str, ...] = None) -> pd.DataFrame:
        if codes is None:
            codes = self.codes
        columns = ["code", "desc", "code_index", "component_version", "component_code"]
        table = pd.DataFrame([(c, self.desc[c], self.index[c], *c.split(self.sep)) for c in codes], columns=columns)
        table.loc[:, "component_scheme"] = list(
            map(lambda v: self.component_scheme_names[v], table["component_version"])
        )
        return table
