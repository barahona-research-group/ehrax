import logging

import pandas as pd

from ..coding_scheme import CodeMap, CodingScheme, CodingSchemesManager, Formatter, FrozenDict1N, HierarchicalScheme
from ..utils import dataframe_log, resources_path


class ICDScheme(CodingScheme, Formatter):
    # def __check_init__(self):
    #     # assert formatted codes.
    #     assert tuple(map(self.reformat, self.legacy_map.keys())) == tuple(self.legacy_map.keys())
    #     assert tuple(map(self.reformat, self.legacy_map.values())) == tuple(self.legacy_map.values())
    #     # assert existence of values in codes.
    #     assert all(c in self.codes for c in self.legacy_map.values())

    @staticmethod
    def deformat(code: str) -> str:
        return code.strip().replace(".", "")


class ICDHierarchicalScheme(HierarchicalScheme, ICDScheme):
    pass


class ICDMapOps:
    @staticmethod
    def load_conversion_table(conversion_filename: str) -> pd.DataFrame:
        df = pd.read_csv(
            resources_path("ICD", conversion_filename), sep=r"\s+", dtype=str, names=["source", "target", "meta"]
        )
        df["approximate"] = df["meta"].apply(lambda s: s[0])
        df["no_map"] = df["meta"].apply(lambda s: s[1])
        df["combination"] = df["meta"].apply(lambda s: s[2])
        df["scenario"] = df["meta"].apply(lambda s: s[3])
        df["choice_list"] = df["meta"].apply(lambda s: s[4])
        return df

    @staticmethod
    def conversion_status(conversion_table: pd.DataFrame) -> dict[str, str]:
        def _get_status(groupby_df: pd.DataFrame):
            if (groupby_df["no_map"] == "1").all():
                return "no_map"
            elif len(groupby_df) == 1:
                return "11_map"
            elif groupby_df["scenario"].nunique() > 1:
                return "ambiguous"
            elif groupby_df["choice_list"].nunique() < len(groupby_df):
                return "1n_map(resolved)"
            else:
                return "1n_map"

        return conversion_table.groupby("source")[["no_map", "scenario", "choice_list"]].apply(_get_status).to_dict()

    @staticmethod
    def register_mappings(
        manager: CodingSchemesManager, source_scheme: str, target_scheme: str, conversion_filename: str
    ) -> CodingSchemesManager:  # expose
        logging.debug(f"[BEGIN] mapping from {source_scheme} to {target_scheme} via {conversion_filename}.")
        source_scheme = manager.scheme[source_scheme]
        target_scheme = manager.scheme[target_scheme]
        assert isinstance(source_scheme, ICDScheme) and isinstance(target_scheme, ICDScheme), (
            f"Expected ICDScheme subclasses. Got {type(source_scheme)} and {type(target_scheme)} instead."
        )
        df = ICDMapOps.load_conversion_table(conversion_filename=conversion_filename).reset_index(drop=True)
        df = df.assign(source=df["source"].map(source_scheme.reformat), target=df["target"].map(target_scheme.reformat))
        valid_target = df["target"].isin(target_scheme.index).values
        valid_source = df["source"].isin(source_scheme.index).values

        table = df[valid_target & valid_source]
        conversion_status = ICDMapOps.conversion_status(table)
        table = table.assign(status=table["source"].map(conversion_status))
        table = table[table["status"] != "no_map"]
        data = FrozenDict1N(table.groupby("source")["target"].apply(set).to_dict())
        m = CodeMap(source_name=source_scheme.name, target_name=target_scheme.name, data=data)
        # Wait. Let's report and log.
        report = df[(~valid_target) | (~valid_source)]
        report = report.assign(
            invalid_target=~valid_target[report.index.values], invalid_source=~valid_source[report.index.values]
        )
        dataframe_log.info(
            f"In processing {conversion_filename}. "
            f"{len(m.domain)} (of {len(source_scheme)}) {m.source_name} source codes were mapped to "
            f"{len(m.range)} {m.target_name} (of {len(target_scheme)}) target codes. "
            f"{(~valid_source).sum()} (of {len(valid_source)}) source code in the conversion table were discarded. "
            f"{(~valid_target).sum()} (of {len(valid_target)}) target code in the conversion table were discarded. ",
            dataframe=report,
            tag=f"conversion_miss_report_{source_scheme.name}_{target_scheme.name}",
        )
        logging.debug(f"[DONE] mapping from {source_scheme} to {target_scheme} via {conversion_filename}.")
        # Carry on. Done report and log.
        return manager.add_map(m)
