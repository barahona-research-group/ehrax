import logging
from typing import Self

import pandas as pd

from ..coding_scheme import FrozenDict11, FrozenDict1N, CodingScheme, CodingSchemesManager, \
    CodeMap
from ..dataset import COLUMN
from ..example_schemes.icd import ICDScheme


class MixedICDScheme(CodingScheme):
    icd_version_schemes: FrozenDict11
    sep: str

    def __init__(self,
                 icd_version_schemes: FrozenDict11,
                 sep: str = ':', *,
                 name: str, codes: tuple[str, ...], desc: FrozenDict11):
        super().__init__(name, codes, desc)
        self.icd_version_schemes = icd_version_schemes
        self.sep = sep

    def icd_schemes(self, manager: CodingSchemesManager) -> dict[str, ICDScheme]:
        return {k: manager.scheme[v] for k, v in self.icd_version_schemes.items()}

    @staticmethod
    def fix_dots(df: pd.DataFrame,
                 icd_schemes: dict[str, ICDScheme]) -> pd.DataFrame:
        df = df.copy()
        add_dots = {v: icd_scheme.ops.add_dots for v, icd_scheme in icd_schemes.items()}
        codes = df[str(COLUMN.code)].str.strip().replace('.', '')
        df[str(COLUMN.code)] = list(map(lambda c, v: add_dots[v](c), codes, df[str(COLUMN.version)]))
        return df


    @classmethod
    def from_selection(cls, manager: CodingSchemesManager, name: str, icd_version_selection: pd.DataFrame,
                       icd_version_schemes: FrozenDict11, sep: str = ':') -> Self:
        # TODO: test this method.
        icd_version_selection = icd_version_selection.sort_values([str(COLUMN.version), str(COLUMN.code)])
        icd_version_selection = icd_version_selection.drop_duplicates([str(COLUMN.version), str(COLUMN.code)]).astype(
            str)
        assert icd_version_selection[str(COLUMN.version)].isin(icd_version_schemes).all(), \
            f"Only {', '.join(map(lambda x: f'ICD-{x}', icd_version_schemes))} are expected."

        # assert no duplicate (icd_code, icd_version)
        assert icd_version_selection.groupby([str(COLUMN.version), str(COLUMN.code)]).size().max() == 1, \
            "Duplicate (icd_code, icd_version) pairs are not allowed."

        icd_schemes_loaded: dict[str, ICDScheme] = {k: manager.scheme[v] for k, v in icd_version_schemes.items()}

        assert all(isinstance(s, ICDScheme) for s in icd_schemes_loaded.values()), \
            "Only ICD schemes are expected."

        df = cls.fix_dots(icd_version_selection, icd_schemes_loaded)
        df[str(COLUMN.code)] = (df[str(COLUMN.version)] + sep + df[str(COLUMN.code)]).tolist()
        desc = df.set_index(str(COLUMN.code))[str(COLUMN.description)].to_dict()

        return cls(name=name, codes=tuple(sorted(df[str(COLUMN.code)].tolist())), desc=FrozenDict11(desc),
                   icd_version_schemes=icd_version_schemes,
                   sep=sep)

    def mixed_code_format_table(self, manager: CodingSchemesManager, table: pd.DataFrame) -> pd.DataFrame:
        # TODO: test this method.
        """
        Format a table with mixed codes to the ICD version:icd_code format and filter out codes that are not in the scheme.
        """
        c_code = str(COLUMN.code)
        c_version = str(COLUMN.version)

        assert c_version in table.columns, f"Column {c_version} not found."
        assert c_code in table.columns, f"Column {c_code} not found."
        icd_schemes = self.icd_schemes(manager)
        assert table[c_version].isin(icd_schemes).all(), \
            f"Only ICD version {list(icd_schemes.keys())} are expected."

        table = self.fix_dots(table, icd_schemes)

        # the version:icd_code format.
        table[c_code] = table[c_version] + self.sep + table[c_code]

        # filter out codes that are not in the scheme.
        return table[table[c_code].isin(self.codes)].reset_index(drop=True)

    def register_standard_icd_maps(self, manager: CodingSchemesManager) -> CodingSchemesManager:
        """
        Register the mappings between the Mixed ICD scheme and the individual ICD scheme.
        For example, if the current `MixedICD` is mixing ICD-9 and ICD-10,
        then register the two mappings between this scheme and ICD-9 and ICD-10 separately.
        This assumes that the current runtime has already registered mappings
        between the individual ICD schemes.
        """
        dataframe = self.as_dataframe()
        icd_schemes = self.icd_schemes(manager)
        for standard_version, standard_scheme in icd_schemes.items():
            # mixed2pure has the form {mixed_code: {icd}}.
            mixed2standard = {}
            for mixed_version, mixed_version_df in dataframe.groupby('icd_version'):
                mixed_format_to_standard_icd = mixed_version_df.set_index('code')['icd_code'].to_dict()
                if mixed_version == standard_version:
                    update = {c: {icd} for c, icd in mixed_format_to_standard_icd.items() if icd in standard_scheme}
                else:
                    # if mixed_version != pure_version, then retrieve
                    # the mapping between ICD-{mixed_version} and ICD-{pure_version}
                    icd_map = manager.map[(icd_schemes[mixed_version].name, icd_schemes[standard_version].name)]
                    update = {c: icd_map[icd] for c, icd in mixed_format_to_standard_icd.items() if icd in icd_map}
                assert len(update) > 0, f"No mapping between ICD-{mixed_version} and ICD-{standard_version} was found."
                mixed2standard.update(update)

            # register the mapping between the mixed and pure ICD schemes.
            manager = manager.add_map(CodeMap(source_name=self.name, target_name=standard_scheme.name,
                                              data=FrozenDict1N(mixed2standard)))

            lost_df = dataframe[~dataframe['code'].isin(mixed2standard)]
            if len(lost_df) > 0:
                n_lost = len(lost_df)
                n_lost_version = {v: (lost_df['icd_version'] == v).sum() for v in icd_schemes}
                n_version = {v: (dataframe['icd_version'] == v).sum() for v in icd_schemes}
                stats0 = map(lambda v: f'v{v} {n_lost_version[v]} ({n_lost_version[v] / n_lost:.2f})', n_version.keys())
                stats1 = map(lambda v: f'v{v} {n_lost_version[v] / n_version[v]: .2f}', n_version.keys())
                logging.warning(f"Lost {n_lost} codes when generating the mapping between the Mixed ICD "
                                f"({self.name}) and the standard ({standard_scheme.name}). "
                                f"Loss stats: {', '.join(stats0)}; "
                                f"Loss ratios: {', '.join(stats1)}.")
                logging.warning(lost_df.to_string().replace('\n', '\n\t'))

        return manager

    def register_map(self, manager: CodingSchemesManager, target_name: str,
                     mapping: pd.DataFrame) -> CodingSchemesManager:
        """
        Register a mapping between the current Mixed ICD scheme and a target scheme.
        """
        # TODO: test this method.
        c_code = str(COLUMN.code)
        c_version = str(COLUMN.version)
        c_target_code = str(COLUMN.mapped_code)
        c_target_desc = str(COLUMN.mapped_description)

        mapping = self.fix_dots(mapping.astype(str), self.icd_schemes(manager))
        mapping[c_code] = (mapping[c_version] + self.sep + mapping[c_code]).tolist()
        mapping = mapping[mapping[c_code].isin(self.codes)]
        assert len(mapping) > 0, "No mapping between the Mixed ICD scheme and the target scheme was found."
        target_codes = tuple(sorted(mapping[c_target_code].drop_duplicates().tolist()))
        target_desc = FrozenDict11(mapping.set_index(c_target_code)[c_target_desc].to_dict())
        manager = manager.add_scheme(CodingScheme(name=target_name, codes=target_codes, desc=target_desc))

        mapping = mapping[[c_code, c_target_code]].astype(str)
        mapping = mapping[mapping[c_code].isin(self.codes) & mapping[c_target_code].isin(target_codes)]
        mapping = FrozenDict1N(mapping.groupby(c_code)[c_target_code].apply(set).to_dict())
        return manager.add_map(CodeMap(source_name=self.name, target_name=target_name, data=mapping))

    def as_dataframe(self):
        columns = ['code', 'desc', 'code_index', 'icd_version', 'icd_code']
        return pd.DataFrame([(c, self.desc[c], self.index[c], *c.split(self.sep)) for c in self.codes],
                            columns=columns)
