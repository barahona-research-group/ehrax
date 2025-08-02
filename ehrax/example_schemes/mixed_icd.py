from typing import Self

import pandas as pd

from ..coding_scheme import CodeMap, CodingScheme, CodingSchemesManager, FrozenDict11, FrozenDict1N
from ..dataset import COLUMN
from ..example_schemes.icd import ICDScheme
from ..utils import dataframe_logger


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
        codes = df[COLUMN.code].str.strip().replace('.', '')
        df[COLUMN.code] = list(map(lambda c, v: add_dots[v](c), codes, df[COLUMN.version]))
        return df

    @classmethod
    def from_selection(cls, manager: CodingSchemesManager, name: str, icd_version_selection: pd.DataFrame,
                       icd_version_schemes: FrozenDict11, sep: str = ':') -> Self:
        icd_version_selection = icd_version_selection.sort_values([str(COLUMN.version), str(COLUMN.code)])
        icd_version_selection = icd_version_selection.drop_duplicates([str(COLUMN.version), str(COLUMN.code)]).astype(
            str)
        assert icd_version_selection[COLUMN.version].isin(icd_version_schemes).all(), \
            f"Only {', '.join(map(lambda x: f'ICD-{x}', icd_version_schemes))} are expected."

        # assert no duplicate (icd_code, icd_version)
        assert icd_version_selection.groupby([str(COLUMN.version), str(COLUMN.code)]).size().max() == 1, \
            "Duplicate (icd_code, icd_version) pairs are not allowed."

        icd_schemes_loaded: dict[str, ICDScheme] = {k: manager.scheme[v] for k, v in icd_version_schemes.items()}

        assert all(isinstance(s, ICDScheme) for s in icd_schemes_loaded.values()), \
            "Only ICD schemes are expected."

        df = cls.fix_dots(icd_version_selection, icd_schemes_loaded)
        df[COLUMN.code] = (df[COLUMN.version] + sep + df[COLUMN.code]).tolist()
        desc = df.set_index(str(COLUMN.code))[COLUMN.description].to_dict()

        return cls(name=name, codes=tuple(sorted(df[COLUMN.code].tolist())), desc=FrozenDict11(desc),
                   icd_version_schemes=icd_version_schemes,
                   sep=sep)

    def mixed_code_format_table(self, manager: CodingSchemesManager, table: pd.DataFrame) -> pd.DataFrame:
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
        dataframe_groupby = [(v, version_df) for v, version_df in dataframe.groupby('icd_version')]
        icd_schemes = self.icd_schemes(manager)
        stats = pd.DataFrame(columns=['count'] + [f'standard-ICD-{v}' for v in icd_schemes.keys()],
                             index=['mixed-ICD'] + [f'mixed-ICD-v{v}' for v in icd_schemes.keys()])
        stats.loc['mixed-ICD', 'count'] = len(dataframe)
        for v, version_df in dataframe_groupby:
            stats.loc[f'mixed-ICD-v{v}', 'count'] = len(version_df)

        for standard_version, standard_scheme in icd_schemes.items():
            # mixed2pure has the form {mixed_code: {icd}}.
            mixed2standard = {}
            for mixed_version, mixed_version_df in dataframe_groupby:
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
            for mixed_version, version_subset_df in dataframe_groupby:
                n_mapped = version_subset_df['code'].isin(mixed2standard).sum()
                stats.loc[f'mixed-ICD-v{mixed_version}', f'standard-ICD-{standard_version}'] = n_mapped
            stats.loc['mixed-ICD', f'standard-ICD-{standard_version}'] = dataframe['code'].isin(mixed2standard).sum()
            lost_codes_df = dataframe[~dataframe['code'].isin(mixed2standard.keys())]
            dataframe_logger.info((
                f"Lost {len(lost_codes_df)} codes when generating the mapping between the Mixed ICD "
                f"({self.name})) and the standard ({icd_schemes[standard_version].name}). ",
                lost_codes_df, f'mixed_to_{icd_schemes[standard_version].name}_lost_codes'))

        lost_stats = len(dataframe) - pd.DataFrame(stats.iloc[:, 1:], columns=[f'Lost {c}' for c in stats.columns[1:]])
        stats = pd.concat([stats, lost_stats], axis=1)
        norm_stats = pd.DataFrame(stats, index=[f'%{i}' for i in stats.index]) / len(dataframe)
        stats = pd.concat([stats, norm_stats], axis=0)
        dataframe_logger.info((
            f"Statistics of the mapping between the Mixed ICD ({self.name}) and the standard ICD schemes.",
            stats, f'mixed_to_standard_stats'
        ))
        return manager

    def register_map(self, manager: CodingSchemesManager, target_name: str,
                     mapping: pd.DataFrame) -> CodingSchemesManager:
        """
        Register a mapping between the current Mixed ICD scheme and a target scheme.
        """
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
