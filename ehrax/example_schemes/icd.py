from abc import abstractmethod

import pandas as pd

from ..coding_scheme import (CodeMap, CodingScheme, CodingSchemesManager, FrozenDict1N, HierarchicalScheme)
from ..utils import resources_path, dataframe_logger


class ICDScheme(CodingScheme):

    @staticmethod
    @abstractmethod
    def add_dots(code: str) -> str:
        raise NotImplementedError("Should be implemented by subclass")


class ICDHierarchicalScheme(HierarchicalScheme, ICDScheme):
    pass


class ICDMapOps:
    @staticmethod
    def load_conversion_table(conversion_filename: str) -> pd.DataFrame:
        df = pd.read_csv(resources_path("ICD", conversion_filename),
                         sep=r'\s+',
                         dtype=str,
                         names=['source', 'target', 'meta'])
        df['approximate'] = df['meta'].apply(lambda s: s[0])
        df['no_map'] = df['meta'].apply(lambda s: s[1])
        df['combination'] = df['meta'].apply(lambda s: s[2])
        df['scenario'] = df['meta'].apply(lambda s: s[3])
        df['choice_list'] = df['meta'].apply(lambda s: s[4])
        return df

    @staticmethod
    def conversion_status(conversion_table: pd.DataFrame) -> dict[str, str]:
        def _get_status(groupby_df: pd.DataFrame):
            if (groupby_df['no_map'] == '1').all():
                return 'no_map'
            elif len(groupby_df) == 1:
                return '11_map'
            elif groupby_df['scenario'].nunique() > 1:
                return 'ambiguous'
            elif groupby_df['choice_list'].nunique() < len(groupby_df):
                return '1n_map(resolved)'
            else:
                return '1n_map'

        return conversion_table.groupby('source')[['no_map', 'scenario', 'choice_list']].apply(_get_status).to_dict()

    @staticmethod
    def register_mappings(manager: CodingSchemesManager, source_scheme: str, target_scheme: str,
                          conversion_filename: str) -> CodingSchemesManager:  # expose
        source_scheme = manager.scheme[source_scheme]
        target_scheme = manager.scheme[target_scheme]
        assert isinstance(source_scheme, ICDScheme) and isinstance(target_scheme, ICDScheme), (
            f"Expected ICDScheme subclasses. Got {type(source_scheme)} and {type(target_scheme)} instead."
        )
        df = ICDMapOps.load_conversion_table(conversion_filename=conversion_filename)
        df['source'] = df['source'].map(source_scheme.add_dots)
        df['target'] = df['target'].map(target_scheme.add_dots)
        valid_target = df['target'].isin(target_scheme.index)
        valid_source = df['source'].isin(source_scheme.index)
        table = df[valid_target & valid_source]
        report = table[(~valid_target) | (~valid_source)]
        report.loc[:, 'invalid_target'] = ~valid_target
        report.loc[:, 'invalid_source'] = ~valid_source
        dataframe_logger.info((f"In processing {conversion_filename}. "
                              f"{(~valid_source).sum()} source code were unsupported. "
                              f"{(~valid_target).sum()} target code were unsupported. ",
                               report, f"conversion_miss_report_{source_scheme.name}_{target_scheme.name}" ))
        conversion_status = ICDMapOps.conversion_status(table)
        table['status'] = table['source'].map(conversion_status)
        table = table[table['status'] != 'no_map']
        data = FrozenDict1N(table.groupby('source')['target'].apply(set).to_dict())
        return manager.add_map(CodeMap(source_name=source_scheme.name, target_name=target_scheme.name, data=data))
