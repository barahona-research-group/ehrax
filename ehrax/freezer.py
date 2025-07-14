from types import MappingProxyType
from typing import TypeVar, Mapping, Self, Iterator, ItemsView, Optional, Iterable

import pandas as pd

from ehrax.base import AbstractWithDataframeEquivalent

K = TypeVar('K')
V = TypeVar('V')


class AbstractFrozenDict(AbstractWithDataframeEquivalent, Mapping[K, V]):
    data: MappingProxyType[K, V]

    def __init__(self, data: Mapping[K, V]):
        self.data = MappingProxyType(data)

    def __getitem__(self, key: str) -> V:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[K]:
        return iter(self.data)

    def __contains__(self, key: K) -> bool:
        return key in self.data

    def get(self, key: str, default: Optional[V] = None) -> Optional[V]:
        return self.data.get(key, default)

    def items(self) -> ItemsView[K, V]:
        return self.data.items()

    def keys(self) -> Iterable[K]:
        return self.data.keys()

    def values(self) -> Iterable[V]:
        return self.data.values()


class FrozenDict11(AbstractFrozenDict[str, V]):
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.data.values()), columns=['value'], index=list(self.data.keys())).sort_index()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        return cls(df['value'].to_dict())


class FrozenDict1N(AbstractFrozenDict[str, set[V]]):
    data: MappingProxyType[str, set[V]]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([(k, item) for k, v in self.data.items() for item in v],
                            columns=['key', 'value']).sort_values(['key', 'value']).reset_index(drop=True)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        return cls(df.groupby('key')['value'].apply(set).to_dict())
