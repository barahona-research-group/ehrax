from collections.abc import ItemsView, Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Self, TypeVar

import pandas as pd

from .base import AbstractWithDataframeEquivalent


K = TypeVar("K")
V = TypeVar("V")


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

    def get(self, key: str, default: V | None = None) -> V | None:
        return self.data.get(key, default)

    def items(self) -> ItemsView[K, V]:
        return self.data.items()

    def keys(self) -> Iterable[K]:
        return self.data.keys()

    def values(self) -> Iterable[V]:
        return self.data.values()


class FrozenDict11(AbstractFrozenDict[str, V]):
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.data.values()), columns=["value"], index=list(self.data.keys())).sort_index()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        return cls(df["value"].to_dict())


class FrozenDict1N(AbstractFrozenDict[str, frozenset[V]]):
    data: MappingProxyType[str, frozenset[V]]

    def __init__(self, data: Mapping[K, frozenset[V] | set[V]]):
        super().__init__({k: frozenset(v) for k, v in data.items()})

    def to_dataframe(self) -> pd.DataFrame:
        return (
            pd.DataFrame([(k, item) for k, v in self.data.items() for item in v], columns=["key", "value"])
            .sort_values(["key", "value"])
            .reset_index(drop=True)
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        return cls(df.groupby("key")["value"].apply(frozenset).to_dict())


class FrozenDict1NM(AbstractFrozenDict[str, AbstractFrozenDict[str, float]]):
    data: MappingProxyType[str, AbstractFrozenDict[str, float]]

    def __init__(self, data: Mapping[K, Mapping[str, float]]):
        super().__init__({k: MappingProxyType(v) for k, v in data.items()})

    def to_dataframe(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                [(k1, k2, v) for k1, kv in self.data.items() for k2, v in kv.items()], columns=["k1", "k2", "v"]
            )
            .sort_values(["k1", "k2", "v"])
            .reset_index(drop=True)
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        return cls(
            {
                k1: MappingProxyType({k2: v["v"].iloc[0].item() for k2, v in df2.groupby("k2")})  # type: ignore
                for k1, df2 in df.groupby("k1")
            }
        )
