from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..coding_scheme import CodingSchemesManager


@dataclass
class CodingSchemeManagerStatisticsInterface:
    m: "CodingSchemesManager"

    def __init__(self, m: "CodingSchemesManager"):
        self.m = m
