from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .core import Feature, Signal
from .operators.base import Operator


@dataclass
class Pipeline:
    operators: list[Operator] = field(default_factory=list)

    def then(self, operator: Operator) -> "Pipeline":
        return Pipeline(self.operators + [operator])

    def run(self, value: Signal | Feature) -> Signal | Feature:
        current = value
        for operator in self.operators:
            current = operator(current)
        return current

    def describe(self) -> list[dict]:
        return [{"operator": op.__class__.__name__, "parameters": op.parameters} for op in self.operators]

