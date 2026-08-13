from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core import Feature, Signal


class Operator(ABC):
    version = "0.1.0"

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @abstractmethod
    def apply(self, value: Signal | Feature) -> Signal | Feature:
        raise NotImplementedError

    def __call__(self, value: Signal | Feature) -> Signal | Feature:
        return self.apply(value)

