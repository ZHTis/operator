from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import Operator
from ..core import Feature, ProvenanceStep, Signal


def _reduce(value: Signal, dim: str, name: str, function) -> Feature:
    axis = value.axis(dim)
    data = function(value.data, axis=axis)
    dims = tuple(d for d in value.dims if d != dim)
    coords = {d: value.coords[d] for d in dims if d in value.coords}
    step = ProvenanceStep(name, "0.1.0", {"dim": dim})
    unit = value.unit if name == "mean" else f"{value.unit}^2"
    return Feature(data, dims, coords, unit, name, {}, value.provenance + (step,))


@dataclass
class Mean(Operator):
    dim: str

    def apply(self, value: Signal) -> Feature:
        return _reduce(value, self.dim, "mean", np.mean)


@dataclass
class Variance(Operator):
    dim: str

    def apply(self, value: Signal) -> Feature:
        return _reduce(value, self.dim, "variance", np.var)

