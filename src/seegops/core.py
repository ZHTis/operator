from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np


class ValidationError(ValueError):
    """Raised when an operation is executable but scientifically invalid."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class ProvenanceStep:
    operator: str
    version: str
    parameters: Mapping[str, Any]
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


@dataclass(frozen=True)
class Signal:
    """A labeled signal tensor. Operators address dimensions by name, never index."""

    data: np.ndarray
    dims: tuple[str, ...]
    coords: Mapping[str, np.ndarray]
    sampling_rate: float
    unit: str = "uV"
    attrs: Mapping[str, Any] = field(default_factory=dict)
    valid_mask: np.ndarray | None = None
    provenance: tuple[ProvenanceStep, ...] = ()

    def __post_init__(self) -> None:
        if self.data.ndim != len(self.dims):
            raise ValidationError("data.ndim must equal len(dims)")
        if len(set(self.dims)) != len(self.dims):
            raise ValidationError("dimension names must be unique")
        for dim, size in zip(self.dims, self.data.shape):
            if dim in self.coords and len(self.coords[dim]) != size:
                raise ValidationError(f"coordinate length mismatch for {dim}")
        if "time" in self.dims and self.sampling_rate <= 0:
            raise ValidationError("sampling_rate must be positive")
        if self.valid_mask is not None and self.valid_mask.shape != self.data.shape:
            raise ValidationError("valid_mask must have the same shape as data")

    def axis(self, dim: str) -> int:
        try:
            return self.dims.index(dim)
        except ValueError as exc:
            raise ValidationError(f"required dimension {dim!r} is absent") from exc

    def coordinate(self, dim: str) -> np.ndarray:
        if dim in self.coords:
            return np.asarray(self.coords[dim])
        return np.arange(self.data.shape[self.axis(dim)])

    def with_step(
        self,
        data: np.ndarray,
        *,
        dims: Sequence[str] | None = None,
        coords: Mapping[str, np.ndarray] | None = None,
        unit: str | None = None,
        attrs: Mapping[str, Any] | None = None,
        valid_mask: np.ndarray | None = None,
        operator: str,
        parameters: Mapping[str, Any],
    ) -> "Signal":
        step = ProvenanceStep(operator=operator, version="0.1.0", parameters=parameters)
        return Signal(
            data=np.asarray(data),
            dims=tuple(dims or self.dims),
            coords=coords or self.coords,
            sampling_rate=self.sampling_rate,
            unit=unit or self.unit,
            attrs=attrs or self.attrs,
            valid_mask=valid_mask,
            provenance=self.provenance + (step,),
        )

    def provenance_json(self) -> str:
        return json.dumps([p.as_dict() for p in self.provenance], ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class Feature:
    data: np.ndarray
    dims: tuple[str, ...]
    coords: Mapping[str, np.ndarray]
    unit: str
    name: str
    attrs: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[ProvenanceStep, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.data.shape),
            "dims": list(self.dims),
            "unit": self.unit,
            "attrs": _jsonable(self.attrs),
            "provenance": [p.as_dict() for p in self.provenance],
        }


def file_fingerprint(path: str | Path, block_size: int = 1 << 20) -> dict[str, Any]:
    """Fast identity hash: metadata plus first and last MiB, without copying large data."""
    p = Path(path)
    size = p.stat().st_size
    h = sha256()
    with p.open("rb") as f:
        h.update(f.read(block_size))
        if size > block_size:
            f.seek(max(0, size - block_size))
            h.update(f.read(block_size))
    return {
        "path": str(p.resolve()),
        "size_bytes": size,
        "mtime_ns": p.stat().st_mtime_ns,
        "edge_sha256": h.hexdigest(),
    }

