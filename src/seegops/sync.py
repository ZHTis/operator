from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np

from .core import ValidationError
from .io.bci2000 import BCI2000Recording


@dataclass(frozen=True)
class StartTimeAlignment:
    offset_s: float
    uncertainty_s: float
    method: str


def align_by_storage_time(source: BCI2000Recording, target: BCI2000Recording) -> StartTimeAlignment:
    """Return target start minus source start; this is a diagnostic, not proof of sample lock."""
    if source.storage_time is None or target.storage_time is None:
        raise ValidationError("both recordings need parseable StorageTime metadata")
    offset = (target.storage_time - source.storage_time).total_seconds()
    uncertainty = max(1 / source.signal.sampling_rate, 1 / target.signal.sampling_rate)
    return StartTimeAlignment(offset, uncertainty, "BCI2000 StorageTime metadata")


def rising_edges(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state)
    return np.flatnonzero((state[1:] != 0) & (state[:-1] == 0)) + 1

