from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .core import ValidationError


def _rising_edges(values) -> list[int]:
    import numpy as np
    values = np.asarray(values)
    return (np.flatnonzero((values[1:] != 0) & (values[:-1] == 0)) + 1).tolist()


@dataclass(frozen=True)
class Trial:
    """One complete paradigm-defined behavioral cycle on the recording clock."""

    trial_id: str
    onset_s: float
    offset_s: float
    condition: str | None = None
    valid: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValidationError("trial_id must not be empty")
        if self.offset_s <= self.onset_s:
            raise ValidationError(f"trial {self.trial_id}: offset_s must exceed onset_s")


@dataclass(frozen=True)
class TrialTable:
    trials: tuple[Trial, ...]

    def __init__(self, trials: Iterable[Trial]):
        rows = tuple(trials)
        ids = [row.trial_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValidationError("TrialTable trial_id values must be unique")
        object.__setattr__(self, "trials", rows)

    def __len__(self) -> int:
        return len(self.trials)

    def by_id(self, trial_id: str) -> Trial:
        for trial in self.trials:
            if trial.trial_id == trial_id:
                return trial
        raise KeyError(f"unknown trial_id {trial_id!r}")

    def valid(self) -> "TrialTable":
        return TrialTable(row for row in self.trials if row.valid)

    def as_records(self) -> list[dict[str, Any]]:
        return [
            {
                "trial_id": row.trial_id,
                "onset_s": row.onset_s,
                "offset_s": row.offset_s,
                "condition": row.condition,
                "valid": row.valid,
                **dict(row.metadata),
            }
            for row in self.trials
        ]


@dataclass(frozen=True)
class Event:
    """A point or interval event, optionally linked to a behavioral trial."""

    event_id: str
    event_type: str
    onset_s: float
    duration_s: float = 0.0
    trial_id: str | None = None
    value: Any = None
    valid: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_type:
            raise ValidationError("event_id and event_type must not be empty")
        if self.duration_s < 0:
            raise ValidationError(f"event {self.event_id}: duration_s must be non-negative")


@dataclass(frozen=True)
class EventTable:
    events: tuple[Event, ...]
    trials: TrialTable | None = None

    def __init__(self, events: Iterable[Event], trials: TrialTable | None = None):
        rows = tuple(events)
        ids = [row.event_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValidationError("EventTable event_id values must be unique")
        if trials is not None:
            for event in rows:
                if event.trial_id is None:
                    continue
                trial = trials.by_id(event.trial_id)
                event_end = event.onset_s + event.duration_s
                if event.onset_s < trial.onset_s or event_end > trial.offset_s:
                    raise ValidationError(
                        f"event {event.event_id} lies outside linked trial {trial.trial_id}"
                    )
        object.__setattr__(self, "events", rows)
        object.__setattr__(self, "trials", trials)

    def __len__(self) -> int:
        return len(self.events)

    def select(self, *, event_type: str | None = None, valid_only: bool = True) -> "EventTable":
        rows = self.events
        if event_type is not None:
            rows = tuple(row for row in rows if row.event_type == event_type)
        if valid_only:
            rows = tuple(row for row in rows if row.valid)
            if self.trials is not None:
                rows = tuple(
                    row for row in rows
                    if row.trial_id is None or self.trials.by_id(row.trial_id).valid
                )
        return EventTable(rows, self.trials)

    def as_records(self) -> list[dict[str, Any]]:
        return [
            {
                "event_id": row.event_id,
                "event_type": row.event_type,
                "onset_s": row.onset_s,
                "duration_s": row.duration_s,
                "trial_id": row.trial_id,
                "value": row.value,
                "valid": row.valid,
                **dict(row.metadata),
            }
            for row in self.events
        ]

    @classmethod
    def from_state_edges(
        cls,
        values,
        *,
        sampling_rate: float,
        event_type: str,
        id_prefix: str | None = None,
        trials: TrialTable | None = None,
    ) -> "EventTable":
        """Build point events from zero-to-nonzero state transitions."""
        if sampling_rate <= 0:
            raise ValidationError("sampling_rate must be positive")
        prefix = id_prefix or event_type
        rows = []
        for index, sample in enumerate(_rising_edges(values), start=1):
            onset = sample / sampling_rate
            trial_id = None
            if trials is not None:
                containing = [
                    trial for trial in trials.trials
                    if trial.onset_s <= onset <= trial.offset_s
                ]
                if len(containing) > 1:
                    raise ValidationError("state edge belongs to multiple overlapping trials")
                trial_id = containing[0].trial_id if containing else None
            rows.append(Event(
                event_id=f"{prefix}-{index:03d}",
                event_type=event_type,
                onset_s=onset,
                trial_id=trial_id,
                metadata={"source_sample": sample, "source": "state_rising_edge"},
            ))
        return cls(rows, trials)
