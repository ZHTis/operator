from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import Signal, ValidationError
from .tables import EventTable, Trial, TrialTable


@dataclass(frozen=True)
class Sample:
    """One analysis segment on a Signal sampling grid."""

    sample_id: str
    sample_kind: str
    start_sample: int
    stop_sample_exclusive: int
    split: str = "unspecified"
    trial_id: str | None = None
    event_id: str | None = None
    anchor_sample: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.sample_kind:
            raise ValidationError("sample_id and sample_kind must not be empty")
        if self.start_sample < 0:
            raise ValidationError(f"sample {self.sample_id}: start_sample must be non-negative")
        if self.stop_sample_exclusive <= self.start_sample:
            raise ValidationError(f"sample {self.sample_id}: stop must exceed start")


@dataclass(frozen=True)
class SampleTable:
    """A paradigm-neutral index of signal segments consumed by feature operators."""

    samples: tuple[Sample, ...]
    sampling_rate: float
    recording_id: str = "recording"

    def __init__(
        self,
        samples: Iterable[Sample],
        *,
        sampling_rate: float,
        recording_id: str = "recording",
    ):
        rows = tuple(samples)
        ids = [row.sample_id for row in rows]
        if sampling_rate <= 0:
            raise ValidationError("sampling_rate must be positive")
        if len(ids) != len(set(ids)):
            raise ValidationError("SampleTable sample_id values must be unique")
        object.__setattr__(self, "samples", rows)
        object.__setattr__(self, "sampling_rate", float(sampling_rate))
        object.__setattr__(self, "recording_id", str(recording_id))

    def __len__(self) -> int:
        return len(self.samples)

    def select(
        self,
        *,
        sample_kind: str | None = None,
        splits: Sequence[str] | None = None,
    ) -> "SampleTable":
        rows = self.samples
        if sample_kind is not None:
            rows = tuple(row for row in rows if row.sample_kind == sample_kind)
        if splits is not None:
            allowed = set(splits)
            rows = tuple(row for row in rows if row.split in allowed)
        return SampleTable(rows, sampling_rate=self.sampling_rate, recording_id=self.recording_id)

    def as_records(self) -> list[dict[str, Any]]:
        fs = self.sampling_rate
        records = []
        for row in self.samples:
            anchor = row.anchor_sample
            records.append({
                "recording": self.recording_id,
                "sample_id": row.sample_id,
                "sample_kind": row.sample_kind,
                "split": row.split,
                "trial_id": row.trial_id,
                "event_id": row.event_id,
                "start_sample": row.start_sample,
                "stop_sample_exclusive": row.stop_sample_exclusive,
                "start_s": row.start_sample / fs,
                "stop_s": row.stop_sample_exclusive / fs,
                "center_s": (row.start_sample + row.stop_sample_exclusive) / (2 * fs),
                "anchor_sample": anchor,
                "anchor_s": None if anchor is None else anchor / fs,
                **dict(row.metadata),
            })
        return records

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.as_records())


def _trial_split(trial: Trial) -> str:
    return str(trial.metadata.get("split", "unspecified"))


@dataclass(frozen=True)
class ContinuousWindowSampler:
    length_s: float
    step_s: float
    include_splits: tuple[str, ...] | None = ("train",)

    def build(
        self,
        signal: Signal,
        *,
        trials: TrialTable | None = None,
        recording_id: str | None = None,
    ) -> SampleTable:
        signal.axis("time")
        fs = signal.sampling_rate
        length = round(self.length_s * fs)
        step = round(self.step_s * fs)
        if length < 2 or step < 1:
            raise ValidationError("window length/step are too small")
        n_time = signal.data.shape[signal.axis("time")]
        if trials is None:
            trial_rows = [Trial("recording", 0.0, n_time / fs, metadata={"split": "unspecified"})]
        else:
            trial_rows = list(trials.trials)
        allowed = None if self.include_splits is None else set(self.include_splits)
        samples: list[Sample] = []
        for trial in trial_rows:
            split = _trial_split(trial)
            if allowed is not None and split not in allowed:
                continue
            trial_start = max(0, round(trial.onset_s * fs))
            trial_stop = min(n_time, round(trial.offset_s * fs))
            starts = np.arange(trial_start, max(trial_start, trial_stop - length + 1), step, dtype=int)
            for window_number, start in enumerate(starts, start=1):
                samples.append(Sample(
                    sample_id=f"{trial.trial_id}-continuous-{window_number:05d}",
                    sample_kind="continuous",
                    start_sample=int(start),
                    stop_sample_exclusive=int(start + length),
                    split=split,
                    trial_id=trial.trial_id,
                    metadata={"window_in_trial": window_number},
                ))
        if not samples:
            raise ValidationError("no continuous windows were generated")
        return SampleTable(
            samples,
            sampling_rate=fs,
            recording_id=recording_id or str(signal.attrs.get("recording_id", "recording")),
        )


@dataclass(frozen=True)
class EventLockedSampler:
    tmin_s: float
    tmax_s: float
    window_length_s: float
    step_s: float
    include_splits: tuple[str, ...] | None = ("train",)
    event_types: tuple[str, ...] | None = None
    trial_boundary: str = "reject"

    def build(
        self,
        signal: Signal,
        *,
        events: EventTable,
        recording_id: str | None = None,
    ) -> SampleTable:
        signal.axis("time")
        if self.tmax_s <= self.tmin_s:
            raise ValidationError("tmax_s must exceed tmin_s")
        if self.trial_boundary not in {"reject", "flag", "ignore"}:
            raise ValidationError("trial_boundary must be reject, flag, or ignore")
        fs = signal.sampling_rate
        n_time = signal.data.shape[signal.axis("time")]
        length = round(self.window_length_s * fs)
        step = round(self.step_s * fs)
        left = round(self.tmin_s * fs)
        right = round(self.tmax_s * fs)
        if length < 2 or step < 1 or right - left < length:
            raise ValidationError("invalid event window length or step")
        allowed_splits = None if self.include_splits is None else set(self.include_splits)
        allowed_types = None if self.event_types is None else set(self.event_types)
        samples: list[Sample] = []
        for event in events.events:
            if not event.valid or (allowed_types is not None and event.event_type not in allowed_types):
                continue
            trial = None
            split = "unspecified"
            if event.trial_id is not None and events.trials is not None:
                trial = events.trials.by_id(event.trial_id)
                if not trial.valid:
                    continue
                split = _trial_split(trial)
            if allowed_splits is not None and split not in allowed_splits:
                continue
            anchor = round(event.onset_s * fs)
            relative_starts = np.arange(left, right - length + 1, step, dtype=int)
            for relative_number, relative_start in enumerate(relative_starts, start=1):
                start = int(anchor + relative_start)
                stop = int(start + length)
                crosses_recording = start < 0 or stop > n_time
                crosses_trial = False
                if trial is not None:
                    trial_start = round(trial.onset_s * fs)
                    trial_stop = round(trial.offset_s * fs)
                    crosses_trial = start < trial_start or stop > trial_stop
                if crosses_recording:
                    continue
                if crosses_trial and self.trial_boundary == "reject":
                    continue
                samples.append(Sample(
                    sample_id=f"{event.event_id}-event-{relative_number:04d}",
                    sample_kind="event_locked",
                    start_sample=start,
                    stop_sample_exclusive=stop,
                    split=split,
                    trial_id=event.trial_id,
                    event_id=event.event_id,
                    anchor_sample=anchor,
                    metadata={
                        "event_type": event.event_type,
                        "event_value": event.value,
                        "relative_start_s": relative_start / fs,
                        "relative_stop_s": (relative_start + length) / fs,
                        "relative_center_s": (relative_start + length / 2) / fs,
                        "crosses_trial_boundary": crosses_trial,
                        **dict(event.metadata),
                    },
                ))
        if not samples:
            raise ValidationError("no event-locked windows were generated")
        return SampleTable(
            samples,
            sampling_rate=fs,
            recording_id=recording_id or str(signal.attrs.get("recording_id", "recording")),
        )
