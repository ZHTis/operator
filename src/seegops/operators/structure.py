from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import Operator
from ..core import Signal, ValidationError
from ..tables import Event, EventTable


@dataclass
class Select(Operator):
    dim: str
    indices: list[int]

    def apply(self, value: Signal) -> Signal:
        axis = value.axis(self.dim)
        data = np.take(value.data, self.indices, axis=axis)
        coords = dict(value.coords)
        coords[self.dim] = value.coordinate(self.dim)[self.indices]
        attrs = dict(value.attrs)
        if self.dim == "channel":
            source_indices = np.asarray(
                attrs.get("source_channel_indices", np.arange(value.data.shape[axis])),
                dtype=int,
            )
            attrs["source_channel_indices"] = source_indices[self.indices].tolist()
        mask = None if value.valid_mask is None else np.take(value.valid_mask, self.indices, axis=axis)
        return value.with_step(data, coords=coords, attrs=attrs, valid_mask=mask, operator="select", parameters=self.parameters)


@dataclass
class Window(Operator):
    length_s: float
    step_s: float
    drop_incomplete: bool = True

    def apply(self, value: Signal) -> Signal:
        axis = value.axis("time")
        if axis != value.data.ndim - 1:
            raise ValidationError("Window currently requires time as the last dimension")
        length = round(self.length_s * value.sampling_rate)
        step = round(self.step_s * value.sampling_rate)
        if length < 2 or step < 1:
            raise ValidationError("window length/step are too small for the sampling rate")
        n = value.data.shape[-1]
        starts = np.arange(0, max(0, n - length + 1), step, dtype=int)
        if len(starts) == 0:
            raise ValidationError("recording is shorter than requested window")
        view = np.lib.stride_tricks.sliding_window_view(value.data, length, axis=-1)
        data = np.take(view, starts, axis=-2)
        dims = value.dims[:-1] + ("window", "time")
        coords = {k: v for k, v in value.coords.items() if k != "time"}
        coords["window"] = starts / value.sampling_rate
        coords["time"] = np.arange(length) / value.sampling_rate
        return value.with_step(data, dims=dims, coords=coords, valid_mask=None, operator="window", parameters=self.parameters)


@dataclass
class Epoch(Operator):
    tmin_s: float
    tmax_s: float
    events: EventTable | None = None
    event_type: str | None = None
    event_samples: np.ndarray | None = None
    trial_boundary: str = "reject"
    overlap: str = "flag"

    def apply(self, value: Signal) -> Signal:
        axis = value.axis("time")
        if axis != value.data.ndim - 1:
            raise ValidationError("Epoch currently requires time as the last dimension")
        left = round(self.tmin_s * value.sampling_rate)
        right = round(self.tmax_s * value.sampling_rate)
        if right <= left:
            raise ValidationError("tmax_s must exceed tmin_s")
        if self.trial_boundary not in {"reject", "flag", "ignore"}:
            raise ValidationError("trial_boundary must be reject, flag, or ignore")
        if self.overlap not in {"reject", "flag", "allow"}:
            raise ValidationError("overlap must be reject, flag, or allow")
        if (self.events is None) == (self.event_samples is None):
            raise ValidationError("provide exactly one of events or event_samples")

        if self.events is not None:
            table = self.events.select(event_type=self.event_type, valid_only=True)
            rows = list(table.events)
            event_samples = np.asarray(
                [round(row.onset_s * value.sampling_rate) for row in rows], dtype=int
            )
        else:
            event_samples = np.asarray(self.event_samples, dtype=int)
            rows = [
                Event(
                    event_id=f"sample-{int(sample)}",
                    event_type=self.event_type or "unspecified",
                    onset_s=float(sample) / value.sampling_rate,
                )
                for sample in event_samples
            ]
            table = EventTable(rows)

        recording_ok = (
            (event_samples + left >= 0)
            & (event_samples + right <= value.data.shape[-1])
        )
        trial_crossing = np.zeros(len(rows), dtype=bool)
        if table.trials is not None:
            for index, row in enumerate(rows):
                if row.trial_id is None:
                    continue
                trial = table.trials.by_id(row.trial_id)
                epoch_start = row.onset_s + self.tmin_s
                epoch_stop = row.onset_s + self.tmax_s
                trial_crossing[index] = (
                    epoch_start < trial.onset_s or epoch_stop > trial.offset_s
                )
        keep = recording_ok.copy()
        if self.trial_boundary == "reject":
            keep &= ~trial_crossing

        intervals = np.column_stack((event_samples + left, event_samples + right))
        overlapping = np.zeros(len(rows), dtype=bool)
        order = np.argsort(intervals[:, 0]) if len(rows) else np.array([], dtype=int)
        for first, second in zip(order[:-1], order[1:]):
            if intervals[second, 0] < intervals[first, 1]:
                overlapping[first] = overlapping[second] = True
        if self.overlap == "reject" and overlapping.any():
            raise ValidationError("requested epochs overlap; use overlap='flag' or 'allow'")

        kept_indices = np.flatnonzero(keep)
        if len(kept_indices) == 0:
            raise ValidationError("no complete epochs remain")
        offsets = np.arange(left, right)
        data = np.stack(
            [np.take(value.data, event_samples[index] + offsets, axis=-1) for index in kept_indices],
            axis=-2,
        )
        dims = value.dims[:-1] + ("epoch", "time")
        coords = {k: v for k, v in value.coords.items() if k != "time"}
        kept_rows = [rows[index] for index in kept_indices]
        coords["epoch"] = np.asarray([row.event_id for row in kept_rows], dtype=object)
        coords["time"] = offsets / value.sampling_rate
        coords["event_id"] = np.asarray([row.event_id for row in kept_rows], dtype=object)
        coords["event_type"] = np.asarray([row.event_type for row in kept_rows], dtype=object)
        coords["trial_id"] = np.asarray([row.trial_id for row in kept_rows], dtype=object)
        coords["event_onset_s"] = np.asarray([row.onset_s for row in kept_rows], dtype=float)
        coords["event_sample"] = event_samples[kept_indices]
        coords["crosses_trial_boundary"] = trial_crossing[kept_indices]
        coords["overlaps_another_epoch"] = overlapping[kept_indices]
        attrs = dict(value.attrs)
        attrs["coordinate_dimensions"] = {
            **dict(attrs.get("coordinate_dimensions", {})),
            "event_id": "epoch",
            "event_type": "epoch",
            "trial_id": "epoch",
            "event_onset_s": "epoch",
            "event_sample": "epoch",
            "crosses_trial_boundary": "epoch",
            "overlaps_another_epoch": "epoch",
        }
        attrs["trial_table"] = table.trials.as_records() if table.trials is not None else None
        attrs["event_table"] = table.as_records()
        params = {
            "event_type": self.event_type,
            "input_events": len(rows),
            "kept_events": len(kept_indices),
            "rejected_recording_boundary": int((~recording_ok).sum()),
            "rejected_trial_boundary": int((trial_crossing & ~keep).sum()),
            "tmin_s": self.tmin_s,
            "tmax_s": self.tmax_s,
            "trial_boundary": self.trial_boundary,
            "overlap": self.overlap,
            "legacy_event_samples": self.events is None,
        }
        return value.with_step(
            data,
            dims=dims,
            coords=coords,
            attrs=attrs,
            valid_mask=None,
            operator="epoch",
            parameters=params,
        )
