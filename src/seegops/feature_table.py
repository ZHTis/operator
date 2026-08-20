from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .core import ValidationError
from .feature_bank import FeatureCollection, TargetCollection
from .qc import ChannelQCTable, SampleQCTable


@dataclass(frozen=True)
class FeatureTable:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]

    def valid(self) -> "FeatureTable":
        if "analysis_valid" not in self.frame:
            return self
        return FeatureTable(
            self.frame.loc[self.frame.analysis_valid].copy(),
            self.feature_columns,
            self.target_columns,
        )

    def to_csv(self, path, **kwargs) -> None:
        self.frame.to_csv(path, index=False, **kwargs)


@dataclass(frozen=True)
class FeatureTableAssembler:
    """Join by declared keys and reject row multiplication."""

    def assemble(
        self,
        features: FeatureCollection,
        *,
        targets: TargetCollection | None = None,
        channel_qc: ChannelQCTable | None = None,
        sample_qc: SampleQCTable | None = None,
    ) -> FeatureTable:
        frame = features.frame.copy()
        if frame.duplicated(["sample_id", "channel_index"]).any():
            raise ValidationError("feature keys sample_id × channel_index are not unique")
        sample_frame = features.sample_table.as_frame()
        if sample_frame.duplicated("sample_id").any():
            raise ValidationError("SampleTable keys are not unique")
        frame = frame.merge(sample_frame, on="sample_id", how="left", validate="many_to_one")
        target_columns: tuple[str, ...] = ()
        if targets is not None:
            if targets.frame.duplicated("sample_id").any():
                raise ValidationError("target keys are not unique")
            frame = frame.merge(targets.frame, on="sample_id", how="left", validate="many_to_one")
            target_columns = targets.target_columns
        if channel_qc is not None:
            keys = ["recording", "channel_index", "source_channel_index", "channel"]
            missing_keys = [
                key for key in keys
                if key not in frame.columns or key not in channel_qc.frame.columns
            ]
            if missing_keys:
                raise ValidationError(f"channel QC identity keys are absent: {missing_keys}")
            if channel_qc.frame.duplicated(keys).any():
                raise ValidationError("channel QC keys are not unique")
            frame = frame.merge(channel_qc.frame, on=keys, how="left", validate="many_to_one")
        else:
            frame["channel_qc_valid"] = True
            frame["channel_qc_reason"] = ""
        if sample_qc is not None:
            keys = ["sample_id", "channel_index"]
            if sample_qc.frame.duplicated(keys).any():
                raise ValidationError("sample QC keys are not unique")
            frame = frame.merge(sample_qc.frame, on=keys, how="left", validate="one_to_one")
        else:
            frame["sample_qc_valid"] = True
            frame["sample_qc_reason"] = ""
        frame["analysis_valid"] = (
            frame["channel_qc_valid"].fillna(False).astype(bool)
            & frame["sample_qc_valid"].fillna(False).astype(bool)
        )
        preferred = [
            "recording", "sample_id", "sample_kind", "split", "trial_id", "event_id",
            "start_sample", "stop_sample_exclusive", "start_s", "stop_s", "center_s",
            "channel_index", "channel",
        ]
        ordered = [column for column in preferred if column in frame]
        ordered += [column for column in frame if column not in ordered]
        frame = frame.loc[:, ordered]
        return FeatureTable(frame, features.feature_columns, target_columns)
