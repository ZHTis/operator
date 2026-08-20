import unittest

import numpy as np

from seegops import (
    BandFeatureSpec,
    ContinuousWindowSampler,
    Event,
    EventLockedSampler,
    EventTable,
    FeatureBank,
    FeatureTableAssembler,
    ForceTargetBank,
    Pipeline,
    Signal,
    Trial,
    TrialSignalProvider,
    TrialTable,
)
from seegops.operators import NotchFilter
from seegops.qc import channel_qc_from_trials, sample_qc_from_features


class FeatureWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fs = 200.0
        time = np.arange(1200) / self.fs
        self.eeg = Signal(
            data=np.stack([
                np.sin(2 * np.pi * 10 * time) + 0.1 * np.sin(2 * np.pi * 50 * time),
                np.sin(2 * np.pi * 20 * time) + 0.1 * np.sin(2 * np.pi * 50 * time),
            ]),
            dims=("channel", "time"),
            coords={"channel": np.array(["A1", "A2"]), "time": time},
            sampling_rate=self.fs,
            attrs={"recording_id": "synthetic"},
        )
        self.force = Signal(
            data=np.linspace(0, 6, len(time)),
            dims=("time",),
            coords={"time": time},
            sampling_rate=self.fs,
            unit="a.u.",
        )
        self.trials = TrialTable([
            Trial("trial-1", 0.0, 3.0, metadata={"split": "train"}),
            Trial("trial-2", 3.0, 6.0, metadata={"split": "test"}),
        ])

    def test_continuous_sampler_respects_trial_and_split(self):
        samples = ContinuousWindowSampler(1.0, 0.5).build(
            self.eeg, trials=self.trials, recording_id="synthetic"
        )
        self.assertTrue(all(row.trial_id == "trial-1" for row in samples.samples))
        self.assertTrue(all(row.stop_sample_exclusive <= 600 for row in samples.samples))
        self.assertEqual(len(samples), 5)

    def test_event_sampler_retains_links_and_excludes_test(self):
        events = EventTable([
            Event("rise-1", "exertion", 1.5, trial_id="trial-1"),
            Event("rise-2", "exertion", 4.5, trial_id="trial-2"),
        ], self.trials)
        samples = EventLockedSampler(
            -0.5, 0.5, window_length_s=0.5, step_s=0.25
        ).build(self.eeg, events=events, recording_id="synthetic")
        self.assertTrue(all(row.event_id == "rise-1" for row in samples.samples))
        self.assertTrue(all(row.trial_id == "trial-1" for row in samples.samples))
        self.assertEqual(len(samples), 3)

    def test_feature_bank_preserves_channels_and_assembler_joins_qc(self):
        samples = ContinuousWindowSampler(1.0, 0.5).build(
            self.eeg, trials=self.trials, recording_id="synthetic"
        )
        provider = TrialSignalProvider(
            self.eeg,
            self.trials,
            Pipeline().then(NotchFilter((50.0,), quality_factor=30.0)),
        )
        bank = FeatureBank(
            bands=(
                BandFeatureSpec("alpha", 8, 12),
                BandFeatureSpec("beta", 14, 30),
            )
        )
        features = bank.transform(provider, samples)
        self.assertEqual(len(features.frame), len(samples) * 2)
        self.assertEqual(features.frame.channel.nunique(), 2)
        targets = ForceTargetBank().transform(self.force, samples)
        channel_qc = channel_qc_from_trials(
            provider,
            trial_ids=["trial-1"],
            recording_id="synthetic",
            line_frequencies_hz=(50.0,),
            maximum_line_noise_ratio=1e9,
        )
        sample_qc = sample_qc_from_features(features)
        table = FeatureTableAssembler().assemble(
            features,
            targets=targets,
            channel_qc=channel_qc,
            sample_qc=sample_qc,
        )
        self.assertEqual(len(table.frame), len(features.frame))
        self.assertIn("force_slope", table.frame)
        self.assertIn("channel_qc_valid", table.frame)
        self.assertIn("source_channel_index", channel_qc.frame)
        self.assertIn("channel", channel_qc.frame)
        self.assertIn("sample_qc_valid", table.frame)
        self.assertIn("analysis_valid", table.frame)


if __name__ == "__main__":
    unittest.main()
