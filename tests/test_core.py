import tempfile
import unittest

import numpy as np

from seegops.core import Signal, ValidationError
from seegops.tables import Event, EventTable, Trial, TrialTable
from seegops.operators import (
    BandPower,
    BipolarReference,
    CommonAverageReference,
    FFTPowerSpectrum,
    Epoch,
    Window,
)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        fs = 1000.0
        t = np.arange(2000) / fs
        data = np.stack([
            np.sin(2 * np.pi * 10 * t),
            2 * np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 40 * t),
        ])
        self.signal = Signal(
            data=data,
            dims=("channel", "time"),
            coords={"channel": np.array(["A1", "A2", "A3"]), "time": t},
            sampling_rate=fs,
            unit="uV",
        )

    def test_bipolar_is_linear_difference(self):
        result = BipolarReference()(self.signal)
        np.testing.assert_allclose(result.data[0], self.signal.data[0] - self.signal.data[1])
        self.assertEqual(result.coords["channel"].tolist(), ["A1-A2", "A2-A3"])
        self.assertEqual(result.attrs["reference_matrix"], [[1, -1, 0], [0, 1, -1]])

    def test_car_has_zero_channel_mean(self):
        result = CommonAverageReference()(self.signal)
        np.testing.assert_allclose(result.data.mean(axis=0), 0.0, atol=1e-12)

    def test_spectrum_recovers_ten_hz(self):
        spectrum = FFTPowerSpectrum()(self.signal)
        peak = spectrum.coords["frequency"][np.argmax(spectrum.data[0])]
        self.assertAlmostEqual(float(peak), 10.0)

    def test_cycle_constraint_rejects_short_window(self):
        spectrum = FFTPowerSpectrum()(self.signal)
        with self.assertRaises(ValidationError):
            BandPower(4, 8, require_cycles=4, source_duration_s=0.2)(spectrum)

    def test_window_uses_named_dimensions(self):
        result = Window(0.5, 0.25)(self.signal)
        self.assertEqual(result.dims, ("channel", "window", "time"))
        self.assertEqual(result.data.shape, (3, 7, 500))

    def test_epoch_is_distinct_from_trial_and_retains_links(self):
        trials = TrialTable([
            Trial("trial-1", 0.5, 1.5, condition="success"),
            Trial("trial-2", 1.5, 2.0, condition="failure"),
        ])
        events = EventTable([
            Event("feedback-1", "feedback", 1.0, trial_id="trial-1"),
            Event("target-1", "target", 0.7, trial_id="trial-1"),
            Event("feedback-2", "feedback", 1.8, trial_id="trial-2"),
        ], trials)
        result = Epoch(
            tmin_s=-0.2,
            tmax_s=0.2,
            events=events,
            event_type="feedback",
        )(self.signal)
        self.assertEqual(result.dims, ("channel", "epoch", "time"))
        self.assertEqual(result.data.shape, (3, 2, 400))
        self.assertEqual(result.coords["epoch"].tolist(), ["feedback-1", "feedback-2"])
        self.assertEqual(result.coords["trial_id"].tolist(), ["trial-1", "trial-2"])
        self.assertEqual(len(result.attrs["trial_table"]), 2)

    def test_epoch_rejects_trial_boundary_crossing(self):
        trials = TrialTable([Trial("trial-1", 0.5, 1.1)])
        events = EventTable([Event("feedback-1", "feedback", 1.0, trial_id="trial-1")], trials)
        with self.assertRaises(ValidationError):
            Epoch(-0.2, 0.2, events=events)(self.signal)

    def test_epoch_can_flag_overlap(self):
        events = EventTable([
            Event("e1", "pulse", 0.8),
            Event("e2", "pulse", 0.9),
        ])
        result = Epoch(-0.2, 0.2, events=events, overlap="flag")(self.signal)
        self.assertEqual(result.coords["overlaps_another_epoch"].tolist(), [True, True])

    def test_legacy_event_samples_now_produce_epoch_dimension(self):
        result = Epoch(-0.1, 0.1, event_samples=np.array([500, 1000]))(self.signal)
        self.assertEqual(result.dims, ("channel", "epoch", "time"))
        self.assertEqual(result.coords["event_sample"].tolist(), [500, 1000])

    def test_event_must_lie_inside_linked_trial(self):
        trials = TrialTable([Trial("trial-1", 0.5, 1.0)])
        with self.assertRaises(ValidationError):
            EventTable([Event("late", "feedback", 1.1, trial_id="trial-1")], trials)

    def test_event_table_from_state_edges(self):
        state = np.array([0, 0, 1, 1, 0, 2, 0])
        events = EventTable.from_state_edges(
            state,
            sampling_rate=2,
            event_type="feedback",
        )
        self.assertEqual([row.onset_s for row in events.events], [1.0, 2.5])
        self.assertEqual(events.events[0].metadata["source_sample"], 2)


if __name__ == "__main__":
    unittest.main()
