import unittest
from pathlib import Path

from seegops.io import read_bci2000


DATA = Path("/Users/heting/Documents/readGripData/0807华山grip flight")


class BCI2000IntegrationTests(unittest.TestCase):
    def test_main_stream_metadata(self):
        rec = read_bci2000(DATA / "testS001R09.dat.larkcache")
        self.assertEqual(rec.signal.data.shape, (256, 286800))
        self.assertEqual(rec.signal.sampling_rate, 2000)
        self.assertEqual(rec.parameters["SubjectRun"], "09")

    def test_task_stream_and_state(self):
        rec = read_bci2000(DATA / "testS001R09_1.dat")
        self.assertEqual(rec.signal.data.shape, (1, 31488))
        self.assertEqual(rec.signal.sampling_rate, 256)
        phase = rec.state("GamePhase")
        self.assertEqual(phase.shape, (31488,))
        self.assertGreaterEqual(int(phase.max()), 1)


if __name__ == "__main__":
    unittest.main()

