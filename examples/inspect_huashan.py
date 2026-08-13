"""Inspect the paired neural and grip-task streams without loading them into RAM."""

from pathlib import Path
import json

from seegops.io import read_bci2000
from seegops.qc import basic_qc
from seegops.sync import align_by_storage_time, rising_edges


ROOT = Path("/Users/heting/Documents/readGripData/0807华山grip flight")


def main():
    for run in ("09", "11"):
        neural = read_bci2000(ROOT / f"testS001R{run}.dat.larkcache")
        task = read_bci2000(ROOT / f"testS001R{run}_1.dat", channel_names=["GripForce1"])
        alignment = align_by_storage_time(neural, task)
        phase = task.state("GamePhase")
        report = {
            "run": run,
            "neural": neural.summary(),
            "task": task.summary(),
            "neural_qc_first_10s": basic_qc(neural.signal, seconds=10),
            "task_states": {
                "GamePhase_values": sorted(set(map(int, phase))),
                "Feedback_rising_edges": rising_edges(task.state("Feedback")).tolist(),
                "Collision_rising_edges": rising_edges(task.state("Collision")).tolist(),
            },
            "start_alignment": alignment.__dict__,
            "warning": (
                "StorageTime only establishes a coarse start offset. Do not resample or merge "
                "streams until a shared state, pulse, or clock mapping validates drift."
            ),
        }
        output = Path(__file__).parents[1] / "reports" / f"run-{run}-inspection.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()

