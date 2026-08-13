from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_bci2000
from .qc import basic_qc


def main() -> None:
    parser = argparse.ArgumentParser(prog="seegops")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="inspect a BCI2000 recording")
    inspect.add_argument("path")
    inspect.add_argument("--qc-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.command == "inspect":
        recording = read_bci2000(args.path)
        result = recording.summary()
        result["qc"] = basic_qc(recording.signal, seconds=args.qc_seconds)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

