"""Minimal deterministic executor for this repository's print-oriented notebook.

This is used only because nbformat/nbclient are unavailable in the bundled runtime.
It executes code cells in one shared namespace, captures stdout/stderr, records
tracebacks, and writes standard Jupyter stream/error outputs back into the ipynb.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import traceback


def execute(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(path)}
    execution_count = 0
    failed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        stdout = io.StringIO()
        stderr = io.StringIO()
        outputs = []
        try:
            source = "".join(cell.get("source", []))
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"{path.name}:cell-{execution_count}", "exec"), namespace)
        except Exception as exc:
            failed = True
            outputs.append({
                "output_type": "error",
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": traceback.format_exc().splitlines(),
            })
        if stdout.getvalue():
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": stdout.getvalue()})
        if stderr.getvalue():
            outputs.append({"output_type": "stream", "name": "stderr", "text": stderr.getvalue()})
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
        if failed:
            break
    notebook.setdefault("metadata", {}).setdefault("execution", {})["stdlib_executor"] = True
    notebook["metadata"]["execution"]["completed"] = not failed
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    if failed:
        raise SystemExit("Notebook execution failed; inspect the final executed cell.")


if __name__ == "__main__":
    execute(Path(sys.argv[1]))

