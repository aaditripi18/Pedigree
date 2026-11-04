"""Start Kaggle jobs one after another as GPU slots free up (the account allows 2 at once,
shared with other projects). Each job is started once; its first status is printed so a
failed start is seen at once.

    python scripts/kaggle_queue.py SHA JOB [JOB ...]
"""
from __future__ import annotations

import sys
import time

from scripts import kaggle_run as k

WAIT = 600


def start(job: str, sha: str) -> bool:
    spec = k.JOBS[job]
    import json
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "run.py").write_text(k.notebook_source(job, sha), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{k.USER}/{spec['kernel']}", "title": spec["kernel"], "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": spec["gpu"], "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [f"{k.USER}/{k.DATASET}"], "competition_sources": [],
            "kernel_sources": [f"{k.USER}/{k.JOBS[s]['kernel']}" for s in spec["sources"]]}), encoding="utf-8")
        r = k.kaggle("kernels", "push", "-p", str(d), check=False)
    out = (r.stdout + r.stderr).strip()
    if "successfully pushed" in out:
        print(f"{time.strftime('%H:%M')} started {job}", flush=True)
        return True
    if "Maximum batch" in out or "session count" in out:
        return False
    print(f"{time.strftime('%H:%M')} FAILED to start {job}: {out}", flush=True)
    raise SystemExit(1)


def main(sha: str, jobs: list[str]) -> None:
    for job in jobs:
        while not start(job, sha):
            time.sleep(WAIT)
        time.sleep(120)
        print(f"{time.strftime('%H:%M')} {job}: {k.status(job)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
