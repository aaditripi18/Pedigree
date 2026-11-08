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
            "kernel_sources": [k.ref(s) for s in spec["sources"]]}), encoding="utf-8")
        r = k.kaggle("kernels", "push", "-p", str(d), check=False)
    out = (r.stdout + r.stderr).strip()
    if "successfully pushed" in out:
        print(f"{time.strftime('%H:%M')} started {job}", flush=True)
        return True
    if "Maximum batch" in out or "session count" in out:
        return False
    if any(t in out for t in ("Connection aborted", "RemoteDisconnected", "timed out", "getaddrinfo",
                              "Max retries", "ConnectionError", "503", "502")):
        print(f"{time.strftime('%H:%M')} network error starting {job}; trying again later", flush=True)
        return False
    print(f"{time.strftime('%H:%M')} FAILED to start {job}: {out}", flush=True)
    raise SystemExit(1)


def sources_done(job: str) -> bool:
    """A job starts only when every session it reads from has finished."""
    for src in k.JOBS[job]["sources"]:
        try:
            st = k.kaggle("kernels", "status", k.ref(src), check=False).stdout
        except Exception:
            return False
        if "COMPLETE" not in st:
            return False
    return True


def main(sha: str, jobs: list[str]) -> None:
    for job in jobs:
        while not (sources_done(job) and start(job, sha)):
            time.sleep(WAIT)
        time.sleep(120)
        print(f"{time.strftime('%H:%M')} {job}: {k.status(job)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
