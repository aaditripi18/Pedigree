"""Start Kaggle jobs once the jobs they read from have finished, without the status endpoint.

Kaggle's status endpoint rate-limits an account for hours after heavy polling. This tells a
finished job by its output instead: a job's small result files (ledger lines, failure list) can
be downloaded only once its session has ended. Each finished source is pulled into the
ledger; each dependent job is started once (retried while GPU slots are full). The script exits
when every job has started, or when a watched job finishes with failures.

    python -m scripts.kaggle_chain SHA JOB [JOB ...]
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from scripts import kaggle_queue as q
from scripts import kaggle_run as k

WAIT = 900


def finished(job: str) -> tuple[bool, list[str]]:
    with tempfile.TemporaryDirectory() as d:
        k.kaggle("kernels", "output", f"{k.USER}/{k.JOBS[job]['kernel']}", "-p", d,
                 "--file-pattern", r".*(\.jsonl|failed\.txt)$", check=False)
        files = [f for f in Path(d).rglob("*") if f.is_file()]
        failed = [ln for f in files if f.name == "failed.txt"
                  for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return bool(files), failed


def main(sha: str, jobs: list[str]) -> None:
    pending, pulled = list(jobs), set()
    while pending:
        for job in list(pending):
            ready = True
            for src in k.JOBS[job]["sources"]:
                if src in pulled:
                    continue
                try:
                    done, failed = finished(src)
                except Exception:
                    done, failed = False, []
                if not done:
                    ready = False
                    break
                k.pull(src)
                pulled.add(src)
                if failed:
                    print(f"{time.strftime('%H:%M')} {src} finished with {len(failed)} failures:",
                          *failed[:10], sep="\n  ", flush=True)
                    raise SystemExit(2)
            if ready and q.start(job, sha):
                pending.remove(job)
        if pending:
            time.sleep(WAIT)
    print(f"{time.strftime('%H:%M')} all started: {', '.join(jobs)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
