"""Every GPU job assigned to a friend (scripts/kaggle_run.py GPU_ASSIGN): its Kaggle status as the
project account sees it ("not shared" until the friend shares the notebook), and whether its output
has been pulled. One line per job; `--changes FILE` prints only jobs whose line differs from FILE's
and updates FILE."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_run as K  # noqa: E402


def state(job: str) -> str:
    r = K.kaggle("kernels", "status", K.ref(job), check=False, timeout=300)
    out = (r.stdout + r.stderr)
    for s in ("COMPLETE", "RUNNING", "QUEUED", "ERROR", "CANCEL"):
        if s in out:
            return s
    if "denied" in out or "403" in out or "404" in out:
        return "not shared"
    return "unknown (network)"


def main() -> None:
    lines = {}
    for friend, jobs in K.GPU_ASSIGN.items():
        for j in jobs:
            pulled = (K.ROOT / "results" / "kaggle" / j.replace(":", "__") / "failed.txt").exists()
            lines[j] = f"{friend} {j}: {state(j)}{' (pulled)' if pulled else ''}"
    if len(sys.argv) > 2 and sys.argv[1] == "--changes":
        f = Path(sys.argv[2])
        old = json.loads(f.read_text()) if f.exists() else {}
        for j, s in lines.items():
            if old.get(j) != s and "unknown" not in s:
                print(s, flush=True)
        f.write_text(json.dumps({j: s if "unknown" not in s else old.get(j, s) for j, s in lines.items()}))
    else:
        print("\n".join(lines.values()))


if __name__ == "__main__":
    main()
