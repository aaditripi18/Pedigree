"""Merge the E2 audit passes into one file of first readings and upload it for the next pass.

    python scripts/e2_previous.py results/kaggle/e2 results/kaggle/e2-2 [...]   (oldest first)

Each model keeps its first reading. Models read again by a later pass (pass 2 could not mount
pass 1, so it re-read 457 of them) are compared field by field; the comparison is written to
results/e2_rereads.json, a measure of how much the Hub moved between passes. Passes from 3 on
start from this file and copy its readings, so only passes 1 and 2 are independent readings:
results/e2_rereads.json is computed from those two.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import kaggle_run as k  # noqa: E402

NAME = "e2_audit_raw.jsonl"


def main(dirs: list[str]) -> None:
    first, later = {}, []
    for d in dirs:
        for line in (Path(d) / NAME).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["id"] in first:
                later.append(r)
            else:
                first[r["id"]] = r
    fields, changed = {}, []
    for r in later:
        diff = sorted(f for f in set(r) | set(first[r["id"]]) if r.get(f) != first[r["id"]].get(f))
        if diff:
            changed.append({"id": r["id"], "fields": diff})
            for f in diff:
                fields[f] = fields.get(f, 0) + 1
    if len(dirs) <= 2:   # later passes copy earlier readings, which are not re-reads
        (ROOT / "results" / "e2_rereads.json").write_text(json.dumps(
            {"passes": dirs, "first_readings": len(first), "reread": len(later),
             "identical": len(later) - len(changed), "changed_fields": fields, "changed": changed}, indent=1) + "\n",
            encoding="utf-8")
    print(f"{len(first)} models; {len(later)} re-read, {len(later) - len(changed)} identical; fields {fields}")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / NAME).write_text("".join(json.dumps(r) + "\n" for r in first.values()), encoding="utf-8")
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": k.E2_PREVIOUS, "id": f"{k.USER}/{k.E2_PREVIOUS}", "licenses": [{"name": "CC0-1.0"}]}),
            encoding="utf-8")
        ds = f"{k.USER}/{k.E2_PREVIOUS}"
        if k.kaggle("datasets", "status", ds, check=False).returncode == 0:
            print(k.kaggle("datasets", "version", "-p", str(d), "-m", f"{len(first)} first readings").stdout.strip())
        else:
            print(k.kaggle("datasets", "create", "-p", str(d)).stdout.strip())
        shutil.copy2(d / NAME, ROOT / "results" / "e2_previous.jsonl")


if __name__ == "__main__":
    main(sys.argv[1:])
