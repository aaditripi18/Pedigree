"""Run Pedigree's heavy experiments on Kaggle (the author's own account only).

  python scripts/kaggle_run.py push                 # upload the committed code (git archive + COMMIT stamp)
  python scripts/kaggle_run.py run JOB              # push, then start JOB's notebook
  python scripts/kaggle_run.py status JOB
  python scripts/kaggle_run.py pull JOB             # new ledger lines -> results/ledger.jsonl; small files -> results/kaggle/JOB/

Jobs:
  ft-<family>   GPU: build E1's ground-truth derivatives of the family's dev and fresh
                bases (experiments/finetune.py). The safetensors stay in the notebook's
                output (they are tens of GB); later jobs mount that output.

Only committed code runs: the upload is `git archive HEAD`, stamped with the commit,
so every result names the exact code. The Kaggle API key stays in ~/.kaggle/kaggle.json
and is read only by the Kaggle CLI.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = os.environ.get("PEDIGREE_KAGGLE_USER", "aadityae5545454")
DATASET = "pedigree-code"
LEDGER = ROOT / "results" / "ledger.jsonl"
FAMILIES = ["text_encoder", "text_decoder", "vision_transformer", "vision_cnn", "speech", "diffusion_unet"]


def _job(kernel: str, commands: list[list[str]], gpu: bool, sources: list[str] | None = None,
         env: dict | None = None) -> dict:
    return {"kernel": kernel, "commands": commands, "gpu": gpu, "sources": sources or [], "env": env or {}}


JOBS = {f"ft-{f}": _job(f"pedigree-ft-{f.replace('_', '-')}",
                        [["-m", "experiments.finetune", f, "dev"], ["-m", "experiments.finetune", f, "fresh"]],
                        gpu=True)
        for f in FAMILIES}
# one base per session, on CPU (no GPU quota), for when the weekly GPU quota is spent;
# the E1 job of a family then mounts both splits' outputs
JOBS.update({f"ft-{f}@{s}": _job(f"pedigree-ft-{f.replace('_', '-')}-{s}", [["-m", "experiments.finetune", f, s]],
                                 gpu=False)
             for f in FAMILIES for s in ("dev", "fresh")})
KINDS = ["ft_short", "ft_long", "lora_r8", "lora_r64", "scratch"]
# one derivative per CPU session, for bases that do not fit in one 12 h session
SHORT = {"text_encoder": "te", "text_decoder": "td", "vision_transformer": "vit", "vision_cnn": "cnn",
         "speech": "sp", "diffusion_unet": "unet"}
JOBS.update({f"ft-{f}@{s}:{k}": _job(f"pedigree-ft-{SHORT[f]}-{s}-{k.replace('_', '-')}",
                                     [["-m", "experiments.finetune", f, s, k]], gpu=False)
             for f in FAMILIES for s in ("dev", "fresh") for k in KINDS})
# E1 comparisons: CPU, reading the derivatives from the matching ft- job's output
JOBS.update({f"e1-{f}": _job(f"pedigree-e1-{f.replace('_', '-')}", [["-m", "experiments.e1_verify", f]], gpu=False,
                             sources=[f"ft-{f}"], env={"PEDIGREE_DERIVATIVES": "find:derivatives"})
             for f in FAMILIES})
JOBS.update({f"e1-{f}@split": _job(f"pedigree-e1-{f.replace('_', '-')}", [["-m", "experiments.e1_verify", f]],
                                   gpu=False, sources=[f"ft-{f}@dev", f"ft-{f}@fresh"],
                                   env={"PEDIGREE_DERIVATIVES": "merge:derivatives"})
             for f in FAMILIES})
# Kaggle refuses a session stopped by its 12 h limit as an input, so a split that hit
# the limit is rebuilt one derivative per session and E1 reads only finished sessions
JOBS["e1-vision_transformer@split"]["sources"] = (["ft-vision_transformer@fresh"]
                                                  + [f"ft-vision_transformer@dev:{k}" for k in KINDS])
# wav2vec2 takes about a minute per step on CPU: its fresh split needs the GPU
JOBS["ft-speech@fresh:gpu"] = _job("pedigree-ft-speech-fresh-gpu", [["-m", "experiments.finetune", "speech", "fresh"]],
                                   gpu=True)
JOBS["e1-speech@split"]["sources"] = [f"ft-speech@dev:{k}" for k in KINDS] + ["ft-speech@fresh:gpu"]
JOBS["e6e7"] = _job("pedigree-e6-e7", [["-m", "experiments.e6_e7"]], gpu=False)
# E5: students (GPU) written to their own folder; image teachers come from the ft- output
E5_FAMILIES = ["text_encoder", "text_decoder", "vision_transformer", "vision_cnn"]
JOBS.update({f"distill-{f}": _job(f"pedigree-distill-{f.replace('_', '-')}",
                                  [["-m", "experiments.distill", f, "dev"], ["-m", "experiments.distill", f, "fresh"]],
                                  gpu=True, sources=[f"ft-{f}"],
                                  env={"PEDIGREE_DERIVATIVES": "/kaggle/working/students",
                                       "PEDIGREE_TEACHERS": "find:derivatives"})
             for f in E5_FAMILIES})
JOBS.update({f"e5-{f}": _job(f"pedigree-e5-{f.replace('_', '-')}", [["-m", "experiments.e5_behaviour", f]],
                             gpu=True, sources=[f"ft-{f}", f"distill-{f}"],
                             env={"PEDIGREE_TEACHERS": "find:derivatives", "PEDIGREE_STUDENTS": "find:students"})
             for f in E5_FAMILIES})

NOTEBOOK = r'''
import os, pathlib, shutil, subprocess, sys
src = next(pathlib.Path("/kaggle/input").rglob("Pedigree/COMMIT")).parent
work = pathlib.Path("/kaggle/working/Pedigree")
shutil.copytree(src, work)
print("code", (work / "COMMIT").read_text().strip(), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(work)], check=True)
# the transformers version every smoke test ran on (Kaggle's 5.20.2 gave wav2vec2 a NaN loss at step 0)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "peft", "timm", "diffusers",
                "soundfile", "gguf", "scipy"], check=True)
# Kaggle's image ships torchao 0.10, which current peft refuses to run LoRA beside; LoRA does not need it
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
# preflight: every library a job uses, on a toy model, before any long run
PREFLIGHT = (
    "import torch, peft, transformers, timm, diffusers, safetensors\n"
    "m = torch.nn.Sequential(torch.nn.Linear(4, 4))\n"
    "p = peft.get_peft_model(m, peft.LoraConfig(r=2, target_modules=['0']))\n"
    "p(torch.randn(2, 4)).sum().backward(); p.merge_and_unload()\n"
    "print('preflight ok: torch', torch.__version__, 'cuda', torch.cuda.is_available(),\n"
    "      'transformers', transformers.__version__, 'peft', peft.__version__)\n")
subprocess.run([sys.executable, "-c", PREFLIGHT], check=True)
os.environ["PEDIGREE_CACHE"] = "/tmp/pedigree-hub"
os.environ["PEDIGREE_DERIVATIVES"] = "/kaggle/working/derivatives"
os.environ["PEDIGREE_TASKS"] = "/tmp/pedigree-tasks"
os.environ["PYTHONUNBUFFERED"] = "1"
for key, value in ENV.items():
    if value.startswith("merge:"):   # several mounted outputs, one family/split tree
        dest = pathlib.Path("/tmp/merged-" + value[6:])
        for root in pathlib.Path("/kaggle/input").rglob(value[6:]):
            # the code upload holds committed meta.json copies (no weights): never use them
            if not root.is_dir() or "Pedigree" in root.parts:
                continue
            # link each finished derivative (family/split/kind with its weights), so one
            # split may come from several sessions (a base split into one kind per session)
            for kind in root.glob("*/*/*"):
                if not (kind / "model.safetensors").exists():
                    continue
                link = dest / kind.relative_to(root)
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(kind, target_is_directory=True)
        value = str(dest)
    elif value.startswith("find:"):
        value = str(next(p for p in pathlib.Path("/kaggle/input").rglob(value[5:])
                         if p.is_dir() and "Pedigree" not in p.parts))
    os.environ[key] = value
    print(key, "=", value, flush=True)
ledger = work / "results" / "ledger.jsonl"
n0 = len(ledger.read_text(encoding="utf-8").splitlines()) if ledger.exists() else 0
failed = []
for cmd in COMMANDS:
    r = subprocess.run([sys.executable, *cmd], cwd=work)
    print("exit", r.returncode, *cmd, flush=True)
    if r.returncode != 0:
        failed.append(" ".join(cmd))
new = ledger.read_text(encoding="utf-8").splitlines()[n0:] if ledger.exists() else []
pathlib.Path("/kaggle/working/new_ledger_lines.jsonl").write_text("\n".join(new) + "\n", encoding="utf-8")
pathlib.Path("/kaggle/working/failed.txt").write_text("\n".join(failed) + "\n", encoding="utf-8")
print(len(new), "new ledger lines;", len(failed), "failed commands", flush=True)
shutil.rmtree(work)
if failed:
    raise SystemExit("failed: " + "; ".join(failed))
'''


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle"))
    env = {**os.environ, "PYTHONUTF8": "1"}
    r = subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if check and r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args[:2])} failed:\n{r.stdout}{r.stderr}")
    return r


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def push() -> str:
    if git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results"):
        sys.exit("commit your changes first: Kaggle runs only committed code")
    sha = git("rev-parse", "--short", "HEAD")
    ds = f"{USER}/{DATASET}"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        archive = d / f"pedigree-{sha}.zip"
        subprocess.run(["git", "archive", "--format=zip", "--prefix=Pedigree/", "-o", str(archive), "HEAD"],
                       cwd=ROOT, check=True)
        with zipfile.ZipFile(archive, "a") as z:
            z.writestr("Pedigree/COMMIT", sha + "\n")
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": DATASET, "id": ds, "licenses": [{"name": "other"}]}), encoding="utf-8")
        if kaggle("datasets", "status", ds, check=False).returncode == 0:
            kaggle("datasets", "version", "-p", str(d), "-m", f"code {sha}")
        else:
            kaggle("datasets", "create", "-p", str(d))  # private unless --public
    for _ in range(90):  # Kaggle can report "ready" before it serves the new version
        with tempfile.TemporaryDirectory() as d:
            r = kaggle("datasets", "download", ds, "-f", "Pedigree/COMMIT", "-p", d, check=False)
            got = [p for p in Path(d).rglob("*") if p.is_file()]
            if r.returncode == 0 and got:
                data = got[0].read_bytes()
                if data[:2] == b"PK":
                    with zipfile.ZipFile(got[0]) as z:
                        data = z.read(z.namelist()[0])
                if data.decode().strip() == sha:
                    print(f"Kaggle serves code {sha}")
                    return sha
        time.sleep(10)
    sys.exit("Kaggle did not serve the new code within 15 minutes")


def notebook_source(job: str, sha: str) -> str:
    """The script Kaggle runs for one job (tests/test_kaggle_run.py checks it)."""
    return (f"# Pedigree {job}, code {sha}\nCOMMANDS = {JOBS[job]['commands']!r}\n"
            f"ENV = {JOBS[job]['env']!r}\n" + NOTEBOOK)


def run(job: str, sha: str | None = None) -> None:
    spec = JOBS[job]
    sha = sha or push()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "run.py").write_text(notebook_source(job, sha), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{spec['kernel']}", "title": spec["kernel"], "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": spec["gpu"], "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [f"{USER}/{DATASET}"], "competition_sources": [],
            "kernel_sources": [f"{USER}/{JOBS[s]['kernel']}" for s in spec["sources"]]}), encoding="utf-8")
        print(kaggle("kernels", "push", "-p", str(d)).stdout.strip())


def status(job: str) -> str:
    out = kaggle("kernels", "status", f"{USER}/{JOBS[job]['kernel']}").stdout.strip()
    print(out)
    return out


def pull(job: str) -> None:
    """Ledger lines, failure list and every derivative's meta.json (not the weights)."""
    dest = ROOT / "results" / "kaggle" / job.replace(":", "__")  # ":" is not allowed in Windows names
    with tempfile.TemporaryDirectory() as d:
        kaggle("kernels", "output", f"{USER}/{JOBS[job]['kernel']}", "-p", d, "--file-pattern",
               r".*(\.jsonl|\.txt|meta\.json|\.log)$", check=False)
        for f in Path(d).rglob("*"):
            if f.is_file() and f.suffix in (".json", ".jsonl", ".txt", ".log"):
                rel = f.relative_to(d)
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest / rel)
    f = dest / "new_ledger_lines.jsonl"
    new = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()] if f.exists() else []
    have = set(LEDGER.read_text(encoding="utf-8").splitlines()) if LEDGER.exists() else set()
    add = [ln for ln in new if ln not in have]
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as out:
        out.writelines(ln + "\n" for ln in add)
    print(f"{job}: appended {len(add)} ledger lines; files in {dest}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "push":
        push()
    else:
        for j in sys.argv[2:]:
            {"run": run, "status": status, "pull": pull}[cmd](j)
