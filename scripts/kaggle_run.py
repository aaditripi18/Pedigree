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


def ref(job: str) -> str:
    """owner/kernel of a job: the project account, or a friend's account that ran it and shared it back"""
    return f"{JOBS[job].get('owner', USER)}/{JOBS[job]['kernel']}"


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
# diffusion's one-session run hit the 12 h limit (2026-11-08; fresh/scratch unfinished) and
# Kaggle refuses such a session as an input: rebuilt as dev in one GPU session and each fresh
# derivative in its own; the two LoRA ones ran on a friend's account (moinuddinhasan8) with this
# code and were shared back
JOBS["ft-diffusion_unet@dev:gpu"] = _job("pedigree-ft-unet-dev-gpu",
                                         [["-m", "experiments.finetune", "diffusion_unet", "dev"]], gpu=True)
for _k in KINDS:
    JOBS[f"ft-diffusion_unet@fresh:{_k}:gpu"] = _job(
        f"pedigree-ft-unet-fresh-{_k.replace('_', '-')}-gpu",
        [["-m", "experiments.finetune", "diffusion_unet", "fresh", _k]], gpu=True)
for _k in ("lora_r8", "lora_r64"):
    JOBS[f"ft-diffusion_unet@fresh:{_k}:gpu"]["owner"] = "moinuddinhasan8"
# the project account's weekly GPU quota ran out (2026-11-10) after ft_long and scratch; the last
# two ran on friends' accounts with this code and were shared back
JOBS["ft-diffusion_unet@dev:gpu"]["owner"] = "rohajraj"
JOBS["ft-diffusion_unet@fresh:ft_short:gpu"]["owner"] = "alubalub"
JOBS["e1-diffusion_unet"]["sources"] = (["ft-diffusion_unet@dev:gpu"]
                                        + [f"ft-diffusion_unet@fresh:{k}:gpu" for k in KINDS])
JOBS["e1-diffusion_unet"]["env"] = {"PEDIGREE_DERIVATIVES": "merge:derivatives"}
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
# image teachers (ft_long) for E5 live in the CPU sessions of each split
TEACHER_SOURCES = {"vision_cnn": ["ft-vision_cnn@dev", "ft-vision_cnn@fresh"],
                   "vision_transformer": ["ft-vision_transformer@dev:ft_long", "ft-vision_transformer@fresh"]}
for _f, _src in TEACHER_SOURCES.items():
    JOBS[f"distill-{_f}"]["sources"] = list(_src)
    JOBS[f"distill-{_f}"]["env"]["PEDIGREE_TEACHERS"] = "merge:derivatives"
    JOBS[f"e5-{_f}"]["sources"] = list(_src) + [f"distill-{_f}"]
    JOBS[f"e5-{_f}"]["env"]["PEDIGREE_TEACHERS"] = "merge:derivatives"

# H12 (PLAN.md amendment 2026-11-08 (3)): new fresh students (seeds 3-7), then off-task probes
# on every student (E5o) and E5's probes on the new ones (E5n)
NEW_SEEDS = [3, 4, 5, 6, 7]
for _f in E5_FAMILIES:
    JOBS[f"distill2-{_f}"] = _job(f"pedigree-distill2-{_f.replace('_', '-')}",
                                  [["-m", "experiments.distill", _f, "fresh"]], gpu=True,
                                  sources=list(JOBS[f"distill-{_f}"]["sources"]),
                                  env={**JOBS[f"distill-{_f}"]["env"], "PEDIGREE_SEEDS": ",".join(map(str, NEW_SEEDS))})
    # three of the four new-student jobs ran on friends' accounts with this code, shared back
    _owner = {"text_encoder": "alien5060707", "vision_transformer": "rohajraj", "vision_cnn": "alubalub"}
    if _f in _owner:
        JOBS[f"distill2-{_f}"]["owner"] = _owner[_f]
    _teach = [s for s in JOBS[f"e5-{_f}"]["sources"] if not s.startswith("distill-")]
    JOBS[f"h12-{_f}"] = _job(f"pedigree-h12-{_f.replace('_', '-')}",
                             [["-m", "experiments.e5_behaviour", _f, "dev", "fresh"],
                              ["-m", "experiments.e5_behaviour", _f, "fresh"]], gpu=False,
                             sources=_teach + [f"distill-{_f}", f"distill2-{_f}"],
                             env={"PEDIGREE_TEACHERS": JOBS[f"e5-{_f}"]["env"]["PEDIGREE_TEACHERS"],
                                  "PEDIGREE_STUDENTS": "merge:students"})
    # CPU: e5_behaviour never uses the GPU (the vision and decoder runs asked for one needlessly)
    # the two commands differ in probes, seeds and ledger name: set per command below
    JOBS[f"h12-{_f}"]["command_env"] = [
        {"PEDIGREE_PROBES": "offtask", "PEDIGREE_EXPERIMENT": "E5o",
         "PEDIGREE_E5_SEEDS": json.dumps({"dev": [0, 1, 2], "fresh": [0, 1, 2] + NEW_SEEDS})},
        {"PEDIGREE_PROBES": "test", "PEDIGREE_EXPERIMENT": "E5n",
         "PEDIGREE_E5_SEEDS": json.dumps({"fresh": NEW_SEEDS})}]

# E1b (H6 whole files, H8 direction): the same inputs as the E1 job each family used
E1_FORM = {"text_encoder": "e1-text_encoder", "text_decoder": "e1-text_decoder",
           "vision_cnn": "e1-vision_cnn@split", "vision_transformer": "e1-vision_transformer@split",
           "speech": "e1-speech@split", "diffusion_unet": "e1-diffusion_unet"}
JOBS.update({f"e1b-{f}": _job(f"pedigree-e1b-{SHORT[f]}", [["-m", "experiments.e1b_whole", f]], gpu=False,
                              sources=list(JOBS[j]["sources"]), env=dict(JOBS[j]["env"]))
             for f, j in E1_FORM.items()})
# E2, the Hub audit: CPU passes of at most 10 h; each later pass mounts the earlier ones
E2_ENV = {"PEDIGREE_WORKERS": "12", "PEDIGREE_BUDGET_HOURS": "10"}
E2_OUT = ["--out", "/kaggle/working/e2_audit_raw.jsonl"]
JOBS["e2"] = _job("pedigree-e2-audit", [["-m", "experiments.e2_audit", *E2_OUT]], gpu=False, env=dict(E2_ENV))
# Kaggle would not mount the earlier passes' outputs as kernel sources (pass 2 started from 0),
# so later passes read them from a private dataset of first readings (scripts/e2_previous.py)
E2_PREVIOUS = "pedigree-e2-previous"
for _k in range(2, 5):
    JOBS[f"e2-{_k}"] = _job(f"pedigree-e2-audit-{_k}", [["-m", "experiments.e2_audit", *E2_OUT]], gpu=False,
                           env={**E2_ENV, "PEDIGREE_E2_PREVIOUS": "/kaggle/input"})
    JOBS[f"e2-{_k}"]["datasets"] = [E2_PREVIOUS]

# frozen configs, second half (the Hub limits each address to 3000 file requests per 5 minutes)
JOBS["e2-configs"] = _job("pedigree-e2-configs", [["-m", "experiments.e2_configs", "--shard", "1/2", "--rows-out",
                                                  "/kaggle/working/e2_configs_shard1.jsonl"]], gpu=False)
# pool v2: metadata of declared bases outside the frame, second half (the Hub limits each address)
JOBS["bases-meta"] = _job("pedigree-bases-meta", [["-m", "experiments.e2_bases_meta", "--shard", "1/2", "--part",
                                                  "/kaggle/working/e2_bases_meta.part.jsonl"]], gpu=False)
# the behaviour fallback of E2 (needs E2 in the committed ledger): 28 pairs of <= 2B parameters
# fit Kaggle's CPU budget, and the statistics do not depend on the device
JOBS["e2-behaviour"] = _job("pedigree-e2-behaviour", [["-m", "experiments.e2_behaviour"]], gpu=False)
# H6 on the Hub: needs E2 in the committed ledger
JOBS["e2-whole"] = _job("pedigree-e2-whole", [["-m", "experiments.e2_whole"]], gpu=False)
# H7 (a): every family's fresh derivatives (their own headers), merged
JOBS["h7"] = _job("pedigree-h7", [["-m", "experiments.h7_pool"]], gpu=False,
                  sources=sorted({s for f in FAMILIES for s in JOBS[E1_FORM[f]]["sources"]}),
                  env={"PEDIGREE_DERIVATIVES": "merge:derivatives"})

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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
for i, cmd in enumerate(COMMANDS):
    extra = CMD_ENV[i] if i < len(CMD_ENV) else {}
    r = subprocess.run([sys.executable, *cmd], cwd=work, env={**os.environ, **extra})
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


def kaggle(*args: str, check: bool = True, timeout: int = 3600) -> subprocess.CompletedProcess:
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle"))
    py = Path(exe).parent / ("python.exe" if os.name == "nt" else "python")
    # run the CLI as a module of its own interpreter: Windows Application Control can block kaggle.exe
    cmd = [str(py), "-m", "kaggle"] if py.exists() else [exe]
    env = {**os.environ, "PYTHONUTF8": "1"}
    try:
        r = subprocess.run([*cmd, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        # a hung connection once froze a chain for days: report it as the network error it is
        r = subprocess.CompletedProcess([*cmd, *args], 1, "", f"kaggle call timed out after {timeout} s")
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
            f"ENV = {JOBS[job]['env']!r}\nCMD_ENV = {JOBS[job].get('command_env', [])!r}\n" + NOTEBOOK)


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
            "dataset_sources": [f"{USER}/{DATASET}", *(f"{USER}/{x}" for x in spec.get("datasets", []))],
            "competition_sources": [],
            "kernel_sources": [ref(s) for s in spec["sources"]]}), encoding="utf-8")
        print(kaggle("kernels", "push", "-p", str(d)).stdout.strip())


def status(job: str) -> str:
    out = kaggle("kernels", "status", ref(job)).stdout.strip()
    print(out)
    return out


def pull(job: str) -> None:
    """Ledger lines, failure list and every derivative's meta.json (not the weights)."""
    dest = ROOT / "results" / "kaggle" / job.replace(":", "__")  # ":" is not allowed in Windows names
    with tempfile.TemporaryDirectory() as d:
        kaggle("kernels", "output", ref(job), "-p", d, "--file-pattern",
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
