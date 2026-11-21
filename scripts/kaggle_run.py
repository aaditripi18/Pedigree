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
TEXT = ("text_encoder", "text_decoder")   # teachers are Hub bases (experiments/distill.py)


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
# E2r: E2's comparisons that ended in a reader error, scored again with the fixed readers
JOBS["e2-rescore"] = _job("pedigree-e2-rescore", [["-m", "experiments.e2_rescore"]], gpu=False,
                          env={"PEDIGREE_WORKERS": "8"})
# E2g: E2's GGUF comparisons that compared too few tensors, with the multimodal name fix (2026-11-17)
JOBS["e2-gguf"] = _job("pedigree-e2-gguf", [["-m", "experiments.e2_gguf_rescore"]], gpu=False,
                       env={"PEDIGREE_WORKERS": "8"})
# E2p: E2's comparisons a legacy (pre-zip) PyTorch file made unreadable, read again (2026-11-17)
JOBS["e2-legacy"] = _job("pedigree-e2-legacy", [["-m", "experiments.e2_rescore", "--match", "legacy pickle",
                                                 "--name", "E2p"]], gpu=False, env={"PEDIGREE_WORKERS": "8"})
# H13: the fresh sample, each model read with its declared bases (pool v2 is judged afterwards)
JOBS["h13"] = _job("pedigree-h13", [["-m", "experiments.h13_fresh", "--out", "/kaggle/working/h13_raw.jsonl"]],
                   gpu=False, env=dict(E2_ENV))
# H15 / H17a (amendment 2026-11-17 (4)): the fresh direction sample, GGUF models also aligned both ways
JOBS["h15"] = _job("pedigree-h15", [["-m", "experiments.h15_fresh", "--out", "/kaggle/working/h15_raw.jsonl"]],
                   gpu=False, env=dict(E2_ENV))
# E2s / H17b: E2's GGUF comparisons that still align too few tensors, with structural alignment
JOBS["e2-struct"] = _job("pedigree-e2-struct", [["-m", "experiments.e2_struct_rescore"]], gpu=False,
                         env={"PEDIGREE_WORKERS": "8"})
# E2f (amendment 2026-11-17 (4)): comparisons that failed on GGUF types newer than the gguf package,
# read again with the types decoded from their public sources (pedigree/ggufextra.py)
JOBS["e2-fork"] = _job("pedigree-e2-fork", [["-m", "experiments.e2_rescore", "--source", "E2r,E2g,E2f", "--match",
                                             "tensor type", "--name", "E2f"]], gpu=False, env={"PEDIGREE_WORKERS": "8"})
# E2s, second pass: E2s comparisons that failed on a format decoded since (GGUF types of pedigree/ggufextra.py,
# safetensors F8_E8M0 scales), scored again with E2s's alignment
JOBS["e2-struct2"] = _job("pedigree-e2-struct2", [["-m", "experiments.e2_rescore", "--source", "E2s", "--match",
                                                   "tensor type|F8_E8M0", "--name", "E2s", "--alignment", "auto"]],
                          gpu=False, env={"PEDIGREE_WORKERS": "8"})
# H19 / H20 (amendment 2026-11-18): the last direction and alignment follow-ups, one fresh sample
JOBS["h19"] = _job("pedigree-h19", [["-m", "experiments.h19_fresh", "--out", "/kaggle/working/h19_raw.jsonl"]],
                   gpu=False, env=dict(E2_ENV))
# amendment 2026-11-20: E2x/E2xp/E2v3 (R1 readers, v3, R2 facts) and H23's fresh sample
# E2x and H23 ran before the readers of amendment 2026-11-20 (2): reproduced with them off
JOBS["e2x"] = _job("pedigree-e2x", [["-m", "experiments.e2x_reaudit", "--out", "/kaggle/working"]],
                   gpu=False, env={**E2_ENV, "PEDIGREE_READERS_V4": "0"})
JOBS["h23"] = _job("pedigree-h23", [["-m", "experiments.h23_fresh", "--out", "/kaggle/working/h23_raw.jsonl"]],
                   gpu=False, env={**E2_ENV, "PEDIGREE_READERS_V4": "0"})
# H25 (amendment 2026-11-20 (2)): the fresh sample, then H23's design pairs under v4
JOBS["h25"] = _job("pedigree-h25", [["-m", "experiments.h25_fresh", "--out", "/kaggle/working/h25_raw.jsonl"],
                                    ["-m", "experiments.h25_fresh", "--design",
                                     "--out", "/kaggle/working/h25_design_raw.jsonl"]],
                   gpu=False, env={**E2_ENV, "PEDIGREE_BUDGET_HOURS": "5.5"})   # two passes in one 12 h session
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



# H14 (PLAN.md amendment 2026-11-17 (2)): error agreement and dark knowledge. E5e scores the dev
# students (seeds 0-2) to choose the rule; new fresh students (seeds 8-12, never seen) judge it.
H14_SEEDS = [8, 9, 10, 11, 12]
for _f in E5_FAMILIES:
    _teach = [s for s in JOBS[f"e5-{_f}"]["sources"] if not s.startswith("distill-")]
    JOBS[f"e5e-dev-{_f}"] = _job(f"pedigree-e5e-dev-{_f.replace('_', '-')}",
                                 [["-m", "experiments.e5_behaviour", _f, "dev"]], gpu=False,
                                 sources=_teach + [f"distill-{_f}"],
                                 env={"PEDIGREE_TEACHERS": JOBS[f"e5-{_f}"]["env"]["PEDIGREE_TEACHERS"],
                                      "PEDIGREE_STUDENTS": "merge:students", "PEDIGREE_EXPERIMENT": "E5e-dev",
                                      "PEDIGREE_E5_SEEDS": json.dumps({"dev": [0, 1, 2]})})
    JOBS[f"distill3-{_f}"] = _job(f"pedigree-distill3-{_f.replace('_', '-')}",
                                  [["-m", "experiments.distill", _f, "fresh"]], gpu=True,
                                  sources=list(JOBS[f"distill-{_f}"]["sources"]),
                                  env={**JOBS[f"distill-{_f}"]["env"], "PEDIGREE_SEEDS": ",".join(map(str, H14_SEEDS))})
    # the new students train on friends' GPUs with this code and are shared back
    JOBS[f"distill3-{_f}"]["owner"] = {"text_encoder": "alien5060707", "text_decoder": "moinuddinhasan8",
                                       "vision_transformer": "rohajraj", "vision_cnn": "alubalub"}[_f]
    JOBS[f"e5e-fresh-{_f}"] = _job(f"pedigree-e5e-fresh-{_f.replace('_', '-')}",
                                   [["-m", "experiments.e5_behaviour", _f, "fresh"]], gpu=False,
                                   sources=_teach + [f"distill3-{_f}"],
                                   env={"PEDIGREE_TEACHERS": JOBS[f"e5-{_f}"]["env"]["PEDIGREE_TEACHERS"],
                                        "PEDIGREE_STUDENTS": "merge:students", "PEDIGREE_EXPERIMENT": "E5e-fresh",
                                        "PEDIGREE_E5_SEEDS": json.dumps({"fresh": H14_SEEDS})})


# --- GPU follow-ups (amendments 2026-11-17 (4) and 2026-11-20): H16, H16b, H24 --------------------
# H16: fresh students seeds 13-17 (suspects) and 18-20 (references), trained exactly as distill3
H16_SEEDS = list(range(13, 21))
for _f in ("text_decoder", "vision_cnn"):
    _teach = [s for s in JOBS[f"e5-{_f}"]["sources"] if not s.startswith("distill-")]
    # 2026-11-21: a text family's teacher is its Hub base (experiments/distill.py), so it mounts no
    # output; parts of <= 2.6 h so a session fits a small weekly GPU quota
    _parts = {"": H16_SEEDS}       # text_decoder's 8 seeds take ~2.4 h, run as one part (2026-11-21)
    for _sfx, _seeds in _parts.items():
        JOBS[f"distill4-{_f}{_sfx}"] = _job(
            f"pedigree-distill4-{_f.replace('_', '-')}{_sfx}", [["-m", "experiments.distill", _f, "fresh"]],
            gpu=True, sources=[] if _f in TEXT else list(JOBS[f"distill-{_f}"]["sources"]),
            env={**{k: v for k, v in JOBS[f"distill-{_f}"]["env"].items() if not (_f in TEXT and k == "PEDIGREE_TEACHERS")},
                 "PEDIGREE_SEEDS": ",".join(map(str, _seeds))})
    JOBS[f"e5e-h16-{_f}"] = _job(f"pedigree-e5e-h16-{_f.replace('_', '-')}",
                                 [["-m", "experiments.e5_behaviour", _f, "fresh"]], gpu=False,
                                 sources=_teach + [j for j in JOBS if j.startswith(f"distill4-{_f}")],
                                 env={"PEDIGREE_TEACHERS": JOBS[f"e5-{_f}"]["env"]["PEDIGREE_TEACHERS"],
                                      "PEDIGREE_STUDENTS": "merge:students", "PEDIGREE_EXPERIMENT": "E5e-h16",
                                      "PEDIGREE_E5_SEEDS": json.dumps({"fresh": H16_SEEDS})})
# H16b: 20,000-step students, dev batch seeds 30-32, fresh 33-40, in parts that fit a 12 h session
LONG = {"PEDIGREE_STEPS": "20000", "PEDIGREE_LR": "5e-4", "PEDIGREE_WARMUP": "1000"}
H16B_PARTS = {"dev": [30, 31, 32], "fresh-a": [33, 34, 35], "fresh-b": [36, 37, 38], "fresh-c": [39, 40]}
# 2026-11-21: a text-encoder seed takes ~2.6 h at 20,000 steps (10x the 2,000-step runs' 930 s), so
# its parts are one seed each; it mounts no output (its teacher is the Hub base)
H16B_FAMILY_PARTS = {"text_encoder": {f"s{x}": [x] for x in range(30, 41)}, "vision_transformer": H16B_PARTS}
for _f in ("text_encoder", "vision_transformer"):
    _teach = [s for s in JOBS[f"e5-{_f}"]["sources"] if not s.startswith("distill-")]
    for _part, _seeds in H16B_FAMILY_PARTS[_f].items():
        JOBS[f"h16b-{_f}-{_part}"] = _job(
            f"pedigree-h16b-{SHORT[_f]}-{_part}", [["-m", "experiments.distill", _f, "h16b"]], gpu=True,
            sources=[] if _f in TEXT else list(JOBS[f"distill-{_f}"]["sources"]),
            env={**{k: v for k, v in JOBS[f"distill-{_f}"]["env"].items() if not (_f in TEXT and k == "PEDIGREE_TEACHERS")},
                 **LONG, "PEDIGREE_SEEDS": ",".join(map(str, _seeds))})
    JOBS[f"e5e-h16b-{_f}"] = _job(f"pedigree-e5e-h16b-{SHORT[_f]}",
                                  [["-m", "experiments.e5_behaviour", _f, "h16b"]], gpu=False,
                                  sources=_teach + [f"h16b-{_f}-{p}" for p in H16B_FAMILY_PARTS[_f]],
                                  env={"PEDIGREE_TEACHERS": JOBS[f"e5-{_f}"]["env"]["PEDIGREE_TEACHERS"],
                                       "PEDIGREE_STUDENTS": "merge:students", "PEDIGREE_EXPERIMENT": "E5e-h16b",
                                       "PEDIGREE_E5_SEEDS": json.dumps({"h16b": list(range(30, 41))})})
# H24: speech and diffusion students (experiments/h24.py), dev seeds 50-52, fresh 53-60
# 2026-11-21: one seed per part (both students), so a session fits a small weekly GPU quota; a
# 300-step timing run per family (seed 99, never scored) measures a step before any long run
H24_PARTS = {f"s{x}": [x] for x in range(50, 61)}
for _f in ("speech", "diffusion_unet"):
    for _part, _seeds in H24_PARTS.items():
        JOBS[f"h24-{_f}-{_part}"] = _job(f"pedigree-h24-{SHORT[_f]}-{_part}",
                                         [["-m", "experiments.h24", "train", _f, ",".join(map(str, _seeds))]], gpu=True,
                                         env={"PEDIGREE_DERIVATIVES": "/kaggle/working/students"})
    JOBS[f"e5e-h24-{_f}"] = _job(f"pedigree-e5e-h24-{SHORT[_f]}",
                                 [["-m", "experiments.h24", "score", _f, ",".join(map(str, range(50, 61)))]],
                                 gpu=False, sources=[f"h24-{_f}-{p}" for p in H24_PARTS],
                                 env={"PEDIGREE_STUDENTS": "merge:students"})
for _f in ("speech", "diffusion_unet"):
    JOBS[f"h24-time-{_f}"] = _job(f"pedigree-h24-{SHORT[_f]}-time",
                                  [["-m", "experiments.h24", "train", _f, "99"]], gpu=True,
                                  env={"PEDIGREE_DERIVATIVES": "/kaggle/working/students", "PEDIGREE_STEPS": "300",
                                       "PEDIGREE_WARMUP": "15"})
# a CPU smoke run of H24's code (3 steps, few probes); its ledger lines are never pulled
JOBS["h24-smoke"] = _job("pedigree-h24-smoke",
                         [["-m", "experiments.h24", "train", "diffusion_unet", "99"],
                          ["-m", "experiments.h24", "score", "diffusion_unet", "99"],
                          ["-m", "experiments.h24", "train", "speech", "99"],
                          ["-m", "experiments.h24", "score", "speech", "99"]], gpu=False,
                         env={"PEDIGREE_STEPS": "3", "PEDIGREE_WARMUP": "1", "PEDIGREE_H24_SHARDS": "1",
                              "PEDIGREE_H24_DIFF_PROBES": "20", "PEDIGREE_H24_SPEECH_PROBES": "3"})

# GPU follow-ups run on friends' accounts (their own keys, never held here): who runs which job, so
# pulls and the CPU scoring jobs mount the right owner's output. Each friend shares the finished
# notebook with the project account as viewer.
GPU_ASSIGN = {   # 2026-11-21: by measured hours and each friend's weekly quota (alien5060707: 6 h this week);
    # only rohajraj's ViT jobs mount outputs (already shared with that account in earlier rounds)
    "alien5060707": ["h16b-text_encoder-s30", "h16b-text_encoder-s31"],
    "rohajraj": ["h16b-vision_transformer-dev", "h16b-vision_transformer-fresh-a",
                 "h16b-vision_transformer-fresh-b", "h16b-vision_transformer-fresh-c",
                 "h16b-text_encoder-s32", "h16b-text_encoder-s33"],
    "alubalub": ["h24-time-speech", "distill4-vision_cnn", "h16b-text_encoder-s34", "h16b-text_encoder-s35",
                 "h16b-text_encoder-s36", "h16b-text_encoder-s39"],
    "moinuddinhasan8": ["h24-time-diffusion_unet", "distill4-text_decoder", "h16b-text_encoder-s37",
                        "h16b-text_encoder-s38", "h16b-text_encoder-s40"],
}
# estimated GPU hours per job, from the measured 2,000-step runs (x10 for 20,000 steps)
GPU_HOURS = {**{f"h16b-text_encoder-s{x}": 2.6 for x in range(30, 41)},
             "h16b-vision_transformer-dev": 2.1, "h16b-vision_transformer-fresh-a": 2.1,
             "h16b-vision_transformer-fresh-b": 2.1, "h16b-vision_transformer-fresh-c": 1.4,
             "distill4-text_decoder": 2.4, "distill4-vision_cnn": 0.2,
             "h24-time-speech": 0.3, "h24-time-diffusion_unet": 0.3}
for _friend, _jobs in GPU_ASSIGN.items():
    for _j in _jobs:
        JOBS[_j]["owner"] = _friend
HANDOFF = ROOT.parent / "Pedigree_handoff"


def kit(job: str, sha: str | None = None) -> Path:
    """A friend's handoff folder for one job: run.py and kernel-metadata.json under their account."""
    spec = JOBS[job]
    owner = spec.get("owner")
    if not owner or owner == USER:
        sys.exit(f"{job} has no friend assigned (GPU_ASSIGN)")
    sha = sha or git("rev-parse", "--short", "HEAD")
    d = HANDOFF / owner / spec["kernel"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(notebook_source(job, sha), encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{spec['kernel']}", "title": spec["kernel"], "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": spec["gpu"], "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{USER}/{DATASET}"], "competition_sources": [],
        "kernel_sources": [ref(s) for s in spec["sources"]]}, indent=1), encoding="utf-8")
    print(f"{job}: {d}")
    return d


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "push":
        push()
    elif cmd == "kits":                      # every assigned GPU job, at the committed code
        for _jobs in GPU_ASSIGN.values():
            for j in _jobs:
                kit(j)
    else:
        for j in sys.argv[2:]:
            {"run": run, "status": status, "pull": pull, "kit": kit}[cmd](j)
