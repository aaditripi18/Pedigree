"""Which files of a Hub repository hold its weights: the rule the audit (E2 and every later run)
applies, shared with `pedigree verify` (moved from experiments/e2_audit.py, unchanged, 2026-11-21)."""
from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.parse

from . import formats, ggufremote, hub


class NetworkError(Exception):
    pass


def _network(e: BaseException) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return e.code in (408, 429) or e.code >= 500
    return isinstance(e, (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError)) or \
        (isinstance(e, OSError) and not isinstance(e, FileNotFoundError))


def weight_files(repo: str, d: dict) -> tuple[str, list[str], dict, dict]:
    """(format, files, sizes, sha256) of a repository; format names why there are none."""
    sib = {s["rfilename"]: s for s in d.get("siblings", [])}
    names = [n for n in sib if not n.split("/")[-1].startswith("adapter_model")]
    prefix = ""
    if "model_index.json" in sib:
        for sub in ("unet", "transformer"):
            if any(n.startswith(sub + "/") for n in names):
                prefix = sub + "/"
                break
    inner = [n[len(prefix):] for n in names if n.startswith(prefix)]
    if prefix:   # diffusers names its weights diffusion_pytorch_model.*; read them as model.*
        inner = [n.replace("diffusion_pytorch_model", "model", 1) for n in inner]

    def real(n):
        return prefix + (n.replace("model", "diffusion_pytorch_model", 1) if prefix and n.startswith("model") else n)

    def read_index(idx):
        with hub._open(f"{hub.HOST}/{repo}/resolve/{d['sha']}/{urllib.parse.quote(real(idx))}") as r:
            j = json.loads(r.read().decode("utf-8"))
        if prefix:
            j["weight_map"] = {k: v.replace("diffusion_pytorch_model", "model", 1)
                               for k, v in j.get("weight_map", {}).items()}
        return j
    def read_index_or_names(idx):
        try:
            return read_index(idx)
        except urllib.error.HTTPError as e:
            if _network(e):
                raise NetworkError(f"{repo}: HTTP {e.code}") from None
            # gated: the index cannot be read, but the shard names are public
            import re
            stem = idx.replace(".index.json", "").rsplit(".", 1)
            pat = re.compile(rf"^{re.escape(stem[0])}-\d{{5}}-of-\d{{5}}\.{re.escape(stem[1])}$")
            return {"weight_map": {n: n for n in inner if pat.match(n)}}
    try:
        files = [real(f) for f in hub.choose_files(inner, read_index_or_names)]
    except hub.HubError:
        files = []
    files = [f for f in files if f.endswith((".safetensors", ".bin", ".gguf"))]
    if any(f not in sib for f in files):   # the index names shards the repository does not hold
        return "not readable (index names missing files)", [], {}, {}
    if not files and not prefix:
        files = _largest_unit(repo, d, names)
    if not files and not prefix and READERS_R1:
        files = other_unit(names, {n: int(sib[n].get("size") or (sib[n].get("lfs") or {}).get("size") or 0)
                                   for n in names})
    if files and not prefix and formats.READERS_V5:
        files = _most_tensors(repo, d, names, files)
    if not files:
        adapters = [n for n in sib if n.split("/")[-1].startswith("adapter_model")]
        top = {n.rsplit(".", 1)[-1] for n in sib}
        if adapters:
            fmt = "adapter"
        elif top & {"onnx"}:
            fmt = "not readable (onnx only)"
        elif top & {"h5", "msgpack"}:
            fmt = "not readable (tf/flax only)"
        elif top & {"pt", "pth", "ckpt", "pkl", "npz", "nemo"}:
            fmt = "not readable (other checkpoint format)"
        else:
            fmt = "not readable (no weight files)"
        return fmt, [], {}, {}
    sizes = {f: int(sib[f].get("size") or (sib[f].get("lfs") or {}).get("size") or 0) for f in files}
    sha = {f: (sib[f].get("lfs") or {}).get("sha256", "") for f in files}
    ext = files[0].rsplit(".", 1)[-1].lower()
    if ext in ("onnx", "npz", "nemo", "pt", "pth", "ckpt", "tflite", "litertlm", "pdiparams"):
        return ext, files, sizes, sha
    if files == ["model.bin"]:
        return "ct2-or-bin", files, sizes, sha
    return ("gguf" if files[0].endswith(".gguf") else "bin" if files[0].endswith(".bin") else "safetensors"),\
        files, sizes, sha


# amendment 2026-11-20 (R1): formats read since then. E2 itself ran without them (READERS_R1=0
# reproduces E2's file choice); the re-audit E2x and every later run use them.
READERS_R1 = os.environ.get("PEDIGREE_READERS_R1", "1") != "0"
_NOT_WEIGHTS = re.compile(r"optim|discriminator|(^|/)disc[_.-]|scheduler|training_state|rng_state", re.I)
_QUANT_ONNX = re.compile(r"quant|q4|q8|int8|uint8|int4|bnb4|fp16|f16", re.I)
_HALF_ONNX = re.compile(r"fp16|f16", re.I)


def other_unit(names: list[str], size: dict) -> list[str]:
    """The weight files of a repository holding none of the three original formats at its top
    level, by format in this order: GGUF (sharded or in subfolders), CTranslate2 model.bin,
    PyTorch .pt/.pth/.ckpt (optimizer and discriminator states left out), NeMo, ONNX (full
    precision first), NumPy .npz, Paddle .pdiparams, LiteRT-LM, TFLite (float models first)."""
    g = ggufremote.choose_variant(names)
    if g:
        return g
    if "model.bin" in names:
        return ["model.bin"]

    def largest(c):
        return [max(c, key=lambda n: size.get(n, 0))] if c else []
    for exts in ((".pt", ".pth", ".ckpt"), (".nemo",)):
        c = [n for n in names if n.lower().endswith(exts) and not _NOT_WEIGHTS.search(n)]
        if c:
            return largest(c)
    onnx = [n for n in names if n.lower().endswith(".onnx") and "with_past" not in n]
    if onnx:
        full = [n for n in onnx if not _QUANT_ONNX.search(n.rsplit("/", 1)[-1])]
        half = [n for n in onnx if _HALF_ONNX.search(n.rsplit("/", 1)[-1])]
        pick = largest(full) or largest(half) or largest(onnx)
        if formats.READERS_V5 and pick:
            pick = _encoder_decoder(pick[0], onnx, size) or pick
        return pick
    for ext in (".npz", ".pdiparams", ".litertlm"):
        c = [n for n in names if n.lower().endswith(ext)]
        if c:
            return largest(c)
    tfl = [n for n in names if n.lower().endswith(".tflite")]
    full = [n for n in tfl if not _QUANT_ONNX.search(n.rsplit("/", 1)[-1])]
    return largest(full) or largest(tfl)


def _encoder_decoder(chosen: str, onnx: list[str], size: dict) -> list[str]:
    """Amendment 2026-11-21, reader (iii): an encoder-decoder export (`encoder_model*.onnx` beside
    `decoder_model*.onnx`) is read as both files of one precision, the decoder without past keys and
    not merged where that file exists (a merged decoder keeps its graph in If branches)."""
    folder, base = (chosen.rsplit("/", 1) if "/" in chosen else ("", chosen))
    stem = re.sub(r"^(encoder|decoder)_model(_merged)?", "", base)          # e.g. "", "_fp16", "_quantized"
    pre = folder + "/" if folder else ""
    enc = pre + "encoder_model" + stem
    decs = [pre + "decoder_model" + stem, pre + "decoder_model_merged" + stem]
    if not base.startswith(("encoder_model", "decoder_model")) or enc not in onnx:
        return []
    dec = next((d for d in decs if d in onnx), None)
    return [enc, dec] if dec else []


def _largest_unit(repo: str, d: dict, names: list[str]) -> list[str]:
    """Weights kept only in subfolders (bundles of several networks): the largest weight
    unit anywhere in the repository, a folder's sharded set or a single file (the main
    network is the largest; amendment 2026-11-08)."""
    sib = {s["rfilename"]: s for s in d.get("siblings", [])}
    size = lambda n: int(sib[n].get("size") or (sib[n].get("lfs") or {}).get("size") or 0)  # noqa: E731
    units = []
    for n in names:
        if "/" not in n:
            continue
        if n.endswith(".index.json"):
            folder = n.rsplit("/", 1)[0] + "/"
            try:
                with hub._open(f"{hub.HOST}/{repo}/resolve/{d['sha']}/{urllib.parse.quote(n)}") as r:
                    shards = sorted({folder + v for v in json.loads(r.read().decode("utf-8"))
                                     .get("weight_map", {}).values()})
            except Exception as e:
                if _network(e):
                    raise NetworkError(str(e)) from None
                continue
            if shards and all(x in sib for x in shards):
                units.append(shards)
        elif n.endswith((".safetensors", ".gguf")) and "-of-0" not in n:
            units.append([n])
        elif n.endswith(".bin") and n.split("/")[-1].startswith(("pytorch_model", "model")) and "-of-0" not in n:
            units.append([n])
    if not units:
        return []
    return max(units, key=lambda u: (sum(size(x) for x in u), -len(u)))


# --- amendment 2026-11-21, reader (vii): the unit with the most tensors -------------------------------

_SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)


def _gguf_units(names: list[str]) -> list[list[str]]:
    """Every GGUF file set of a repository (a single file, or the shards of one split file), the
    multimodal projectors left out."""
    groups: dict[str, list[str]] = {}
    for n in names:
        if n.lower().endswith(".gguf") and "mmproj" not in n.lower():
            groups.setdefault(_SHARD.sub("", n), []).append(n)
    return [sorted(g) for g in groups.values()]


def _count(repo: str, commit: str, unit: list[str], size: dict) -> int:
    """How many tensors a weight unit holds, from its headers alone."""
    from .remote import read_header, resolve_url
    if unit[0].lower().endswith(".gguf"):
        n, split = ggufremote.tensor_count(resolve_url(repo, unit[0], commit), size[unit[0]])
        if split is not None:
            return split
        if len(unit) == 1:
            return n
        return n + sum(ggufremote.tensor_count(resolve_url(repo, f, commit), size[f])[0] for f in unit[1:])
    total = 0
    for f in unit:
        url = resolve_url(repo, f, commit)
        if f.endswith(".safetensors"):
            _, h = read_header(url)
            total += sum(1 for k in h if k != "__metadata__")
        elif f.endswith(".bin"):
            from . import torchzip
            _, h = torchzip.read_header(url, size[f])
            total += len(h)
    return total


def _most_tensors(repo: str, d: dict, names: list[str], files: list[str]) -> list[str]:
    """When a repository holds several weight units (a top-level safetensors or PyTorch set and
    GGUF sets, or GGUF sets of different files), the one whose headers list the most tensors is the
    model: a draft or multi-token-prediction head, a VAE, a classifier head, a control vector or a
    lookup table beside it holds fewer. Units with equally many tensors (one model's quantisation
    variants) are chosen among as before."""
    sib = {s["rfilename"]: s for s in d.get("siblings", [])}
    size = {n: int(sib[n].get("size") or (sib[n].get("lfs") or {}).get("size") or 0) for n in sib}
    plain = [f for f in files if f.endswith((".safetensors", ".bin"))]
    if not plain and not all(f.endswith(".gguf") for f in files):
        return files                                        # other formats: one unit already
    chosen = plain or ggufremote.choose_variant(files)
    units = ([plain] if plain else []) + _gguf_units(names)
    if len(units) < 2:
        return chosen
    counts = []
    for u in units:
        try:
            counts.append(_count(repo, d["sha"], u, size))
        except Exception as e:                              # noqa: BLE001
            if _network(e):
                raise NetworkError(f"{repo}: {type(e).__name__}: {e}") from None
            counts.append(-1)                               # an unreadable header: never preferred
    best = max(counts)
    tied = [u for u, c in zip(units, counts) if c == best]
    if any(sorted(u) == sorted(chosen) for u in tied):
        return chosen
    gg = [n for u in tied for n in u if n.endswith(".gguf")]
    return ggufremote.choose_variant(gg) if gg else tied[0]
