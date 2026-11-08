"""The browser checker (site/checker.js) aligns and samples exactly like the Python verifier."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pedigree.align import align
from pedigree.verify import _spaced, cosine, decode

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

MODEL = {f"bert.encoder.layer.{i}.attention.self.query.weight": [768, 768] for i in range(12)}
MODEL.update({f"bert.encoder.layer.{i}.output.LayerNorm.gamma": [768] for i in range(12)})
MODEL.update({f"bert.encoder.layer.{i}.intermediate.dense.weight": [3072, 768] for i in range(12)})
BASE = {f"layers.{i}.attention.q_proj.weight": [768, 768] for i in range(12)}
BASE.update({f"layers.{i}.output_layernorm.weight": [768] for i in range(12)})
BASE.update({f"layers.{i}.mlp.fc1.weight": [3072, 768] for i in range(12)})


def _node(code: str):
    js = ROOT / "site" / "checker.js"
    script = f"globalThis.THRESHOLDS={{plain:0.03}};const m=await import({json.dumps(js.as_uri())});{code}"
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_alignment_and_spacing_match():
    py = {m: b for m, b in align({k: ("float", tuple(v)) for k, v in MODEL.items()},
                                 {k: ("float", tuple(v)) for k, v in BASE.items()})[0].items()}
    js = _node(f"console.log(JSON.stringify(m.align({json.dumps(MODEL)}, {json.dumps(BASE)})))")
    assert js == py and len(py) == 36
    items = sorted(py)
    assert _node(f"console.log(JSON.stringify(m.spaced({json.dumps(items)}, 24)))") == _spaced(items, 24)


def test_decoding_and_cosine_match():
    rng = np.random.default_rng(0)
    x = rng.normal(size=64).astype(np.float32)
    y = x + rng.normal(scale=0.1, size=64).astype(np.float32)
    for dtype, arr in (("F32", x), ("F16", x.astype(np.float16)), ("BF16", None)):
        raw = arr.tobytes() if arr is not None else (x.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        js = _node(f"const b=new Uint8Array({json.dumps(list(raw))}).buffer;"
                   f"console.log(JSON.stringify(Array.from(m.decode('{dtype}', b))))")
        np.testing.assert_allclose(js, decode(dtype, raw), rtol=0, atol=0)
    c = _node(f"console.log(JSON.stringify(m.cosine({json.dumps(x.tolist())}, {json.dumps(y.tolist())})))")
    assert c == pytest.approx(cosine(x.astype(np.float64), y.astype(np.float64)), rel=1e-12)
