// Pedigree's browser checker: the same comparison as `pedigree verify`, in the page.
// Reads the Hub's metadata, each safetensors header (a few KB) and the first 8,192
// values of up to 24 matched tensors per model, with HTTP range requests. Nothing is
// uploaded anywhere; the only network requests go to huggingface.co.
// Port of pedigree/align.py and pedigree/verify.py (plain score); the thresholds are
// the published ones (pedigree/thresholds.json), injected at build time as THRESHOLDS.

const HOST = "https://huggingface.co";
const N_TENSORS = 24, SAMPLE = 8192, MIN_ELEMENTS = 4096, MIN_COMPARED = 8, MAX_DROP = 2;
const FLOATS = new Set(["F64", "F32", "F16", "BF16"]);
const ITEM = { F64: 8, F32: 4, F16: 2, BF16: 2 };

const RENAMES = [
  [/(^|\.)encoder\.layer\.(\d+)\./g, "$1layers.$2."],
  [/\.attention\.(?:attention|self)\.query\./g, ".attention.q_proj."],
  [/\.attention\.(?:attention|self)\.key\./g, ".attention.k_proj."],
  [/\.attention\.(?:attention|self)\.value\./g, ".attention.v_proj."],
  [/\.attention\.output\.dense\./g, ".attention.o_proj."],
  [/\.attention\.output\.LayerNorm\./g, ".attention_layernorm."],
  [/(layers\.\d+)\.intermediate\.dense\./g, "$1.mlp.fc1."],
  [/(layers\.\d+)\.output\.dense\./g, "$1.mlp.fc2."],
  [/(layers\.\d+)\.output\.LayerNorm\./g, "$1.output_layernorm."],
];
const LEGACY = { gamma: "weight", beta: "bias" };

export function canon(name, drop) {
  for (const [re, rep] of RENAMES) name = name.replace(re, rep);
  const parts = name.split(".").slice(drop);
  if (parts.length && LEGACY[parts[parts.length - 1]]) parts[parts.length - 1] = LEGACY[parts[parts.length - 1]];
  return parts.join(".");
}

// model / base: name -> shape (array). Same rule as align.py: try dropping 0..2 leading
// name parts on each side, keep the choice pairing the most same-shape tensors.
export function align(model, base) {
  let best = { pairs: {}, n: 0 };
  for (let dm = 0; dm <= MAX_DROP; dm++) for (let db = 0; db <= MAX_DROP; db++) {
    const bk = new Map(), mk = new Map();
    for (const b of Object.keys(base)) { const k = canon(b, db); bk.set(k, (bk.get(k) || []).concat([b])); }
    for (const m of Object.keys(model)) { const k = canon(m, dm); mk.set(k, (mk.get(k) || []).concat([m])); }
    const pairs = {};
    for (const [k, ms] of mk) {
      const bs = bk.get(k);
      if (k && ms.length === 1 && bs && bs.length === 1 && String(model[ms[0]]) === String(base[bs[0]])) pairs[ms[0]] = bs[0];
    }
    const n = Object.keys(pairs).length;
    if (n > best.n) best = { pairs, n };
  }
  return best.pairs;
}

export function spaced(items, k) {
  if (items.length <= k) return items.slice();
  const idx = new Set();
  for (let i = 0; i < k; i++) idx.add(Math.round(i * (items.length - 1) / (k - 1)));
  return [...idx].sort((a, b) => a - b).map(i => items[i]);
}

export function decode(dtype, buf) {
  const dv = new DataView(buf), n = buf.byteLength / ITEM[dtype], out = new Float64Array(n);
  const bf = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < n; i++) {
    if (dtype === "F32") out[i] = dv.getFloat32(i * 4, true);
    else if (dtype === "F64") out[i] = dv.getFloat64(i * 8, true);
    // bfloat16 is the high half of a float32 (big-endian view: high half first)
    else if (dtype === "BF16") { bf.setUint16(0, dv.getUint16(i * 2, true)); bf.setUint16(2, 0); out[i] = bf.getFloat32(0); }
    else out[i] = f16(dv.getUint16(i * 2, true));
  }
  return out;
}

function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export function cosine(a, b) {
  const n = Math.min(a.length, b.length);
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) continue;
    ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i];
  }
  return aa > 0 && bb > 0 ? ab / Math.sqrt(aa * bb) : NaN;
}

export function median(xs) {
  const s = xs.slice().sort((a, b) => a - b), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

let onWait = () => {};
export function setWaitLog(fn) { onWait = fn; }

async function range(url, start, end) {
  for (let attempt = 0; attempt < 5; attempt++) {
    const r = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
    if (r.status === 206 || (r.status === 200 && start === 0)) return (await r.arrayBuffer()).slice(0, end - start + 1);
    if (r.status === 429 || r.status >= 500) {
      const wait = 2000 * 2 ** attempt;
      onWait(`the Hub asked this browser to slow down; retrying in ${wait / 1000} s`);
      await new Promise(res => setTimeout(res, wait));
      continue;
    }
    throw new Error(r.status === 401 || r.status === 403 ? "gated or private model (read it with the CLI and a token)"
      : `HTTP ${r.status} for ${url}`);
  }
  throw new Error("the Hub is rate-limiting this browser; try again in a few minutes");
}

export async function openModel(repo, log = () => {}) {
  const r = await fetch(`${HOST}/api/models/${repo}/revision/main`);
  if (!r.ok) throw new Error(`${repo}: HTTP ${r.status}`);
  const info = await r.json();
  const names = info.siblings.map(s => s.rfilename);
  let files = [];
  if (names.includes("model.safetensors")) files = ["model.safetensors"];
  else if (names.includes("model.safetensors.index.json")) {
    const idx = await (await fetch(`${HOST}/${repo}/resolve/${info.sha}/model.safetensors.index.json`)).json();
    files = [...new Set(Object.values(idx.weight_map))].sort();
  } else files = names.filter(n => !n.includes("/") && n.endsWith(".safetensors") && !n.startsWith("adapter_model")).sort();
  if (!files.length) throw new Error(`${repo}: no safetensors weights (the CLI also reads PyTorch, GGUF and MLX files)`);
  const tensors = {};
  for (const f of files) {
    const url = `${HOST}/${repo}/resolve/${info.sha}/${f}`;
    const n = Number(new DataView(await range(url, 0, 7)).getBigUint64(0, true));
    if (n > 100 * 2 ** 20) throw new Error(`${f}: implausible header`);
    const header = JSON.parse(new TextDecoder().decode(await range(url, 8, 8 + n - 1)));
    for (const [name, s] of Object.entries(header)) if (name !== "__metadata__") tensors[name] = { url, start: 8 + n, ...s };
    log(`${repo}: read the header of ${f}`);
  }
  const base = (info.cardData || {}).base_model;
  return { repo, commit: info.sha, tensors, declared: base ? [].concat(base) : [] };
}

function floatShapes(m) {
  const out = {};
  for (const [n, s] of Object.entries(m.tensors)) if (FLOATS.has(s.dtype)) out[n] = s.shape;
  return out;
}

async function sample(m, name) {
  const s = m.tensors[name], it = ITEM[s.dtype];
  const len = Math.min(s.data_offsets[1] - s.data_offsets[0], SAMPLE * it);
  if (len <= 0) return new Float64Array(0);
  const a = s.start + s.data_offsets[0];
  return decode(s.dtype, await range(s.url, a, a + len - 1));
}

export async function compare(model, base, log = () => {}) {
  const ms = floatShapes(model), bs = floatShapes(base);
  const pairs = align(ms, bs);
  const big = Object.keys(pairs).sort().filter(n => ms[n].reduce((a, b) => a * b, 1) >= MIN_ELEMENTS);
  const names = spaced(big, N_TENSORS);
  const cos = [];
  for (const n of names) {
    const [a, b] = await Promise.all([sample(model, n), sample(base, pairs[n])]);
    const c = cosine(a, b);
    if (Number.isFinite(c)) cos.push(c);
    log(`compared ${cos.length} of ${names.length} tensors`);
  }
  const plain = cos.length ? median(cos) : null;
  const t = THRESHOLDS.plain;
  let verdict = "not comparable";
  if (plain !== null && plain > t) verdict = "related";
  else if (plain !== null && cos.length >= MIN_COMPARED) verdict = "unrelated";
  return { matched: Object.keys(pairs).length, compared: cos.length, plain, verdict,
           partial: verdict === "related" && cos.length < MIN_COMPARED };
}
