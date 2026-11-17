"""GGUF tensor types newer than the published `gguf` package, decoded from public sources only
(PLAN.md, amendment 2026-11-17 (4), H17: "a type is decoded only from a public source").

  42  Q2_0      llama.cpp master (ggml/src/ggml-common.h, ggml-quants.c: dequantize_row_q2_0)
  43  F8_E4M3   llama.cpp master (ggml-impl.h: ggml_f8_e4m3_to_fp32, OCP FP8)
  142 PQ2_0     PrismML-Eng/llama.cpp, branch prism (dequantize_row_pq2_0): Q2_0's codec, group 128
  143 PTQ1_0    PrismML-Eng/llama.cpp, branch prism (dequantize_row_ptq1_0): TQ1_0's trits, group 128

tests/test_ggufextra.py compiles those reference functions (copied from the sources above, MIT
licensed) where a C compiler exists, and requires byte-for-byte equal output on random blocks.
Types with no public source (seen on the Hub: 45, 101, 103) are not decoded, and say so.
"""
from __future__ import annotations

import numpy as np

# type id -> (name, values per block, bytes per block)
TYPES = {42: ("Q2_0", 64, 18), 43: ("F8_E4M3", 1, 1), 142: ("PQ2_0", 128, 34), 143: ("PTQ1_0", 128, 28)}
SOURCES = {42: "llama.cpp master", 43: "llama.cpp master", 142: "PrismML-Eng/llama.cpp (prism)",
           143: "PrismML-Eng/llama.cpp (prism)"}
# seen on the Hub, no public definition found (searched llama.cpp, PrismML-Eng/llama.cpp, ik_llama.cpp)
UNKNOWN = {45, 101, 103}
# defined publicly but not decoded here (an interleaved row-packed IQ4_K of ik_llama.cpp)
NOT_YET = {339: "IQ4_K_R4 (ikawrakow/ik_llama.cpp)"}


def _q2(raw: bytes, qk: int, size: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)          # (nb, 1)
    qs = b[:, 2:]
    q = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(-1, qk)   # element j: byte j//4, shift 2*(j%4)
    return ((q.astype(np.int32) - 1).astype(np.float32) * d).reshape(-1)


def _f8_e4m3(raw: bytes) -> np.ndarray:
    x = np.frombuffer(raw, dtype=np.uint8)
    ax = x & 0x7F
    e = (ax >> 3) & 0xF
    m = (ax & 0x7).astype(np.float32)
    val = np.where(e == 0, m / 512.0, (1 + m / 8) * np.exp2(e.astype(np.float32) - 7)).astype(np.float32)
    val = np.where(x & 0x80, -val, val)
    return np.where(ax == 0x7F, np.float32(np.nan), val).astype(np.float32)


def _trits(qbytes: np.ndarray, n: int) -> np.ndarray:
    """The n-th trit of each byte, as ggml's TQ1_0 decoder takes it: ((uint8)(q * 3^n) * 3) >> 8."""
    q = (qbytes.astype(np.uint16) * (3 ** n)) & 0xFF
    return ((q * 3) >> 8).astype(np.int16)


def _ptq1(raw: bytes) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 28)
    qs, qh = b[:, :24], b[:, 24:26]
    d = b[:, 26:28].copy().view(np.float16).astype(np.float32)
    out = []
    j = 0
    for c in (32, 16, 8):                       # ptq1_0_stages
        while j + c <= 24:
            for n in range(5):
                out.append(_trits(qs[:, j:j + c], n))
            j += c
    for n in range(4):
        out.append(_trits(qh, n))
    t = np.concatenate(out, axis=1)             # (nb, 128) in the reference's output order
    return ((t - 1).astype(np.float32) * d).reshape(-1)


def dequantize(raw: bytes, qtype: int) -> np.ndarray:
    if qtype == 42:
        return _q2(raw, 64, 18)
    if qtype == 142:
        return _q2(raw, 128, 34)
    if qtype == 43:
        return _f8_e4m3(raw)
    if qtype == 143:
        return _ptq1(raw)
    raise ValueError(f"GGUF tensor type {qtype} has no public specification Pedigree can decode")
