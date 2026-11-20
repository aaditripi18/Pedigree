"""GGUF tensor types newer than the published `gguf` package, decoded from public sources only
(PLAN.md, amendment 2026-11-17 (4), H17: "a type is decoded only from a public source").

  41  Q1_0      llama.cpp master (ggml/src/ggml-common.h block_q1_0, ggml-quants.c: dequantize_row_q1_0);
                the gguf package lists it but cannot decode it
  42  Q2_0      llama.cpp master (ggml/src/ggml-common.h, ggml-quants.c: dequantize_row_q2_0)
  43  F8_E4M3   llama.cpp master (ggml-impl.h: ggml_f8_e4m3_to_fp32, OCP FP8)
  142 PQ2_0     PrismML-Eng/llama.cpp, branch prism (dequantize_row_pq2_0): Q2_0's codec, group 128
  143 PTQ1_0    PrismML-Eng/llama.cpp, branch prism (dequantize_row_ptq1_0): TQ1_0's trits, group 128
  139 IQ4_K     ikawrakow/ik_llama.cpp (ggml-common.h block_iq4_k, iqk_quantize.cpp dequantize_row_iq4_k)
  339 IQ4_K_R4  ikawrakow/ik_llama.cpp (ggml-common.h block_iq4_k_r4, iqk_quantize.cpp
                dequantize_row_iq4_k_r4): IQ4_K with four rows interleaved in each 576-byte block

tests/test_ggufextra.py compiles those reference functions (copied from the sources above, MIT
licensed) where a C compiler exists, and requires byte-for-byte equal output on random blocks.
Types with no public source (seen on the Hub: 45, 101, 103) are not decoded, and say so.

A row-interleaved type (ROW_GROUP) stores ROW_GROUP rows together, so the first values of a tensor
can only be decoded from whole groups of rows: dequantize() then needs the row length.
"""
from __future__ import annotations

import numpy as np

# type id -> (name, values per block, bytes per block)
TYPES = {41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18), 43: ("F8_E4M3", 1, 1), 142: ("PQ2_0", 128, 34), 143: ("PTQ1_0", 128, 28),
         139: ("IQ4_K", 256, 144), 339: ("IQ4_K_R4", 256, 144)}
SOURCES = {41: "llama.cpp master", 42: "llama.cpp master", 43: "llama.cpp master", 142: "PrismML-Eng/llama.cpp (prism)",
           143: "PrismML-Eng/llama.cpp (prism)", 139: "ikawrakow/ik_llama.cpp", 339: "ikawrakow/ik_llama.cpp"}
ROW_GROUP = {339: 4}
IQ4K_VALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
                        -123, -100, -79, -61, -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117], dtype=np.float32)
# seen on the Hub, no public definition found (searched llama.cpp, PrismML-Eng/llama.cpp, ik_llama.cpp)
UNKNOWN = {45, 101, 103}


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


def _iq4_k(raw: bytes) -> np.ndarray:
    """dequantize_row_iq4_k: 144-byte blocks of 256 values."""
    B = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 144)
    d = B[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    extra = B[:, 2:4].copy().view("<u2")[:, 0].astype(np.int64)
    sh_all, sl, qs = B[:, 4:8], B[:, 8:16], B[:, 16:144]
    y = np.empty((B.shape[0], 256), dtype=np.float32)
    for ib in range(8):
        sh = (sh_all[:, ib // 2] >> (4 * (ib % 2))).astype(np.int64)
        dl1 = d * ((((sl[:, ib] & 0xF).astype(np.int64)) | ((sh << 4) & 0x30)) - 32).astype(np.float32)
        dl2 = d * ((((sl[:, ib] >> 4).astype(np.int64)) | ((sh << 2) & 0x30)) - 32).astype(np.float32)
        o1 = np.where((extra >> (2 * ib)) & 1, 16, 0)
        o2 = np.where((extra >> (2 * ib)) & 2, 16, 0)
        q = qs[:, 16 * ib:16 * ib + 16]
        y[:, 32 * ib:32 * ib + 16] = dl1[:, None] * IQ4K_VALUES[o1[:, None] + (q & 0xF)]
        y[:, 32 * ib + 16:32 * ib + 32] = dl2[:, None] * IQ4K_VALUES[o2[:, None] + (q >> 4)]
    return y.reshape(-1)


def _iq4_k_r4(raw: bytes, cols: int) -> np.ndarray:
    """dequantize_row_iq4_k_r4, for whole groups of 4 rows of `cols` values: row-major output."""
    nb = cols // 256
    B = np.frombuffer(raw, dtype=np.uint8).reshape(-1, nb, 576)          # (groups, blocks per row, bytes)
    G = B.shape[0]
    d = B[..., 0:8].copy().view(np.float16).astype(np.float32)          # (G, nb, 4)
    extra, sh, sl, qs = B[..., 8:16], B[..., 16:32], B[..., 32:64], B[..., 64:576]
    y = np.empty((G, 4, nb, 256), dtype=np.float32)
    for k in range(4):
        for ib in range(8):
            dls = []
            for is_ in (8 * ib + k, 8 * ib + k + 4):
                sc = ((sl[..., is_ % 32] >> (4 * (is_ // 32))) & 0xF) | (((sh[..., is_ % 16] >> (2 * (is_ // 16))) & 3) << 4)
                dls.append(d[..., k] * (sc.astype(np.float32) - 32))
            v1 = np.where((extra[..., k] >> ib) & 1, 16, 0)
            v2 = np.where((extra[..., k + 4] >> ib) & 1, 16, 0)
            for i in range(4):
                for off, src, shift, dl, v in ((0, 0, 0, 0, v1), (8, 0, 4, 0, v1), (16, 16, 0, 1, v2), (24, 16, 4, 1, v2),
                                               (4, 32, 0, 0, v1), (12, 32, 4, 0, v1), (20, 48, 0, 1, v2),
                                               (28, 48, 4, 1, v2)):
                    q = (qs[..., 64 * ib + 4 * k + i + src] >> shift) & 0xF
                    y[:, k, :, 32 * ib + i + off] = dls[dl] * IQ4K_VALUES[v + q]
    return y.reshape(G * 4, cols).reshape(-1)


def _q1(raw: bytes) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)          # (nb, 1)
    bits = np.unpackbits(b[:, 2:], axis=1, bitorder="little")        # element j: byte j//8, bit j%8
    return np.where(bits == 1, d, -d).astype(np.float32).reshape(-1)


def dequantize(raw: bytes, qtype: int, cols: int | None = None) -> np.ndarray:
    if qtype == 41:
        return _q1(raw)
    if qtype == 42:
        return _q2(raw, 64, 18)
    if qtype == 142:
        return _q2(raw, 128, 34)
    if qtype == 43:
        return _f8_e4m3(raw)
    if qtype == 143:
        return _ptq1(raw)
    if qtype == 139:
        return _iq4_k(raw)
    if qtype == 339:
        if not cols or cols % 256:
            raise ValueError("IQ4_K_R4 needs the row length (a multiple of 256)")
        return _iq4_k_r4(raw, cols)
    raise ValueError(f"GGUF tensor type {qtype} has no public specification Pedigree can decode")
