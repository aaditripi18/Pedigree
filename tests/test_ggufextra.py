"""GGUF types newer than the gguf package (pedigree/ggufextra.py) decode exactly as their public
reference implementations do. The C below is copied from those sources (llama.cpp master and
PrismML-Eng/llama.cpp branch prism, both MIT): dequantize_row_q2_0, dequantize_row_pq2_0,
dequantize_row_ptq1_0 and ggml_f8_e4m3_to_fp32. Compiled where a C compiler exists (CI: gcc)."""
import shutil
import struct
import subprocess

import numpy as np
import pytest

from pedigree import ggufextra

CC = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")

REF = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef _Float16 ggml_half;
#define GGML_FP16_TO_FP32(x) ((float)(x))
#define QK2_0 64
typedef struct { ggml_half d; uint8_t qs[QK2_0 / 4]; } block_q2_0;
#define QK_PQ2_0 128
typedef struct { ggml_half d; uint8_t qs[QK_PQ2_0 / 4]; } block_pq2_0;
#define QK_PTQ1_0 128
typedef struct { uint8_t qs[(QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5]; uint8_t qh[QK_PTQ1_0/64]; ggml_half d; } block_ptq1_0;
static const size_t ptq1_0_stages[3] = {32, 16, 8};
static inline float fp32_from_bits(uint32_t w) { float f; memcpy(&f, &w, 4); return f; }

void dequantize_row_q2_0(const block_q2_0 * x, float * y, int64_t k) {
    static const int qk = QK2_0; const int nb = k / qk;
    for (int i = 0; i < nb; i++) { const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) { const int byte_index = j / 4; const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03; y[i*qk + j] = ((int)q - 1) * d; } }
}
void dequantize_row_pq2_0(const block_pq2_0 * x, float * y, int64_t k) {
    static const int qk = QK_PQ2_0; const int nb = k / qk;
    for (int i = 0; i < nb; i++) { const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) { const int byte_index = j / 4; const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03; y[i*qk + j] = ((int)q - 1) * d; } }
}
void dequantize_row_ptq1_0(const block_ptq1_0 * x, float * y, int64_t k) {
    const int64_t nb = k / QK_PTQ1_0; const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    for (int64_t i = 0; i < nb; ++i) { const float d = GGML_FP16_TO_FP32(x[i].d); size_t j = 0;
        for (size_t s = 0; s < 3; ++s) { const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(x->qs); j += c) { for (size_t n = 0; n < 5; ++n) { for (size_t m = 0; m < c; ++m) {
                uint8_t q = x[i].qs[j + m] * pow3[n]; int16_t xi = ((uint16_t) q * 3) >> 8; *y++ = (float) (xi - 1) * d; } } } }
        for (size_t n = 0; n < 4; ++n) { for (size_t h = 0; h < sizeof(x->qh); ++h) {
            uint8_t q = x[i].qh[h] * pow3[n]; int16_t xi = ((uint16_t) q * 3) >> 8; *y++ = (float) (xi - 1) * d; } } }
}
static inline float ggml_f8_e4m3_to_fp32(uint8_t x) {
    const uint8_t ax = x & 0x7F; if (ax == 0x7F) return NAN;
    const uint32_t exp = (ax >> 3) & 0xF; const uint32_t man = ax & 0x7;
    const float value = exp == 0 ? (float) man * (1.0f / 512.0f) : fp32_from_bits(((exp + 120) << 23) | (man << 20));
    return (x & 0x80) ? -value : value;
}
int main(int argc, char ** argv) {
    int type = atoi(argv[1]); long n = atol(argv[2]);         /* n values; raw bytes on stdin */
    static uint8_t buf[1 << 20]; size_t got = fread(buf, 1, sizeof buf, stdin);
    float * y = malloc(n * sizeof(float));
    if (type == 42) dequantize_row_q2_0((block_q2_0 *) buf, y, n);
    else if (type == 142) dequantize_row_pq2_0((block_pq2_0 *) buf, y, n);
    else if (type == 143) dequantize_row_ptq1_0((block_ptq1_0 *) buf, y, n);
    else if (type == 43) for (long i = 0; i < n; i++) y[i] = ggml_f8_e4m3_to_fp32(buf[i]);
    fwrite(y, sizeof(float), n, stdout); (void) got; return 0;
}
"""


def _random_blocks(qtype: int, nblocks: int, rng) -> bytes:
    _, qk, size = ggufextra.TYPES[qtype]
    raw = bytearray(rng.integers(0, 256, nblocks * size, dtype=np.uint8).tobytes())
    if qtype in (42, 142, 143):     # put a finite fp16 scale where the format keeps it
        off = 26 if qtype == 143 else 0
        for b in range(nblocks):
            raw[b * size + off: b * size + off + 2] = struct.pack("<e", float(rng.uniform(-0.1, 0.1)))
    return bytes(raw)


def test_block_sizes_match_the_layouts():
    assert ggufextra.TYPES[42][2] == 2 + 64 // 4 and ggufextra.TYPES[142][2] == 2 + 128 // 4
    assert ggufextra.TYPES[143][2] == (128 - 4 * 128 // 64) // 5 + 128 // 64 + 2


def test_q2_codec_by_hand():
    # one Q2_0 block, scale 0.5, codes 0,1,2,3 repeated: values -0.5, 0, 0.5, 1.0
    raw = struct.pack("<e", 0.5) + bytes([0b11100100] * 16)
    assert np.allclose(ggufextra.dequantize(raw, 42)[:4], [-0.5, 0.0, 0.5, 1.0])


@pytest.mark.skipif(CC is None, reason="no C compiler")
@pytest.mark.parametrize("qtype", [42, 43, 142, 143])
def test_equals_the_reference_implementation(tmp_path, qtype):
    src = tmp_path / "ref.c"
    src.write_text(REF)
    exe = tmp_path / "ref"
    subprocess.run([CC, "-O1", "-o", str(exe), str(src), "-lm"], check=True)
    rng = np.random.default_rng(qtype)
    _, qk, size = ggufextra.TYPES[qtype]
    nb = 64 if qk > 1 else 4096
    raw = _random_blocks(qtype, nb, rng)
    n = nb * qk
    ref = np.frombuffer(subprocess.run([str(exe), str(qtype), str(n)], input=raw, capture_output=True,
                                       check=True).stdout, dtype=np.float32)
    ours = ggufextra.dequantize(raw, qtype).astype(np.float32)
    assert ref.shape == ours.shape
    assert np.array_equal(np.isnan(ref), np.isnan(ours))
    assert np.array_equal(ref[~np.isnan(ref)], ours[~np.isnan(ours)])
