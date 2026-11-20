"""GGUF types newer than the gguf package (pedigree/ggufextra.py) decode exactly as their public
reference implementations do. The C below is copied from those sources (llama.cpp master and
PrismML-Eng/llama.cpp branch prism, ikawrakow/ik_llama.cpp, all MIT): dequantize_row_q1_0, dequantize_row_q2_0,
dequantize_row_pq2_0, dequantize_row_ptq1_0, ggml_f8_e4m3_to_fp32, dequantize_row_iq4_k and
dequantize_row_iq4_k_r4 (C++
`auto` written out as its C type, nothing else changed). Compiled where a C compiler exists; CI
sets PEDIGREE_REQUIRE_CC=1, so there a missing compiler fails the test instead of skipping it."""
import os
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
#define QK1_0 128
typedef struct { ggml_half d; uint8_t qs[QK1_0 / 8]; } block_q1_0;
#define QK2_0 64
typedef struct { ggml_half d; uint8_t qs[QK2_0 / 4]; } block_q2_0;
#define QK_PQ2_0 128
typedef struct { ggml_half d; uint8_t qs[QK_PQ2_0 / 4]; } block_pq2_0;
#define QK_PTQ1_0 128
typedef struct { uint8_t qs[(QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5]; uint8_t qh[QK_PTQ1_0/64]; ggml_half d; } block_ptq1_0;
static const size_t ptq1_0_stages[3] = {32, 16, 8};
static inline float fp32_from_bits(uint32_t w) { float f; memcpy(&f, &w, 4); return f; }
#define QK_K 256
typedef struct { ggml_half d[4]; uint8_t extra[8]; uint8_t scales_h[QK_K/16]; uint8_t scales_l[QK_K/8]; uint8_t qs[QK_K*2]; } block_iq4_k_r4;
static const int8_t iq4k_values[32] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117 };
typedef struct { ggml_half d; uint16_t extra; uint8_t scales_h[QK_K/64]; uint8_t scales_l[QK_K/32]; uint8_t qs[QK_K/2]; } block_iq4_k;
void dequantize_row_iq4_k(const block_iq4_k * x, float * y, int64_t k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t * qs = x[i].qs;
        const float d = GGML_FP16_TO_FP32(x[i].d);
        uint16_t extra = x[i].extra;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint8_t sh = x[i].scales_h[ib/2] >> 4*(ib%2);
            const float dl1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const float dl2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
            const int8_t * values1 = extra & 1 ? iq4k_values + 16 : iq4k_values;
            const int8_t * values2 = extra & 2 ? iq4k_values + 16 : iq4k_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[qs[j] & 0xf];
                y[j+16] = dl2 * values2[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}
void dequantize_row_iq4_k_r4(const block_iq4_k_r4 * x, float * y, int64_t k) {
    int64_t n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                is += 4;
                float dl2 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                const int8_t * values1 = iq4k_values + (x[ibl].extra[k+0] & (1 << ib) ? 16 : 0);
                const int8_t * values2 = iq4k_values + (x[ibl].extra[k+4] & (1 << ib) ? 16 : 0);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+ 0] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+ 0] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+16] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+16] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+32] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+32] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+48] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+48] >>  4];
                }
            }
        }
    }
}

void dequantize_row_q1_0(const block_q1_0 * x, float * y, int64_t k) {
    static const int qk = QK1_0; const int nb = k / qk;
    for (int i = 0; i < nb; i++) { const float d = GGML_FP16_TO_FP32(x[i].d); const float neg_d = -d;
        for (int j = 0; j < qk; ++j) { const int byte_index = j / 8; const int bit_offset = j % 8;
            const uint8_t bit = (x[i].qs[byte_index] >> bit_offset) & 1; y[i*qk + j] = bit ? d : neg_d; } }
}
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
    if (type == 41) dequantize_row_q1_0((block_q1_0 *) buf, y, n);
    else if (type == 42) dequantize_row_q2_0((block_q2_0 *) buf, y, n);
    else if (type == 142) dequantize_row_pq2_0((block_pq2_0 *) buf, y, n);
    else if (type == 143) dequantize_row_ptq1_0((block_ptq1_0 *) buf, y, n);
    else if (type == 43) for (long i = 0; i < n; i++) y[i] = ggml_f8_e4m3_to_fp32(buf[i]);
    else if (type == 139) dequantize_row_iq4_k((block_iq4_k *) buf, y, n);
    else if (type == 339) { long cols = atol(argv[3]); for (long g = 0; g < n / (4 * cols); g++)
        dequantize_row_iq4_k_r4((block_iq4_k_r4 *) (buf + g * (cols / 256) * 576), y + g * 4 * cols, 4 * cols); }
    fwrite(y, sizeof(float), n, stdout); (void) got; return 0;
}
"""


def _random_blocks(qtype: int, nblocks: int, rng) -> bytes:
    _, qk, size = ggufextra.TYPES[qtype]
    raw = bytearray(rng.integers(0, 256, nblocks * size, dtype=np.uint8).tobytes())
    if qtype in (41, 42, 142, 143):     # put a finite fp16 scale where the format keeps it
        off = 26 if qtype == 143 else 0
        for b in range(nblocks):
            raw[b * size + off: b * size + off + 2] = struct.pack("<e", float(rng.uniform(-0.1, 0.1)))
    if qtype == 139:                # 144-byte blocks: one fp16 scale first
        for b in range(nblocks):
            raw[b * size: b * size + 2] = struct.pack("<e", float(rng.uniform(-0.01, 0.01)))
    if qtype == 339:                # 576-byte blocks of 4 rows: four fp16 scales first
        for b in range(len(raw) // 576):
            raw[b * 576: b * 576 + 8] = struct.pack("<4e", *(float(x) for x in rng.uniform(-0.01, 0.01, 4)))
    return bytes(raw)


def test_block_sizes_match_the_layouts():
    assert ggufextra.TYPES[41][2] == 2 + 128 // 8
    assert ggufextra.TYPES[42][2] == 2 + 64 // 4 and ggufextra.TYPES[142][2] == 2 + 128 // 4
    assert ggufextra.TYPES[143][2] == (128 - 4 * 128 // 64) // 5 + 128 // 64 + 2


def test_q2_codec_by_hand():
    # one Q2_0 block, scale 0.5, codes 0,1,2,3 repeated: values -0.5, 0, 0.5, 1.0
    raw = struct.pack("<e", 0.5) + bytes([0b11100100] * 16)
    assert np.allclose(ggufextra.dequantize(raw, 42)[:4], [-0.5, 0.0, 0.5, 1.0])


def test_q1_codec_by_hand():
    # one Q1_0 block, scale 0.25, byte 0 = 0b00000101: weights 0 and 2 are +d, 1 and 3..7 are -d
    raw = struct.pack("<e", 0.25) + bytes([0b00000101] + [0] * 15)
    assert np.allclose(ggufextra.dequantize(raw, 41)[:4], [0.25, -0.25, 0.25, -0.25])


def test_ci_has_a_compiler():
    if os.environ.get("PEDIGREE_REQUIRE_CC") and CC is None:
        pytest.fail("PEDIGREE_REQUIRE_CC is set but no C compiler was found: the parity tests would not run")


@pytest.mark.skipif(CC is None and not os.environ.get("PEDIGREE_REQUIRE_CC"), reason="no C compiler")
@pytest.mark.parametrize("qtype", [41, 42, 43, 139, 142, 143, 339])
def test_equals_the_reference_implementation(tmp_path, qtype):
    assert CC is not None, "no C compiler"
    src = tmp_path / "ref.c"
    src.write_text(REF)
    exe = tmp_path / "ref"
    subprocess.run([CC, "-O1", "-o", str(exe), str(src), "-lm"], check=True)
    rng = np.random.default_rng(qtype)
    _, qk, size = ggufextra.TYPES[qtype]
    cols = 512                                   # IQ4_K_R4: rows of 512 values, two groups of 4 rows
    nb = (2 * 4 * cols // qk) if qtype == 339 else (64 if qk > 1 else 4096)
    raw = _random_blocks(qtype, nb, rng)
    n = nb * qk
    ref = np.frombuffer(subprocess.run([str(exe), str(qtype), str(n), str(cols)], input=raw, capture_output=True,
                                       check=True).stdout, dtype=np.float32)
    ours = ggufextra.dequantize(raw, qtype, cols if qtype == 339 else None).astype(np.float32)
    assert ref.shape == ours.shape
    assert np.array_equal(np.isnan(ref), np.isnan(ours))
    assert np.array_equal(ref[~np.isnan(ref)], ours[~np.isnan(ours)])
