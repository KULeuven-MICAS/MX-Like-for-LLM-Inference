#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""MX and element-wise float/integer quantization utilities."""

import numpy as np

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


class E5M2:
    """FP8: 1S 5E 2M, bias=15, max=57344."""
    BIAS = 15
    MAX_EXP_BIASED = 30
    MAX_VAL = 57344.0

    def __init__(self, sign=0, exponent=0, mantissa=0):
        self.sign = int(sign); self.exponent = int(exponent); self.mantissa = int(mantissa)

    def pack(self):
        return np.uint8((self.sign << 7) | (self.exponent << 2) | self.mantissa)

    def unpack(self, value):
        self.sign = (value >> 7) & 0x1
        self.exponent = (value >> 2) & 0x1F
        self.mantissa = value & 0x3
        return self

    def to_float(self):
        if self.exponent == 0:
            val = (2.0 ** (1 - self.BIAS)) * (self.mantissa / 4.0)
        else:
            val = (2.0 ** (self.exponent - self.BIAS)) * (1.0 + self.mantissa / 4.0)
        return -val if self.sign else val

    def quantize(self, x):
        if x == 0.0:
            self.sign, self.exponent, self.mantissa = 0, 0, 0; return self
        sign = int(x < 0); abs_x = abs(x)
        if abs_x >= self.MAX_VAL:
            self.sign, self.exponent, self.mantissa = sign, self.MAX_EXP_BIASED, 3; return self
        exp = int(np.floor(np.log2(abs_x))); exp_enc = exp + self.BIAS
        if exp_enc <= 0:
            mant = int(np.round(abs_x / (2.0 ** (1 - self.BIAS)) * 4))
            if mant == (1 << 2):
                self.sign, self.exponent, self.mantissa = sign, 1, 0
            else:
                self.sign, self.exponent, self.mantissa = sign, 0, mant
        else:
            mant = int(np.round((abs_x / (2.0 ** exp) - 1.0) * 4))
            if mant >= 4: exp_enc += 1; mant = 0
            exp_enc = min(exp_enc, self.MAX_EXP_BIASED)
            self.sign, self.exponent, self.mantissa = sign, exp_enc, mant
        return self


class E4M3:
    """FP8: 1S 4E 3M, bias=7, max=448. exp=15,mant=7 is NaN."""
    BIAS = 7
    MAX_EXP_BIASED = 15
    MAX_VAL = 448.0

    def __init__(self, sign=0, exponent=0, mantissa=0):
        self.sign = int(sign); self.exponent = int(exponent); self.mantissa = int(mantissa)

    def pack(self):
        return np.uint8((self.sign << 7) | (self.exponent << 3) | self.mantissa)

    def unpack(self, value):
        self.sign = (value >> 7) & 0x1
        self.exponent = (value >> 3) & 0x0F
        self.mantissa = value & 0x7
        return self

    def to_float(self):
        if self.exponent == 0:
            val = (2.0 ** (1 - self.BIAS)) * (self.mantissa / 8.0)
        else:
            val = (2.0 ** (self.exponent - self.BIAS)) * (1.0 + self.mantissa / 8.0)
        return -val if self.sign else val

    def quantize(self, x):
        if x == 0.0:
            self.sign, self.exponent, self.mantissa = 0, 0, 0; return self
        sign = int(x < 0); abs_x = abs(x)
        if abs_x >= self.MAX_VAL:
            self.sign, self.exponent, self.mantissa = sign, 15, 6; return self
        exp = int(np.floor(np.log2(abs_x))); exp_enc = exp + self.BIAS
        if exp_enc <= 0:
            mant = int(np.round(abs_x / (2.0 ** (1 - self.BIAS)) * 8))
            if mant == (1 << 3):
                self.sign, self.exponent, self.mantissa = sign, 1, 0
            else:
                self.sign, self.exponent, self.mantissa = sign, 0, mant
        else:
            mant = int(np.round((abs_x / (2.0 ** exp) - 1.0) * 8))
            if mant >= 8: exp_enc += 1; mant = 0
            if exp_enc > 15 or (exp_enc == 15 and mant > 6): exp_enc, mant = 15, 6
            self.sign, self.exponent, self.mantissa = sign, exp_enc, mant
        return self


class E3M2:
    """FP6: 1S 3E 2M, bias=3, max=28. Packed in lower 6 bits."""
    BIAS = 3
    MAX_EXP_BIASED = 7
    MAX_VAL = 28.0

    def __init__(self, sign=0, exponent=0, mantissa=0):
        self.sign = int(sign); self.exponent = int(exponent); self.mantissa = int(mantissa)

    def pack(self):
        return np.uint8((self.sign << 5) | (self.exponent << 2) | self.mantissa)

    def unpack(self, value):
        self.sign = (value >> 5) & 0x1
        self.exponent = (value >> 2) & 0x7
        self.mantissa = value & 0x3
        return self

    def to_float(self):
        if self.exponent == 0:
            val = (2.0 ** (1 - self.BIAS)) * (self.mantissa / 4.0)
        else:
            val = (2.0 ** (self.exponent - self.BIAS)) * (1.0 + self.mantissa / 4.0)
        return -val if self.sign else val

    def quantize(self, x):
        if x == 0.0:
            self.sign, self.exponent, self.mantissa = 0, 0, 0; return self
        sign = int(x < 0); abs_x = abs(x)
        if abs_x >= self.MAX_VAL:
            self.sign, self.exponent, self.mantissa = sign, self.MAX_EXP_BIASED, 3; return self
        exp = int(np.floor(np.log2(abs_x))); exp_enc = exp + self.BIAS
        if exp_enc <= 0:
            mant = int(np.round(abs_x / (2.0 ** (1 - self.BIAS)) * 4))
            if mant == (1 << 2):
                self.sign, self.exponent, self.mantissa = sign, 1, 0
            else:
                self.sign, self.exponent, self.mantissa = sign, 0, mant
        else:
            mant = int(np.round((abs_x / (2.0 ** exp) - 1.0) * 4))
            if mant >= 4: exp_enc += 1; mant = 0
            exp_enc = min(exp_enc, self.MAX_EXP_BIASED)
            self.sign, self.exponent, self.mantissa = sign, exp_enc, mant
        return self


class E2M3:
    """FP6: 1S 2E 3M, bias=1, max=7.5. Packed in lower 6 bits."""
    BIAS = 1
    MAX_EXP_BIASED = 3
    MAX_VAL = 7.5

    def __init__(self, sign=0, exponent=0, mantissa=0):
        self.sign = int(sign); self.exponent = int(exponent); self.mantissa = int(mantissa)

    def pack(self):
        return np.uint8((self.sign << 5) | (self.exponent << 3) | self.mantissa)

    def unpack(self, value):
        self.sign = (value >> 5) & 0x1
        self.exponent = (value >> 3) & 0x3
        self.mantissa = value & 0x7
        return self

    def to_float(self):
        if self.exponent == 0:
            val = (2.0 ** (1 - self.BIAS)) * (self.mantissa / 8.0)
        else:
            val = (2.0 ** (self.exponent - self.BIAS)) * (1.0 + self.mantissa / 8.0)
        return -val if self.sign else val

    def quantize(self, x):
        if x == 0.0:
            self.sign, self.exponent, self.mantissa = 0, 0, 0; return self
        sign = int(x < 0); abs_x = abs(x)
        if abs_x >= self.MAX_VAL:
            self.sign, self.exponent, self.mantissa = sign, self.MAX_EXP_BIASED, 7; return self
        exp = int(np.floor(np.log2(abs_x))); exp_enc = exp + self.BIAS
        if exp_enc <= 0:
            mant = int(np.round(abs_x / (2.0 ** (1 - self.BIAS)) * 8))
            if mant == (1 << 3):
                self.sign, self.exponent, self.mantissa = sign, 1, 0
            else:
                self.sign, self.exponent, self.mantissa = sign, 0, mant
        else:
            mant = int(np.round((abs_x / (2.0 ** exp) - 1.0) * 8))
            if mant >= 8: exp_enc += 1; mant = 0
            exp_enc = min(exp_enc, self.MAX_EXP_BIASED)
            self.sign, self.exponent, self.mantissa = sign, exp_enc, mant
        return self


class E2M1:
    """FP4: 1S 2E 1M, bias=1, max=6. Packed in lower 4 bits."""
    BIAS = 1
    MAX_EXP_BIASED = 3
    MAX_VAL = 6.0

    def __init__(self, sign=0, exponent=0, mantissa=0):
        self.sign = int(sign); self.exponent = int(exponent); self.mantissa = int(mantissa)

    def pack(self):
        return np.uint8((self.sign << 3) | (self.exponent << 1) | self.mantissa)

    def unpack(self, value):
        self.sign = (value >> 3) & 0x1
        self.exponent = (value >> 1) & 0x3
        self.mantissa = value & 0x1
        return self

    def to_float(self):
        if self.exponent == 0:
            val = (2.0 ** (1 - self.BIAS)) * (self.mantissa / 2.0)
        else:
            val = (2.0 ** (self.exponent - self.BIAS)) * (1.0 + self.mantissa / 2.0)
        return -val if self.sign else val

    def quantize(self, x):
        if x == 0.0:
            self.sign, self.exponent, self.mantissa = 0, 0, 0; return self
        sign = int(x < 0); abs_x = abs(x)
        if abs_x >= self.MAX_VAL:
            self.sign, self.exponent, self.mantissa = sign, self.MAX_EXP_BIASED, 1; return self
        exp = int(np.floor(np.log2(abs_x))); exp_enc = exp + self.BIAS
        if exp_enc <= 0:
            subnormal_step = (2.0 ** (1 - self.BIAS)) / 2.0
            mant = int(np.round(abs_x / subnormal_step))
            if mant == 0:
                self.sign, self.exponent, self.mantissa = sign, 0, 0
            elif mant == 1:
                self.sign, self.exponent, self.mantissa = sign, 0, 1
            else:
                self.sign, self.exponent, self.mantissa = sign, 1, 0
        else:
            mant = int(np.round((abs_x / (2.0 ** exp) - 1.0) * 2))
            if mant >= 2: exp_enc += 1; mant = 0
            exp_enc = min(exp_enc, self.MAX_EXP_BIASED)
            self.sign, self.exponent, self.mantissa = sign, exp_enc, mant
        return self


class MXINT8:
    """MXINT8: 1S 7M two's complement, implicit scale 2^-6, range ≈ [-1.984, 1.984]."""

    def __init__(self, sign=0, exponent=0):
        self.sign = int(sign); self.exponent = int(exponent)

    def pack(self):
        if self.sign == 0:
            return np.uint8(self.exponent)
        if self.exponent == 0:
            return np.uint8(0)
        return np.uint8((~self.exponent + 1) & 0xFF)

    def unpack(self, value):
        value = int(value) & 0xFF
        self.sign = (value >> 7) & 0x1
        self.exponent = (256 - value) & 0x7F if self.sign else value & 0x7F
        return self

    def to_float(self):
        val = self.exponent / 64.0
        return -val if self.sign else val

    def quantize(self, x):
        if x == 0:
            self.sign, self.exponent = 0, 0; return self
        self.sign = int(x < 0); abs_x = abs(x)
        if abs_x >= 2.0:
            self.exponent = 127; return self
        self.exponent = int(np.round(abs_x * 64))
        return self


class ScaleFormat:
    """Unsigned 8-bit shared scale: ebits + mbits == 8. E8M0: raw=255 is NaN (OCP MX)."""

    def __init__(self, ebits: int, mbits: int):
        assert ebits + mbits == 8
        self.ebits = ebits; self.mbits = mbits
        if ebits == 0:
            self.bias = 0; self.max_val = 255.0 / 128.0
            self.saturate_raw = 255; self.min_raw = 1; self.min_val = 1.0 / 128.0
        elif mbits == 0:
            self.bias = 127; self.max_val = 2.0 ** 127
            self.saturate_raw = 254; self.min_raw = 0; self.min_val = 2.0 ** (-self.bias)
        else:
            self.bias     = (1 << (ebits - 1)) - 1
            max_exp_b     = (1 << ebits) - 1
            max_mant      = (1 << mbits) - 1
            self.max_val  = (2.0 ** (max_exp_b - self.bias)) * (1.0 + max_mant / (2 ** mbits))
            self.saturate_raw = (max_exp_b << mbits) | max_mant
            self.min_raw = 1; self.min_val = (2.0 ** (1 - self.bias)) / (2 ** mbits)

    def _raw_to_float(self, raw: int) -> float:
        if self.ebits == 0: return raw / 128.0
        if self.mbits == 0: return 2.0 ** (raw - self.bias)
        exp_biased = raw >> self.mbits
        mant_bits  = raw & ((1 << self.mbits) - 1)
        if exp_biased == 0:
            return (2.0 ** (1 - self.bias)) * (mant_bits / (2 ** self.mbits))
        return (2.0 ** (exp_biased - self.bias)) * (1.0 + mant_bits / (2 ** self.mbits))

    def quantize(self, ideal_scale: float) -> tuple:
        if ideal_scale <= self.min_val: return self.min_val, self.min_raw
        if ideal_scale >= self.max_val:
            return self._raw_to_float(self.saturate_raw), self.saturate_raw
        if self.ebits == 0:
            raw = int(np.clip(int(np.round(ideal_scale * 128.0)), 0, 255))
            return raw / 128.0, raw
        if self.mbits == 0:
            exp = int(np.floor(np.log2(ideal_scale + 1e-45)))
            raw = int(np.clip(exp + self.bias, 0, self.saturate_raw))
            return 2.0 ** (raw - self.bias), raw
        exp = int(np.floor(np.log2(ideal_scale + 1e-45))); exp_biased = exp + self.bias
        if exp_biased <= 0:
            subnorm_unit = 2.0 ** (1 - self.bias) / (2 ** self.mbits)
            mant = int(np.round(ideal_scale / subnorm_unit))
            if mant >= (1 << self.mbits):
                raw = 1 << self.mbits
            else:
                raw = int(np.clip(mant, 0, (1 << self.mbits) - 1))
            return self._raw_to_float(raw), raw
        frac = ideal_scale / (2.0 ** exp)
        mant = int(np.round((frac - 1.0) * (2 ** self.mbits)))
        if mant >= (1 << self.mbits): mant = 0; exp_biased += 1
        max_exp_b = (1 << self.ebits) - 1
        if exp_biased > max_exp_b:
            return self._raw_to_float(self.saturate_raw), self.saturate_raw
        raw = (exp_biased << self.mbits) | mant
        return self._raw_to_float(raw), raw


VALID_SCALE_FORMATS = {
    'UE8M0': ScaleFormat(8, 0),
    'UE7M1': ScaleFormat(7, 1),
    'UE6M2': ScaleFormat(6, 2),
    'UE5M3': ScaleFormat(5, 3),
    'UE4M4': ScaleFormat(4, 4),
    'UE3M5': ScaleFormat(3, 5),
    'UE2M6': ScaleFormat(2, 6),
    'UE1M7': ScaleFormat(1, 7),
    'UE0M8': ScaleFormat(0, 8),
}

_DATA_CLASSES = {
    'fp8_e4m3': E4M3, 'fp8_e5m2': E5M2,
    'fp6_e3m2': E3M2, 'fp6_e2m3': E2M3,
    'fp4_e2m1': E2M1, 'mxint8':   MXINT8,
}

# Max unbiased exponent per element format (shared_exp = floor(log2(amax)) - EMAX).
_EMAX = {
    'fp8_e5m2': 15, 'fp8_e4m3': 8,
    'fp6_e3m2': 4,  'fp6_e2m3': 2,
    'fp4_e2m1': 2,  'mxint8':   0,
}

# Logical bit width per element format.
_DTYPE_BITS = {
    'fp8_e4m3': 8, 'fp8_e5m2': 8,
    'fp6_e3m2': 6, 'fp6_e2m3': 6,
    'fp4_e2m1': 4, 'mxint8':   8,
}

VALID_DTYPES = list(_DATA_CLASSES.keys()) + ['int8']

# Max representable value per element format (used for round-up block scale).
_DTYPE_MAX_VAL = {
    'fp8_e5m2': E5M2.MAX_VAL,
    'fp8_e4m3': E4M3.MAX_VAL,
    'fp6_e3m2': E3M2.MAX_VAL,
    'fp6_e2m3': E2M3.MAX_VAL,
    'fp4_e2m1': E2M1.MAX_VAL,
    'mxint8':   127.0 / 64.0,
    'int8':     127.0,
}


def _pack_raw_1d(raw_bytes: np.ndarray, dtype: str) -> np.ndarray:
    """Tight bit-pack: FP4 → 2 elems/byte, FP6 → 4 elems/3 bytes, FP8/int8 → no-op."""
    raw = raw_bytes.astype(np.uint8)
    if dtype == 'fp4_e2m1':
        if len(raw) % 2:
            raw = np.concatenate([raw, np.zeros(1, dtype=np.uint8)])
        return ((raw[0::2] & 0xF) | ((raw[1::2] & 0xF) << 4)).astype(np.uint8)
    if dtype in ('fp6_e3m2', 'fp6_e2m3'):
        pad = (4 - len(raw) % 4) % 4
        if pad:
            raw = np.concatenate([raw, np.zeros(pad, dtype=np.uint8)])
        raw = raw & 0x3F
        e0, e1, e2, e3 = raw[0::4], raw[1::4], raw[2::4], raw[3::4]
        out = np.zeros(len(raw) // 4 * 3, dtype=np.uint8)
        out[0::3] = e0        | ((e1 & 0x03) << 6)
        out[1::3] = (e1 >> 2) | ((e2 & 0x0F) << 4)
        out[2::3] = (e2 >> 4) | (e3 << 2)
        return out
    return raw


def quantize(arr: np.ndarray, dtype: str, packed: bool = False):
    """Element-wise quantization. Returns (q_float32, q_raw) or (q_float32, scale, q_raw_int8) for int8."""
    arr = np.asarray(arr, dtype=np.float32)
    if dtype == 'int8':
        max_abs = float(np.max(np.abs(arr)))
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        q_int8 = np.round(arr / scale).clip(-128, 127).astype(np.int8)
        return q_int8.astype(np.float32) * scale, scale, q_int8
    if dtype not in _DATA_CLASSES:
        raise ValueError(f"Unknown dtype {dtype!r}. Choose from {VALID_DTYPES}")
    cls = _DATA_CLASSES[dtype]
    flat = arr.flatten()
    q_flat   = np.empty(flat.shape, dtype=np.float32)
    raw_flat = np.empty(flat.shape, dtype=np.uint8)
    for i, val in enumerate(flat):
        obj = cls().quantize(float(val))
        q_flat[i] = obj.to_float(); raw_flat[i] = obj.pack()
    q_float = q_flat.reshape(arr.shape)
    raw = _pack_raw_1d(raw_flat, dtype) if packed else raw_flat.reshape(arr.shape)
    return q_float, raw


def _quantize_block_by_scale(block: np.ndarray, dtype: str, scale: float) -> tuple:
    """Normalize block by scale, quantize element-wise, denormalize."""
    if scale == 0.0:
        zero_raw = np.zeros(block.shape, dtype=np.int8 if dtype == 'int8' else np.uint8)
        return np.zeros(block.shape, dtype=np.float32), zero_raw
    norm = block / scale
    if dtype == 'int8':
        q_int8 = np.round(norm).clip(-128, 127).astype(np.int8)
        return q_int8.astype(np.float32) * scale, q_int8
    cls = _DATA_CLASSES[dtype]
    q_float = np.empty(block.shape, dtype=np.float32)
    raw     = np.empty(block.shape, dtype=np.uint8)
    for i, val in enumerate(norm.flat):
        obj = cls().quantize(float(val))
        q_float.flat[i] = obj.to_float() * scale; raw.flat[i] = obj.pack()
    return q_float, raw


def quantize_round_up(ideal_scale: float, scale_fmt: 'ScaleFormat') -> tuple:
    """Ceil-round scale quantization: guarantees quantized_scale >= ideal_scale."""
    if ideal_scale <= scale_fmt.min_val:
        return scale_fmt._raw_to_float(scale_fmt.min_raw), scale_fmt.min_raw
    if ideal_scale >= scale_fmt.max_val:
        return scale_fmt._raw_to_float(scale_fmt.saturate_raw), scale_fmt.saturate_raw
    if scale_fmt.mbits == 0:
        mant_s, exp_raw = np.frexp(np.float32(ideal_scale))
        is_exact = (abs(float(mant_s) - 0.5) < 1e-7)
        raw = int(exp_raw - 1 + scale_fmt.bias) if is_exact else int(exp_raw + scale_fmt.bias)
        raw = int(np.clip(raw, 0, scale_fmt.saturate_raw))
        return scale_fmt._raw_to_float(raw), raw
    if scale_fmt.ebits == 0:
        raw = int(np.clip(int(np.ceil(ideal_scale * 128.0)), 1, 255))
        return raw / 128.0, raw
    max_exp_b = (1 << scale_fmt.ebits) - 1
    exp = int(np.floor(np.log2(ideal_scale + 1e-45))); exp_biased = exp + scale_fmt.bias
    if exp_biased <= 0:
        subnorm_unit = (2.0 ** (1 - scale_fmt.bias)) / (2 ** scale_fmt.mbits)
        mant = int(np.ceil(ideal_scale / subnorm_unit))
        if mant >= (1 << scale_fmt.mbits):
            raw = 1 << scale_fmt.mbits
        else:
            raw = int(np.clip(mant, 0, (1 << scale_fmt.mbits) - 1))
        return scale_fmt._raw_to_float(raw), raw
    frac = ideal_scale / (2.0 ** exp)
    mant = int(np.ceil((frac - 1.0) * (2 ** scale_fmt.mbits)))
    if mant >= (1 << scale_fmt.mbits): mant = 0; exp_biased += 1
    if exp_biased > max_exp_b:
        return scale_fmt._raw_to_float(scale_fmt.saturate_raw), scale_fmt.saturate_raw
    raw = (exp_biased << scale_fmt.mbits) | mant
    return scale_fmt._raw_to_float(raw), raw


def quantize_mx(
    arr: np.ndarray,
    dtype: str,
    block_size: int = 32,
    axis: int = 1,
    packed: bool = False,
    scale_format: str = 'UE8M0',
) -> tuple:
    """MX block quantization with NVFP4-style round-up block scale.
    Returns (q_arr, scale_shared, scale_raw, q_raw)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"quantize_mx expects a 2D array, got shape {arr.shape}")
    if dtype not in VALID_DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}. Choose from {VALID_DTYPES}")
    if scale_format not in VALID_SCALE_FORMATS:
        raise ValueError(f"Unknown scale_format {scale_format!r}")

    scale_fmt        = VALID_SCALE_FORMATS[scale_format]
    private_elem_max = _DTYPE_MAX_VAL[dtype]
    Row, Col  = arr.shape
    raw_dtype = np.int8 if dtype == 'int8' else np.uint8
    q_arr     = np.zeros_like(arr)
    raw_arr   = np.zeros((Row, Col), dtype=raw_dtype)

    if axis == 1:
        n_blocks     = int(np.ceil(Col / block_size))
        scale_shared = np.zeros((Row, n_blocks), dtype=np.float32)
        scale_raw    = np.zeros((Row, n_blocks), dtype=np.uint8)
        for i in range(Row):
            for b in range(n_blocks):
                j0 = b * block_size; j1 = min(j0 + block_size, Col)
                block   = arr[i, j0:j1]
                max_abs = float(np.max(np.abs(block)))
                sc_f, sc_r = (0.0, 0) if max_abs == 0.0 else quantize_round_up(max_abs / private_elem_max, scale_fmt)
                scale_shared[i, b] = sc_f; scale_raw[i, b] = sc_r
                q_arr[i, j0:j1], raw_arr[i, j0:j1] = _quantize_block_by_scale(block, dtype, sc_f)
    elif axis == 0:
        n_blocks     = int(np.ceil(Row / block_size))
        scale_shared = np.zeros((n_blocks, Col), dtype=np.float32)
        scale_raw    = np.zeros((n_blocks, Col), dtype=np.uint8)
        for b in range(n_blocks):
            i0 = b * block_size; i1 = min(i0 + block_size, Row)
            for j in range(Col):
                block   = arr[i0:i1, j]
                max_abs = float(np.max(np.abs(block)))
                sc_f, sc_r = (0.0, 0) if max_abs == 0.0 else quantize_round_up(max_abs / private_elem_max, scale_fmt)
                scale_shared[b, j] = sc_f; scale_raw[b, j] = sc_r
                q_arr[i0:i1, j], raw_arr[i0:i1, j] = _quantize_block_by_scale(block, dtype, sc_f)
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if packed and dtype not in ('fp8_e4m3', 'fp8_e5m2', 'int8'):
        q_raw = np.stack([_pack_raw_1d(raw_arr[i], dtype) for i in range(Row)], axis=0)
    else:
        q_raw = raw_arr

    return q_arr, scale_shared, scale_raw, q_raw


# Hardware simulation: SNAX BFP_PE dot-product / GEMM.
# Replicates: CustomOperator → ScaleAddition → ScaleToFP32 → FP32 tree-reduce → accumulator.
# Ref: hw/chisel_acc/src/main/scala/mx/mac/

_HW_ELEM = {
    'INT8': dict(ebits=0, mbits=7, bias=0,  impl_exp=-6),
    'E5M2': dict(ebits=5, mbits=2, bias=15, impl_exp=0),
    'E4M3': dict(ebits=4, mbits=3, bias=7,  impl_exp=0),
    'E3M2': dict(ebits=3, mbits=2, bias=3,  impl_exp=0),
    'E2M3': dict(ebits=2, mbits=3, bias=1,  impl_exp=0),
    'E2M1': dict(ebits=2, mbits=1, bias=1,  impl_exp=0),
}

_HW_SCALE = {
    'UE8M0': dict(ebits=8, mbits=0, bias=127),
    'UE7M1': dict(ebits=7, mbits=1, bias=63),
    'UE6M2': dict(ebits=6, mbits=2, bias=31),
    'UE5M3': dict(ebits=5, mbits=3, bias=15),
    'UE4M4': dict(ebits=4, mbits=4, bias=7),
    'UE3M5': dict(ebits=3, mbits=5, bias=3),
    'UE2M6': dict(ebits=2, mbits=6, bias=1),
}


def _hw_decode_elem(raw_vec, etype):
    """Decode raw uint8 element bits → (sign, adj_exp, full_mant). Matches CustomOperator.scala."""
    p = _HW_ELEM[etype]
    ebits, mbits, bias, impl_exp = p['ebits'], p['mbits'], p['bias'], p['impl_exp']
    raw  = raw_vec.astype(np.int64)
    sign = ((raw >> (ebits + mbits)) & 1).astype(bool)
    if etype == 'INT8':
        raw7      = raw & 0x7F
        full_mant = np.where(sign, (~raw7 + 1) & 0x7F, raw7)
        adj_exp   = np.full(len(raw_vec), impl_exp, dtype=np.int64)
    else:
        exp       = (raw >> mbits) & ((1 << ebits) - 1)
        mant      = raw & ((1 << mbits) - 1)
        implicit  = (exp > 0).astype(np.int64)
        full_mant = (implicit << mbits) | mant
        unbiased  = np.where(exp == 0, np.int64(1 - bias), exp - np.int64(bias))
        adj_exp   = (unbiased + impl_exp - mbits).astype(np.int64)
    return sign, adj_exp, full_mant.astype(np.int64)


def _hw_decode_scale(scale_raw, stype):
    """Decode uint8 scale → (adj_exp_s, full_mant_s). Matches ScaleAddition.scala."""
    p = _HW_SCALE[stype]
    mbits, bias = p['mbits'], p['bias']
    s = int(scale_raw); exp_s = s >> mbits
    if mbits == 0:
        return exp_s - bias, 1
    mant_s      = s & ((1 << mbits) - 1)
    impl        = 1 if exp_s > 0 else 0
    full_mant_s = (impl << mbits) | mant_s
    adj_exp_s   = ((1 - bias) if exp_s == 0 else (exp_s - bias)) - mbits
    return adj_exp_s, full_mant_s


def _fp32_tree_reduce(lane_fp32):
    """Balanced binary tree FP32 reduction, matching FusedDotProductUnit.scala."""
    arr = list(lane_fp32.astype(np.float32))
    while len(arr) > 1:
        arr = [np.float32(arr[i]) + np.float32(arr[i+1]) if i+1 < len(arr) else arr[i]
               for i in range(0, len(arr), 2)]
    return np.float32(arr[0])


def hw_snax_dot(a_raws, b_raws, scale_a_raw, scale_b_raw,
                etype_a='INT8', etype_b='E2M1', stype='UE8M0'):
    """Simulate one BFP_PE vector dot-product."""
    sign_a, exp_a, mant_a = _hw_decode_elem(a_raws, etype_a)
    sign_b, exp_b, mant_b = _hw_decode_elem(b_raws, etype_b)
    adj_exp_sA, fmant_sA  = _hw_decode_scale(scale_a_raw, stype)
    adj_exp_sB, fmant_sB  = _hw_decode_scale(scale_b_raw, stype)
    exp_total  = np.int64(adj_exp_sA + adj_exp_sB) + exp_a + exp_b
    mant_total = np.int64(fmant_sA) * np.int64(fmant_sB) * mant_a * mant_b
    sign_out   = sign_a ^ sign_b
    sign_f     = np.where(sign_out, np.float64(-1.0), np.float64(1.0))
    lane_fp32  = (sign_f * np.ldexp(mant_total.astype(np.float64),
                                    exp_total.astype(np.int32))).astype(np.float32)
    return _fp32_tree_reduce(lane_fp32)


def hw_snax_gemm(A_raw, B_raw, scale_A, scale_B,
                 etype_a='INT8', etype_b='E2M1', stype='UE8M0'):
    """Simulate SNAX BFP_PE GEMM: result[m, n] = Σ_k A[m,k] * B[n,k]."""
    M, K_blocks, _ = A_raw.shape
    N      = B_raw.shape[0]
    result = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            acc = np.float32(0.0)
            for kb in range(K_blocks):
                acc = np.float32(acc) + np.float32(hw_snax_dot(
                    A_raw[m, kb], B_raw[n, kb],
                    scale_A[m, kb], scale_B[n, kb],
                    etype_a, etype_b, stype,
                ))
            result[m, n] = acc
    return result


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    np.random.seed(42)
    arr2d = (np.random.randn(32, 32) * 3).astype(np.float32)
    arr1d = arr2d.flatten()

    print("=== element-wise quantization ===")
    for dt in ['fp8_e4m3', 'fp8_e5m2', 'fp6_e3m2', 'fp6_e2m3', 'fp4_e2m1', 'mxint8']:
        q, raw = quantize(arr1d, dt)
        print(f"  {dt:12s}  rmse={_rmse(arr1d, q):.6f}  raw:{raw.dtype}{list(raw.shape)}")

    print("=== MX quantization ===")
    for dt in ['fp8_e4m3', 'fp6_e2m3', 'fp4_e2m1', 'mxint8']:
        q, sc, sc_r, raw = quantize_mx(arr2d, dt, block_size=32, axis=1)
        print(f"  {dt:12s}  rmse={_rmse(arr2d, q):.6f}  scale_raw:{sc_r.shape}  raw:{raw.dtype}{list(raw.shape)}")

    print("All done.")


if __name__ == "__main__":
    main()
