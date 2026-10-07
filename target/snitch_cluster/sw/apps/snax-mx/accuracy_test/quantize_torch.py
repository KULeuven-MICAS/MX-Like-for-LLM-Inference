# Copyright 2026 KU Leuven
# Licensed under the Apache License, Version 2.0
# Zijun Zhao <zijun.zhao@student.kuleuven.be>
# quantize_torch.py — PyTorch vectorized MX block quantization.

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union

_EMAX = {
    'fp8_e5m2': 15,
    'fp8_e4m3':  8,
    'fp6_e3m2':  4,
    'fp6_e2m3':  2,
    'fp4_e2m1':  2,
    'mxint8':    0,
}

_FP_PARAMS = {
    'fp8_e5m2': (5, 2, 15, 30,  3, 57344.0, False),
    'fp8_e4m3': (4, 3,  7, 15,  6,   448.0, True),
    'fp6_e3m2': (3, 2,  3,  7,  3,    28.0, False),
    'fp6_e2m3': (2, 3,  1,  3,  7,     7.5, False),
    'fp4_e2m1': (2, 1,  1,  3,  1,     6.0, False),
}

# Scale formats: (ebits, mbits)
VALID_SCALE_FORMATS = {
    'UE8M0': (8, 0),
    'UE7M1': (7, 1),
    'UE6M2': (6, 2),
    'UE5M3': (5, 3),
    'UE4M4': (4, 4),
    'UE3M5': (3, 5),
    'UE2M6': (2, 6),
    'UE1M7': (1, 7),
    'UE0M8': (0, 8),
    'UE4M3': (4, 3),
}

VALID_DTYPES = list(_EMAX.keys()) + ['int8']

# max representable value per element format
_DTYPE_MAX_VAL = {
    'fp8_e5m2': 57344.0,
    'fp8_e4m3':   448.0,
    'fp6_e3m2':    28.0,
    'fp6_e2m3':     7.5,
    'fp4_e2m1':     6.0,
    'mxint8':   127.0 / 64.0,
    'int8':     127.0,
}

# max representable value per scale format
_SCALE_FORMAT_MAX_VAL = {
    'UE8M0': 2.0 ** 127,
    'UE7M1': (2.0 ** 63)  * (1.0 + 1.0/2),
    'UE6M2': (2.0 ** 31)  * (1.0 + 3.0/4),
    'UE5M3': (2.0 ** 15)  * (1.0 + 7.0/8),
    'UE4M4': (2.0 ** 7)   * (1.0 + 15.0/16),
    'UE3M5': (2.0 ** 3)   * (1.0 + 31.0/32),
    'UE2M6': (2.0 ** 1)   * (1.0 + 63.0/64),
    'UE1M7': (2.0 ** 0)   * (1.0 + 127.0/128),
    'UE0M8': 255.0 / 128.0,
    'UE4M3': 448
}

FP32_MAX  = torch.finfo(torch.float32).max
FP32_TINY = torch.finfo(torch.float32).tiny


def _raw_to_float_tensor(raw_int, ebits, mbits, bias):
    """Vectorized decode: 8-bit unsigned raw → float. raw_int dtype: int32."""
    if ebits == 0:
        return raw_int.float() / 128.0
    if mbits == 0:
        return torch.exp2(raw_int.float() - bias)
    exp_b  = (raw_int >> mbits)
    mant_b = raw_int & ((1 << mbits) - 1)
    val_sub  = (2.0 ** (1 - bias)) * mant_b.float() / (2 ** mbits)
    val_norm = torch.exp2((exp_b - bias).float()) * (1.0 + mant_b.float() / (2 ** mbits))
    return torch.where(exp_b == 0, val_sub, val_norm)


def _quantize_scale_tensor(
    ideal_scale: torch.Tensor, ebits: int, mbits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Round-nearest scale quantization → (scale_float [R,NB], scale_raw [R,NB] uint8)."""
    if mbits == 0:
        bias    = 127
        min_val = 2.0 ** (-bias)
        underflow_mask = (ideal_scale <= min_val)
        _, exp_raw = torch.frexp(ideal_scale.clamp(min=min_val))
        raw = (exp_raw - 1 + bias).clamp(0, 254).to(torch.int32)
        raw = torch.where(underflow_mask, torch.zeros_like(raw), raw)
        scale_float = torch.exp2(raw.float() - bias)
        return scale_float, raw.to(torch.uint8)

    if ebits == 0:
        min_val = 1.0 / 128.0
        underflow_mask = (ideal_scale <= min_val)
        raw = ideal_scale.mul(128.0).round().clamp(0, 255).to(torch.int32)
        raw = torch.where(underflow_mask, torch.ones_like(raw), raw)
        scale_float = raw.float() / 128.0
        return scale_float, raw.to(torch.uint8)

    bias       = (1 << (ebits - 1)) - 1
    max_exp_b  = (1 << ebits) - 1
    max_mant_s = (1 << mbits) - 1
    max_val_s  = (2.0 ** (max_exp_b - bias)) * (1.0 + max_mant_s / (2 ** mbits))
    min_val    = (2.0 ** (1 - bias)) / (2 ** mbits)
    underflow_mask = (ideal_scale <= min_val)
    sat_mask   = (~underflow_mask) & (ideal_scale >= max_val_s)

    mant_s_frac, exp_raw = torch.frexp(ideal_scale.clamp(min=min_val))
    exp        = exp_raw - 1
    exp_biased = exp + bias

    subnorm_unit = (2.0 ** (1 - bias)) / (2 ** mbits)
    mant_sub  = ideal_scale.div(subnorm_unit).round().to(torch.int32)
    carry_sub = (mant_sub >= (1 << mbits))
    raw_sub = torch.where(
        carry_sub,
        torch.full_like(mant_sub, 1 << mbits),
        mant_sub.clamp(0, max_mant_s),
    )

    mant_norm    = ((2.0 * mant_s_frac - 1.0) * (2 ** mbits)).round().to(torch.int32)
    carry_norm   = (mant_norm >= (1 << mbits))
    exp_biased_n = torch.where(carry_norm, exp_biased + 1, exp_biased).clamp(max=max_exp_b)
    mant_norm_e  = torch.where(carry_norm, torch.zeros_like(mant_norm), mant_norm)
    raw_norm     = (exp_biased_n << mbits) | mant_norm_e

    is_sub  = (~underflow_mask) & (~sat_mask) & (exp_biased <= 0)
    raw_int = torch.where(
        underflow_mask, torch.ones_like(raw_norm),
        torch.where(
            sat_mask,  torch.full_like(raw_norm, 255),
            torch.where(is_sub, raw_sub, raw_norm),
        )
    )

    scale_float = _raw_to_float_tensor(raw_int, ebits, mbits, bias)
    return scale_float, raw_int.to(torch.uint8)


def _quantize_scale_tensor_round_up(
    ideal_scale: torch.Tensor, ebits: int, mbits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ceil-round scale quantization; guarantees scale >= ideal_scale (NVFP4-style)."""
    if mbits == 0:
        bias           = 127
        min_val        = 2.0 ** (-bias)
        underflow_mask = (ideal_scale <= min_val)
        mant_s, exp_raw = torch.frexp(ideal_scale.clamp(min=min_val))
        is_exact = (mant_s == 0.5)
        raw = torch.where(is_exact,
                          exp_raw - 1 + bias,
                          exp_raw     + bias).clamp(0, 254).to(torch.int32)
        raw = torch.where(underflow_mask, torch.zeros_like(raw), raw)
        scale_float = torch.exp2(raw.float() - bias)
        return scale_float, raw.to(torch.uint8)

    if ebits == 0:
        min_val        = 1.0 / 128.0
        underflow_mask = (ideal_scale <= min_val)
        raw = torch.ceil(ideal_scale.mul(128.0)).clamp(0, 255).to(torch.int32)
        raw = torch.where(underflow_mask, torch.ones_like(raw), raw)
        scale_float = raw.float() / 128.0
        return scale_float, raw.to(torch.uint8)

    bias       = (1 << (ebits - 1)) - 1
    max_exp_b  = (1 << ebits) - 1
    max_mant_s = (1 << mbits) - 1
    max_val_s  = (2.0 ** (max_exp_b - bias)) * (1.0 + max_mant_s / (2 ** mbits))
    min_val    = (2.0 ** (1 - bias)) / (2 ** mbits)
    underflow_mask = (ideal_scale <= min_val)
    sat_mask   = (~underflow_mask) & (ideal_scale >= max_val_s)

    mant_s_frac, exp_raw = torch.frexp(ideal_scale.clamp(min=min_val))
    exp        = exp_raw - 1
    exp_biased = exp + bias

    subnorm_unit = (2.0 ** (1 - bias)) / (2 ** mbits)
    mant_sub  = torch.ceil(ideal_scale.div(subnorm_unit)).to(torch.int32)
    carry_sub = (mant_sub >= (1 << mbits))
    raw_sub   = torch.where(
        carry_sub,
        torch.full_like(mant_sub, 1 << mbits),
        mant_sub.clamp(0, max_mant_s),
    )

    mant_norm    = torch.ceil((2.0 * mant_s_frac - 1.0) * (2 ** mbits)).to(torch.int32)
    carry_norm   = (mant_norm >= (1 << mbits))
    exp_biased_n = torch.where(carry_norm, exp_biased + 1, exp_biased).clamp(max=max_exp_b)
    mant_norm_e  = torch.where(carry_norm, torch.zeros_like(mant_norm), mant_norm)
    raw_norm     = (exp_biased_n << mbits) | mant_norm_e
    max_raw_s = (max_exp_b << mbits) | max_mant_s

    is_sub  = (~underflow_mask) & (~sat_mask) & (exp_biased <= 0)
    raw_int = torch.where(
        underflow_mask, torch.ones_like(raw_norm),
        torch.where(
            sat_mask,  torch.full_like(raw_norm, max_raw_s),
            torch.where(is_sub, raw_sub, raw_norm),
        )
    )

    scale_float = _raw_to_float_tensor(raw_int, ebits, mbits, bias)
    return scale_float, raw_int.to(torch.uint8)


def _quantize_fp_elements(
    norm, ebits, mbits, bias, max_exp_biased, max_mant, max_val, is_e4m3,
    return_raw=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized FP element quant-dequant on [R,NB,BS]; returns (q_dequant, q_raw_int32)."""
    sign  = (norm < 0)
    abs_x = norm.abs().clamp(max=max_val)

    mant_frac, exp_raw = torch.frexp(abs_x)
    exp        = exp_raw - 1
    exp_biased = exp + bias

    subnorm_step = (2.0 ** (1 - bias)) / (2 ** mbits)
    mant_sub_raw = abs_x.div(subnorm_step).round().to(torch.int32)
    carry_sub    = (mant_sub_raw >= (1 << mbits))
    sub_exp_enc  = torch.where(carry_sub, torch.ones_like(exp),         torch.zeros_like(exp))
    sub_mant_enc = torch.where(carry_sub, torch.zeros_like(mant_sub_raw), mant_sub_raw)

    mant_norm_raw = ((2.0 * mant_frac - 1.0) * (2 ** mbits)).round().to(torch.int32)
    carry_norm    = (mant_norm_raw >= (1 << mbits))
    exp_n  = torch.where(carry_norm, exp_biased + 1, exp_biased).clamp(max=max_exp_biased)
    mant_n = torch.where(carry_norm, torch.zeros_like(mant_norm_raw), mant_norm_raw)

    # FP8-E4M3: (exp=15, mant=7) is NaN — clamp mant to ≤ 6
    if is_e4m3:
        e4m3_nan = (exp_n == 15) & (mant_n > 6)
        mant_n = torch.where(e4m3_nan, torch.full_like(mant_n, 6), mant_n)

    is_zero = (abs_x == 0)
    is_sub  = (~is_zero) & (exp_biased <= 0)

    final_exp  = torch.where(is_zero, torch.zeros_like(exp_n),
                 torch.where(is_sub,  sub_exp_enc, exp_n))
    final_mant = torch.where(is_zero, torch.zeros_like(mant_n),
                 torch.where(is_sub,  sub_mant_enc, mant_n))

    subnorm_val = (2.0 ** (1 - bias)) * final_mant.float() / (2 ** mbits)
    normal_val  = torch.exp2((final_exp - bias).float()) * (1.0 + final_mant.float() / (2 ** mbits))
    q_abs = torch.where(final_exp == 0, subnorm_val, normal_val)
    q_abs = torch.where(is_zero, torch.zeros_like(q_abs), q_abs)
    q_dequant = torch.where(sign, -q_abs, q_abs)

    if return_raw:
        sign_int = sign.to(torch.int32)
        q_raw    = (sign_int << (ebits + mbits)) | (final_exp << mbits) | final_mant
    else:
        q_raw = None

    return q_dequant, q_raw


def quantize_mx_v1(
    arr: Union[torch.Tensor, np.ndarray],
    dtype: str,
    block_size: int = 32,
    axis: int = 1,
    packed: bool = False,
    scale_format: str = 'UE8M0',
    compute_raw: bool = False,
) -> Tuple:
    """MX block quantization with round-nearest scale (v1); returns (q_arr, scale_shared, scale_raw, q_raw)."""
    is_numpy = isinstance(arr, np.ndarray)
    if is_numpy:
        arr_t = torch.from_numpy(np.asarray(arr, dtype=np.float32).copy())
        compute_raw = True
    else:
        arr_t = arr.float()

    if arr_t.ndim != 2:
        raise ValueError(f"quantize_mx_v1 expects a 2D array, got shape {arr_t.shape}")
    if dtype not in VALID_DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}. Choose from {VALID_DTYPES}")
    if scale_format not in VALID_SCALE_FORMATS:
        raise ValueError(f"Unknown scale_format {scale_format!r}. Choose from {list(VALID_SCALE_FORMATS)}")

    ebits_sc, mbits_sc = VALID_SCALE_FORMATS[scale_format]

    transposed = False
    if axis == 0:
        arr_t = arr_t.T.contiguous()
        transposed = True
    elif axis != 1:
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    Row, Col = arr_t.shape
    n_blocks   = math.ceil(Col / block_size)
    padded_col = n_blocks * block_size
    if padded_col > Col:
        arr_pad = F.pad(arr_t, (0, padded_col - Col))
    else:
        arr_pad = arr_t
    arr_blocked = arr_pad.view(Row, n_blocks, block_size)

    max_abs = arr_blocked.abs().amax(dim=-1)

    emax = _EMAX.get(dtype, None)
    if dtype == 'int8':
        ideal_scale = max_abs / 127.0
    else:
        ideal_scale = torch.exp2(torch.log2(max_abs.clamp(min=FP32_TINY)) - emax)
        ideal_scale = torch.where(max_abs == 0, torch.zeros_like(ideal_scale), ideal_scale)

    scale_shared, scale_raw = _quantize_scale_tensor(ideal_scale, ebits_sc, mbits_sc)
    if not compute_raw:
        scale_raw = None

    scale_exp = scale_shared.unsqueeze(-1)
    norm = torch.where(
        scale_exp == 0,
        torch.zeros_like(arr_blocked),
        arr_blocked / scale_exp.clamp(min=1e-45),
    )

    if dtype in _FP_PARAMS:
        ebits, mbits, bias, max_exp_biased, max_mant, max_val, is_e4m3 = _FP_PARAMS[dtype]
        q_norm, q_raw_blocked = _quantize_fp_elements(
            norm, ebits, mbits, bias, max_exp_biased, max_mant, max_val, is_e4m3,
            return_raw=compute_raw,
        )
    elif dtype == 'mxint8':
        mag    = norm.abs().mul(64.0).round().clamp(0, 127).to(torch.int32)
        q_norm = torch.where(norm < 0, -mag.float() / 64.0, mag.float() / 64.0)
        if compute_raw:
            neg_raw       = (~mag + 1) & 0xFF
            q_raw_blocked = torch.where(norm < 0, neg_raw, mag)
        else:
            q_raw_blocked = None
    else:
        q_norm_i      = norm.round().clamp(-128, 127).to(torch.int32)
        q_norm        = q_norm_i.float()
        q_raw_blocked = q_norm_i if compute_raw else None

    q_blocked = q_norm * scale_shared.unsqueeze(-1)
    q_arr = q_blocked.reshape(Row, padded_col)[:, :Col].contiguous()
    if compute_raw:
        raw_dtype = torch.int8 if dtype == 'int8' else torch.uint8
        q_raw_out = q_raw_blocked.reshape(Row, padded_col)[:, :Col].contiguous().to(raw_dtype)
    else:
        q_raw_out = None

    if transposed:
        q_arr        = q_arr.T.contiguous()
        scale_shared = scale_shared.T.contiguous()
        if compute_raw:
            q_raw_out = q_raw_out.T.contiguous()
            scale_raw = scale_raw.T.contiguous()

    if is_numpy:
        return (q_arr.numpy(), scale_shared.numpy(), scale_raw.numpy(), q_raw_out.numpy())

    return q_arr.bfloat16(), scale_shared, scale_raw, q_raw_out

def quantize_mx_v2(
    arr: Union[torch.Tensor, np.ndarray],
    dtype: str = 'fp4_e2m1',
    block_size: int = 16,
    axis: int = 1,
    packed: bool = False,
    scale_format: str = 'UE8M0',
    compute_raw: bool = False,
) -> Tuple:
    """Single-level MX quant with NVFP4 block-scale formula (amax/elem_max, round-up)."""
    is_numpy = isinstance(arr, np.ndarray)
    if is_numpy:
        arr_t = torch.from_numpy(arr.astype(np.float32))
        compute_raw = True
    else:
        arr_t = arr.float()

    if arr_t.ndim != 2:
        raise ValueError(f"quantize_mx_v2 expects 2D input, got shape {arr_t.shape}")
    if dtype not in VALID_DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}")
    if scale_format not in VALID_SCALE_FORMATS:
        raise ValueError(f"Unknown scale_format {scale_format!r}")

    ebits, mbits = VALID_SCALE_FORMATS[scale_format]
    private_elem_max = _DTYPE_MAX_VAL[dtype]

    transposed = (axis == 0)
    if transposed:
        arr_t = arr_t.T.contiguous()

    Row, Col = arr_t.shape
    NB = math.ceil(Col / block_size)
    padded_col = NB * block_size

    if padded_col > Col:
        arr_t = F.pad(arr_t, (0, padded_col - Col))

    arr_blocked = arr_t.reshape(Row, NB, block_size)

    max_abs     = arr_blocked.abs().amax(dim=-1)
    ideal_scale = max_abs / private_elem_max

    scale_shared, scale_raw = _quantize_scale_tensor_round_up(ideal_scale, ebits, mbits)

    scale_nz = scale_shared.masked_fill(scale_shared == 0, 1.0)
    norm = arr_blocked / scale_nz.unsqueeze(-1)

    if dtype in _FP_PARAMS:
        fp_ebits, fp_mbits, fp_bias, fp_max_exp_b, fp_max_mant, fp_max_val, fp_is_e4m3 = _FP_PARAMS[dtype]
        q_norm, q_raw_blocked = _quantize_fp_elements(
            norm, fp_ebits, fp_mbits, fp_bias, fp_max_exp_b,
            fp_max_mant, fp_max_val, fp_is_e4m3, return_raw=compute_raw,
        )
    elif dtype == 'mxint8':
        q_norm_i      = norm.clamp(-127/64, 127/64).mul(64).round().to(torch.int32)
        q_norm        = q_norm_i.float() / 64.0
        q_raw_blocked = q_norm_i.to(torch.uint8) if compute_raw else None
    else:
        q_norm_i      = norm.round().clamp(-128, 127).to(torch.int32)
        q_norm        = q_norm_i.float()
        q_raw_blocked = q_norm_i if compute_raw else None

    q_blocked = q_norm * scale_shared.unsqueeze(-1)

    q_arr = q_blocked.reshape(Row, padded_col)[:, :Col].contiguous()
    if compute_raw:
        raw_dtype = torch.int8 if dtype == 'int8' else torch.uint8
        q_raw_out = q_raw_blocked.reshape(Row, padded_col)[:, :Col].contiguous().to(raw_dtype)
    else:
        q_raw_out = None

    if transposed:
        q_arr        = q_arr.T.contiguous()
        scale_shared = scale_shared.T.contiguous()
        if compute_raw:
            q_raw_out  = q_raw_out.T.contiguous()
            scale_raw  = scale_raw.T.contiguous()

    if is_numpy:
        return (q_arr.float().numpy(), scale_shared.numpy(), scale_raw.numpy(), q_raw_out.numpy())

    return q_arr.bfloat16(), scale_shared, scale_raw, q_raw_out

def quantize_mx_v3(
    arr: Union[torch.Tensor, np.ndarray],
    dtype: str = 'fp4_e2m1',
    block_size: int = 16,
    axis: int = 1,
    packed: bool = False,
    scale_format: str = 'UE8M0',
    compute_raw: bool = False,
    per_tensor: bool = True,
    per_tensor_scale=None,
) -> Tuple:
    """NVFP4-style two-level MX quant: per-tensor scale + per-block scale (round-up)."""
    is_numpy = isinstance(arr, np.ndarray)
    if is_numpy:
        arr_t = torch.from_numpy(arr.astype(np.float32))
        compute_raw = True
    else:
        arr_t = arr.float()

    if arr_t.ndim != 2:
        raise ValueError(f"quantize_mx_v3 expects 2D input, got shape {arr_t.shape}")
    if dtype not in VALID_DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}")
    if scale_format not in VALID_SCALE_FORMATS:
        raise ValueError(f"Unknown scale_format {scale_format!r}")

    ebits, mbits = VALID_SCALE_FORMATS[scale_format]
    private_elem_max = _DTYPE_MAX_VAL[dtype]
    scale_fmt_max    = _SCALE_FORMAT_MAX_VAL[scale_format]

    if per_tensor_scale is not None:
        if isinstance(per_tensor_scale, torch.Tensor):
            per_tensor_scale = per_tensor_scale.float().to(arr_t.device).clamp(min=FP32_TINY)
        else:
            per_tensor_scale = torch.tensor(float(per_tensor_scale), dtype=torch.float32,
                                            device=arr_t.device).clamp(min=FP32_TINY)
    elif per_tensor:
        tensor_abs_max = arr_t.abs().amax()
        per_tensor_scale = (tensor_abs_max / (scale_fmt_max * private_elem_max)).clamp(FP32_TINY, FP32_MAX).to(torch.float32)
    else:
        per_tensor_scale = torch.tensor(1.0, dtype=torch.float32, device=arr_t.device)

    arr_scaled = arr_t / per_tensor_scale

    transposed = (axis == 0)
    if transposed:
        arr_scaled = arr_scaled.T.contiguous()

    Row, Col = arr_scaled.shape
    NB = math.ceil(Col / block_size)
    padded_col = NB * block_size

    if padded_col > Col:
        arr_scaled = F.pad(arr_scaled, (0, padded_col - Col))

    arr_blocked = arr_scaled.reshape(Row, NB, block_size)

    max_abs = arr_blocked.abs().amax(dim=-1)
    ideal_scale = max_abs / private_elem_max

    scale_shared, scale_raw = _quantize_scale_tensor_round_up(ideal_scale, ebits, mbits)

    scale_nz = scale_shared.masked_fill(scale_shared == 0, 1.0)
    norm = arr_blocked / scale_nz.unsqueeze(-1)

    if dtype in _FP_PARAMS:
        fp_ebits, fp_mbits, fp_bias, fp_max_exp_b, fp_max_mant, fp_max_val, fp_is_e4m3 = _FP_PARAMS[dtype]
        q_norm, q_raw_blocked = _quantize_fp_elements(
            norm, fp_ebits, fp_mbits, fp_bias, fp_max_exp_b,
            fp_max_mant, fp_max_val, fp_is_e4m3, return_raw=compute_raw,
        )
    elif dtype == 'mxint8':
        q_norm_i     = norm.clamp(-127/64, 127/64).mul(64).round().to(torch.int32)
        q_norm       = q_norm_i.float() / 64.0
        q_raw_blocked = q_norm_i.to(torch.uint8) if compute_raw else None
    else:
        q_norm_i     = norm.round().clamp(-128, 127).to(torch.int32)
        q_norm       = q_norm_i.float()
        q_raw_blocked = q_norm_i if compute_raw else None

    q_blocked = q_norm * scale_shared.unsqueeze(-1)

    q_arr_scaled = q_blocked.reshape(Row, padded_col)[:, :Col].contiguous()
    if compute_raw:
        raw_dtype = torch.int8 if dtype == 'int8' else torch.uint8
        q_raw_out = q_raw_blocked.reshape(Row, padded_col)[:, :Col].contiguous().to(raw_dtype)
    else:
        q_raw_out = None

    q_arr = q_arr_scaled * per_tensor_scale

    if transposed:
        q_arr        = q_arr.T.contiguous()
        scale_shared = scale_shared.T.contiguous()
        if compute_raw:
            q_raw_out = q_raw_out.T.contiguous()
            scale_raw = scale_raw.T.contiguous()

    if is_numpy:
        return (
            q_arr.float().numpy(), scale_shared.numpy(),
            scale_raw.numpy(), q_raw_out.numpy(),
            np.float32(per_tensor_scale.item()),
        )

    return q_arr.bfloat16(), scale_shared, scale_raw, q_raw_out, per_tensor_scale

