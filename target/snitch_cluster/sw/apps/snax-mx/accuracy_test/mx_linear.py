# mx_linear.py — MX Linear modules, calibration helpers, and activation scale tracker.

import os
import math
import csv
import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from quantize_torch import (
    quantize_mx_v3,
    _DTYPE_MAX_VAL, _SCALE_FORMAT_MAX_VAL, FP32_TINY,
)
from eval_utils import _load_testencs


def get_format_max(element_dtype: str, scale_format: str = 'UE8M0'):
    """Return per-tensor scale denominator (dtype_max × scale_fmt_max); None for UE8M0."""
    if scale_format == 'UE8M0':
        return None
    d_max = _DTYPE_MAX_VAL.get(element_dtype, 1.0)
    s_max = _SCALE_FORMAT_MAX_VAL.get(scale_format, 1.0)
    return d_max * s_max


class ActivationScaleTracker(torch.nn.Module):
    """Per-layer activation per-tensor scale manager for two-phase (calib + inference) quant."""

    def __init__(self, fmt: str, scale_format: str = 'UE8M0', momentum: float = 0.99,
                 device=None):
        super().__init__()
        self.fmt      = fmt
        self.momentum = momentum
        fmt_max = get_format_max(fmt, scale_format)
        self._enabled    = (fmt_max is not None)
        self._format_max = fmt_max if self._enabled else 1.0
        self._calibrating = False
        self._frozen = False

        self.register_buffer('amax_global', torch.tensor(0.0, device=device))
        self.register_buffer('amax_ema',    torch.tensor(0.0, device=device))

    def start_calibration(self):
        self._calibrating = True
        self.amax_global.zero_()

    def end_calibration(self):
        self._calibrating = False
        self.amax_ema.copy_(self.amax_global)

    def freeze(self):
        """Lock scale permanently; observe() becomes a no-op (for offline sweep)."""
        self._frozen = True

    def observe(self, x: torch.Tensor):
        if not self._enabled or self._frozen:
            return
        cur = x.abs().max()
        if self._calibrating:
            self.amax_global.clamp_(min=cur)
        else:
            self.amax_ema.mul_(self.momentum).add_(cur * (1.0 - self.momentum))

    @property
    def scale(self) -> torch.Tensor:
        """pts = amax / format_max; uses amax_global during calibration, amax_ema during inference."""
        if not self._enabled:
            return 1.0
        amax = self.amax_global if self._calibrating else self.amax_ema
        return torch.where(amax > 0,
                           amax / self._format_max,
                           torch.ones_like(amax))


class CustomMXLinear(torch.nn.Module):
    """Linear layer with offline weight quantization and online activation quantization."""

    def __init__(self, in_features, out_features, weight_data, bias_data,
                 w_dtype, a_dtype, block_size=32, scale_format='UE8M0'):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.w_dtype      = w_dtype
        self.a_dtype      = a_dtype
        self.block_size   = block_size
        self.scale_format = scale_format

        from run_mx_quantized import _mx_quantize
        q_w, _, _, _ = _mx_quantize(weight_data.detach().float(), self.w_dtype,
                                    block_size=block_size, axis=1, scale_format=scale_format)
        self.register_buffer('q_weight', q_w.to(dtype=torch.bfloat16))

        if bias_data is not None:
            self.register_buffer('bias', bias_data.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        from run_mx_quantized import _mx_quantize
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).float()
        q_x, _, _, _ = _mx_quantize(x_2d, self.a_dtype, block_size=self.block_size,
                                    axis=1, scale_format=self.scale_format)
        q_x_t = q_x.to(dtype=x.dtype).reshape(orig_shape)
        return F.linear(q_x_t, self.q_weight, self.bias)


class CalibratedMXLinear(torch.nn.Module):
    """MX Linear with offline calibration + EMA activation per-tensor scale."""

    def __init__(self, in_features, out_features, weight_data, bias_data,
                 w_dtype, a_dtype, block_size=32, scale_format='UE8M0', momentum=0.99):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.w_dtype      = w_dtype
        self.a_dtype      = a_dtype
        self.block_size   = block_size
        self.scale_format = scale_format

        from run_mx_quantized import _mx_quantize
        q_w, _, _, _ = _mx_quantize(weight_data.detach().float(), w_dtype,
                                     block_size=block_size, axis=1, scale_format=scale_format)
        self.register_buffer('q_weight', q_w.to(dtype=torch.bfloat16))

        if bias_data is not None:
            self.register_buffer('bias', bias_data.clone())
        else:
            self.register_parameter('bias', None)

        self.act_tracker = ActivationScaleTracker(a_dtype, scale_format=scale_format,
                                                   momentum=momentum, device=q_w.device)

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        if self.act_tracker._calibrating:
            self.act_tracker.observe(x_2d)
            return F.linear(x, self.q_weight.to(dtype=x.dtype), self.bias)

        scale = self.act_tracker.scale
        q_x, _, _, _, _ = quantize_mx_v3(
            x_2d.float(), self.a_dtype,
            block_size=self.block_size, axis=1, scale_format=self.scale_format,
            per_tensor_scale=scale,
        )
        self.act_tracker.observe(x_2d)
        return F.linear(q_x.to(dtype=x.dtype).reshape(orig_shape),
                        self.q_weight.to(dtype=x.dtype), self.bias)


def replace_linear_with_custom_mx(model, w_dtype, a_dtype, block_size=32, scale_format='UE8M0',
                                   skip_names=('lm_head', 'embed_tokens')):
    """Recursively replace nn.Linear with CustomMXLinear."""
    for name, child in list(model.named_children()):
        if name in skip_names:
            continue
        if isinstance(child, torch.nn.Linear):
            setattr(model, name, CustomMXLinear(
                child.in_features, child.out_features,
                child.weight.data,
                child.bias.data if child.bias is not None else None,
                w_dtype=w_dtype, a_dtype=a_dtype,
                block_size=block_size, scale_format=scale_format,
            ))
        else:
            replace_linear_with_custom_mx(child, w_dtype, a_dtype, block_size, scale_format,
                                          skip_names=skip_names)
    return model


def replace_linear_with_calibrated_mx(model, w_dtype, a_dtype, block_size=32,
                                       scale_format='UE8M0', momentum=0.99,
                                       skip_names=('lm_head', 'embed_tokens')):
    """Recursively replace nn.Linear with CalibratedMXLinear."""
    for name, child in list(model.named_children()):
        if name in skip_names:
            continue
        if isinstance(child, torch.nn.Linear):
            setattr(model, name, CalibratedMXLinear(
                child.in_features, child.out_features,
                child.weight.data,
                child.bias.data if child.bias is not None else None,
                w_dtype=w_dtype, a_dtype=a_dtype,
                block_size=block_size, scale_format=scale_format,
                momentum=momentum,
            ))
        else:
            replace_linear_with_calibrated_mx(child, w_dtype, a_dtype, block_size,
                                               scale_format, momentum, skip_names)
    return model


def set_a_dtype(model, a_dtype):
    """Hot-swap a_dtype on all CustomMXLinear layers without re-quantizing weights."""
    for mod in model.modules():
        if isinstance(mod, CustomMXLinear):
            mod.a_dtype = a_dtype


def set_a_dtype_calibrated(model, a_dtype, scale_format):
    """Switch CalibratedMXLinear a_dtype and re-initialize tracker from amax_global."""
    for mod in model.modules():
        if isinstance(mod, CalibratedMXLinear):
            mod.a_dtype = a_dtype
            fmt_max = get_format_max(a_dtype, scale_format)
            mod.act_tracker._enabled    = (fmt_max is not None)
            mod.act_tracker._format_max = fmt_max if fmt_max is not None else 1.0
            mod.act_tracker.amax_ema.copy_(mod.act_tracker.amax_global)


def run_calibration(model, input_ids: torch.Tensor, n_batches=32, seqlen=128):
    """Run calibration forward passes to collect per-layer activation amax."""
    calib_layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, CalibratedMXLinear):
            mod.act_tracker.start_calibration()
            calib_layers[name] = mod

    model.eval()
    with torch.no_grad():
        for i in range(min(n_batches, input_ids.numel() // seqlen)):
            model(input_ids[:, i * seqlen : (i + 1) * seqlen])

    stats = {}
    n_disabled = 0
    for name, mod in calib_layers.items():
        mod.act_tracker.end_calibration()
        if not mod.act_tracker._enabled:
            n_disabled += 1
            stats[name] = {'amax': None, 'scale': None, 'disabled': True}
        else:
            sc = mod.act_tracker.scale
            stats[name] = {
                'amax':  mod.act_tracker.amax_global.item(),
                'scale': sc.item() if isinstance(sc, torch.Tensor) else sc,
                'disabled': False,
            }

    enabled_stats = {k: v for k, v in stats.items() if not v['disabled']}

    print(f"\n{'='*65}")
    print(f"  Calibration done  n_batches={n_batches}  seqlen={seqlen}")
    if n_disabled == len(calib_layers):
        print(f"  [NOTE] scale_format has unlimited dynamic range (e.g. UE8M0).")
        print(f"         Per-tensor scale disabled for all {n_disabled} layers.")
        print(f"         Inference uses standard per-block MX (scale=1.0).")
    elif n_disabled > 0:
        print(f"  [{n_disabled} layers disabled: scale_format → per-block MX only]")
    if enabled_stats:
        print(f"  {'Layer':<50s}  {'amax':>8}  {'scale':>10}")
        print(f"  {'-'*63}")
        for name, s in enabled_stats.items():
            parts = name.split(".")
            idx   = next((p for p in parts if p.isdigit()), "")
            sn    = f"L{idx}.{parts[-1]}" if idx else parts[-1]
            amax_val  = s['amax'].item()  if isinstance(s['amax'],  torch.Tensor) else s['amax']
            scale_val = s['scale'].item() if isinstance(s['scale'], torch.Tensor) else s['scale']
            print(f"  {sn:<50s}  {amax_val:8.4f}  {scale_val:10.6f}")
    print(f"{'='*65}\n")
    return stats


def _collect_calib_amax(model_str: str, calib_enc: torch.Tensor,
                        n_batches: int, seqlen: int, device,
                        base_dtype=torch.bfloat16) -> dict:
    """Run BF16 forward passes on train data; return {layer_name: amax_float}."""
    model = AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype=base_dtype,
        low_cpu_mem_usage=True, device_map=device, tie_word_embeddings=True,
    ).eval()

    amax_dict = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            amax_dict[name] = 0.0
            def _hook(_, inp, __, _name=name):
                cur = inp[0].detach().float().abs().max().item()
                if cur > amax_dict[_name]:
                    amax_dict[_name] = cur
            hooks.append(mod.register_forward_hook(_hook))

    with torch.no_grad():
        for i in range(min(n_batches, calib_enc.numel() // seqlen)):
            model(calib_enc[:, i * seqlen : (i + 1) * seqlen])

    for h in hooks:
        h.remove()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  [calib] collected amax for {len(amax_dict)} layers "
          f"(n_batches={n_batches}, seqlen={seqlen})")
    return amax_dict


def _apply_calib_amax(model, calib_amax: dict) -> None:
    """Initialize CalibratedMXLinear trackers from pre-collected amax dict."""
    for name, mod in model.named_modules():
        if isinstance(mod, CalibratedMXLinear) and name in calib_amax:
            mod.act_tracker.amax_global.fill_(calib_amax[name])
            mod.act_tracker.end_calibration()
            mod.act_tracker.freeze()


def dump_calib_stats(
    model_str: str = "facebook/opt-125m",
    a_fmt: str = 'mxint8',
    w_fmt: str = 'mxint8',
    scale_fmt: str = 'UE8M0',
    block_size: int = 32,
    base_dtype=torch.bfloat16,
    calib_batches: int = 32,
    calib_seqlen: int = 128,
    momentum: float = 0.99,
    save_csv: str = None,
) -> str:
    """Load model, run calibration, save per-layer amax/scale to CSV; returns CSV path."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== [dump_calib_stats] {model_str}  "
          f"a={a_fmt} w={w_fmt} scale={scale_fmt} blk={block_size} ===")

    model = AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype=base_dtype,
        low_cpu_mem_usage=True, device_map='auto', tie_word_embeddings=True,
    )
    model = replace_linear_with_calibrated_mx(
        model, w_dtype=w_fmt, a_dtype=a_fmt,
        block_size=block_size, scale_format=scale_fmt, momentum=momentum,
    )

    calib_enc = _load_testencs(model_str, device, ('wikitext',))['wikitext']
    stats = run_calibration(model, calib_enc, n_batches=calib_batches, seqlen=calib_seqlen)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if save_csv is None:
        model_tag = model_str.replace('/', '_')
        save_csv  = f"{model_tag}_calib_{a_fmt}_{scale_fmt}_blk{block_size}.csv"

    with open(save_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'amax', 'scale', 'inv_scale',
                         'ceil_log2_scale', 'diff_inv_ceil_log2', 'disabled'])
        for name, s in stats.items():
            if s['disabled'] or s['scale'] is None:
                writer.writerow([name, s['amax'], s['scale'],
                                 None, None, None, s['disabled']])
            else:
                sc           = float(s['scale'])
                inv_sc       = 1.0 / sc
                ceil_log2_sc = math.ceil(math.log2(sc))
                diff         = inv_sc - ceil_log2_sc
                writer.writerow([name, s['amax'], sc,
                                 inv_sc, ceil_log2_sc, diff, s['disabled']])

    print(f"  [dump_calib_stats] wrote {len(stats)} layers → {save_csv}")
    return save_csv
