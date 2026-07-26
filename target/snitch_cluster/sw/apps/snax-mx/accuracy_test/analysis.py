# analysis.py — W/A quantization block-level analysis and visualization tools.

import os
import math
import csv
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from quantize_torch import quantize_mx_v1, quantize_mx_v2, quantize_mx_v3
from eval_utils import _load_testencs, _run_eval_dataset
from mx_linear import CustomMXLinear


def _block_amax_stats(x: torch.Tensor, block_size: int):
    """[R, K] float32 → block amax stats dict (mean, std, cv, max_min_ratio, kurtosis)."""
    R, K = x.shape
    NB = math.ceil(K / block_size)
    pad = NB * block_size - K
    xp  = F.pad(x, (0, pad)) if pad else x
    blk = xp.reshape(R * NB, block_size)
    amaxes     = blk.abs().amax(dim=1).float()
    amaxes_pos = amaxes[amaxes > 0]
    mean_ = float(amaxes_pos.mean()) if len(amaxes_pos) else 0.0
    std_  = float(amaxes_pos.std())  if len(amaxes_pos) > 1 else 0.0
    cv    = std_ / mean_ if mean_ > 0 else 0.0
    mmr   = float(amaxes_pos.max() / amaxes_pos.min()) if len(amaxes_pos) > 1 else 1.0
    # excess kurtosis (Fisher): E[(x-μ)^4]/σ^4 - 3
    if len(amaxes_pos) > 3 and std_ > 0:
        z4   = ((amaxes_pos - mean_) / std_).pow(4).mean().item()
        kurt = z4 - 3.0
    else:
        kurt = 0.0
    return dict(amax_mean=mean_, amax_std=std_, cv=cv, max_min_ratio=mmr, kurtosis=kurt)


def _tensor_kurtosis(x: torch.Tensor) -> float:
    """Excess kurtosis of all elements in x."""
    xf = x.float().reshape(-1)
    if xf.numel() < 4:
        return 0.0
    mu  = xf.mean()
    std = xf.std()
    if std == 0:
        return 0.0
    return float(((xf - mu) / std).pow(4).mean()) - 3.0


def summarize_layer_distributions(
    model_str: str = "facebook/opt-125m",
    n_batches: int = 4,
    seqlen: int = 128,
    device: str = None,
    save_csv: str = None,
):
    """Collect per-layer activation and weight distribution stats (min, max, mean, var, std)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", tie_word_embeddings=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False, trust_remote_code=True)
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt").input_ids.to(device)

    layer_names  = []
    act_buffers  = {}
    weight_cache = {}

    def _make_hook(name):
        def hook(mod, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            act_buffers[name].append(x)
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if mod.weight.is_meta:
                continue
            layer_names.append(name)
            act_buffers[name] = []
            weight_cache[name] = mod.weight.detach()
            hooks.append(mod.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        for i in range(min(n_batches, testenc.numel() // seqlen)):
            model(testenc[:, i * seqlen : (i + 1) * seqlen])

    for h in hooks:
        h.remove()
    del model

    def _stats(t: torch.Tensor) -> dict:
        t = t.float()
        return {
            "min":  t.min().item(),
            "max":  t.max().item(),
            "mean": t.mean().item(),
            "var":  t.var().item(),
            "std":  t.std().item(),
        }

    results = []
    for name in layer_names:
        if not act_buffers[name]:
            continue
        act = torch.cat(act_buffers[name], dim=0)
        w   = weight_cache[name]
        row = {"layer": name,
               **{f"act_{k}": v for k, v in _stats(act).items()},
               **{f"w_{k}":   v for k, v in _stats(w).items()}}
        results.append(row)

    def _short_name(full_name):
        parts = full_name.split(".")
        idx   = next((p for p in parts if p.isdigit()), "")
        return f"L{idx}.{parts[-1]}" if idx else parts[-1]

    hdr = (f"  {'Layer':<40s}"
           f"  {'act_min':>9}  {'act_max':>9}  {'act_mean':>9}  {'act_std':>9}"
           f"  {'w_min':>9}  {'w_max':>9}  {'w_mean':>9}  {'w_std':>9}")
    sep = "=" * (len(hdr) - 2)
    print(f"\n{'='*len(sep)}")
    print(f"  Layer distributions: {model_str}   n_batches={n_batches}  seqlen={seqlen}")
    print(f"{'='*len(sep)}")
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for r in results:
        sn = _short_name(r["layer"])
        print(f"  {sn:<40s}"
              f"  {r['act_min']:9.4f}  {r['act_max']:9.4f}"
              f"  {r['act_mean']:9.4f}  {r['act_std']:9.4f}"
              f"  {r['w_min']:9.4f}  {r['w_max']:9.4f}"
              f"  {r['w_mean']:9.4f}  {r['w_std']:9.4f}")
    print(f"{'='*len(sep)}")

    if save_csv is not None:
        model_tag = model_str.replace("/", "_")
        base, ext = os.path.splitext(save_csv)
        save_csv_out = f"{base}_{model_tag}{ext}"
        fields = ["layer", "act_min", "act_max", "act_mean", "act_var", "act_std",
                  "w_min",   "w_max",   "w_mean",   "w_var",   "w_std"]
        out_dir = os.path.dirname(os.path.abspath(save_csv_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(save_csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"  [csv] saved to {os.path.abspath(save_csv_out)}")

    return results


def compare_wa_block_stats(
    model_str: str,
    a_fmt: str = 'fp4_e2m1',
    w_fmt: str = 'fp4_e2m1',
    scale_format: str = 'UE5M3',
    block_size: int = 16,
    calib_batches: int = 4,
    calib_seqlen: int = 128,
    base_dtype=torch.bfloat16,
    save_csv: str = None,
):
    """Compare per-layer W and A block amax CV and kurtosis."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[compare_wa_block_stats] {model_str}  fmt={a_fmt}/{w_fmt}  scale={scale_format}  blk={block_size}")

    model = AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype=base_dtype,
        low_cpu_mem_usage=True, device_map="auto", tie_word_embeddings=True,
    ).eval()

    enc = _load_testencs(model_str, device, ('wikitext',))['wikitext']

    act_bufs = {}
    hooks    = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            act_bufs[name] = []
            def _h(m, inp, out, _n=name):
                act_bufs[_n].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu())
            hooks.append(mod.register_forward_hook(_h))

    with torch.no_grad():
        for i in range(min(calib_batches, enc.numel() // calib_seqlen)):
            model(enc[:, i * calib_seqlen : (i + 1) * calib_seqlen])

    for h in hooks:
        h.remove()

    rows = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if name not in act_bufs or not act_bufs[name]:
            continue

        w = mod.weight.detach().cpu().float()
        w_stats = _block_amax_stats(w, block_size)
        w_kurt  = _tensor_kurtosis(w)

        act = torch.cat(act_bufs[name], dim=0)
        a_stats = _block_amax_stats(act, block_size)
        a_kurt  = _tensor_kurtosis(act)

        rows.append(dict(
            layer             = name,
            w_amax_mean       = w_stats['amax_mean'],
            w_amax_std        = w_stats['amax_std'],
            w_cv              = w_stats['cv'],
            w_max_min_ratio   = w_stats['max_min_ratio'],
            w_kurtosis        = w_kurt,
            a_amax_mean       = a_stats['amax_mean'],
            a_amax_std        = a_stats['amax_std'],
            a_cv              = a_stats['cv'],
            a_max_min_ratio   = a_stats['max_min_ratio'],
            a_kurtosis        = a_kurt,
        ))

    del model

    if save_csv:
        with open(save_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  → {save_csv}  ({len(rows)} layers)")

    if rows:
        wcv = [r['w_cv'] for r in rows]
        acv = [r['a_cv'] for r in rows]
        print(f"  W  CV: mean={sum(wcv)/len(wcv):.3f}  max={max(wcv):.3f}")
        print(f"  A  CV: mean={sum(acv)/len(acv):.3f}  max={max(acv):.3f}")
    return rows


def compare_wa_quant_mse(
    model_str: str,
    a_fmt: str = 'fp4_e2m1',
    w_fmt: str = 'fp4_e2m1',
    scale_format: str = 'UE5M3',
    block_size: int = 16,
    quant_version: str = 'v3',
    calib_batches: int = 4,
    calib_seqlen: int = 128,
    base_dtype=torch.bfloat16,
    save_csv: str = None,
):
    """Per-layer W and A normalized quantization MSE (NMSE = MSE / mean(x^2))."""
    import run_mx_quantized as rmx
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[compare_wa_quant_mse] {model_str}  {a_fmt}/{w_fmt}  scale={scale_format}  {quant_version}  blk={block_size}")

    orig_version = rmx.QUANT_VERSION
    rmx.QUANT_VERSION = quant_version

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_str, torch_dtype=base_dtype,
            low_cpu_mem_usage=True, device_map="auto", tie_word_embeddings=True,
        ).eval()

        enc = _load_testencs(model_str, device, ('wikitext',))['wikitext']

        act_bufs = {}
        hooks    = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                act_bufs[name] = []
                def _h(m, inp, out, _n=name):
                    act_bufs[_n].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu())
                hooks.append(mod.register_forward_hook(_h))

        with torch.no_grad():
            for i in range(min(calib_batches, enc.numel() // calib_seqlen)):
                model(enc[:, i * calib_seqlen : (i + 1) * calib_seqlen])

        for h in hooks:
            h.remove()

        rows = []
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if name not in act_bufs or not act_bufs[name]:
                continue

            w_orig = mod.weight.detach().cpu().float()
            if quant_version == 'v1':
                q_w, _, _, _ = quantize_mx_v1(w_orig, w_fmt, block_size=block_size,
                                               axis=1, scale_format=scale_format)
            elif quant_version == 'v2':
                q_w, _, _, _ = quantize_mx_v2(w_orig, w_fmt, block_size=block_size,
                                               axis=1, scale_format=scale_format)
            else:
                q_w, _, _, _, _ = quantize_mx_v3(w_orig, w_fmt, block_size=block_size,
                                                  axis=1, scale_format=scale_format)
            w_mse  = float(((w_orig - q_w.float()) ** 2).mean())
            w_norm = float(w_orig.pow(2).mean())
            w_nmse = w_mse / w_norm if w_norm > 0 else 0.0
            del w_orig, q_w

            act = torch.cat(act_bufs[name], dim=0).float()
            if quant_version == 'v1':
                q_a, _, _, _ = quantize_mx_v1(act, a_fmt, block_size=block_size,
                                               axis=1, scale_format=scale_format)
            elif quant_version == 'v2':
                q_a, _, _, _ = quantize_mx_v2(act, a_fmt, block_size=block_size,
                                               axis=1, scale_format=scale_format)
            else:
                q_a, _, _, _, _ = quantize_mx_v3(act, a_fmt, block_size=block_size,
                                                  axis=1, scale_format=scale_format)
            a_mse  = float(((act - q_a.float()) ** 2).mean())
            a_norm = float(act.pow(2).mean())
            a_nmse = a_mse / a_norm if a_norm > 0 else 0.0
            ratio  = w_nmse / a_nmse if a_nmse > 0 else float('inf')

            rows.append(dict(
                layer      = name,
                w_mse      = w_mse,
                w_nmse     = w_nmse,
                a_mse      = a_mse,
                a_nmse     = a_nmse,
                nmse_ratio = ratio,
            ))

        del model

    finally:
        rmx.QUANT_VERSION = orig_version

    if save_csv and rows:
        with open(save_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  → {save_csv}  ({len(rows)} layers)")

    if rows:
        ratios = [r['nmse_ratio'] for r in rows if math.isfinite(r['nmse_ratio'])]
        above1 = sum(1 for r in ratios if r > 1)
        print(f"  nmse_ratio > 1 (W harder): {above1}/{len(ratios)} layers")
        print(f"  median nmse_ratio: {sorted(ratios)[len(ratios)//2]:.3f}")
    return rows


def layer_sensitivity_scan(
    model_str: str,
    a_fmt: str = 'fp4_e2m1',
    w_fmt: str = 'fp4_e2m1',
    scale_format: str = 'UE5M3',
    block_size: int = 16,
    base_dtype=torch.bfloat16,
    testenc=None,
    max_samples: int = 10,
    device: str = None,
    save_csv: str = None,
):
    """Per-layer PPL sensitivity scan: W_only vs A_only quantization per Linear layer."""
    import gc
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[layer_sensitivity_scan] {model_str}  {a_fmt}/{w_fmt}  scale={scale_format}")

    if testenc is None:
        testenc = _load_testencs(model_str, device, ('wikitext',))['wikitext']

    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype=base_dtype,
        low_cpu_mem_usage=True, device_map="auto", tie_word_embeddings=True,
    ).eval()
    ppl_base = _run_eval_dataset(model_bf16, device, "BF16 baseline", model_str,
                                 testenc=testenc, max_samples=max_samples)
    del model_bf16

    model_tmp = AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype=base_dtype,
        low_cpu_mem_usage=True, device_map="auto", tie_word_embeddings=True,
    ).eval()
    linear_names = [n for n, m in model_tmp.named_modules()
                    if isinstance(m, torch.nn.Linear)
                    and not any(s in n for s in ('lm_head', 'embed_tokens'))]
    del model_tmp
    gc.collect()

    rows = []
    for idx, layer_name in enumerate(linear_names):
        print(f"  [{idx+1}/{len(linear_names)}] {layer_name}")
        for mode in ('W_only', 'A_only'):
            model = AutoModelForCausalLM.from_pretrained(
                model_str, torch_dtype=base_dtype,
                low_cpu_mem_usage=True, device_map="auto", tie_word_embeddings=True,
            ).eval()

            parent = model
            parts  = layer_name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            orig_linear = getattr(parent, parts[-1])

            if mode == 'W_only':
                new_layer = CustomMXLinear(
                    orig_linear.in_features, orig_linear.out_features,
                    orig_linear.weight.data,
                    orig_linear.bias.data if orig_linear.bias is not None else None,
                    w_dtype=w_fmt, a_dtype='identity',
                    block_size=block_size, scale_format=scale_format,
                )
            else:
                new_layer = CustomMXLinear(
                    orig_linear.in_features, orig_linear.out_features,
                    orig_linear.weight.data,
                    orig_linear.bias.data if orig_linear.bias is not None else None,
                    w_dtype='identity', a_dtype=a_fmt,
                    block_size=block_size, scale_format=scale_format,
                )
            setattr(parent, parts[-1], new_layer)

            ppl = _run_eval_dataset(model, device, f"{mode} {layer_name}", model_str,
                                    testenc=testenc, max_samples=max_samples)
            del model; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            existing = next((r for r in rows if r['layer'] == layer_name), None)
            if existing is None:
                existing = dict(layer=layer_name, ppl_W_only=None, delta_ppl_W=None,
                                ppl_A_only=None, delta_ppl_A=None, sensitivity_ratio=None)
                rows.append(existing)
            if mode == 'W_only':
                existing['ppl_W_only']  = ppl
                existing['delta_ppl_W'] = ppl - ppl_base
            else:
                existing['ppl_A_only']  = ppl
                existing['delta_ppl_A'] = ppl - ppl_base

        r = rows[-1]
        if r['delta_ppl_A'] and abs(r['delta_ppl_A']) > 0:
            r['sensitivity_ratio'] = r['delta_ppl_W'] / r['delta_ppl_A']

    if save_csv and rows:
        with open(save_csv, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print(f"  → {save_csv}")
    return rows


def plot_wa_comparison(
    block_stats_rows=None,
    mse_rows=None,
    sensitivity_rows=None,
    save_path: str = 'wa_comparison.png',
) -> 'plt.Figure':
    """Visualize W vs A quantization symmetry: up to 4 subplots (CV, NMSE, PPL, CV-NMSE scatter)."""
    import matplotlib.pyplot as plt

    panels = []
    if block_stats_rows:
        panels.append('cv')
    if mse_rows:
        panels.append('nmse')
        if block_stats_rows:
            panels.append('corr')
    if sensitivity_rows:
        panels.append('ppl')

    if not panels:
        raise ValueError("All input rows are None — nothing to plot.")

    ncols = min(len(panels), 2)
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows))
    if len(panels) == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    ax_list = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    def _short(name):
        parts = name.split('.')
        idx = next((p for p in parts if p.isdigit()), '')
        return f"L{idx}.{parts[-1]}" if idx else parts[-1]

    for pi, panel in enumerate(panels):
        ax = ax_list[pi]

        if panel == 'cv':
            names = [_short(r['layer']) for r in block_stats_rows]
            x     = range(len(names))
            ax.bar([i - 0.2 for i in x], [r['w_cv'] for r in block_stats_rows],
                   0.4, label='Weight CV', color='#4499FF', alpha=0.8)
            ax.bar([i + 0.2 for i in x], [r['a_cv'] for r in block_stats_rows],
                   0.4, label='Activation CV', color='#FF6666', alpha=0.8)
            ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_ylabel('Block amax CV (std/mean)'); ax.set_title('Block amax Variation (CV)')
            ax.legend(fontsize=8)

        elif panel == 'nmse':
            names = [_short(r['layer']) for r in mse_rows]
            x     = range(len(names))
            ax.bar([i - 0.2 for i in x], [r['w_nmse'] for r in mse_rows],
                   0.4, label='W NMSE', color='#4499FF', alpha=0.8)
            ax.bar([i + 0.2 for i in x], [r['a_nmse'] for r in mse_rows],
                   0.4, label='A NMSE', color='#FF6666', alpha=0.8)
            ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_yscale('log'); ax.set_ylabel('NMSE (log scale)'); ax.set_title('Normalized Quantization MSE')
            ax.legend(fontsize=8)

        elif panel == 'ppl':
            names = [_short(r['layer']) for r in sensitivity_rows]
            x     = range(len(names))
            dW = [r['delta_ppl_W'] or 0 for r in sensitivity_rows]
            dA = [r['delta_ppl_A'] or 0 for r in sensitivity_rows]
            ax.bar([i - 0.2 for i in x], dW, 0.4, label='ΔPPL (W only)', color='#4499FF', alpha=0.8)
            ax.bar([i + 0.2 for i in x], dA, 0.4, label='ΔPPL (A only)', color='#FF6666', alpha=0.8)
            ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_ylabel('ΔPPL vs BF16'); ax.set_title('Per-Layer PPL Sensitivity')
            ax.legend(fontsize=8)

        elif panel == 'corr':
            bs_map = {r['layer']: r for r in block_stats_rows}
            for r in mse_rows:
                if r['layer'] not in bs_map:
                    continue
                bs = bs_map[r['layer']]
                ax.scatter(bs['w_cv'], r['w_nmse'], c='#4499FF', alpha=0.7, s=20)
                ax.scatter(bs['a_cv'], r['a_nmse'], c='#FF6666', alpha=0.7, s=20)
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#4499FF', markersize=8, label='Weight'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor='#FF6666', markersize=8, label='Activation'),
            ], fontsize=8)
            ax.set_xlabel('Block amax CV'); ax.set_ylabel('NMSE')
            ax.set_yscale('log'); ax.set_title('CV vs NMSE Correlation')

    for pi in range(len(panels), nrows * ncols):
        ax_list[pi].set_visible(False)

    fig.suptitle('W vs A Quantization Symmetry Analysis', fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=130)
        print(f"  [plot_wa_comparison] → {save_path}")
    return fig
