# Zijun Zhao <zijun.zhao@student.kuleuven.be>

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import torch
from transformers import AutoModelForCausalLM

from quantize_torch import (quantize_mx_v1, quantize_mx_v2, quantize_mx_v3,
                            VALID_SCALE_FORMATS, _SCALE_FORMAT_MAX_VAL, _DTYPE_MAX_VAL,
                            FP32_TINY)
from mx_linear import (CustomMXLinear, CalibratedMXLinear,
                       replace_linear_with_custom_mx, replace_linear_with_calibrated_mx,
                       set_a_dtype, set_a_dtype_calibrated,
                       run_calibration, dump_calib_stats,
                       _collect_calib_amax, _apply_calib_amax)
from eval_utils import (_load_testencs, _load_calib_enc, _run_eval_dataset, _append_csv)


QUANT_VERSION: str = 'v3'

_FP32_MAX = torch.finfo(torch.float32).max

_DATA_CLASSES = [
    'fp8_e4m3',
    'fp8_e5m2',
    'fp6_e3m2',
    'fp6_e2m3',
    'fp4_e2m1',
    'mxint8',
]


def _v3_overflows(scale_format: str, dtype: str) -> bool:
    """Return True when v3 per_tensor_scale denominator overflows FP32."""
    if QUANT_VERSION != 'v3':
        return False
    return float(_SCALE_FORMAT_MAX_VAL[scale_format]) * float(_DTYPE_MAX_VAL[dtype]) > float(_FP32_MAX)


def _mx_quantize(arr, dtype, block_size, axis, scale_format):
    """Dispatch quantization to the active QUANT_VERSION; returns (q_arr, scale_shared, scale_raw, q_raw)."""
    if QUANT_VERSION == 'v3':
        q, sc, sc_r, q_r, _ = quantize_mx_v3(arr, dtype, block_size=block_size,
                                               axis=axis, scale_format=scale_format)
        return q, sc, sc_r, q_r
    if QUANT_VERSION == 'v2':
        return quantize_mx_v2(arr, dtype, block_size=block_size,
                               axis=axis, scale_format=scale_format)
    return quantize_mx_v1(arr, dtype, block_size=block_size,
                          axis=axis, scale_format=scale_format)


def scan_scale_format(device, model_str="facebook/opt-125m", ppl_base=None, w_fmt='fp6_e2m3',
                      a_fmt='fp8_e5m2', block_size=32, base_dtype=torch.bfloat16,
                      save_csv=None, dataset='wikitext', batch_size=1,
                      calibrate=False, calib_batches=32, calib_seqlen=128, momentum=0.99):
    """Sweep all scale formats for a fixed (w_fmt, a_fmt) pair."""
    calib_tag = " [calibrated]" if calibrate else ""
    print(f"\n=== scan_scale_format: {a_fmt} x {w_fmt} | block_size={block_size} | dataset={dataset}{calib_tag} ===")
    testencs = _load_testencs(model_str, device, (dataset,))
    testenc  = testencs[dataset]
    rows = []
    for scale_fmt in VALID_SCALE_FORMATS:
        model = AutoModelForCausalLM.from_pretrained(
            model_str, torch_dtype=base_dtype,
            low_cpu_mem_usage=True, device_map='auto', tie_word_embeddings=True
        )
        if calibrate:
            model = replace_linear_with_calibrated_mx(
                model, w_dtype=w_fmt, a_dtype=a_fmt,
                block_size=block_size, scale_format=scale_fmt, momentum=momentum,
            )
            run_calibration(model, testenc, n_batches=calib_batches, seqlen=calib_seqlen)
        else:
            model = replace_linear_with_custom_mx(
                model, w_dtype=w_fmt, a_dtype=a_fmt,
                block_size=block_size, scale_format=scale_fmt,
            )
        label = f"{a_fmt} x {w_fmt} | scale={scale_fmt}{calib_tag}"
        model = model.eval()
        ppl   = _run_eval_dataset(model, device, label, model_str, testenc, 2048, dataset,
                                  batch_size=batch_size)
        delta = ppl - ppl_base
        print(f"  {'Δppl vs BF16:':20s} {delta:+.4f}\n")
        rows.append(dict(dataset=dataset, model=model_str, a_fmt=a_fmt, w_fmt=w_fmt,
                         scale_fmt=scale_fmt, block_size=block_size,
                         ppl=f"{ppl:.4f}", delta_ppl=f"{delta:+.4f}",
                         baseline_ppl=f"{ppl_base:.4f}", calibrate=calibrate))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if save_csv is not None:
        _append_csv(save_csv, rows)


def scan_W(device, model_str="facebook/opt-125m", ppl_base=None, a_fmt='mxint8',
           scale_fmt='UE8M0', block_size=32, base_dtype=torch.bfloat16,
           save_csv=None, dataset='wikitext', batch_size=1):
    """Sweep each weight element format for a fixed (a_fmt, scale_fmt) pair."""
    print(f"\n=== scan_W: a_fmt={a_fmt} | block_size={block_size} | dataset={dataset} ===")
    testencs = _load_testencs(model_str, device, (dataset,), c4_n_samples=30)
    testenc  = testencs[dataset]
    rows = []
    for w_fmt in _DATA_CLASSES:
        model = AutoModelForCausalLM.from_pretrained(
            model_str, torch_dtype=base_dtype,
            low_cpu_mem_usage=True, device_map='auto', tie_word_embeddings=True
        )
        model = replace_linear_with_custom_mx(
            model, w_dtype=w_fmt, a_dtype=a_fmt,
            block_size=block_size, scale_format=scale_fmt,
        )
        label = f"{a_fmt} x {w_fmt} | scale={scale_fmt}"
        model = model.eval()
        ppl   = _run_eval_dataset(model, device, label, model_str, testenc, 2048, dataset,
                                  batch_size=batch_size)
        delta = ppl - ppl_base
        print(f"  {'Δppl vs BF16:':20s} {delta:+.4f}\n")
        rows.append(dict(dataset=dataset, model=model_str, a_fmt=a_fmt, w_fmt=w_fmt,
                         scale_fmt=scale_fmt, block_size=block_size,
                         ppl=f"{ppl:.4f}", delta_ppl=f"{delta:+.4f}",
                         baseline_ppl=f"{ppl_base:.4f}"))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if save_csv is not None:
        _append_csv(save_csv, rows)


def scan_W_and_scale(device, model_str="facebook/opt-125m", ppl_base=None, a_fmt='fp8_e5m2',
                     block_size=32, base_dtype=torch.bfloat16, save_csv=None,
                     dataset='wikitext', threshold=50.0, batch_size=1,
                     calibrate=False, calib_batches=32, calib_seqlen=128, momentum=0.99):
    """2D sweep: weight format × scale format."""
    calib_tag = " [calibrated]" if calibrate else ""
    print(f"\n=== scan_W_and_scale: a_fmt={a_fmt} | block_size={block_size} | dataset={dataset}{calib_tag} ===")
    testencs = _load_testencs(model_str, device, (dataset,), c4_n_samples=30)
    testenc  = testencs[dataset]
    rows = []
    for w_fmt in _DATA_CLASSES:
        for scale_fmt in VALID_SCALE_FORMATS:
            model = AutoModelForCausalLM.from_pretrained(
                model_str, torch_dtype=base_dtype,
                low_cpu_mem_usage=True, device_map='auto', tie_word_embeddings=True
            )
            if calibrate:
                model = replace_linear_with_calibrated_mx(
                    model, w_dtype=w_fmt, a_dtype=a_fmt,
                    block_size=block_size, scale_format=scale_fmt, momentum=momentum,
                )
                run_calibration(model, testenc, n_batches=calib_batches, seqlen=calib_seqlen)
            else:
                model = replace_linear_with_custom_mx(
                    model, w_dtype=w_fmt, a_dtype=a_fmt,
                    block_size=block_size, scale_format=scale_fmt,
                )
            label = f"{a_fmt} x {w_fmt} | scale={scale_fmt}{calib_tag}"
            model = model.eval()
            ppl      = _run_eval_dataset(model, device, label, model_str, testenc, 2048, dataset,
                                         batch_size=batch_size)
            ppl_diff = ppl - ppl_base
            print(f"  {'Δppl vs BF16:':20s} {ppl_diff:+.4f}\n")
            rows.append(dict(dataset=dataset, model=model_str, a_fmt=a_fmt, w_fmt=w_fmt,
                             scale_fmt=scale_fmt, block_size=block_size,
                             ppl=f"{ppl:.4f}", delta_ppl=f"{ppl_diff:+.4f}",
                             baseline_ppl=f"{ppl_base:.4f}", calibrate=calibrate))
            del model
            if ppl_diff > threshold:
                break
    if save_csv is not None:
        _append_csv(save_csv, rows)


def scan_all(device, model_str="facebook/opt-125m", ppl_base=None, block_size=32,
             base_dtype=torch.bfloat16, save_csv=None,
             w_fmts=None, a_fmts=None, scale_fmts=None,
             max_samples=30, ppl_threshold=50.0, dataset='wikitext', testenc=None,
             batch_size=1,
             calibrate=False, calib_batches=32, calib_seqlen=128, momentum=0.99):
    """3D sweep: scale_fmt × w_fmt × a_fmt; supports resume and per-tensor scale overflow guard."""
    import csv as _csv
    if w_fmts     is None: w_fmts     = _DATA_CLASSES
    if a_fmts     is None: a_fmts     = _DATA_CLASSES
    if scale_fmts is None: scale_fmts = list(VALID_SCALE_FORMATS.keys())

    total = len(scale_fmts) * len(w_fmts) * len(a_fmts)
    print(f"\n=== scan_all: block_size={block_size}  total={total} combinations | dataset={dataset} ===")
    print(f"  scale_fmts : {scale_fmts}")
    print(f"  w_fmts     : {w_fmts}")
    print(f"  a_fmts     : {a_fmts}")
    print(f"  max_samples: {max_samples}\n")

    done_pairs: set = set()
    if save_csv is not None and os.path.exists(save_csv) and os.path.getsize(save_csv) > 0:
        with open(save_csv, newline='') as _f:
            for row in _csv.DictReader(_f):
                done_pairs.add((row['scale_fmt'], row['w_fmt']))
        if done_pairs:
            print(f"  [resume] skipping {len(done_pairs)} already-completed (scale_fmt, w_fmt) pairs\n")

    if testenc is None:
        testencs = _load_testencs(model_str, device, (dataset,))
        testenc  = testencs[dataset]

    calib_amax = None
    if calibrate:
        print(f"  [calib] loading train split for calibration (dataset={dataset}) ...")
        calib_enc = _load_calib_enc(
            model_str, device, dataset=dataset,
            n_batches=calib_batches, seqlen=calib_seqlen,
        )
        calib_amax = _collect_calib_amax(
            model_str, calib_enc, n_batches=calib_batches,
            seqlen=calib_seqlen, device=device, base_dtype=base_dtype,
        )
        del calib_enc

    rows = []
    done = 0

    for scale_fmt in scale_fmts:
        for w_fmt in w_fmts:

            if (scale_fmt, w_fmt) in done_pairs:
                done += len(a_fmts)
                continue

            if _v3_overflows(scale_fmt, w_fmt):
                reason = f"v3 per_tensor_scale overflow: scale_fmt_max×{w_fmt}_max > FP32_MAX"
                print(f"  [no_need] {scale_fmt}×{w_fmt}: {reason}")
                for a_fmt in a_fmts:
                    rows.append(dict(dataset=dataset, model=model_str, a_fmt=a_fmt, w_fmt=w_fmt,
                                     scale_fmt=scale_fmt, block_size=block_size,
                                     ppl='no_need', delta_ppl='no_need',
                                     baseline_ppl=f"{ppl_base:.4f}", calibrate=calibrate))
                    done += 1
                if save_csv is not None and rows:
                    _append_csv(save_csv, rows)
                    rows = []
                continue

            model = AutoModelForCausalLM.from_pretrained(
                model_str, torch_dtype=base_dtype,
                low_cpu_mem_usage=True, device_map=device, tie_word_embeddings=True
            )
            if calibrate:
                replace_linear_with_calibrated_mx(
                    model, w_dtype=w_fmt, a_dtype=a_fmts[0],
                    block_size=block_size, scale_format=scale_fmt, momentum=momentum,
                )
                _apply_calib_amax(model, calib_amax)
            else:
                replace_linear_with_custom_mx(
                    model, w_dtype=w_fmt, a_dtype=a_fmts[0],
                    block_size=block_size, scale_format=scale_fmt,
                )
            model = model.eval()

            skip_remaining = False
            for a_fmt in a_fmts:
                if skip_remaining:
                    break

                if _v3_overflows(scale_fmt, a_fmt):
                    reason = f"v3 per_tensor_scale overflow: scale_fmt_max×{a_fmt}_max > FP32_MAX"
                    print(f"  [no_need] {scale_fmt}×W={w_fmt} A={a_fmt}: {reason}")
                    rows.append(dict(dataset=dataset, model=model_str, a_fmt=a_fmt, w_fmt=w_fmt,
                                     scale_fmt=scale_fmt, block_size=block_size,
                                     ppl='no_need', delta_ppl='no_need',
                                     baseline_ppl=f"{ppl_base:.4f}", calibrate=calibrate))
                    done += 1
                    continue

                if calibrate:
                    set_a_dtype_calibrated(model, a_fmt, scale_fmt)
                else:
                    set_a_dtype(model, a_fmt)

                calib_tag = " [calib]" if calibrate else ""
                label = f"W={w_fmt:<12} A={a_fmt:<8} scale={scale_fmt}{calib_tag}"
                with torch.no_grad():
                    ppl = _run_eval_dataset(model, device, label, model_str,
                                            testenc, 2048, dataset,
                                            max_samples=max_samples,
                                            batch_size=batch_size)
                ppl_diff = ppl - ppl_base
                print(f"  Δppl = {ppl_diff:+.4f}")
                rows.append(dict(dataset=dataset, model=model_str, a_fmt=a_fmt, w_fmt=w_fmt,
                                 scale_fmt=scale_fmt, block_size=block_size,
                                 ppl=f"{ppl:.4f}", delta_ppl=f"{ppl_diff:+.4f}",
                                 baseline_ppl=f"{ppl_base:.4f}", calibrate=calibrate))
                done += 1

                if ppl_diff > ppl_threshold:
                    print(f"  [skip] Δppl={ppl_diff:+.1f} > {ppl_threshold}, skipping remaining a_fmts for {scale_fmt}×{w_fmt}")
                    skip_remaining = True

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if save_csv is not None and rows:
                _append_csv(save_csv, rows)
                rows = []

    print(f"\n  Scan complete: {done} / {total} combinations")
    if save_csv is not None and rows:
        _append_csv(save_csv, rows)


def run_spec_setting(device, model_str="facebook/opt-125m", ppl_base=None,
                     a_fmt='mxint8', w_fmt='mxint8', scale_fmt='UE8M0',
                     block_size=32, base_dtype=torch.bfloat16, dataset='wikitext',
                     batch_size=1, max_samples=30,
                     calibrate=False, calib_batches=32, calib_seqlen=128, momentum=0.99):
    """Run a single (w_fmt, a_fmt, scale_fmt) combination; calibrate=True uses CalibratedMXLinear."""
    calib_tag = " [calibrated]" if calibrate else ""
    print(f"\n=== w_fmt={w_fmt} | a_fmt={a_fmt} | scale={scale_fmt} | block_size={block_size}{calib_tag} ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype=base_dtype,
        low_cpu_mem_usage=True, device_map='auto', tie_word_embeddings=True
    )
    if calibrate:
        model = replace_linear_with_calibrated_mx(
            model, w_dtype=w_fmt, a_dtype=a_fmt,
            block_size=block_size, scale_format=scale_fmt, momentum=momentum,
        )
        calib_enc = _load_testencs(model_str, device, ('wikitext',))['wikitext']
        run_calibration(model, calib_enc, n_batches=calib_batches, seqlen=calib_seqlen)
    else:
        model = replace_linear_with_custom_mx(
            model, w_dtype=w_fmt, a_dtype=a_fmt,
            block_size=block_size, scale_format=scale_fmt,
        )
    label = f"{a_fmt} x {w_fmt} | scale={scale_fmt}{calib_tag}"
    model = model.eval()
    ppl = _run_eval_dataset(model, device, label=label, seqlen=2048, model_str=model_str,
                            dataset=dataset, batch_size=batch_size, max_samples=max_samples)
    print(f"  {'Δppl vs BF16:':20s} {ppl - ppl_base:+.4f}\n")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MX quantization PPL sweep")
    parser.add_argument('--model',         type=str,   default="Qwen/Qwen2.5-3B")
    parser.add_argument('--dataset',       type=str,   default="wikitext",
                        choices=["wikitext", "c4"])
    parser.add_argument('--seqlen',        type=int,   default=2048)
    parser.add_argument('--max-samples',   type=int,   default=30,
                        help="Max chunks to evaluate per model. 0 = full dataset.")
    parser.add_argument('--block-size',    type=int,   default=16)
    parser.add_argument('--dtype',         type=str,   default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument('--ppl-threshold', type=float, default=50.0)
    parser.add_argument('--w-fmts',        type=str,   default=None)
    parser.add_argument('--a-fmts',        type=str,   default=None)
    parser.add_argument('--scale-fmts',    type=str,   default="UE8M0,UE7M1,UE6M2,UE5M3,UE4M4,UE4M3")
    parser.add_argument('--output-dir',    type=str,   default=".")
    parser.add_argument('--batch-size',    type=int,   default=1)
    parser.add_argument('--quant-version', type=str,   default='v3', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--calibrate',     action='store_true')
    parser.add_argument('--calib-batches', type=int,   default=32)
    parser.add_argument('--calib-seqlen',  type=int,   default=128)
    parser.add_argument('--momentum',      type=float, default=0.99)
    parser.add_argument('--dump-calib',    action='store_true')
    parser.add_argument('--dump-calib-csv', type=str,  default=None)
    args = parser.parse_args()

    global QUANT_VERSION
    QUANT_VERSION = args.quant_version

    _DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    base_dtype  = _DTYPE_MAP[args.dtype]
    max_samples = args.max_samples if args.max_samples > 0 else None
    w_fmts      = args.w_fmts.split(',')     if args.w_fmts     else None
    a_fmts      = args.a_fmts.split(',')     if args.a_fmts     else None
    scale_fmts  = args.scale_fmts.split(',') if args.scale_fmts else None

    model_tag   = args.model.replace('/', '_')
    samples_tag = str(max_samples) if max_samples is not None else "full"
    calib_tag   = "_calib" if args.calibrate else ""
    out_dir     = os.path.join(args.output_dir, model_tag)
    os.makedirs(out_dir, exist_ok=True)
    save_csv    = os.path.join(
        out_dir,
        f"{model_tag}_result_blk{args.block_size}_{args.dataset}_{samples_tag}_{QUANT_VERSION}{calib_tag}.csv",
    )

    _calib_kwargs = dict(
        calibrate=args.calibrate,
        calib_batches=args.calib_batches,
        calib_seqlen=args.calib_seqlen,
        momentum=args.momentum,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    print(f"Results will be saved to: {os.path.abspath(save_csv)}\n")

    if args.dump_calib:
        a_fmt_calib     = (a_fmts or _DATA_CLASSES)[0]
        w_fmt_calib     = (w_fmts or _DATA_CLASSES)[0]
        scale_fmt_calib = (scale_fmts or ['UE4M4'])[0]
        calib_csv = args.dump_calib_csv or os.path.join(
            out_dir,
            f"{model_tag}_calib_{a_fmt_calib}_{scale_fmt_calib}_blk{args.block_size}.csv",
        )
        dump_calib_stats(
            model_str=args.model,
            a_fmt=a_fmt_calib,
            w_fmt=w_fmt_calib,
            scale_fmt=scale_fmt_calib,
            block_size=args.block_size,
            base_dtype=base_dtype,
            calib_batches=args.calib_batches,
            calib_seqlen=args.calib_seqlen,
            momentum=args.momentum,
            save_csv=calib_csv,
        )
        return

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=base_dtype,
                                                  low_cpu_mem_usage=True, device_map='auto',
                                                  tie_word_embeddings=True)
    with torch.no_grad():
        model    = model.eval()
        testencs = _load_testencs(args.model, device, (args.dataset,), seqlen=args.seqlen)
        testenc  = testencs[args.dataset]
        ppl_base = _run_eval_dataset(model, device, label="BF16 baseline", model_str=args.model,
                                     testenc=testenc, seqlen=args.seqlen, dataset=args.dataset,
                                     max_samples=max_samples, batch_size=args.batch_size)
        del model

        scan_all(
            device=device,
            model_str=args.model,
            ppl_base=ppl_base,
            block_size=args.block_size,
            base_dtype=base_dtype,
            save_csv=save_csv,
            w_fmts=w_fmts,
            a_fmts=a_fmts,
            scale_fmts=scale_fmts,
            max_samples=max_samples,
            ppl_threshold=args.ppl_threshold,
            dataset=args.dataset,
            testenc=testenc,
            batch_size=args.batch_size,
            **_calib_kwargs,
        )


if __name__ == "__main__":
    main()
