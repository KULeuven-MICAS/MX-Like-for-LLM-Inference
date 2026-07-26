# eval_utils.py — dataset loading, PPL evaluation, and CSV helpers.

import os
import math
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


_CSV_FIELDS = ['dataset', 'model', 'a_fmt', 'w_fmt', 'scale_fmt', 'block_size', 'ppl', 'delta_ppl', 'baseline_ppl']


def _append_csv(path: str, rows: list):
    """Append rows to CSV; write header only when the file is new/empty."""
    import csv
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction='ignore', restval='')
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    print(f"  [csv] {len(rows)} row(s) appended to {os.path.abspath(path)}")


def load_c4_testenc(model_str: str, seqlen: int = 2048, n_samples: int = 256,
                    seed: int = 0, cache_dir: str = './data_cache') -> torch.Tensor:
    """Build a C4 validation token tensor by randomly sampling n_samples chunks; cached to disk."""
    import random
    os.makedirs(cache_dir, exist_ok=True)
    model_tag  = model_str.replace('/', '_').replace('-', '_').lower()
    cache_path = os.path.join(cache_dir, f'testenc_{model_tag}_c4_{seqlen}_n{n_samples}_s{seed}.pt')

    if os.path.exists(cache_path):
        testenc = torch.load(cache_path, weights_only=True)
        print(f"  [C4] loaded {n_samples}×{seqlen} tokens from cache: {cache_path}")
        return testenc

    print(f"  [C4] building testenc ({n_samples} samples × {seqlen} tokens, seed={seed})...")
    tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False, trust_remote_code=True)
    tokenizer.model_max_length = int(1e30)

    raw = load_dataset(
        "allenai/c4",
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation',
    )

    random.seed(seed)
    chunks = []
    with tqdm(total=n_samples, desc="[C4] sampling chunks", unit="chunk") as pbar:
        while len(chunks) < n_samples:
            i   = random.randint(0, len(raw) - 1)
            ids = tokenizer(raw[i]['text'], return_tensors='pt').input_ids
            if ids.shape[1] <= seqlen:
                continue
            start = random.randint(0, ids.shape[1] - seqlen - 1)
            chunks.append(ids[:, start : start + seqlen])
            pbar.update(1)

    testenc = torch.hstack(chunks)
    torch.save(testenc, cache_path)
    print(f"  [C4] saved to cache: {cache_path}  shape={tuple(testenc.shape)}")
    return testenc


def _load_testencs(model_str: str, device, datasets: tuple, seqlen: int = 2048,
                   c4_n_samples: int = 256, c4_seed: int = 0,
                   cache_dir: str = './data_cache') -> dict:
    """Load and tokenize all requested datasets once; returns dict[dataset_name → tensor]."""
    result = {}
    if 'wikitext' in datasets:
        tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False, trust_remote_code=True)
        tokenizer.model_max_length = int(1e30)
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        result['wikitext'] = tokenizer(
            "\n\n".join(raw["text"]), return_tensors="pt"
        ).input_ids.to(device)
    if 'c4' in datasets:
        result['c4'] = load_c4_testenc(
            model_str, seqlen=seqlen, n_samples=c4_n_samples,
            seed=c4_seed, cache_dir=cache_dir,
        ).to(device)
    return result


def _load_calib_enc(model_str: str, device, dataset: str = 'wikitext',
                    n_batches: int = 32, seqlen: int = 128,
                    c4_seed: int = 42, cache_dir: str = './data_cache') -> torch.Tensor:
    """Load calibration data from train split (separate from eval test split)."""
    tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False, trust_remote_code=True)
    tokenizer.model_max_length = int(1e30)
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tokenizer("\n\n".join(raw["text"]), return_tensors="pt").input_ids
    return enc.to(device)


def run_eval(model, device, label="", model_str="facebook/opt-125m", seqlen=2048,
             testenc=None, max_samples=None, batch_size=1):
    """WikiText-2 PPL evaluation."""
    if testenc is None:
        tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False, trust_remote_code=True)
        tokenizer.model_max_length = int(1e30)
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(raw["text"]), return_tensors="pt").input_ids.to(device)

    nsamples = testenc.numel() // seqlen
    if max_samples is not None:
        nsamples = min(nsamples, max_samples)
    nsamples = (nsamples // batch_size) * batch_size
    loss_fct = torch.nn.CrossEntropyLoss()

    all_seqs   = testenc[:, : nsamples * seqlen].view(nsamples, seqlen).to(device)
    all_labels = all_seqs[:, 1:]

    nlls = []
    n_batches = nsamples // batch_size
    for i in tqdm(range(n_batches), desc="evaluating...", leave=False):
        batch  = all_seqs  [i * batch_size : (i + 1) * batch_size]
        labels = all_labels[i * batch_size : (i + 1) * batch_size]
        with torch.no_grad():
            lm_logits    = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            del lm_logits
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.reshape(-1))
            del shift_logits
        nlls.append(loss * batch_size * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(f"{label:<55s}  wikitext_ppl = {ppl.item():.4f}")
    return ppl.item()


def run_eval_c4(model, device, label="", model_str="facebook/opt-125m", seqlen=2048,
                testenc=None, n_samples=256, seed=0, cache_dir='./data_cache',
                batch_size=1, max_samples=None):
    """C4 PPL evaluation (n_samples independent random chunks of seqlen tokens each)."""
    if testenc is None:
        testenc = load_c4_testenc(model_str, seqlen=seqlen,
                                  n_samples=n_samples, seed=seed,
                                  cache_dir=cache_dir)

    actual_n  = (testenc.numel() // seqlen // batch_size) * batch_size
    if max_samples is not None:
        actual_n = min(actual_n, (max_samples // batch_size) * batch_size)
    all_seqs   = testenc[:, : actual_n * seqlen].view(actual_n, seqlen).to(device)
    all_labels = all_seqs[:, 1:]

    loss_fct  = torch.nn.CrossEntropyLoss()
    nlls      = []
    n_batches = actual_n // batch_size
    for i in tqdm(range(n_batches), desc="evaluating C4...", leave=False):
        batch  = all_seqs  [i * batch_size : (i + 1) * batch_size]
        labels = all_labels[i * batch_size : (i + 1) * batch_size]
        with torch.no_grad():
            lm_logits    = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            del lm_logits
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.reshape(-1))
            del shift_logits
        nlls.append(loss * batch_size * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (actual_n * seqlen))
    print(f"{label:<55s}  c4_ppl = {ppl.item():.4f}")
    return ppl.item()


def _run_eval_dataset(model, device, label, model_str, testenc=None, seqlen=2048,
                      dataset='wikitext', max_samples=None, batch_size=1):
    """Dispatch to run_eval (wikitext) or run_eval_c4 (c4)."""
    if dataset == 'c4':
        return run_eval_c4(model, device, label=label, model_str=model_str, seqlen=seqlen,
                           testenc=testenc, max_samples=max_samples, batch_size=batch_size)
    return run_eval(model, device, label=label, seqlen=seqlen,
                    model_str=model_str, testenc=testenc, max_samples=max_samples,
                    batch_size=batch_size)
