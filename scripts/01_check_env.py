#!/usr/bin/env python
"""M1 - environment and GPU proof.

Answers, before any training code exists:

  1. Does PyTorch see the RTX 3050?
  2. Do both candidate models load with the installed `transformers`?
  3. Does one training-style step (forward + backward, AMP) fit in 6 GB at
     the batch sizes in configs/, and what is the peak VRAM?
  4. Is Ollama reachable and is llama3.2 pulled?

Run from the project root with the venv active:

    python scripts/01_check_env.py

First run downloads ~1.5 GB of model weights into .cache/huggingface.
Pass --skip-models to check only torch/CUDA/Ollama.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
# hf_xet (HuggingFace's chunked downloader) stalls partway through the
# "reconstructing file" stage on Windows - the blob never lands on disk and
# the run hangs with a frozen progress bar. Fall back to plain HTTP.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from osint_shield.config import load_config  # noqa: E402

CANDIDATES = ["mmbert_small.yaml", "xlmr_base.yaml"]
MAX_LEN = 512
OK, FAIL, WARN = "OK  ", "FAIL", "WARN"


def line(status: str, msg: str) -> None:
    print(f"  [{status}] {msg}")


# ------------------------------------------------------------------ torch / cuda
def check_torch() -> dict:
    import torch

    out = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
           "cuda_available": torch.cuda.is_available()}
    line(OK, f"torch {torch.__version__}  (CUDA build: {torch.version.cuda})")

    if not out["cuda_available"]:
        line(FAIL, "torch.cuda.is_available() is False")
        if "+cpu" in torch.__version__:
            print("         -> you have the CPU-only wheel. Fix:")
            print("            pip uninstall -y torch")
            print("            pip install torch --index-url https://download.pytorch.org/whl/cu128")
        else:
            print("         -> CUDA wheel installed but driver can't serve it. Update the NVIDIA")
            print("            driver, or try the cu126 wheel instead of cu128.")
        return out

    props = torch.cuda.get_device_properties(0)
    out.update({"gpu": props.name, "vram_gb": round(props.total_memory / 1024**3, 2),
                "bf16": torch.cuda.is_bf16_supported()})
    line(OK, f"GPU: {props.name}  |  VRAM {out['vram_gb']} GB  |  bf16 supported: {out['bf16']}")
    return out


# ------------------------------------------------------------------ transformers
def check_transformers() -> dict:
    import transformers

    line(OK, f"transformers {transformers.__version__}")
    return {"transformers": transformers.__version__}


# ------------------------------------------------------------------ one model
def probe_model(cfg_name: str, device: str) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cfg = load_config(cfg_name)
    name = cfg["model"]["name"]
    batch = cfg["training"]["batch_size"]
    use_amp = bool(cfg["training"].get("amp", False)) and device == "cuda"
    res: dict = {"config": cfg_name, "model": name, "batch": batch}
    print(f"\n  --- {name}  (from {cfg_name}) ---")

    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(
            name, attn_implementation=cfg["model"].get("attn_implementation", "eager")
        )
    except Exception as e:  # noqa: BLE001
        line(FAIL, f"load failed: {type(e).__name__}: {e}")
        res["load"] = f"{type(e).__name__}: {e}"
        if "trust_remote_code" in str(e) or "does not recognize" in str(e):
            print("         -> architecture unknown to this transformers version. Try:")
            print("            pip install -U transformers")
        return res
    res["load_s"] = round(time.time() - t0, 1)

    n_total = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for n, p in model.named_parameters() if "embed" in n.lower())
    res.update({"params_total_M": round(n_total / 1e6, 1),
                "params_body_M": round((n_total - n_emb) / 1e6, 1),
                "hidden": model.config.hidden_size,
                "max_position": getattr(model.config, "max_position_embeddings", None)})
    line(OK, f"loaded in {res['load_s']}s  |  {res['params_total_M']} M params "
             f"({res['params_body_M']} M body)  |  hidden {res['hidden']}  |  "
             f"max positions {res['max_position']}")

    # sanity: a real sentence tokenises and the max_length we plan to use is legal
    enc = tok("DRDO conducts salvo launch of two Pralay missiles.", return_tensors="pt")
    line(OK, f"tokenizer ok  (sep token: {tok.sep_token!r}, {enc['input_ids'].shape[1]} tokens)")
    if res["max_position"] and res["max_position"] < MAX_LEN:
        line(WARN, f"max_position {res['max_position']} < planned max_len {MAX_LEN}")

    model.to(device).train()
    head = torch.nn.Linear(model.config.hidden_size, 5).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # training-style step at the configured batch, then at half of it
    for bs in sorted({batch, max(1, batch // 2)}, reverse=True):
        ids = torch.randint(5, tok.vocab_size, (bs, MAX_LEN), device=device)
        mask = torch.ones_like(ids)
        y = torch.randint(0, 5, (bs,), device=device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                h = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
                loss = torch.nn.functional.cross_entropy(head(h), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
            res[f"step_bs{bs}"] = {"peak_vram_gb": round(peak, 2), "seconds": round(dt, 2)}
            status = OK if peak < 5.5 else WARN
            line(status, f"train step  batch {bs} x {MAX_LEN} tok  amp={use_amp}  "
                         f"->  peak VRAM {peak:.2f} GB   {dt:.2f}s")
        except torch.cuda.OutOfMemoryError:
            res[f"step_bs{bs}"] = "OOM"
            line(FAIL, f"train step  batch {bs}  ->  OUT OF MEMORY")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    del model, head, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ ollama
def check_ollama() -> dict:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    want = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
            tags = json.load(r)
    except Exception as e:  # noqa: BLE001
        line(FAIL, f"Ollama not reachable at {host}: {e}")
        print("         -> start it: open the Ollama app, or run `ollama serve`")
        return {"reachable": False}
    names = [m["name"] for m in tags.get("models", [])]
    line(OK, f"Ollama reachable at {host}  |  models: {', '.join(names) or 'none'}")
    if want in names:
        line(OK, f"{want} is pulled")
    else:
        line(FAIL, f"{want} not found  ->  ollama pull {want}")
    return {"reachable": True, "models": names, "has_target": want in names}


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true", help="only torch/CUDA/Ollama")
    args = ap.parse_args()

    print("=" * 72)
    print("M1  environment check")
    print(f"    {platform.system()} {platform.release()}  |  Python {platform.python_version()}")
    print("=" * 72)

    report: dict = {"python": platform.python_version(), "platform": platform.platform()}

    print("\n[1] PyTorch / CUDA")
    report["torch"] = check_torch()
    device = "cuda" if report["torch"]["cuda_available"] else "cpu"

    print("\n[2] transformers")
    report["transformers"] = check_transformers()

    if not args.skip_models:
        print(f"\n[3] candidate models  (device={device})")
        if device == "cpu":
            line(WARN, "no GPU - loading models but skipping the VRAM measurement")
        report["models"] = [probe_model(c, device) for c in CANDIDATES]

    print("\n[4] Ollama")
    report["ollama"] = check_ollama()

    out = ROOT / "runs" / "env_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  report written to {out.relative_to(ROOT)}")

    ok = report["torch"]["cuda_available"] and report["ollama"].get("has_target", False)
    if not args.skip_models:
        ok = ok and all("load_s" in m for m in report["models"])
    print("\n" + ("M1 PASSED - ready for M2" if ok else "M1 NOT PASSED - fix the [FAIL] lines above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
