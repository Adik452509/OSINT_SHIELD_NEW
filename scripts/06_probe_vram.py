#!/usr/bin/env python
"""Measure a configuration's real VRAM need before committing to a long run.

Builds the actual model from the config (pooling, frozen embeddings, keyword
fusion, attention backend), runs two full training steps - the second includes
AdamW state, which a single fp16 step can skip and under-measure (D9) - and
reports peak memory and what is left free on the card.

    python scripts/06_probe_vram.py --max-length 1024 --batch 8 4
    python scripts/06_probe_vram.py --config xlmr_base.yaml --batch 4 2

Each batch size runs in a fresh process: a CUDA out-of-memory error can leave
the context unusable for anything after it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osint_shield.runtime import configure_environment  # noqa: E402

configure_environment()

GB = 1024**3
#: Free memory to keep in reserve: the display shares this GPU and moves.
SAFE_MARGIN_GB = 1.5


def probe_once(cfg_name: str, overrides: list[str], max_length: int, batch: int) -> None:
    """One measurement. Prints a single RESULT line for the parent to read."""
    import torch

    from osint_shield.config import apply_overrides, load_config
    from osint_shield.keywords import build_features
    from osint_shield.models import MultiTaskLoss, MultiTaskModel
    from osint_shield.runtime import is_cuda_oom
    from osint_shield.training.trainer import param_groups

    cfg = apply_overrides(load_config(cfg_name), overrides)
    kw = cfg["keywords"]
    n_kw = 0
    if kw.get("enabled") and kw.get("mode") in {"group_counts", "per_keyword"}:
        n_kw = build_features(["x"], mode=kw["mode"])[0].shape[1]

    device = torch.device("cuda")
    dtype = torch.bfloat16 if cfg["training"].get("amp_dtype", "bf16") == "bf16" else torch.float16
    heads = {t: int(s["n_classes"]) for t, s in cfg["heads"].items() if s.get("enabled", True)}

    model = MultiTaskModel.from_config(cfg, n_kw).to(device).train()
    loss_fn = MultiTaskLoss({t: torch.ones(n) for t, n in heads.items()},
                            {t: 1.0 for t in heads}).to(device)
    opt = torch.optim.AdamW(param_groups(model, 0.01), lr=2e-5)

    vocab = model.encoder.config.vocab_size
    ids = torch.randint(5, min(vocab, 30_000), (batch, max_length), device=device)
    mask = torch.ones_like(ids)
    mask[0, max_length // 2:] = 0                       # real batches are padded
    feats = torch.rand(batch, n_kw, device=device) if n_kw else None
    targets = {t: torch.zeros(batch, dtype=torch.long, device=device) for t in heads}

    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(2):                              # step 2 carries AdamW state
            with torch.autocast("cuda", dtype=dtype):
                logits = model(ids, mask, feats)
            loss, _ = loss_fn(logits, targets)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        free = torch.cuda.mem_get_info()[0] / GB
        print(f"RESULT ok {torch.cuda.max_memory_allocated() / GB:.2f} "
              f"{torch.cuda.max_memory_reserved() / GB:.2f} {free:.2f}")
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        print("RESULT oom")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="mmbert_small.yaml")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--_one", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._one is not None:
        probe_once(args.config, args.overrides, args.max_length, args._one)
        return 0

    import torch

    if not torch.cuda.is_available():
        print("no GPU - nothing to probe")
        return 1
    free0, total = (x / GB for x in torch.cuda.mem_get_info())
    print("=" * 78)
    print(f"VRAM probe  {args.config}   max_length {args.max_length}"
          + (f"   {' '.join(args.overrides)}" if args.overrides else ""))
    print(f"  free before loading anything: {free0:.2f} of {total:.2f} GB")
    print("=" * 78)

    fits = []
    for batch in args.batch:
        cmd = [sys.executable, __file__, "--config", args.config,
               "--max-length", str(args.max_length), "--_one", str(batch)]
        for o in args.overrides:
            cmd += ["--set", o]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")), None)
        if line is None:
            print(f"  batch {batch}: probe crashed\n{out.stderr[-1500:]}")
            continue
        parts = line.split()
        if parts[1] == "oom":
            print(f"  batch {batch:>2}:  OUT OF MEMORY")
            continue
        peak, reserved, free = map(float, parts[2:5])
        safe = free >= SAFE_MARGIN_GB
        fits.append((batch, safe))
        print(f"  batch {batch:>2}:  peak {peak:.2f} GB   reserved {reserved:.2f} GB   "
              f"free during step {free:.2f} GB   "
              + ("[ok] safe" if safe else f"[!!] under the {SAFE_MARGIN_GB} GB safety margin"))

    safe = [b for b, ok in fits if ok]
    print()
    if safe:
        b = max(safe)
        accum = max(1, 16 // b)
        print(f"  use batch {b} x grad_accum {accum} (effective batch {b * accum})")
        print(f"      --set training.batch_size={b} --set training.grad_accum={accum}")
    else:
        print("  nothing fits safely - try smaller batches, or add "
              "--set training.gradient_checkpointing=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
