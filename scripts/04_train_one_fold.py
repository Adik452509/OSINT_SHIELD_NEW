#!/usr/bin/env python
"""M4 - the multi-task model, end to end on the GPU.

Two modes, run in this order:

  1. Overfit check - train on 40 articles and score on the same 40. A working
     gradient path memorises them (narrative macro-F1 >= 0.95). If it cannot,
     the bug is in the loop, and we find out in a minute rather than two hours
     into a bake-off.

         python scripts/04_train_one_fold.py --overfit 40

  2. One real fold - train on four folds, early-stop on a group-aware inner
     split, predict the held-out fold once.

         python scripts/04_train_one_fold.py --fold 0

Options: --config xlmr_base.yaml, --mode all, --seed, --epochs, --max-length.
Results land in runs/m4/<tag>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Must run before anything imports transformers - HF_HOME is read at import time.
from osint_shield.runtime import configure_environment, get_device  # noqa: E402

configure_environment()

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from osint_shield.config import load_config  # noqa: E402
from osint_shield.data.loaders import load_gold  # noqa: E402
from osint_shield.evaluation.metrics import LEAKAGE_ALARM_ACCURACY  # noqa: E402
from osint_shield.models import load_tokenizer  # noqa: E402
from osint_shield.paths import DATA_PROCESSED  # noqa: E402
from osint_shield.training import (  # noqa: E402
    TrainSettings,
    prepare_data,
    run_fold,
    run_overfit_check,
)

#: tfidf_logreg on the clean folds, from M3 - the bar a real fold is read against.
CLEAN_BAR = {
    "body_only": {"narrative_macro_f1": 0.466, "narrative_collapsed_macro_f1": 0.604,
                  "severity_f1_high": 0.687},
    "all": {"narrative_macro_f1": 0.449, "narrative_collapsed_macro_f1": 0.607,
            "severity_f1_high": 0.603},
}
OVERFIT_TARGET = 0.95
VRAM_LIMIT_GB = 6.0
#: Measured need is ~2.7 GB plus ~0.5 GB of CUDA context; below this, warn.
VRAM_FREE_WARN_GB = 3.5
OK, BAD = "  [ok]  ", "  [!!]  "

OOM_HELP = """
  The GPU ran out of memory. On this laptop the RTX 3050 also drives the
  display, so every open window shares its 6 GB - browsers and VS Code alone
  can take 1.5 GB or more.

  1. Close Brave/Edge/Chrome, Copilot and the Xbox app, then rerun.
  2. Still failing: halve the batch and double accumulation (same effective
     batch of 16) in configs/mmbert_small.yaml:
         training: {batch_size: 4, grad_accum: 4}
  3. Last resort: --max-length 384
"""


def is_oom(exc: BaseException) -> bool:
    """CUDA OOM surfaces as OutOfMemoryError from the allocator, or as a generic
    AcceleratorError when a library such as cuBLAS fails to get workspace."""
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def preflight_vram() -> None:
    """Report free VRAM before anything is loaded, and warn if it is tight."""
    free, total = (x / 1024**3 for x in torch.cuda.mem_get_info())
    used = total - free
    line = f"free VRAM {free:.2f} of {total:.2f} GB ({used:.2f} GB already used by other apps)"
    if free < VRAM_FREE_WARN_GB:
        print(f"{BAD}{line}")
        print("        less than 3.5 GB free - close browsers and other GPU apps first")
    else:
        print(f"{OK}{line}")


def check(label: str, passed: bool) -> bool:
    print(f"{OK if passed else BAD}{label}")
    return passed


def load_frame(mode: str) -> pd.DataFrame:
    """Join the persisted M2 fold assignments with article text."""
    path = DATA_PROCESSED / f"folds_{mode}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run: python scripts/02_build_folds.py --mode both")
    folds = pd.read_csv(path)
    text = load_gold()[["id", "clean_headline", "clean_text"]]
    df = folds.merge(text, on="id", how="left", validate="one_to_one")
    if df["clean_headline"].isna().any():
        raise SystemExit("some fold ids are missing from the corpus - rebuild the folds")
    return df


def save(out_dir: Path, payload: dict, history: list[dict],
         predictions: pd.DataFrame | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str),
                                          encoding="utf-8")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    if predictions is not None:
        predictions.to_csv(out_dir / "predictions.csv", index=False)
    print(f"\n  written to {out_dir.relative_to(ROOT)}")


def run_overfit(args, cfg, data, device) -> bool:
    print(f"\n[3] overfit check - {args.overfit} articles, {args.overfit_epochs} epochs, "
          f"lr {args.overfit_lr}, scored on the SAME rows")
    res = run_overfit_check(data, args.overfit, cfg, seed=args.seed, device=device,
                            epochs=args.overfit_epochs, lr=args.overfit_lr)
    m = res["metrics"]
    print(f"\n  narrative macro-F1 {m['narrative_macro_f1']:.3f}   "
          f"severity F1(High) {m['severity_f1_high']:.3f}   "
          f"propaganda predicted-positive {m.get('propaganda_n_pred_pos', 0)} "
          f"of {m.get('propaganda_n_pos', 0)}")

    print("\n[4] verification")
    ok = check(f"narrative macro-F1 {m['narrative_macro_f1']:.3f} >= {OVERFIT_TARGET}",
               m["narrative_macro_f1"] >= OVERFIT_TARGET)
    ok &= check(f"severity F1(High) {m['severity_f1_high']:.3f} >= {OVERFIT_TARGET}",
                m["severity_f1_high"] >= OVERFIT_TARGET)
    ok &= check(f"loss fell {res['first_loss']:.3f} -> {res['last_loss']:.3f}",
                res["last_loss"] < 0.5 * res["first_loss"])
    if device.type == "cuda":
        ok &= check(f"peak VRAM {res['peak_vram_gb']:.2f} GB < {VRAM_LIMIT_GB}",
                    res["peak_vram_gb"] < VRAM_LIMIT_GB)
    print(f"  [..]  took {res['seconds']:.0f}s")

    tag = f"overfit_{cfg['model']['name'].split('/')[-1]}_{args.overfit}"
    save(ROOT / "runs" / "m4" / tag,
         {k: v for k, v in res.items() if k != "history"}, res["history"])
    print("\n" + ("OVERFIT CHECK PASSED - gradient path works. Next: --fold 0"
                  if ok else "OVERFIT CHECK FAILED - do not run a real fold until this passes"))
    return ok


def run_one_fold(args, cfg, data, device) -> bool:
    print(f"\n[3] fold {args.fold}  seed {args.seed}")
    res = run_fold(data, args.fold, cfg, seed=args.seed, device=device,
                   settings=TrainSettings.from_config(cfg, epochs=args.epochs))
    m = res.metrics

    print(f"\n[4] test fold {args.fold}  (n={res.n_test}, never seen in training)")
    bar = CLEAN_BAR.get(args.mode, {})
    for key, label in [("narrative_macro_f1", "narrative macro-F1, 5-class"),
                       ("narrative_collapsed_macro_f1", "narrative macro-F1, 3-class"),
                       ("severity_f1_high", "severity F1(High)")]:
        ref = bar.get(key)
        cmp = f"   tfidf bar {ref:.3f}  ({m[key] - ref:+.3f})" if ref is not None else ""
        print(f"    {label:<30} {m[key]:.3f}{cmp}")
    print(f"    {'propaganda F1(pos)':<30} {m['propaganda_f1_pos']:.3f}   "
          f"[diagnostic - {m['propaganda_n_pos']} positive(s) in this fold, "
          f"{m['propaganda_n_pred_pos']} predicted]")
    for regime, s in m.get("severity_by_has_body", {}).items():
        print(f"    severity on {regime:<17} n={s['n']:<4} F1(High) {s['f1_high']:.3f}")

    print("\n[5] verification")
    ok = check(f"train {res.n_train} / inner-val {res.n_val} / test {res.n_test} - "
               "disjoint rows and duplicate groups (asserted)", True)
    cw = res.class_weights
    cap = cfg["heads"]["propaganda"].get("class_weight_cap")
    if "propaganda" in cw and cap is not None:
        ok &= check(f"propaganda class weights {cw['propaganda']} - positive capped at {cap}",
                    cw["propaganda"][1] <= cap + 1e-6)
    print(f"  [..]  narrative class weights {cw.get('narrative')} (training fold only)")
    ok &= check(f"best epoch {res.best_epoch} of {len(res.history)} run",
                1 <= res.best_epoch <= len(res.history))
    if device.type == "cuda":
        ok &= check(f"peak VRAM {res.peak_vram_gb:.2f} GB < {VRAM_LIMIT_GB}",
                    res.peak_vram_gb < VRAM_LIMIT_GB)
    alarm = m["narrative_accuracy"] > LEAKAGE_ALARM_ACCURACY
    ok &= check(f"narrative accuracy {m['narrative_accuracy']:.3f} below the "
                f"{LEAKAGE_ALARM_ACCURACY} leakage alarm", not alarm)
    print(f"  [..]  fold took {res.seconds:.0f}s")
    print("\n  One fold, one seed: expect +/- 0.05 noise. Read it as 'is it learning',")
    print("  not 'has it beaten the bar' - that verdict is M5's 15-run mean.")

    tag = (f"{cfg['model']['name'].split('/')[-1]}_{args.mode}"
           f"_fold{args.fold}_seed{args.seed}")
    save(ROOT / "runs" / "m4" / tag, res.summary(), res.history, res.predictions)
    print("\n" + ("M4 PASSED - ready for M5 (full cross-validation)"
                  if ok else "M4 FAILED - see the [!!] lines above"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="mmbert_small.yaml")
    ap.add_argument("--mode", choices=["body_only", "all"], default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--overfit", type=int, default=0, metavar="N",
                    help="run the N-article overfit check instead of a fold")
    ap.add_argument("--overfit-epochs", type=int, default=30)
    ap.add_argument("--overfit-lr", type=float, default=1e-4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    args.mode = args.mode or cfg["data"]["mode"]
    args.seed = cfg["seed"] if args.seed is None else args.seed
    if args.max_length:
        cfg["tokenizer"]["max_length"] = args.max_length

    device = get_device()
    s = TrainSettings.from_config(cfg)
    print("=" * 78)
    print(f"M4  {cfg['model']['name']}   mode={args.mode}")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "none"
    print(f"    device {device} ({gpu})   amp={s.amp} {s.amp_dtype}   "
          f"batch {s.batch_size} x accum {s.grad_accum}   lr {s.lr}")
    print(f"    keywords: {cfg['keywords']['mode'] if cfg['keywords']['enabled'] else 'off'}")
    print("=" * 78)
    if device.type != "cuda":
        print("  WARNING: no GPU - this will be very slow. Check M1.")
    else:
        preflight_vram()
    frozen = "frozen" if cfg["model"].get("freeze_embeddings") else "trainable"
    print(f"  [..]  attention {cfg['model'].get('attn_implementation', 'eager')}   "
          f"word embeddings {frozen}")

    print("\n[1] tokenizer")
    tok = load_tokenizer(cfg["model"]["name"])
    print(f"  [ok]  {type(tok).__name__}   sep {tok.sep_token!r}   pad id {tok.pad_token_id}")

    print("\n[2] data")
    df = load_frame(args.mode)
    data = prepare_data(df, tok, cfg)
    print(f"  [ok]  {len(df)} articles from data/processed/folds_{args.mode}.csv")
    print(f"  [ok]  max_length {data.max_length}: {data.truncated.mean() * 100:.1f}% "
          "of inputs truncated")
    print(f"  [ok]  keyword features: {data.n_keyword_features} per article, "
          "computed over the untruncated text")

    try:
        ok = (run_overfit(args, cfg, data, device) if args.overfit
              else run_one_fold(args, cfg, data, device))
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if not is_oom(exc):
            raise
        print(OOM_HELP)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
