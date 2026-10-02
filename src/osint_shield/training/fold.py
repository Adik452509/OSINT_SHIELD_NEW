"""One fold, end to end - and the overfit check that proves the loop works.

Each fold run enforces the protocol rather than trusting the caller:

* the inner validation split is **group-aware**, so near-duplicates never
  straddle inner-train and inner-val;
* train / val / test indices and their duplicate groups are asserted disjoint;
* class weights and keyword standardisation statistics come from the
  **inner-train rows only**;
* the outer test fold is predicted exactly once, after training finishes.
"""

from __future__ import annotations

import gc
import time
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from ..config import narrative_label_maps, severity_label_maps
from ..data.loaders import build_text
from ..data.splits import stratification_labels
from ..evaluation.metrics import score_multitask
from ..keywords import KeywordMatcher, build_features
from ..models import MultiTaskLoss, MultiTaskModel, class_weights
from ..runtime import set_seed
from .dataset import ArticleDataset, tokenize_texts
from .trainer import Trainer, TrainSettings

#: Corpus column holding each task's integer label.
TASK_COLUMNS = {"narrative": "y_narrative", "severity": "y_severity",
                "propaganda": "y_propaganda"}


# --------------------------------------------------------------- preparation
@dataclass
class PreparedData:
    """A corpus frame tokenised once, ready to be sliced into folds.

    Attributes:
        frame: one row per article - ids, labels, ``fold``, ``dupe_group``,
            ``has_body``, ``narrative``.
        input_ids: token ids per row, truncated to ``max_length``.
        truncated: whether each row hit ``max_length``.
        keyword_feats: ``(n, k)`` matrix computed over the **untruncated**
            text, or ``None`` when fusion is off.
        pad_id: tokenizer padding id.
        max_length: the length actually used after clamping to the model.
    """

    frame: pd.DataFrame
    input_ids: list[list[int]]
    truncated: np.ndarray
    keyword_feats: np.ndarray | None
    pad_id: int
    max_length: int

    @property
    def n_keyword_features(self) -> int:
        return 0 if self.keyword_feats is None else int(self.keyword_feats.shape[1])

    def dataset(self, idx: np.ndarray) -> ArticleDataset:
        """Slice out an :class:`ArticleDataset` for the given row positions."""
        idx = np.asarray(idx, dtype=int)
        return ArticleDataset(
            [self.input_ids[i] for i in idx],
            {task: self.frame[col].to_numpy()[idx] for task, col in TASK_COLUMNS.items()},
            None if self.keyword_feats is None else self.keyword_feats[idx],
        )


def prepare_data(df: pd.DataFrame, tokenizer, cfg: dict) -> PreparedData:
    """Build model inputs and keyword features for every row of ``df``.

    Keyword features are computed on the full headline + body string *before*
    tokenisation truncates it. That is the property the fusion arm tests: at
    512 tokens only 17.3% of bodied articles are seen in full, but the keyword
    vector still reflects the whole document.
    """
    frame = df.reset_index(drop=True)
    marker = cfg["data"].get("no_body_marker", "[NO_BODY]")
    full_text = build_text(frame, sep=" ", no_body_marker=marker)
    model_text = build_text(frame, sep=tokenizer.sep_token or " ", no_body_marker=marker)

    kw_cfg = cfg.get("keywords", {})
    kw_mode = kw_cfg.get("mode", "off") if kw_cfg.get("enabled", False) else "off"
    exclude = kw_cfg.get("exclude_low_precision", False)

    keyword_feats = None
    if kw_mode in {"group_counts", "per_keyword"}:
        keyword_feats, _ = build_features(full_text, mode=kw_mode,
                                          transform=kw_cfg.get("transform", "log1p"),
                                          exclude_low_precision=exclude)
    elif kw_mode == "markers":
        # markers only reach terms that survive truncation - a separate ablation arm
        matcher = KeywordMatcher(exclude_low_precision=exclude)
        model_text = model_text.map(matcher.mark)
    elif kw_mode != "off":
        raise ValueError(f"unknown keywords.mode: {kw_mode!r}")

    max_length = int(cfg["tokenizer"]["max_length"])
    model_limit = getattr(tokenizer, "model_max_length", None)
    if model_limit and model_limit < 100_000:      # unset limits are ~1e30
        max_length = min(max_length, int(model_limit))

    input_ids, truncated = tokenize_texts(tokenizer, model_text, max_length)
    return PreparedData(frame, input_ids, truncated, keyword_feats,
                        tokenizer.pad_token_id, max_length)


# ------------------------------------------------------------------- helpers
def inner_split(strat_labels: np.ndarray, groups: np.ndarray, val_size: float,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware stratified train/validation split of a training fold.

    Returns positions relative to the arrays passed in.
    """
    n_splits = max(2, int(round(1.0 / val_size)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        # the propaganda stratum has fewer members than n_splits by design
        warnings.simplefilter("ignore", UserWarning)
        train_pos, val_pos = next(splitter.split(np.zeros(len(strat_labels)),
                                                 strat_labels, groups))
    return train_pos, val_pos


def assert_disjoint(frame: pd.DataFrame, *index_sets: np.ndarray) -> None:
    """Fail loudly if any two index sets share a row or a duplicate group."""
    groups = frame["dupe_group"].to_numpy()
    for i in range(len(index_sets)):
        for j in range(i + 1, len(index_sets)):
            a, b = index_sets[i], index_sets[j]
            if set(a) & set(b):
                raise AssertionError(f"index sets {i} and {j} share rows")
            if set(groups[a]) & set(groups[b]):
                raise AssertionError(f"index sets {i} and {j} share a duplicate group")


def fold_class_weights(frame: pd.DataFrame, train_idx: np.ndarray,
                       cfg: dict) -> dict[str, torch.Tensor]:
    """Class weights per enabled head, from the given training rows only."""
    out = {}
    for task, spec in cfg["heads"].items():
        if not spec.get("enabled", True):
            continue
        n = int(spec["n_classes"])
        if spec.get("class_weight") == "balanced":
            out[task] = class_weights(frame[TASK_COLUMNS[task]].to_numpy()[train_idx], n,
                                      cap=spec.get("class_weight_cap"))
        else:
            out[task] = torch.ones(n)
    return out


def task_loss_weights(cfg: dict) -> dict[str, float]:
    return {task: float(spec.get("loss_weight", 1.0))
            for task, spec in cfg["heads"].items() if spec.get("enabled", True)}


def default_model_builder(cfg: dict, n_keyword_features: int) -> MultiTaskModel:
    return MultiTaskModel.from_config(cfg, n_keyword_features)


def _build(cfg, data: PreparedData, train_idx, model_builder) -> MultiTaskModel:
    model = (model_builder or default_model_builder)(cfg, data.n_keyword_features)
    if data.n_keyword_features:
        feats = data.keyword_feats[train_idx]
        model.set_keyword_stats(feats.mean(axis=0), feats.std(axis=0))
    return model


def _free_memory() -> None:
    """Return cached GPU blocks to the driver.

    Callers must ``del`` their own references to the model and trainer first -
    a helper cannot drop names that live in its caller's frame, and
    ``empty_cache`` frees nothing that is still referenced.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predictions_frame(frame: pd.DataFrame, idx: np.ndarray, probs: dict,
                      *, fold: int, seed: int) -> pd.DataFrame:
    """Out-of-fold predictions with confidences, using label names."""
    _, narr = narrative_label_maps()
    _, sev = severity_label_maps()
    rows = frame.iloc[idx]
    out = pd.DataFrame({"id": rows["id"].to_numpy(), "fold": fold, "seed": seed,
                        "has_body": rows["has_body"].to_numpy()})
    if "narrative" in probs:
        p = probs["narrative"]
        out["true_narrative"] = [narr[i] for i in rows["y_narrative"]]
        out["pred_narrative"] = [narr[i] for i in p.argmax(1)]
        out["conf_narrative"] = p.max(1)
    if "severity" in probs:
        p = probs["severity"]
        out["true_severity"] = [sev[i] for i in rows["y_severity"]]
        out["pred_severity"] = [sev[i] for i in p.argmax(1)]
        out["p_high"] = p[:, 1]
    if "propaganda" in probs:
        p = probs["propaganda"]
        out["true_propaganda"] = rows["y_propaganda"].to_numpy()
        out["pred_propaganda"] = p.argmax(1)
        out["p_propaganda"] = p[:, 1]
    return out


# ---------------------------------------------------------------------- fold
@dataclass
class FoldResult:
    """Everything one fold run produces."""

    fold: int
    seed: int
    metrics: dict
    history: list[dict]
    best_epoch: int
    class_weights: dict[str, list[float]]
    n_train: int
    n_val: int
    n_test: int
    peak_vram_gb: float
    seconds: float
    train_ids: list[str] = field(repr=False)
    val_ids: list[str] = field(repr=False)
    test_ids: list[str] = field(repr=False)
    predictions: pd.DataFrame = field(repr=False, default=None)

    def summary(self) -> dict:
        d = asdict(self)
        for key in ("predictions", "train_ids", "val_ids", "test_ids"):
            d.pop(key, None)
        return d


def run_fold(data: PreparedData, test_fold: int, cfg: dict, *, seed: int, device,
             model_builder=None, settings: TrainSettings | None = None,
             log=print) -> FoldResult:
    """Train on every fold but ``test_fold``, then predict ``test_fold`` once."""
    t0 = time.time()
    set_seed(seed)
    frame = data.frame
    folds = frame["fold"].to_numpy()
    test_idx = np.where(folds == test_fold)[0]
    pool_idx = np.where(folds != test_fold)[0]
    if len(test_idx) == 0:
        raise ValueError(f"fold {test_fold} has no rows")

    settings = settings or TrainSettings.from_config(cfg)
    if settings.early_stopping:
        strat = stratification_labels(frame.iloc[pool_idx],
                                      stratify_col=cfg["split"]["stratify_on"])
        tr_pos, va_pos = inner_split(strat, frame["dupe_group"].to_numpy()[pool_idx],
                                     cfg["training"].get("inner_val_size", 0.12), seed)
        train_idx, val_idx = pool_idx[tr_pos], pool_idx[va_pos]
    else:
        # Fixed schedule: nothing is held back to pick an epoch, so every
        # non-test row trains and the final epoch is kept.
        train_idx, val_idx = pool_idx, np.array([], dtype=int)
    assert_disjoint(frame, train_idx, val_idx, test_idx)

    weights = fold_class_weights(frame, train_idx, cfg)
    model = _build(cfg, data, train_idx, model_builder)
    loss_fn = MultiTaskLoss(weights, task_loss_weights(cfg))
    trainer = Trainer(model, loss_fn, settings, device, data.pad_id, log=log)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    test_ds = data.dataset(test_idx)
    # Optional per-epoch scoring of the test fold - recorded, never used for
    # selection. Measures how well early stopping picks (docs/DECISIONS.md D11).
    track = test_ds if cfg["training"].get("track_test_fold", False) else None
    val_ds = data.dataset(val_idx) if len(val_idx) else None
    history = trainer.fit(data.dataset(train_idx), val_ds, track_ds=track,
                          early_stopping=settings.early_stopping)

    probs = trainer.predict_proba(test_ds)
    preds = {task: p.argmax(axis=-1) for task, p in probs.items()}
    targets = {task: frame[col].to_numpy()[test_idx] for task, col in TASK_COLUMNS.items()}
    metrics = score_multitask(targets, preds, has_body=frame["has_body"].to_numpy()[test_idx])

    peak = (torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0
    ids = frame["id"].to_numpy()
    result = FoldResult(
        fold=int(test_fold), seed=int(seed), metrics=metrics, history=history,
        best_epoch=int(trainer.best_epoch or len(history)),
        class_weights={t: [round(float(x), 4) for x in w] for t, w in weights.items()},
        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
        peak_vram_gb=round(peak, 2), seconds=round(time.time() - t0, 1),
        train_ids=list(ids[train_idx]), val_ids=list(ids[val_idx]),
        test_ids=list(ids[test_idx]),
        predictions=predictions_frame(frame, test_idx, probs, fold=test_fold, seed=seed),
    )
    del trainer, model, loss_fn
    _free_memory()
    return result


# ------------------------------------------------------------ overfit check
def select_overfit_rows(frame: pd.DataFrame, n_rows: int, seed: int) -> np.ndarray:
    """Pick ``n_rows`` covering every narrative class as evenly as possible."""
    rng = np.random.default_rng(seed)
    labels = frame["y_narrative"].to_numpy()
    classes = np.unique(labels)
    per_class = max(1, n_rows // len(classes))

    chosen: list[int] = []
    for c in classes:
        members = np.where(labels == c)[0]
        chosen.extend(rng.choice(members, min(per_class, len(members)), replace=False))
    remaining = np.setdiff1d(np.arange(len(frame)), chosen)
    shortfall = n_rows - len(chosen)
    if shortfall > 0:
        chosen.extend(rng.choice(remaining, min(shortfall, len(remaining)), replace=False))
    return np.sort(np.asarray(chosen[:n_rows], dtype=int))


def run_overfit_check(data: PreparedData, n_rows: int, cfg: dict, *, seed: int, device,
                      epochs: int = 30, lr: float = 1e-4, model_builder=None,
                      log=print) -> dict:
    """Train on a tiny subset and score on the same rows.

    A healthy gradient path memorises 40 articles. If it cannot, the bug is in
    the loop - labels, loss, optimiser or masking - and no amount of
    cross-validation would reveal that as clearly.
    """
    set_seed(seed)
    idx = select_overfit_rows(data.frame, n_rows, seed)
    settings = TrainSettings.from_config(cfg, epochs=epochs, lr=lr, grad_accum=1,
                                         warmup_ratio=0.0, min_epochs=0)
    weights = fold_class_weights(data.frame, idx, cfg)
    model = _build(cfg, data, idx, model_builder)
    trainer = Trainer(model, MultiTaskLoss(weights, task_loss_weights(cfg)), settings,
                      device, data.pad_id, log=log)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    history = trainer.fit(data.dataset(idx), val_ds=None, early_stopping=False)
    metrics = trainer.evaluate(data.dataset(idx))
    peak = (torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0

    result = {
        "n_rows": len(idx),
        "narrative_classes": int(len(np.unique(data.frame["y_narrative"].to_numpy()[idx]))),
        "metrics": metrics,
        "history": history,
        "first_loss": history[0]["train_loss"],
        "last_loss": history[-1]["train_loss"],
        "peak_vram_gb": round(peak, 2),
        "seconds": round(time.time() - t0, 1),
    }
    del trainer, model
    _free_memory()
    return result
