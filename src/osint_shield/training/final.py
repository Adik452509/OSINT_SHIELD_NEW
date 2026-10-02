"""The final model: train on every development article, save it, load it back.

Uses the fixed schedule (D13) because nothing is held out to early-stop on -
the held-out ``test.csv`` is for scoring once, never for training decisions.

A saved model is a directory:

    model.pt     state dict, including the keyword-standardisation buffers
    meta.json    config, label maps, keyword feature names, provenance

so M9's inference service can rebuild the exact architecture and inputs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..config import narrative_label_maps, severity_label_maps
from ..models import MultiTaskLoss, MultiTaskModel
from ..runtime import set_seed
from .fold import PreparedData, _build, fold_class_weights, task_loss_weights
from .trainer import Trainer, TrainSettings

MODEL_FILE = "model.pt"
META_FILE = "meta.json"


def train_final(data: PreparedData, train_idx, cfg: dict, *, seed: int, device,
                model_builder=None, settings: TrainSettings | None = None,
                log=print) -> tuple[Trainer, list[dict]]:
    """Train on exactly ``train_idx`` with the fixed schedule.

    Class weights and keyword standardisation come from these rows only.

    Raises:
        ValueError: if the settings ask for early stopping - there is no
            validation set to stop on, by design.
    """
    settings = settings or TrainSettings.from_config(cfg)
    if settings.early_stopping:
        raise ValueError("the final model uses a fixed schedule (D13) - "
                         "nothing is held out to early-stop on")
    set_seed(seed)
    weights = fold_class_weights(data.frame, train_idx, cfg)
    model = _build(cfg, data, train_idx, model_builder)
    trainer = Trainer(model, MultiTaskLoss(weights, task_loss_weights(cfg)), settings,
                      device, data.pad_id, log=log)
    history = trainer.fit(data.dataset(train_idx), None, early_stopping=False)
    return trainer, history


def save_model(model: MultiTaskModel, cfg: dict, out_dir: Path, *, seed: int,
               n_train: int, keyword_feature_names: list[str]) -> Path:
    """Write ``model.pt`` and ``meta.json`` to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / MODEL_FILE)
    _, narrative = narrative_label_maps()
    _, severity = severity_label_maps()
    meta = {
        "model_name": cfg["model"]["name"],
        "seed": seed,
        "n_train": n_train,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labels": {"narrative": narrative, "severity": severity,
                   "propaganda": {0: "No", 1: "Yes"}},
        "n_keyword_features": model.n_keyword_features,
        "keyword_features": keyword_feature_names,
        "max_length": cfg["tokenizer"]["max_length"],
        "config": cfg,
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out


def load_model(model_dir: Path, *, device="cpu",
               model_builder=None) -> tuple[MultiTaskModel, dict]:
    """Rebuild a saved model in eval mode. Returns ``(model, meta)``.

    ``model_builder(cfg, n_keyword_features)`` overrides construction - tests
    use it to avoid downloading the encoder.
    """
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
    cfg, n_kw = meta["config"], int(meta["n_keyword_features"])
    model = (model_builder(cfg, n_kw) if model_builder
             else MultiTaskModel.from_config(cfg, n_kw))
    state = torch.load(model_dir / MODEL_FILE, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), meta
