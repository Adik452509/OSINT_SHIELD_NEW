"""Training loop, datasets and fold orchestration."""

from .callbacks import EarlyStopping
from .dataset import ArticleDataset, make_collate, tokenize_texts
from .fold import (
    FoldResult,
    PreparedData,
    prepare_data,
    run_fold,
    run_overfit_check,
)
from .trainer import Trainer, TrainSettings

__all__ = [
    "ArticleDataset",
    "EarlyStopping",
    "FoldResult",
    "PreparedData",
    "Trainer",
    "TrainSettings",
    "make_collate",
    "prepare_data",
    "run_fold",
    "run_overfit_check",
    "tokenize_texts",
]
