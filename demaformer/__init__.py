"""DemaFormer: Damped Exponential Moving Average Transformer with Energy-Based Modeling."""

from .model import (
    DemaFormer,
    DemaFormerOutput,
    EBMHead,
    matching_loss,
    salience_bce_loss,
    ebm_nll_loss,
    langevin_sample,
)
from .dataset import (
    QVHighlightsDataset,
    CharadesSTADataset,
    TLGSample,
    build_collate_fn,
)

__version__ = "0.1.0"

__all__ = [
    "DemaFormer",
    "DemaFormerOutput",
    "EBMHead",
    "matching_loss",
    "salience_bce_loss",
    "ebm_nll_loss",
    "langevin_sample",
    "QVHighlightsDataset",
    "CharadesSTADataset",
    "TLGSample",
    "build_collate_fn",
]
