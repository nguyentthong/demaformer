"""Prediction extraction + evaluation metrics.

The model emits per-clip (salience, center, center_offset, width) where
center & width are normalized to [0, 1] w.r.t. video duration. The
predicted window for clip i is

    c_pred = clip_center_normalized_i + center_offset_i
    w_pred = width_i
    start  = (c_pred - w_pred/2) * duration
    end    = (c_pred + w_pred/2) * duration

We rank clips by salience and take the top-Lm distinct windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .model import DemaFormerOutput


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    qid: int
    vid: str
    duration: float
    pred_windows: list[tuple[float, float, float]]   # (start_s, end_s, score)
    pred_saliency: list[float]                       # per-clip
    gt_windows: list[tuple[float, float]]


def _nms(
    windows: list[tuple[float, float, float]],
    iou_thresh: float = 0.7,
) -> list[tuple[float, float, float]]:
    """Greedy NMS: keep the highest-scoring window, drop anything with IoU
    above the threshold, repeat. `windows` is (start, end, score)."""
    sorted_w = sorted(windows, key=lambda x: -x[2])
    kept: list[tuple[float, float, float]] = []
    for w in sorted_w:
        if all(_iou_1d((w[0], w[1]), (k[0], k[1])) <= iou_thresh for k in kept):
            kept.append(w)
    return kept


def _iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def extract_predictions(
    out: DemaFormerOutput,
    meta: list[dict[str, Any]],
    top_k: int = 10,
    nms_iou: float = 0.7,
) -> list[Prediction]:
    """Convert model outputs into ranked window predictions."""
    salience = out.salience.detach().cpu().numpy()              # (B, Lv)
    center = out.center.detach().cpu().numpy()
    offset = out.center_offset.detach().cpu().numpy()
    width = out.width.detach().cpu().numpy()
    vmask = out.video_mask.detach().cpu().numpy()               # True = pad

    preds: list[Prediction] = []
    B, Lv = salience.shape
    for b in range(B):
        m = meta[b]
        duration = float(m["duration"])
        clip_len = float(m["clip_len"])

        valid = ~vmask[b]
        idx = np.where(valid)[0]
        if len(idx) == 0:
            preds.append(Prediction(m["qid"], m["vid"], duration, [], [], m["gt_windows"]))
            continue

        # Decode windows. Use center + offset (interpreted as a small
        # adjustment) to get the predicted window center in [0, 1].
        cands: list[tuple[float, float, float]] = []
        for i in idx:
            c_pred = float(center[b, i] + offset[b, i])
            w_pred = float(width[b, i])
            c_pred = max(0.0, min(1.0, c_pred))
            w_pred = max(1e-3, min(1.0, w_pred))
            start_s = max(0.0, (c_pred - 0.5 * w_pred) * duration)
            end_s = min(duration, (c_pred + 0.5 * w_pred) * duration)
            if end_s > start_s:
                cands.append((start_s, end_s, float(salience[b, i])))

        kept = _nms(cands, iou_thresh=nms_iou)[:top_k]
        preds.append(
            Prediction(
                qid=m["qid"],
                vid=m["vid"],
                duration=duration,
                pred_windows=kept,
                pred_saliency=salience[b].tolist(),
                gt_windows=m["gt_windows"],
            )
        )
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k_iou(
    predictions: list[Prediction],
    k: int,
    iou_threshold: float,
) -> float:
    """Fraction of queries with at least one top-k prediction whose IoU
    against any gt window is >= threshold."""
    if not predictions:
        return 0.0
    hits = 0
    for p in predictions:
        topk = p.pred_windows[:k]
        ok = False
        for (s, e, _) in topk:
            for (gs, ge) in p.gt_windows:
                if _iou_1d((s, e), (gs, ge)) >= iou_threshold:
                    ok = True
                    break
            if ok:
                break
        hits += int(ok)
    return hits / len(predictions)


def mean_average_precision(
    predictions: list[Prediction],
    iou_threshold: float,
) -> float:
    """Per-query average precision @ IoU >= threshold, averaged across
    queries. We compute AP via the 11-point interpolation that's standard
    in QVHighlights (Lei et al. 2021)."""
    if not predictions:
        return 0.0
    aps = []
    for p in predictions:
        if not p.pred_windows or not p.gt_windows:
            aps.append(0.0)
            continue
        scores = np.array([w[2] for w in p.pred_windows])
        order = np.argsort(-scores)
        matched = [False] * len(p.gt_windows)
        tp = np.zeros(len(p.pred_windows))
        fp = np.zeros(len(p.pred_windows))
        for rank, idx in enumerate(order):
            s, e, _ = p.pred_windows[idx]
            best_iou, best_j = 0.0, -1
            for j, (gs, ge) in enumerate(p.gt_windows):
                iou = _iou_1d((s, e), (gs, ge))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j >= 0 and not matched[best_j]:
                tp[rank] = 1
                matched[best_j] = True
            else:
                fp[rank] = 1
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / max(1, len(p.gt_windows))
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = rec >= t
            ap += (prec[mask].max() if mask.any() else 0.0) / 11
        aps.append(ap)
    return float(np.mean(aps))


def hit_at_1(
    predictions: list[Prediction],
    salience_gt: list[np.ndarray],     # per-prediction (Lv,) raw scores
    salience_thresh: float = 4.0,
) -> float:
    """Highest-salience clip is a 'hit' if its annotated salience >= thresh."""
    if not predictions:
        return 0.0
    hits = 0
    for p, sg in zip(predictions, salience_gt):
        scores = np.array(p.pred_saliency)
        if scores.size == 0:
            continue
        top = int(np.argmax(scores))
        if top < len(sg) and sg[top] >= salience_thresh:
            hits += 1
    return hits / len(predictions)


# ---------------------------------------------------------------------------
# QVHighlights metric bundle
# ---------------------------------------------------------------------------


def compute_qvhighlights_metrics(
    predictions: list[Prediction],
    salience_gt: list[np.ndarray],
    salience_hit_thresh: float = 4.0,
) -> dict[str, float]:
    """Returns the standard QVHighlights metrics reported in Table 1."""
    return {
        "R1@0.5": 100 * recall_at_k_iou(predictions, k=1, iou_threshold=0.5),
        "R1@0.7": 100 * recall_at_k_iou(predictions, k=1, iou_threshold=0.7),
        "mAP@0.5": 100 * mean_average_precision(predictions, 0.5),
        "mAP@0.75": 100 * mean_average_precision(predictions, 0.75),
        "mAP_avg": 100 * float(
            np.mean([mean_average_precision(predictions, t)
                     for t in np.arange(0.5, 1.0, 0.05)])
        ),
        "HIT@1": 100 * hit_at_1(predictions, salience_gt, salience_hit_thresh),
    }


def compute_charades_metrics(predictions: list[Prediction]) -> dict[str, float]:
    """Returns the standard Charades-STA metrics reported in Table 2."""
    return {
        "R1@0.5": 100 * recall_at_k_iou(predictions, k=1, iou_threshold=0.5),
        "R1@0.7": 100 * recall_at_k_iou(predictions, k=1, iou_threshold=0.7),
        "R5@0.5": 100 * recall_at_k_iou(predictions, k=5, iou_threshold=0.5),
        "R5@0.7": 100 * recall_at_k_iou(predictions, k=5, iou_threshold=0.7),
    }
