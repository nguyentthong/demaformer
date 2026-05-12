"""Evaluate a trained DemaFormer checkpoint.

Usage:
    python scripts/evaluate.py --dataset qvhighlights \
        --data-root data/qvhighlights \
        --features-root data/qvhighlights/features \
        --ckpt runs/qvhl_base/best.pt \
        --split val
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demaformer import DemaFormer, QVHighlightsDataset, CharadesSTADataset, build_collate_fn
from demaformer.evaluation import (
    extract_predictions,
    compute_qvhighlights_metrics,
    compute_charades_metrics,
)


def build_qvhighlights(data_root: str, features_root: str, split: str, use_audio: bool):
    split_files = {
        "train": "highlight_train_release.jsonl",
        "val": "highlight_val_release.jsonl",
        "test": "highlight_test_release.jsonl",
    }
    ds = QVHighlightsDataset(
        annotations_path=os.path.join(data_root, "annotations", split_files[split]),
        features_root=features_root,
        use_audio=use_audio,
    )
    return ds


def build_charades(data_root: str, split: str):
    fname = f"charades_sta_{split}.txt"
    return CharadesSTADataset(
        annotations_path=os.path.join(data_root, fname),
        features_root=data_root,
    )


@torch.no_grad()
def run_eval(model, dataset, dataset_name: str, batch_size: int, device: torch.device,
             top_k: int, nms_iou: float, hit_thresh: float, save_preds: str | None) -> dict:
    has_audio = getattr(dataset, "use_audio", False)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=build_collate_fn(has_audio=has_audio),
    )
    model.eval()
    all_preds = []
    all_sal_gt = []
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        out = model(
            video_feat=batch["video_feat"].float(),
            video_mask=batch["video_mask"],
            text_feat=batch["text_feat"].float(),
            text_mask=batch["text_mask"],
            audio_feat=(
                batch["audio_feat"].float() if batch.get("audio_feat") is not None else None
            ),
        )
        preds = extract_predictions(out, batch["meta"], top_k=top_k, nms_iou=nms_iou)
        all_preds.extend(preds)
        for sg in batch["salience_gt"].cpu().numpy():
            all_sal_gt.append(sg)

    if dataset_name == "qvhighlights":
        metrics = compute_qvhighlights_metrics(all_preds, all_sal_gt, hit_thresh)
    else:
        metrics = compute_charades_metrics(all_preds)

    if save_preds:
        out_lines = []
        for p in all_preds:
            out_lines.append({
                "qid": p.qid, "vid": p.vid, "duration": p.duration,
                "pred_relevant_windows": [[s, e, sc] for (s, e, sc) in p.pred_windows],
                "pred_saliency_scores": p.pred_saliency,
            })
        with open(save_preds, "w") as f:
            for line in out_lines:
                f.write(json.dumps(line) + "\n")
        print(f"[wrote] {save_preds} ({len(out_lines)} predictions)")

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["qvhighlights", "charades"], required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--features-root", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--hit-thresh", type=float, default=4.0)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-preds", default=None,
                    help="Optional JSONL path for predictions (for test submission)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == "qvhighlights":
        features_root = args.features_root or os.path.join(args.data_root, "features")
        ds = build_qvhighlights(args.data_root, features_root, args.split,
                                use_audio=not args.no_audio)
        model = DemaFormer(
            video_dim=ds.video_dim, text_dim=ds.text_dim, audio_dim=ds.audio_dim,
            d_model=256, n_heads=8, ffn_dim=1024,
            n_enc_layers=2, n_dec_layers=2,
            max_v_len=ds.max_v_len, max_q_len=ds.max_q_len,
            use_audio=not args.no_audio,
        )
    else:
        ds = build_charades(args.data_root, args.split)
        model = DemaFormer(
            video_dim=ds.video_dim, text_dim=ds.text_dim, audio_dim=0,
            d_model=256, n_heads=8, ffn_dim=1024,
            n_enc_layers=2, n_dec_layers=2,
            max_v_len=ds.max_v_len, max_q_len=ds.max_q_len,
            use_audio=False,
        )

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    metrics = run_eval(
        model, ds, args.dataset, args.batch_size, device,
        args.top_k, args.nms_iou, args.hit_thresh, args.save_preds,
    )
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")


if __name__ == "__main__":
    main()
