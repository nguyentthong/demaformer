"""Train DemaFormer on QVHighlights or Charades-STA.

Usage:
    python scripts/train.py --dataset qvhighlights \
        --data-root data/qvhighlights \
        --features-root data/qvhighlights/features \
        --out runs/qvhl_base

    python scripts/train.py --dataset charades \
        --data-root data/charades_sta \
        --out runs/charades_base
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as a script without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demaformer.trainer import (
    TrainConfig,
    build_qvhighlights_trainer,
    build_charades_trainer,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["qvhighlights", "charades"], required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--features-root", default=None,
                    help="QVHighlights only; defaults to {data-root}/features")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--k-steps", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--lam-nll", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    cfg = TrainConfig(
        lr=args.lr,
        k_steps=args.k_steps,
        gamma=args.gamma,
        lam_nll=args.lam_nll,
        out_dir=args.out,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    if args.dataset == "qvhighlights":
        features_root = args.features_root or os.path.join(args.data_root, "features")
        trainer = build_qvhighlights_trainer(
            data_root=args.data_root,
            features_root=features_root,
            cfg=cfg,
            use_audio=not args.no_audio,
        )
    else:
        trainer = build_charades_trainer(data_root=args.data_root, cfg=cfg)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args) | cfg.__dict__, f, indent=2)

    state = trainer.fit()
    print(f"\nBest {trainer.dataset_name} metric: {state.best_metric:.4f} "
          f"at epoch {state.best_epoch}")


if __name__ == "__main__":
    main()
