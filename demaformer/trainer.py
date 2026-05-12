"""Training and validation loops for DemaFormer."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import QVHighlightsDataset, CharadesSTADataset, build_collate_fn
from .evaluation import (
    extract_predictions,
    compute_qvhighlights_metrics,
    compute_charades_metrics,
)
from .model import (
    DemaFormer,
    EBMHead,
    ebm_nll_loss,
    matching_loss,
    salience_bce_loss,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Optimisation
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    epochs: int = 200
    batch_size: int = 32
    num_workers: int = 4

    # Loss weights (Appendix B)
    lam_center: float = 1 / 3
    lam_width: float = 0.01
    lam_offset: float = 1 / 3
    lam_nll: float = 0.1
    lam_bce: float = 0.0           # optional auxiliary; off by default

    # EBM
    k_steps: int = 100
    gamma: float = 0.1
    pos_threshold: float = 4.0     # 4.0 for QVHL, 1.0 for Charades, 0.4 for TVSum
    alpha_min: float = 0.1

    # Eval
    top_k: int = 10
    nms_iou: float = 0.7
    salience_hit_thresh: float = 4.0
    eval_every: int = 1

    # IO
    out_dir: str = "runs/exp"
    device: str = "cuda"
    seed: int = 0


@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = -1.0
    best_epoch: int = -1
    history: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(
        self,
        model: DemaFormer,
        cfg: TrainConfig,
        train_ds,
        val_ds,
        dataset_name: str = "qvhighlights",
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        has_audio = getattr(train_ds, "use_audio", False)
        collate = build_collate_fn(has_audio=has_audio)
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True,
        )
        self.val_ds = val_ds

        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.ebm = EBMHead(self.model.salience_head)

        self.state = TrainState()
        os.makedirs(cfg.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def _forward(self, batch: dict[str, Any]):
        return self.model(
            video_feat=batch["video_feat"].float(),
            video_mask=batch["video_mask"],
            text_feat=batch["text_feat"].float(),
            text_mask=batch["text_mask"],
            audio_feat=(
                batch["audio_feat"].float() if batch.get("audio_feat") is not None else None
            ),
        )

    # ------------------------------------------------------------------
    # Train / Validate
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        tot = {"loss": 0.0, "match": 0.0, "nll": 0.0, "bce": 0.0, "n": 0}
        t0 = time.time()
        for batch in self.train_loader:
            batch = self._move(batch)
            out = self._forward(batch)

            l_match = matching_loss(
                out,
                gt_salience=batch["salience_gt"].float(),
                gt_center=batch["center_gt"].float(),
                gt_width=batch["width_gt"].float(),
                gt_offset=batch["offset_gt"].float(),
                pos_mask=batch["pos_mask"],
                lam_center=self.cfg.lam_center,
                lam_width=self.cfg.lam_width,
                lam_offset=self.cfg.lam_offset,
            )
            l_nll = ebm_nll_loss(
                self.ebm,
                decoder_out=out.decoder_out,
                salience_gt=batch["salience_gt"].float(),
                video_mask=batch["video_mask"],
                pos_threshold=self.cfg.pos_threshold,
                k_steps=self.cfg.k_steps,
                gamma=self.cfg.gamma,
                epoch=self.state.epoch,
                alpha_min=self.cfg.alpha_min,
            )
            l_bce = (
                salience_bce_loss(out, batch["salience_gt_norm"].float())
                if self.cfg.lam_bce > 0
                else torch.zeros((), device=self.device)
            )
            loss = l_match + self.cfg.lam_nll * l_nll + self.cfg.lam_bce * l_bce

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self.opt.step()

            bsz = batch["video_feat"].size(0)
            tot["loss"] += loss.item() * bsz
            tot["match"] += l_match.item() * bsz
            tot["nll"] += l_nll.item() * bsz
            tot["bce"] += l_bce.item() * bsz
            tot["n"] += bsz

        n = max(1, tot["n"])
        return {
            "train/loss": tot["loss"] / n,
            "train/match": tot["match"] / n,
            "train/nll": tot["nll"] / n,
            "train/bce": tot["bce"] / n,
            "train/time_s": time.time() - t0,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        all_preds = []
        all_sal_gt = []
        for batch in self.val_loader:
            batch = self._move(batch)
            out = self._forward(batch)
            preds = extract_predictions(
                out, batch["meta"],
                top_k=self.cfg.top_k, nms_iou=self.cfg.nms_iou,
            )
            all_preds.extend(preds)
            for sg in batch["salience_gt"].cpu().numpy():
                all_sal_gt.append(sg)

        if self.dataset_name == "qvhighlights":
            return compute_qvhighlights_metrics(
                all_preds, all_sal_gt, self.cfg.salience_hit_thresh
            )
        return compute_charades_metrics(all_preds)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self) -> TrainState:
        cfg = self.cfg
        primary = "R1@0.7" if self.dataset_name == "qvhighlights" else "R1@0.5"
        for epoch in range(cfg.epochs):
            self.state.epoch = epoch
            tr = self.train_one_epoch()
            log = dict(tr)
            log["epoch"] = epoch
            if (epoch + 1) % cfg.eval_every == 0:
                va = self.validate()
                for k, v in va.items():
                    log[f"val/{k}"] = v
                key_metric = va[primary]
                if key_metric > self.state.best_metric:
                    self.state.best_metric = key_metric
                    self.state.best_epoch = epoch
                    torch.save(
                        {"model": self.model.state_dict(),
                         "epoch": epoch, "cfg": cfg.__dict__},
                        os.path.join(cfg.out_dir, "best.pt"),
                    )
            self.state.history.append(log)
            self._log_line(log)

        torch.save(
            {"model": self.model.state_dict(), "cfg": cfg.__dict__},
            os.path.join(cfg.out_dir, "last.pt"),
        )
        return self.state

    @staticmethod
    def _log_line(log: dict[str, float]) -> None:
        parts = []
        for k, v in log.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" | ".join(parts), flush=True)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def build_qvhighlights_trainer(
    data_root: str,
    features_root: str,
    cfg: TrainConfig | None = None,
    use_audio: bool = True,
) -> Trainer:
    cfg = cfg or TrainConfig()
    cfg.pos_threshold = 4.0
    cfg.salience_hit_thresh = 4.0

    train_ds = QVHighlightsDataset(
        annotations_path=os.path.join(data_root, "annotations",
                                       "highlight_train_release.jsonl"),
        features_root=features_root,
        use_audio=use_audio,
    )
    val_ds = QVHighlightsDataset(
        annotations_path=os.path.join(data_root, "annotations",
                                       "highlight_val_release.jsonl"),
        features_root=features_root,
        use_audio=use_audio,
    )

    model = DemaFormer(
        video_dim=train_ds.video_dim,
        text_dim=train_ds.text_dim,
        audio_dim=train_ds.audio_dim,
        d_model=256, n_heads=8, ffn_dim=1024,
        n_enc_layers=2, n_dec_layers=2,
        max_v_len=train_ds.max_v_len, max_q_len=train_ds.max_q_len,
        use_audio=use_audio,
    )
    return Trainer(model, cfg, train_ds, val_ds, dataset_name="qvhighlights")


def build_charades_trainer(
    data_root: str,
    cfg: TrainConfig | None = None,
) -> Trainer:
    cfg = cfg or TrainConfig(batch_size=8, epochs=100)
    cfg.pos_threshold = 1.0

    train_ds = CharadesSTADataset(
        annotations_path=os.path.join(data_root, "charades_sta_train.txt"),
        features_root=data_root,
    )
    val_ds = CharadesSTADataset(
        annotations_path=os.path.join(data_root, "charades_sta_test.txt"),
        features_root=data_root,
    )

    model = DemaFormer(
        video_dim=train_ds.video_dim,
        text_dim=train_ds.text_dim,
        audio_dim=0,
        d_model=256, n_heads=8, ffn_dim=1024,
        n_enc_layers=2, n_dec_layers=2,
        max_v_len=train_ds.max_v_len, max_q_len=train_ds.max_q_len,
        use_audio=False,
    )
    return Trainer(model, cfg, train_ds, val_ds, dataset_name="charades_sta")
