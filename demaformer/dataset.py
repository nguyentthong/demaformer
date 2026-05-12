"""Dataset loaders.

Both QVHighlights and Charades-STA distribute *pre-extracted features*: the
images are NOT released, only frame-level feature tensors plus annotation
JSONLs. We follow that convention. See `scripts/download_data.py` for the
download URLs.

Expected layout (after `download_data.py`):

    data/
        qvhighlights/
            features/
                clip_features/{vid}.npz       # 'features': (Lv, 512), 'mask': (Lv,)
                slowfast_features/{vid}.npz   # (Lv, 2304)
                clip_text_features/{qid}.npz  # (Lq, 512)
                pann_features/{vid}.npz       # (Lv, 2050)  -- optional
            annotations/
                highlight_train_release.jsonl
                highlight_val_release.jsonl
                highlight_test_release.jsonl

        charades_sta/
            vgg_features/{vid}.npy            # (Lv, 4096)
            optical_flow/{vid}.npy            # (Lv, 1024)
            glove_features/{qid}.npy          # (Lq, 300)
            charades_sta_train.txt
            charades_sta_test.txt

JSONL fields used (QVHighlights):
    qid               int
    query             str
    vid               str (video id)
    duration          float, seconds
    relevant_clip_ids list[int]         (every 2-second clip in the gt span)
    relevant_windows  list[[s, e]]      (in seconds)
    saliency_scores   list[list[int]]   (three annotators, per relevant clip)

Charades-STA TXT format: each line is "<vid> <start> <end>##<query>".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_npz_features(path: str, key: str = "features") -> np.ndarray:
    """Load features from .npz/.npy. Returns float32 array of shape (L, D)."""
    if path.endswith(".npz"):
        with np.load(path) as f:
            arr = f[key] if key in f.files else f[f.files[0]]
    else:
        arr = np.load(path)
    return arr.astype(np.float32)


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def _pad_seq(arr: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad/truncate `arr` (L, D) to (max_len, D). Returns (padded, mask).

    mask: True at padding positions (PyTorch's `key_padding_mask` convention).
    """
    L, D = arr.shape
    if L >= max_len:
        return arr[:max_len], np.zeros(max_len, dtype=bool)
    padded = np.zeros((max_len, D), dtype=arr.dtype)
    padded[:L] = arr
    mask = np.ones(max_len, dtype=bool)
    mask[:L] = False
    return padded, mask


# ---------------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------------


@dataclass
class TLGSample:
    qid: int
    vid: str
    duration: float
    clip_len: float                      # seconds per moment
    video_feat: np.ndarray               # (Lv, video_dim)
    video_mask: np.ndarray               # (Lv,) True = pad
    text_feat: np.ndarray                # (Lq, text_dim)
    text_mask: np.ndarray                # (Lq,) True = pad
    audio_feat: np.ndarray | None        # (Lv, audio_dim) or None

    # Per-moment targets (length Lv, padded slots are zero / False)
    salience_gt: np.ndarray              # (Lv,) raw scores
    salience_gt_norm: np.ndarray         # (Lv,) in [0, 1]
    pos_mask: np.ndarray                 # (Lv,) True = moment overlaps gt window
    center_gt: np.ndarray                # (Lv,) center of containing gt window, 0..1
    width_gt: np.ndarray                 # (Lv,) width of containing gt window, 0..1
    offset_gt: np.ndarray                # (Lv,) center - moment_center

    # Raw gt windows for evaluation, in seconds
    gt_windows: list[tuple[float, float]]


# ---------------------------------------------------------------------------
# QVHighlights
# ---------------------------------------------------------------------------


class QVHighlightsDataset(Dataset):
    """QVHighlights TLG dataset.

    Each video is split into 2-second clips. The paper uses 75 clips max
    (150 s) as the standard cap.
    """

    SALIENCE_MAX = 4.0                   # annotator scale is 0..4
    SALIENCE_POS_THRESHOLD = 4.0         # used by the EBM (Appendix B)
    CLIP_LEN = 2.0

    def __init__(
        self,
        annotations_path: str,
        features_root: str,
        max_v_len: int = 75,
        max_q_len: int = 32,
        use_audio: bool = True,
        use_slowfast: bool = True,
        use_clip_video: bool = True,
    ) -> None:
        self.max_v_len = max_v_len
        self.max_q_len = max_q_len
        self.use_audio = use_audio
        self.use_slowfast = use_slowfast
        self.use_clip_video = use_clip_video
        self.features_root = features_root

        self.clip_video_dir = os.path.join(features_root, "clip_features")
        self.slowfast_dir = os.path.join(features_root, "slowfast_features")
        self.text_dir = os.path.join(features_root, "clip_text_features")
        self.audio_dir = os.path.join(features_root, "pann_features")

        self.samples = self._load_annotations(annotations_path)

    @staticmethod
    def _load_annotations(path: str) -> list[dict[str, Any]]:
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @property
    def video_dim(self) -> int:
        d = 0
        if self.use_slowfast:
            d += 2304
        if self.use_clip_video:
            d += 512
        return d

    @property
    def text_dim(self) -> int:
        return 512

    @property
    def audio_dim(self) -> int:
        return 2050        # PANN cnn14 pooled features

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TLGSample:
        r = self.samples[idx]
        qid, vid = r["qid"], r["vid"]
        duration = float(r["duration"])

        # ---- Video features ----
        feats = []
        if self.use_slowfast:
            feats.append(_load_npz_features(
                os.path.join(self.slowfast_dir, f"{vid}.npz")))
        if self.use_clip_video:
            feats.append(_load_npz_features(
                os.path.join(self.clip_video_dir, f"{vid}.npz")))
        video_feat = np.concatenate(feats, axis=-1) if feats else np.zeros((0, 0), np.float32)
        video_feat = _l2_normalize(video_feat)

        video_feat, video_mask = _pad_seq(video_feat, self.max_v_len)

        # ---- Text features ----
        text_feat = _load_npz_features(
            os.path.join(self.text_dir, f"qid{qid}.npz"))
        text_feat = _l2_normalize(text_feat)
        text_feat, text_mask = _pad_seq(text_feat, self.max_q_len)

        # ---- Audio (optional) ----
        audio_feat = None
        if self.use_audio:
            apath = os.path.join(self.audio_dir, f"{vid}.npz")
            if os.path.exists(apath):
                a = _load_npz_features(apath)
                a = _l2_normalize(a)
                audio_feat, _ = _pad_seq(a, self.max_v_len)

        # ---- Targets ----
        Lv = self.max_v_len
        salience_gt = np.zeros(Lv, dtype=np.float32)
        pos_mask = np.zeros(Lv, dtype=bool)
        center_gt = np.zeros(Lv, dtype=np.float32)
        width_gt = np.zeros(Lv, dtype=np.float32)
        offset_gt = np.zeros(Lv, dtype=np.float32)

        windows = r.get("relevant_windows", [])
        # Saliency is annotated per *relevant clip*, averaged across 3 raters.
        if "saliency_scores" in r and "relevant_clip_ids" in r:
            rel_ids = r["relevant_clip_ids"]
            scores = r["saliency_scores"]
            for clip_id, s_triplet in zip(rel_ids, scores):
                if 0 <= clip_id < Lv:
                    salience_gt[clip_id] = float(np.mean(s_triplet))

        # For each ground-truth window, mark every clip whose center lies
        # inside it as positive and record (center, width, offset).
        for (ws, we) in windows:
            ws, we = float(ws), float(we)
            c_norm = 0.5 * (ws + we) / max(duration, 1e-6)
            w_norm = (we - ws) / max(duration, 1e-6)
            i_start = max(0, int(np.floor(ws / self.CLIP_LEN)))
            i_end = min(Lv - 1, int(np.ceil(we / self.CLIP_LEN)) - 1)
            for i in range(i_start, i_end + 1):
                clip_center_t = (i + 0.5) * self.CLIP_LEN
                if ws <= clip_center_t <= we:
                    pos_mask[i] = True
                    center_gt[i] = c_norm
                    width_gt[i] = w_norm
                    offset_gt[i] = c_norm - clip_center_t / max(duration, 1e-6)

        salience_gt_norm = salience_gt / self.SALIENCE_MAX

        return TLGSample(
            qid=int(qid),
            vid=str(vid),
            duration=duration,
            clip_len=self.CLIP_LEN,
            video_feat=video_feat,
            video_mask=video_mask,
            text_feat=text_feat,
            text_mask=text_mask,
            audio_feat=audio_feat,
            salience_gt=salience_gt,
            salience_gt_norm=salience_gt_norm,
            pos_mask=pos_mask,
            center_gt=center_gt,
            width_gt=width_gt,
            offset_gt=offset_gt,
            gt_windows=[(float(s), float(e)) for s, e in windows],
        )


# ---------------------------------------------------------------------------
# Charades-STA
# ---------------------------------------------------------------------------


class CharadesSTADataset(Dataset):
    """Charades-STA TLG dataset.

    Charades-STA has only one gt window per query and no per-clip salience
    annotations, so we synthesise salience as 1.0 inside the gt window and
    0.0 elsewhere (this is the standard convention used by Moment-DETR /
    UMT). The EBM threshold is therefore set to 1.0 (Appendix B).
    """

    SALIENCE_POS_THRESHOLD = 1.0
    CLIP_LEN = 1.0                       # 1-second clips by default

    def __init__(
        self,
        annotations_path: str,
        features_root: str,
        max_v_len: int = 64,
        max_q_len: int = 32,
        clip_len: float = 1.0,
    ) -> None:
        self.max_v_len = max_v_len
        self.max_q_len = max_q_len
        self.CLIP_LEN = clip_len
        self.features_root = features_root

        self.vgg_dir = os.path.join(features_root, "vgg_features")
        self.flow_dir = os.path.join(features_root, "optical_flow")
        self.text_dir = os.path.join(features_root, "glove_features")

        self.samples = self._load_annotations(annotations_path)

    @staticmethod
    def _load_annotations(path: str) -> list[dict[str, Any]]:
        records = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                head, query = line.split("##", 1)
                parts = head.split()
                vid, s, e = parts[0], float(parts[1]), float(parts[2])
                records.append(
                    {"qid": i, "vid": vid, "query": query, "start": s, "end": e}
                )
        return records

    @property
    def video_dim(self) -> int:
        return 4096 + 1024

    @property
    def text_dim(self) -> int:
        return 300

    @property
    def audio_dim(self) -> int:
        return 0       # Charades-STA: no audio

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TLGSample:
        r = self.samples[idx]
        vid = r["vid"]
        s, e = r["start"], r["end"]

        vgg = _load_npz_features(os.path.join(self.vgg_dir, f"{vid}.npy"))
        flow = _load_npz_features(os.path.join(self.flow_dir, f"{vid}.npy"))
        # Align lengths (in case of off-by-one)
        L = min(len(vgg), len(flow))
        video_feat = np.concatenate([vgg[:L], flow[:L]], axis=-1)
        video_feat = _l2_normalize(video_feat)
        duration = float(L) * self.CLIP_LEN
        video_feat, video_mask = _pad_seq(video_feat, self.max_v_len)

        text_feat = _load_npz_features(
            os.path.join(self.text_dir, f"qid{r['qid']}.npy"))
        text_feat, text_mask = _pad_seq(text_feat, self.max_q_len)

        # Targets
        Lv = self.max_v_len
        salience_gt = np.zeros(Lv, dtype=np.float32)
        pos_mask = np.zeros(Lv, dtype=bool)
        center_gt = np.zeros(Lv, dtype=np.float32)
        width_gt = np.zeros(Lv, dtype=np.float32)
        offset_gt = np.zeros(Lv, dtype=np.float32)

        c_norm = 0.5 * (s + e) / max(duration, 1e-6)
        w_norm = (e - s) / max(duration, 1e-6)
        i_start = max(0, int(np.floor(s / self.CLIP_LEN)))
        i_end = min(Lv - 1, int(np.ceil(e / self.CLIP_LEN)) - 1)
        for i in range(i_start, i_end + 1):
            clip_center_t = (i + 0.5) * self.CLIP_LEN
            if s <= clip_center_t <= e:
                pos_mask[i] = True
                salience_gt[i] = 1.0
                center_gt[i] = c_norm
                width_gt[i] = w_norm
                offset_gt[i] = c_norm - clip_center_t / max(duration, 1e-6)

        return TLGSample(
            qid=int(r["qid"]),
            vid=str(vid),
            duration=duration,
            clip_len=self.CLIP_LEN,
            video_feat=video_feat,
            video_mask=video_mask,
            text_feat=text_feat,
            text_mask=text_mask,
            audio_feat=None,
            salience_gt=salience_gt,
            salience_gt_norm=salience_gt,
            pos_mask=pos_mask,
            center_gt=center_gt,
            width_gt=width_gt,
            offset_gt=offset_gt,
            gt_windows=[(s, e)],
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def build_collate_fn(has_audio: bool = True):
    """Returns a collate_fn that stacks a list of TLGSample into a dict of
    tensors plus a metadata list (for evaluation)."""

    def collate(batch: list[TLGSample]) -> dict[str, Any]:
        def stack(field: str) -> torch.Tensor:
            return torch.from_numpy(np.stack([getattr(b, field) for b in batch]))

        out: dict[str, Any] = {
            "video_feat": stack("video_feat"),
            "video_mask": stack("video_mask"),
            "text_feat": stack("text_feat"),
            "text_mask": stack("text_mask"),
            "salience_gt": stack("salience_gt"),
            "salience_gt_norm": stack("salience_gt_norm"),
            "pos_mask": stack("pos_mask"),
            "center_gt": stack("center_gt"),
            "width_gt": stack("width_gt"),
            "offset_gt": stack("offset_gt"),
            "meta": [
                {
                    "qid": b.qid,
                    "vid": b.vid,
                    "duration": b.duration,
                    "clip_len": b.clip_len,
                    "gt_windows": b.gt_windows,
                }
                for b in batch
            ],
        }
        if has_audio and batch[0].audio_feat is not None:
            out["audio_feat"] = torch.from_numpy(
                np.stack([b.audio_feat for b in batch])
            )
        else:
            out["audio_feat"] = None
        return out

    return collate
