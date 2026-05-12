# DemaFormer

An official implementation of **DemaFormer: Damped Exponential Moving Average Transformer with Energy-Based Modeling for Temporal Language Grounding** (Nguyen et al., arXiv:2312.02549, 2023).

Given a natural-language query and a video, Temporal Language Grounding (TLG) asks: *which moments of the video correspond to the query?* DemaFormer addresses this with two ideas:

1. **DEMA attention** — replace the attention layer's value tensor with a learned exponential moving average over the sequence, with both rate and damping coefficients parameterized in (0, 1) per feature dimension. The recurrence injects local-temporal inductive bias that vanilla attention lacks.
2. **Energy-based modeling (EBM)** — train the model with an auxiliary contrastive-divergence loss so the joint moment-query representations of relevant clips cluster apart from irrelevant ones. Negatives are produced by Langevin sampling seeded at the encoded outputs.

This repo implements both, the matching loss from Eq. (30), and the standard QVHighlights / Charades-STA evaluation metrics (R@k, mAP, HIT@1).

## Repository layout

```
demaformer/
├── demaformer/
│   ├── __init__.py
│   ├── dema.py            # DEMA layer + DEMAAttention (§3.1)
│   ├── model.py           # Encoder/decoder, heads, EBM (§3.2-3.3)
│   ├── dataset.py         # QVHighlights & Charades-STA loaders
│   ├── evaluation.py      # window extraction + R@k / mAP / HIT@1
│   └── trainer.py         # training + validation loops
├── scripts/
│   ├── download_data.py   # fetch QVHighlights / Charades-STA
│   ├── train.py           # CLI training entry point
│   └── evaluate.py        # CLI evaluation entry point
├── pyproject.toml         # uv / pip metadata
├── requirements.txt
└── README.md
```

## Installation

### With uv (recommended)

```bash
git clone <this-repo> demaformer && cd demaformer
uv venv
source .venv/bin/activate
uv pip install -e .
```

If you need to re-extract features from raw videos:

```bash
uv pip install -e ".[features]"
```

### With pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Verify the install:

```bash
python -c "from demaformer import DemaFormer; print(DemaFormer.__doc__)"
```

## Data

DemaFormer is evaluated on four benchmarks. The dataset releases distribute **pre-extracted features** rather than raw videos, and our loaders consume those features directly.

### QVHighlights (Lei et al., 2021)

```bash
python scripts/download_data.py --dataset qvhighlights --out data/qvhighlights
```

This downloads:

* `data/qvhighlights/annotations/highlight_{train,val,test}_release.jsonl`
* `data/qvhighlights/features/clip_features/{vid}.npz`         (CLIP, 512-d)
* `data/qvhighlights/features/slowfast_features/{vid}.npz`     (SlowFast, 2304-d)
* `data/qvhighlights/features/clip_text_features/qid{qid}.npz` (CLIP text, 512-d)
* `data/qvhighlights/features/pann_features/{vid}.npz`         (PANN audio, 2050-d)

Use `--annotations-only` to skip the ~30 GB feature bundle, or `--skip-audio` to skip PANN.

### Charades-STA (Gao et al., 2017)

```bash
python scripts/download_data.py --dataset charades --out data/charades_sta
```

This downloads the annotation TXTs. The feature files (`vgg_features/`, `optical_flow/`, `glove_features/`) need to be placed under the same directory — community releases vary; see the printed instructions.

### YouTube Highlights & TVSum

Not bundled in this download script (each is hosted by a different group). The loaders in `demaformer/dataset.py` can be extended in the same style; we welcome PRs.

## Training

### QVHighlights

```bash
python scripts/train.py \
  --dataset qvhighlights \
  --data-root data/qvhighlights \
  --features-root data/qvhighlights/features \
  --out runs/qvhl_base
```

Default hyperparameters follow Appendix B of the paper: `d_model=256`, `n_heads=8`, 2 encoder + 2 decoder layers, Adam(lr=1e-3, wd=1e-4), batch size 32, 200 epochs, `K=100` Langevin steps with `γ=0.1`, `λ_NLL=0.1`, positive-sample threshold ρ=4.

Useful flags:

* `--no-audio` — disable the PANN audio cross-attention branch.
* `--k-steps 50` — fewer Langevin steps (faster training, slight quality drop; see Figure 4).
* `--epochs 100 --batch-size 16` — quicker experimentation.

### Charades-STA

```bash
python scripts/train.py \
  --dataset charades \
  --data-root data/charades_sta \
  --out runs/charades_base
```

Defaults: batch size 8, 100 epochs, ρ=1.0.

## Evaluation

```bash
python scripts/evaluate.py \
  --dataset qvhighlights \
  --data-root data/qvhighlights \
  --features-root data/qvhighlights/features \
  --ckpt runs/qvhl_base/best.pt \
  --split val
```

For the QVHighlights *test* submission file, add `--save-preds preds.jsonl` and upload `preds.jsonl` to the official server.

Charades-STA:

```bash
python scripts/evaluate.py \
  --dataset charades \
  --data-root data/charades_sta \
  --ckpt runs/charades_base/best.pt \
  --split test
```

## What's in the model

* `demaformer.dema.DEMA` — the recurrence `l_i = α·g_i + (1 − α·δ)·l_{i-1}` with sigmoid-parameterized α, δ (so they stay in (0, 1)^d). Layered on top of a Linear-in / Linear-out projection (Eqs. 1–3).
* `demaformer.dema.DEMAAttention` — DEMA output feeds the value tensor of a standard multi-head attention; queries and keys come from the raw input; outputs are combined with the residual through a learned sigmoid gate (Eqs. 4–12).
* `demaformer.model.DemaFormer` — projects video / text (and optionally audio) into a shared 256-d space, runs N encoder layers over the concatenated sequence, takes the first L_v outputs as moment-query representations and runs N decoder layers, then emits per-clip (salience, center, offset, width).
* `demaformer.model.ebm_nll_loss` — contrastive divergence with the convention `E(o) = −salience(o)` (Eq. 27). Negatives are drawn via `langevin_sample`; the negative term has a decay coefficient α (Eq. 28) that anneals from 1 toward `alpha_min` as training proceeds.

## Losses

The total objective is

```
L = L_match + λ_NLL · L_NLL
```

where `L_match` is the salience-plus-localization regression loss from Eq. (30) computed over the positive moments, and `L_NLL` is the EBM negative log-likelihood implemented as contrastive divergence (Eq. 29). An optional auxiliary BCE on salience is available via `lam_bce`.

## Reproducing the headline numbers

The paper reports R1@0.5 / R1@0.7 of 62.39 / 43.94 on QVHighlights and 52.63 / 32.15 on Charades-STA. To get close to these, you need (i) the official pre-extracted features (the loaders L2-normalize them, matching common practice in this family of models), (ii) all four loss terms enabled, (iii) sufficient Langevin steps (K=100 is the convergence threshold in Figure 4), and (iv) the per-dataset positive-sample threshold ρ from Appendix B (4.0 for QVHighlights, 1.0 for Charades-STA, 0.4 for TVSum, 1.0 for YouTube Highlights).

## Citing

```bibtex
@inproceedings{nguyen2023demaformer,
  title={Demaformer: Damped exponential moving average transformer with energy-based modeling for temporal language grounding},
  author={Nguyen, Thong and Wu, Xiaobao and Dong, Xinshuai and Nguyen, Cong-Duy and Ng, See Kiong and Tuan, Luu Anh},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={3635--3649},
  year={2023}
}
```

## Caveats

* This is an **unofficial** reimplementation. Numbers and detailed choices that aren't fully pinned down by the paper (e.g. exact NMS thresholds, exact LayerNorm placements) follow the conventions of Moment-DETR / UMT, which the paper builds on.
* The Langevin sampler costs roughly `K × forward_through_salience_head` per training step. For `K=100`, expect training to be ~1.5× slower than a vanilla Moment-DETR-style trainer.
