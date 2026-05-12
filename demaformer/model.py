"""DemaFormer model (Section 3.2) and Energy-Based Modeling loss (Section 3.3).

Architecture overview:
    audio  -> 1-layer cross-attention -> F' (audio-dependent video features)
    text   -> token embeddings T
    Xe = [F'; T]                                          (Eq. 14)
    Oe = DemaFormerEncoder(Xe)                            (Eqs. 15-17)
    Xd = first Lv tokens of Oe                            (moment-query reps)
    Od = DemaFormerDecoder(Xd)                            (Eqs. 18-20)

Prediction heads emit (salience, center, center_offset, width) per moment.

The EBM treats Eθ(o) = -salience(o) (Eq. 27) and trains via contrastive
divergence (Eq. 26). Negatives are produced by K-step Langevin dynamics
seeded at the encoded representations (Eq. 24).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dema import DEMAAttention


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class FeatureProjector(nn.Module):
    """Project raw modality features into the shared d_model space."""

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AudioCrossAttention(nn.Module):
    """Eq. (13): 1-layer attention to inject audio into the video stream.

    F' = F + softmax(A F^T / sqrt(d)) F
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scale = d_model**-0.5
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        f_video: torch.Tensor,           # (B, Lv, D)
        a_audio: torch.Tensor,           # (B, Lv, D) or None
        video_mask: Optional[torch.Tensor] = None,  # (B, Lv), True = pad
    ) -> torch.Tensor:
        if a_audio is None:
            return f_video
        scores = torch.matmul(a_audio, f_video.transpose(-2, -1)) * self.scale
        if video_mask is not None:
            scores = scores.masked_fill(video_mask[:, None, :], float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = f_video + torch.matmul(attn, f_video)
        return self.norm(out)


class DemaFormerLayer(nn.Module):
    """A single encoder/decoder layer.

    DEMA-attention -> Norm -> ReLU non-linear -> Add & Norm   (Eqs. 15-16)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.attn = DEMAAttention(d_model, n_heads, dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.attn(x, key_padding_mask=key_padding_mask)
        h = self.norm1(h)
        out = self.norm2(self.ffn(h) + h)
        return out


# ---------------------------------------------------------------------------
# Prediction head outputs
# ---------------------------------------------------------------------------


@dataclass
class DemaFormerOutput:
    salience: torch.Tensor          # (B, Lv) raw scores
    center: torch.Tensor            # (B, Lv) in (0, 1)
    center_offset: torch.Tensor     # (B, Lv) small offset
    width: torch.Tensor             # (B, Lv) in (0, 1)
    decoder_out: torch.Tensor       # (B, Lv, D)  -- used by EBM
    video_mask: torch.Tensor        # (B, Lv) True = padding


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class DemaFormer(nn.Module):
    """Full DemaFormer architecture."""

    def __init__(
        self,
        video_dim: int = 2816,         # SlowFast(2304) + CLIP(512) for QVHL
        text_dim: int = 512,           # CLIP text
        audio_dim: int = 2048,         # PANN
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        max_v_len: int = 75,
        max_q_len: int = 32,
        dropout: float = 0.1,
        use_audio: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_v_len = max_v_len
        self.max_q_len = max_q_len
        self.use_audio = use_audio

        # Modality projectors
        self.video_proj = FeatureProjector(video_dim, d_model, dropout)
        self.text_proj = FeatureProjector(text_dim, d_model, dropout)
        if use_audio:
            self.audio_proj = FeatureProjector(audio_dim, d_model, dropout)
            self.audio_xattn = AudioCrossAttention(d_model, dropout)

        # Positional encodings (separate for video and text so the model
        # always knows which modality a token belongs to)
        self.video_pos = PositionalEncoding(d_model, max_len=max_v_len)
        self.text_pos = PositionalEncoding(d_model, max_len=max_q_len)
        self.modality_emb = nn.Embedding(2, d_model)     # 0 = video, 1 = text

        # Encoder / decoder
        self.encoder = nn.ModuleList(
            [
                DemaFormerLayer(d_model, n_heads, ffn_dim, dropout)
                for _ in range(n_enc_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DemaFormerLayer(d_model, n_heads, ffn_dim, dropout)
                for _ in range(n_dec_layers)
            ]
        )

        # Prediction heads. Eqs. (21)-(22).
        self.salience_head = nn.Linear(d_model, 1)
        self.center_head = nn.Linear(d_model, 1)
        self.offset_head = nn.Linear(d_model, 1)
        self.width_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        video_feat: torch.Tensor,           # (B, Lv, video_dim)
        video_mask: torch.Tensor,           # (B, Lv) True = pad
        text_feat: torch.Tensor,            # (B, Lq, text_dim)
        text_mask: torch.Tensor,            # (B, Lq) True = pad
        audio_feat: Optional[torch.Tensor] = None,   # (B, Lv, audio_dim)
    ) -> DemaFormerOutput:
        B, Lv, _ = video_feat.shape
        Lq = text_feat.size(1)

        # ---- Project & add positional / modality info -------------------
        v = self.video_proj(video_feat)
        v = self.video_pos(v) + self.modality_emb.weight[0]
        if self.use_audio and audio_feat is not None:
            a = self.audio_proj(audio_feat)
            v = self.audio_xattn(v, a, video_mask=video_mask)

        t = self.text_proj(text_feat)
        t = self.text_pos(t) + self.modality_emb.weight[1]

        # ---- Concatenate and encode -------------------------------------
        x_enc = torch.cat([v, t], dim=1)                   # (B, Lv+Lq, D)
        enc_pad_mask = torch.cat([video_mask, text_mask], dim=1)
        for layer in self.encoder:
            x_enc = layer(x_enc, key_padding_mask=enc_pad_mask)

        # ---- Decoder takes first Lv encoder outputs ---------------------
        x_dec = x_enc[:, :Lv]
        for layer in self.decoder:
            x_dec = layer(x_dec, key_padding_mask=video_mask)

        # ---- Heads ------------------------------------------------------
        salience = self.salience_head(x_dec).squeeze(-1)            # (B, Lv)
        # constrain center & width to (0, 1) via sigmoid
        center = torch.sigmoid(self.center_head(x_dec).squeeze(-1))
        width = torch.sigmoid(self.width_head(x_dec).squeeze(-1))
        # offset is a small adjustment, no activation
        offset = self.offset_head(x_dec).squeeze(-1)

        return DemaFormerOutput(
            salience=salience,
            center=center,
            center_offset=offset,
            width=width,
            decoder_out=x_dec,
            video_mask=video_mask,
        )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def matching_loss(
    out: DemaFormerOutput,
    gt_salience: torch.Tensor,           # (B, Lv) in [0, scale]
    gt_center: torch.Tensor,             # (B, Lv) in (0, 1)
    gt_width: torch.Tensor,              # (B, Lv) in (0, 1)
    gt_offset: torch.Tensor,             # (B, Lv) small reals
    pos_mask: torch.Tensor,              # (B, Lv) True = positive (gt) moment
    lam_center: float = 1 / 3,
    lam_width: float = 0.01,
    lam_offset: float = 1 / 3,
) -> torch.Tensor:
    """Eq. (30). Computed only over the positive (groundtruth) moments.

    L_match = - (1 / Lm) sum_i [ s_i - lam1*|c_i - c_i^| - lam2*|w_i - w_i^|
                                       - lam3*|co_i - (co_i^ - c_i^)| ]
    The paper writes the last term as |co - (co_pred - c_pred)| ; we follow
    that literally.
    """
    pos = pos_mask.float()
    n = pos.sum().clamp(min=1.0)

    s_term = (out.salience * pos).sum() / n
    c_term = ((out.center - gt_center).abs() * pos).sum() / n
    w_term = ((out.width - gt_width).abs() * pos).sum() / n
    o_term = ((gt_offset - (out.center_offset - out.center)).abs() * pos).sum() / n

    loss = -(s_term - lam_center * c_term - lam_width * w_term - lam_offset * o_term)
    return loss


def salience_bce_loss(
    out: DemaFormerOutput,
    gt_salience: torch.Tensor,           # (B, Lv) in [0, 1] (normalized)
) -> torch.Tensor:
    """Auxiliary BCE on salience so the score is well-calibrated across all
    moments, not only positives. The paper relies on the matching-loss
    salience term plus EBM; we keep BCE optional via the trainer config.
    """
    valid = (~out.video_mask).float()
    n = valid.sum().clamp(min=1.0)
    bce = F.binary_cross_entropy_with_logits(
        out.salience, gt_salience, reduction="none"
    )
    return (bce * valid).sum() / n


# ---------------------------------------------------------------------------
# Energy-Based Modeling
# ---------------------------------------------------------------------------


class EBMHead(nn.Module):
    """Wraps the salience head as an energy function: E(o) = -salience(o).

    Sampling is performed in the decoder-output space (`decoder_out`) via
    K-step Langevin dynamics (Eq. 24). Because we want gradients w.r.t. the
    sample (not the parameters), we detach the salience-head parameters
    during the sampling loop.
    """

    def __init__(self, salience_head: nn.Linear) -> None:
        super().__init__()
        self.salience_head = salience_head

    def energy(self, o: torch.Tensor) -> torch.Tensor:
        # (B, Lv, D) -> (B, Lv)
        return -self.salience_head(o).squeeze(-1)


def langevin_sample(
    ebm: EBMHead,
    o_init: torch.Tensor,              # (B, Lv, D)
    k_steps: int = 100,
    gamma: float = 0.1,
    clamp: Optional[float] = None,
) -> torch.Tensor:
    """K-step Langevin MCMC starting from `o_init`.

    o_{k+1} = o_k - (gamma/2) * grad_o E(o_k) + sqrt(gamma) * eps
    """
    o = o_init.detach().clone().requires_grad_(True)
    noise_std = gamma**0.5
    for _ in range(k_steps):
        e = ebm.energy(o).sum()
        grad = torch.autograd.grad(e, o, create_graph=False)[0]
        with torch.no_grad():
            o = o - 0.5 * gamma * grad + noise_std * torch.randn_like(o)
            if clamp is not None:
                o = o.clamp_(-clamp, clamp)
        o.requires_grad_(True)
    return o.detach()


def ebm_nll_loss(
    ebm: EBMHead,
    decoder_out: torch.Tensor,       # (B, Lv, D)
    salience_gt: torch.Tensor,       # (B, Lv) raw annotated scores
    video_mask: torch.Tensor,        # (B, Lv) True = pad
    pos_threshold: float,
    k_steps: int = 100,
    gamma: float = 0.1,
    epoch: int = 0,
    alpha_min: float = 0.1,
) -> torch.Tensor:
    """Contrastive-divergence NLL (Eq. 29).

    Positive samples: decoder outputs whose annotated salience exceeds
    `pos_threshold` (and which are not padding).
    Negative samples: K-step Langevin draws seeded at the decoder outputs.

    alpha = max( 1 / (1 + 0.5 * epoch), alpha_min )    -- Eq. (28)
    """
    valid = ~video_mask                                  # (B, Lv)
    pos = valid & (salience_gt >= pos_threshold)         # (B, Lv)

    # Decay coefficient on the negative term.
    alpha = max(1.0 / (1.0 + 0.5 * float(epoch)), alpha_min)

    # Positive energy. Only over positive positions.
    e_pos = ebm.energy(decoder_out)                      # (B, Lv)
    n_pos = pos.float().sum().clamp(min=1.0)
    pos_term = (e_pos * pos.float()).sum() / n_pos

    # Negative samples via Langevin starting from the same decoder outputs.
    o_neg = langevin_sample(ebm, decoder_out, k_steps=k_steps, gamma=gamma)
    e_neg = ebm.energy(o_neg)                            # (B, Lv)
    n_valid = valid.float().sum().clamp(min=1.0)
    neg_term = (e_neg * valid.float()).sum() / n_valid

    return pos_term - alpha * neg_term
