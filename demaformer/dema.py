"""Damped Exponential Moving Average (DEMA) attention.

Implements Section 3.1 of the paper. The DEMA computation runs a learnable
exponentially-decaying recurrence over the input sequence with a damping
factor, then feeds the result as the *value* tensor of a self-attention
layer. The recurrence is:

    g_i = Linear(x_i)
    l_i = alpha * g_i + (1 - alpha * delta) * l_{i-1}
    x'_i = Linear(l_i)

where alpha, delta in (0,1)^d are learnable and applied element-wise. We
parameterize them with sigmoid so they always lie in (0,1).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DEMA(nn.Module):
    """Damped Exponential Moving Average over a sequence.

    Args:
        d_model: feature dimension of the input.
        causal: if True, run the recurrence strictly left-to-right (causal).
            Set False for non-autoregressive encoders/decoders used in TLG —
            we still scan left-to-right, the flag is reserved for clarity.
    """

    def __init__(self, d_model: int, causal: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.causal = causal

        # Map x_i -> g_i and l_i -> x'_i. Eq. (1) and (3).
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Pre-sigmoid parameters so alpha, delta stay in (0, 1)^d. Init near 0
        # gives alpha ~ 0.5, delta ~ 0.5 at start.
        self.alpha_logit = nn.Parameter(torch.zeros(d_model))
        self.delta_logit = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            mask: (B, L) with 1 for valid tokens, 0 for padding. Padding
                positions still get a hidden state (the recurrence simply
                passes the previous state through) but their *output* is
                zeroed so downstream attention masking handles them.

        Returns:
            x': (B, L, D)
        """
        B, L, D = x.shape
        assert D == self.d_model, f"expected feature dim {self.d_model}, got {D}"

        g = self.in_proj(x)                              # (B, L, D)
        alpha = torch.sigmoid(self.alpha_logit)          # (D,)
        delta = torch.sigmoid(self.delta_logit)          # (D,)
        decay = 1.0 - alpha * delta                      # (D,)

        # Sequential scan. L is small for TLG (75 moments + ~32 query tokens),
        # so a Python loop is fine and keeps the code obviously correct.
        h = x.new_zeros(B, D)
        outs = []
        for t in range(L):
            h = alpha * g[:, t] + decay * h
            outs.append(h)
        l = torch.stack(outs, dim=1)                     # (B, L, D)

        out = self.out_proj(l)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class DEMAAttention(nn.Module):
    """DEMA + multi-head self-attention with the adaptive gate from Eqs. (4)-(12).

    The DEMA output Z (after a SiLU non-linearity) is used as the *value*
    tensor of the attention layer, while Q and K come straight from the raw
    input X. The attention output is then combined with X through a learned
    sigmoid gate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.dema = DEMA(d_model)

        # Z = act(Linear(X'))    -- Eq. (5)
        self.z_proj = nn.Linear(d_model, d_model)

        # Q, K from raw input X; V from Z. Eqs. (6)-(8).
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Adaptive aggregation. Eqs. (10)-(12).
        self.gate_proj = nn.Linear(d_model, d_model)     # -> lambda
        self.p_proj_a = nn.Linear(d_model, d_model)      # Linear(X') in P
        self.p_proj_b = nn.Linear(d_model, d_model)      # Linear(Z ⊙ Z')

        self.dropout = nn.Dropout(dropout)
        self.act = _get_activation(activation)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            key_padding_mask: (B, L) with True for *padding* positions
                (standard PyTorch convention).
        """
        B, L, D = x.shape
        valid_mask = None
        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).to(x.dtype)  # (B, L), 1 for valid

        # ---- DEMA branch -------------------------------------------------
        x_prime = self.dema(x, mask=valid_mask)           # (B, L, D)
        z = self.act(self.z_proj(x_prime))                # (B, L, D)

        # ---- Attention ---------------------------------------------------
        q = self._split_heads(self.q_proj(x))             # (B, H, L, dh)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(z))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if key_padding_mask is not None:
            # mask: (B, 1, 1, L)
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        z_prime = torch.matmul(attn, v)                   # (B, H, L, dh)
        z_prime = self._merge_heads(z_prime)              # (B, L, D)

        # ---- Adaptive aggregation ---------------------------------------
        lam = torch.sigmoid(self.gate_proj(x_prime))
        p = self.act(self.p_proj_a(x_prime) + self.p_proj_b(z * z_prime))
        h = lam * p + (1.0 - lam) * x

        if valid_mask is not None:
            h = h * valid_mask.unsqueeze(-1)
        return h

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, L, _ = t.shape
        return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, H, L, dh = t.shape
        return t.transpose(1, 2).contiguous().view(B, L, H * dh)


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")
