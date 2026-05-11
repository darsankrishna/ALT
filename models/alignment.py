# models/alignment.py
"""
Entity-to-Region Alignment — Eq. (4), (5), (6).

Eq. (4): A_{i,j} = Gumbel-Softmax(cosine_sim(patch_i, entity_j) / tau)
         → soft assignment of each patch region to one corpus entity

Eq. (5): Â = one_hot(argmax A) + A - stop_gradient(A)
         → differentiable straight-through one-hot:
           forward = hard assignment, backward = soft gradient

Eq. (6): q  = MLP(mean_over_regions(Â · S))   → query per frame   (B, T, d)
         ê  = MLP(CLS token)                   → K/V per frame     (B, T, d)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax(logits: torch.Tensor, tau: float, training: bool) -> torch.Tensor:
    """
    Gumbel-Softmax over last dim.
    Training  → add Gumbel noise (stochastic, encourages exploration)
    Inference → plain softmax (deterministic argmax equivalent)
    """
    if training:
        # Sample Gumbel noise: -log(-log(U)),  U ~ Uniform(0,1)
        u = torch.zeros_like(logits).uniform_().clamp_(1e-10, 1 - 1e-10)
        gumbel = -torch.log(-torch.log(u))
        logits = logits + gumbel
    return F.softmax(logits / tau, dim=-1)


def straight_through_onehot(A: torch.Tensor) -> torch.Tensor:
    """
    Eq. (5): Â = one_hot(argmax A) + A - stop_gradient(A)
    Forward : hard one-hot  (each region assigned to exactly one entity)
    Backward: gradients flow through soft A
    """
    A_hard = torch.zeros_like(A).scatter_(-1, A.argmax(dim=-1, keepdim=True), 1.0)
    return A_hard + A - A.detach()


class EntityToRegionAlignment(nn.Module):
    """
    Aligns patch tokens E ∈ R^{B×T×N×d} to corpus S ∈ R^{K×d}.

    Returns
    -------
    Q0    : (B, T, d)  — entity-aligned query per frame   [Eq. 6, q]
    E_hat : (B, T, d)  — CLS-based K/V for adapter        [Eq. 6, ê]
    """

    def __init__(self, d: int = 512, tau: float = 1.0, alignment_topk: Optional[int] = None) -> None:
        super().__init__()
        self.tau: float = float(tau)
        self.alignment_topk: Optional[int] = alignment_topk

        # Eq. (6) — query MLP:  aligned entity mean → query vector
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Eq. (6) — KV MLP: CLS token → key/value input for adapter
        self.kv_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(
        self,
        E: torch.Tensor,   # (B, T, N+1, d)  — all tokens; index 0 = CLS
        S: torch.Tensor,   # (K, d)           — corpus entity embeddings
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N1, d = E.shape
        S = S.float().to(device=E.device)                         # keep corpus in fp32 for stable cosine sim

        # ── patch tokens only (skip CLS at index 0) ─────────────────────────
        patches = E[:, :, 1:, :].reshape(B * T, N1 - 1, d)  # (BT, N, d)

        # ── Eq. (4): cosine sim then Gumbel-Softmax ──────────────────────────
        p_norm = F.normalize(patches, dim=-1)                # (BT, N, d)
        s_norm = F.normalize(S,       dim=-1)                # (K,  d)
        sim = torch.einsum("bnd,kd->bnk", p_norm.float(), s_norm.float())  # (BT, N, K)

        if self.alignment_topk is not None:
            topk = min(int(self.alignment_topk), sim.shape[-1])
            if topk > 0 and topk < sim.shape[-1]:
                topk_vals, topk_idx = sim.topk(k=topk, dim=-1)
                masked_sim = torch.full_like(sim, float("-inf"))
                masked_sim.scatter_(-1, topk_idx, topk_vals)
                sim = masked_sim

        A = gumbel_softmax(sim.to(E.dtype), tau=self.tau, training=self.training)  # (BT, N, K)

        # ── Eq. (5): differentiable straight-through one-hot ─────────────────
        A_hat = straight_through_onehot(A)                   # (BT, N, K)

        # ── Eq. (6): weighted entity sum → mean over regions → MLP → Q0 ─────
        aligned = torch.einsum("bnk,kd->bnd", A_hat.float(), S)    # (BT, N, d)
        q_frame  = aligned.mean(dim=1)                       # (BT, d)
        Q0       = self.query_mlp(q_frame.to(E.dtype)).reshape(B, T, d)  # (B, T, d)

        # ── Eq. (6): CLS token → MLP → E_hat (K/V for adapter) ──────────────
        cls    = E[:, :, 0, :].reshape(B * T, d)            # (BT, d)
        E_hat  = self.kv_mlp(cls).reshape(B, T, d)          # (B, T, d)

        return Q0, E_hat
