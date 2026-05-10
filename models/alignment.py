"""
Entity-to-Region Alignment.

Eq. (4): A_{i,j} = Gumbel-Softmax(cosine_sim(e_i, s_j)) over K entities
         → soft assignment of each region to one entity

Eq. (5): Â = one_hot(A_argmax) + A - detach(A)
         → differentiable one-hot: each region dominated by best entity,
           gradient flows through soft assignment

Eq. (6): q = MLP(Â·S)    (weighted-sum entity embedding → query vector per frame)
         ê = MLP(E)        (visual key/value projection)

GPU optimization:
  - S (corpus) stays on GPU as fp16, no gradient
  - Gumbel samples drawn at training only (use argmax at inference)
  - MLP is tiny (d → d) — negligible memory
"""
# models/alignment.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityToRegionAlignment(nn.Module):
    """
    Aligns visual region tokens to entity corpus entries via soft cross-attention.
    Outputs:
      Q0    : (B, T, d) — query initialized from mean visual token via MLP
      E_hat : (B, T, d) — entity-aligned visual summary per frame
    """
    def __init__(self, d=768):
        super().__init__()
        self.scale = d ** -0.5
        self.query_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, E, S):
        """
        E : (B, T, N, d)  — visual tokens per frame
        S : (K, d)        — corpus entity embeddings (fp32, from .pt file)
        """
        B, T, N, d = E.shape

        # Normalize for cosine similarity
        E_norm = F.normalize(E, dim=-1)                          # (B, T, N, d)
        S_norm = F.normalize(S.to(E.dtype), dim=-1)             # (K, d)

        # Entity attention: each region attends to corpus
        A = torch.einsum('btnd,kd->btnk', E_norm, S_norm) * self.scale  # (B, T, N, K)
        A = F.softmax(A, dim=-1)                                 # (B, T, N, K)

        # Entity-enriched visual feature per region, then pool spatially
        E_hat = torch.einsum('btnk,kd->btnd', A, S.to(E.dtype)) # (B, T, N, d)
        E_hat = E_hat.mean(dim=2)                                # (B, T, d)

        # Query: project mean visual token
        Q0 = self.query_proj(E.mean(dim=2))                      # (B, T, d)

        return Q0, E_hat