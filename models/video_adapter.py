"""
Multi-Modal Video Adapter.

Eq. (7): Q_0 = [q^1, ..., q^T]     — initial queries (entity-aligned, per frame)
         Ê   = [ê^1, ..., ê^T]     — projected visual embeddings (per frame)

Eq. (8): For each block m = 1,...,M:
         Q'_m  = SA_m(Q_{m-1})                    — self-attn: temporal query interaction
         Ê_m   = Ê + 1D-Conv_m(Ê)                — temporal conv on visual side
         Q_m   = CA_m(Q'_m, Ê_m)                 — cross-attn: entities query visuals

Eq. (9): z = MLP(AvgPool(Q_M))                   — final video representation

GPU optimization:
  - Flash Attention 2 for SA and CA (if available)
  - Small d_model (matches CLIP: 512 for B, 768 for L)
  - M=4 blocks for B/16 (paper default); reduce to M=2 if VRAM < 8GB
"""
# models/video_adapter.py
import torch
import torch.nn as nn


class AdapterBlock(nn.Module):
    """One block of [Depthwise-1D-Conv + Self-Attention + Cross-Attention + FFN]."""
    def __init__(self, d=768, n_heads=8):
        super().__init__()
        # Temporal depthwise conv (per-channel, along T)
        self.conv      = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d)
        self.conv_norm = nn.LayerNorm(d)

        # Self-attention over time
        self.sa      = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.0)
        self.sa_norm = nn.LayerNorm(d)

        # Cross-attention: query attends to entity-aligned features
        self.ca      = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.0)
        self.ca_norm = nn.LayerNorm(d)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )
        self.ffn_norm = nn.LayerNorm(d)

    def forward(self, Q, E_hat):
        """
        Q     : (B, T, d)
        E_hat : (B, T, d)
        Returns Q: (B, T, d)
        """
        # Depthwise temporal conv
        x = Q.permute(0, 2, 1)                     # (B, d, T)
        x = self.conv(x).permute(0, 2, 1)          # (B, T, d)
        Q = self.conv_norm(Q + x)

        # Self-attention
        sa_out, _ = self.sa(Q, Q, Q)
        Q = self.sa_norm(Q + sa_out)

        # Cross-attention
        ca_out, _ = self.ca(Q, E_hat, E_hat)
        Q = self.ca_norm(Q + ca_out)

        # FFN
        Q = self.ffn_norm(Q + self.ffn(Q))

        return Q


class VideoRepresentationAdapter(nn.Module):
    def __init__(self, d=768, n_heads=8, n_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([AdapterBlock(d, n_heads) for _ in range(n_blocks)])

    def forward(self, Q, E_hat):
        for block in self.blocks:
            Q = block(Q, E_hat)
        return Q   # (B, T, d)