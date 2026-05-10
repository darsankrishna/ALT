# models/video_adapter.py
"""
Multi-Modal Video Adapter — Eq. (7), (8), (9).

Eq. (7): Q_0 = entity-aligned queries per frame   (B, T, d)
         Ê   = CLS-based K/V per frame            (B, T, d)

Eq. (8): For each block m = 1…M:
         Ê_m  = Ê + DepthwiseConv1D_m(Ê)         ← conv on visual/entity side
         Q'_m = SA_m(Q_{m-1})                     ← self-attn over time
         Q_m  = CA_m(Q'_m, Ê_m)                   ← cross-attn: Q queries Ê

Eq. (9): z = MLP(AvgPool_T(Q_M))                  ← final video representation
"""

import torch
import torch.nn as nn


class AdapterBlock(nn.Module):
    """
    One adapter block.

    Note on conv placement: the depthwise temporal conv is applied to Ê (the
    entity-aligned visual side), NOT to Q.  This matches Eq. (8) in the paper —
    the conv enriches the key/value stream with local temporal context before
    Q attends to it via cross-attention.
    """

    def __init__(self, d: int = 512, n_heads: int = 8) -> None:
        super().__init__()

        # Temporal depthwise conv — operates on Ê (key/value stream)
        self.conv      = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d)
        self.conv_norm = nn.LayerNorm(d)

        # Self-attention on Q (temporal query context)
        self.sa      = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.0)
        self.sa_norm = nn.LayerNorm(d)

        # Cross-attention: Q attends to Ê
        self.ca      = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.0)
        self.ca_norm = nn.LayerNorm(d)

        # FFN on Q
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )
        self.ffn_norm = nn.LayerNorm(d)

    def forward(self, Q: torch.Tensor, E_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Q, E_hat : (B, T, d)
        Returns  : Q_out (B, T, d), E_hat_out (B, T, d)
        """
        # 1. Temporal conv on Ê (enriches K/V with local motion context)
        e = E_hat.permute(0, 2, 1)                           # (B, d, T)
        E_hat = self.conv_norm(E_hat + self.conv(e).permute(0, 2, 1))

        # 2. Self-attention on Q (temporal query interactions)
        sa_out, _ = self.sa(Q, Q, Q)
        Q = self.sa_norm(Q + sa_out)

        # 3. Cross-attention: Q queries the convolved Ê
        ca_out, _ = self.ca(Q, E_hat, E_hat)
        Q = self.ca_norm(Q + ca_out)

        # 4. FFN
        Q = self.ffn_norm(Q + self.ffn(Q))

        return Q, E_hat                                       # pass E_hat through


class VideoRepresentationAdapter(nn.Module):
    """M stacked AdapterBlocks → AvgPool over T → MLP → z (B, d)."""

    def __init__(
        self,
        d: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        use_temporal_pe: bool = False,
    ) -> None:
        super().__init__()
        self.d: int = d
        self.use_temporal_pe: bool = use_temporal_pe
        self.blocks = nn.ModuleList(
            [AdapterBlock(d, n_heads) for _ in range(n_blocks)]
        )
        # Eq. (9) MLP
        self.out_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def _temporal_sinusoidal_pe(self, timesteps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build sinusoidal temporal position encoding `(T, d)`."""
        position = torch.arange(timesteps, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d, 2, device=device, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0, device=device)) / self.d)
        )
        pe = torch.zeros(timesteps, self.d, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(dtype=dtype)

    def forward(self, Q0: torch.Tensor, E_hat: torch.Tensor) -> torch.Tensor:
        """
        Q0, E_hat : (B, T, d)
        Returns   : z (B, d) — L2 norm applied in ALT.forward
        """
        Q = Q0
        if self.use_temporal_pe:
            pe = self._temporal_sinusoidal_pe(
                timesteps=Q.shape[1],
                device=Q.device,
                dtype=Q.dtype,
            )
            Q = Q + pe.unsqueeze(0)
        for block in self.blocks:
            Q, E_hat = block(Q, E_hat)
        return self.out_mlp(Q.mean(dim=1))                   # (B, d)
