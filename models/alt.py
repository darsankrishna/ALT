# models/alt.py
"""
ALT — Align-Before-Adapt (CVPR 2024).
Orchestrates: encoder → alignment → adapter → z.

Dimension flow (all 512, consistent with CLIP ViT-B/16 output):
  x            : (B, T, 3, 224, 224)
  E            : (B, T, N'+1, 512)   ← N' may be < 196 after ToMe merging
  Q0, E_hat    : (B, T, 512)
  z            : (B, 512)            ← L2-normalized, ready for cosine loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip_encoder  import RegionAwareCLIPEncoder
from models.alignment     import EntityToRegionAlignment
from models.video_adapter import VideoRepresentationAdapter


class ALT(nn.Module):
    def __init__(self, cfg: dict, S: torch.Tensor) -> None:
        """
        cfg : yaml config dict
        S   : corpus tensor (K, 512) — stored as non-trainable fp32 buffer
        """
        super().__init__()
        d = 512   # CLIP ViT-B/16 output_dim; all modules share this

        self.encoder = RegionAwareCLIPEncoder(
            clip_model_name=cfg.get("clip_model", "ViT-B/16"),
            freeze=cfg.get("freeze_backbone", True),
            tome_r=cfg.get("tome_r", 8),              # 0 to disable ToMe
            tome_layers=cfg.get("tome_layers", None),
        )
        self.alignment = EntityToRegionAlignment(
            d=d,
            tau=cfg.get("tau", 1.0),
            alignment_topk=cfg.get("alignment_topk", None),
        )
        self.adapter = VideoRepresentationAdapter(
            d=d,
            n_heads=cfg.get("n_heads", 8),
            n_blocks=cfg.get("adapter_blocks", 4),
            use_temporal_pe=cfg.get("use_temporal_pe", False),
        )

        # Corpus: non-trainable, moves with .to(device)
        self.register_buffer("S", S.float())           # (K, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, 3, 224, 224)
        Returns z : (B, 512) — L2-normalized video representation
        """
        E            = self.encoder(x)                # (B, T, N'+1, 512)
        Q0, E_hat    = self.alignment(E, self.S)      # (B, T, 512) each
        z            = self.adapter(Q0, E_hat)         # (B, 512)
        return F.normalize(z, dim=-1)
