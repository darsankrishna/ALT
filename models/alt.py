"""
Full ALT model: orchestrates encoder → align → adapt → z.
GPU-constrained defaults:
  - backbone=frozen by default (set freeze_backbone=False for full finetune)
  - bf16 autocast at training
  - corpus embeddings preloaded as fp16 non-gradient buffer
"""
# models/alt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.clip_encoder import RegionAwareCLIPEncoder
from models.alignment    import EntityToRegionAlignment
from models.video_adapter import VideoRepresentationAdapter


class ALT(nn.Module):
    """
    Align-before-Adapt (ALT) for video action recognition.
    Visual d=768, CLIP embedding d=512 (projected for contrastive loss).
    """
    def __init__(self, cfg, S):
        """
        cfg : yaml config dict
        S   : corpus tensor (K, 512) — entity embeddings, fp16 or fp32
        """
        super().__init__()
        d_vis  = 768   # ViT-B/16 hidden dim
        d_clip = 512   # CLIP projection dim

        self.encoder   = RegionAwareCLIPEncoder(
            model_name=cfg.get('clip_model', 'openai/clip-vit-base-patch16'),
            freeze=cfg.get('freeze_backbone', True),
        )
        self.alignment = EntityToRegionAlignment(d=d_vis)
        self.adapter   = VideoRepresentationAdapter(
            d=d_vis,
            n_heads=cfg.get('n_heads', 8),
            n_blocks=cfg.get('adapter_blocks', 4),
        )
        # Project from ViT hidden dim to CLIP embedding dim
        self.proj = nn.Sequential(
            nn.Linear(d_vis, d_clip),
            nn.GELU(),
            nn.Linear(d_clip, d_clip),
        )

        # Corpus: non-trainable buffer
        self.register_buffer('S', S.float())   # (K, d_clip=512)

    def forward(self, x):
        """x : (B, T, 3, 224, 224)"""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # Region-aware encoding
        E_flat = self.encoder(x_flat)          # (B*T, 197, 768)
        N, d   = E_flat.shape[1], E_flat.shape[2]
        E      = E_flat.view(B, T, N, d)       # (B, T, N, 768)

        # Entity-to-region alignment
        Q0, E_hat = self.alignment(E, self.S)  # (B, T, 768), (B, T, 768)

        # Video adapter
        Q_M = self.adapter(Q0, E_hat)          # (B, T, 768)

        # Pool over time + project to CLIP space
        z = Q_M.mean(dim=1)                    # (B, 768)
        z = self.proj(z)                        # (B, 512)
        z = F.normalize(z, dim=-1)             # (B, 512) — unit norm for cosine

        return z