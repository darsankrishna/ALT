# models/clip_encoder.py
import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class RegionAwareCLIPEncoder(nn.Module):
    """
    CLIP ViT-B/16 visual encoder returning all patch tokens.
    d_model = 768 (ViT hidden dim), d_clip = 512 (CLIP projection).
    We use hidden tokens (768-dim) for spatial alignment,
    and project to 512 for contrastive loss.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze=True):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        self.d = self.encoder.config.hidden_size   # 768

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Enable gradient checkpointing to save VRAM even when frozen
        if hasattr(self.encoder.vision_model.encoder, 'gradient_checkpointing_enable'):
            self.encoder.vision_model.encoder.gradient_checkpointing_enable()

    def forward(self, x):
        # x: (B*T, 3, 224, 224)
        out = self.encoder(pixel_values=x)
        return out.last_hidden_state   # (B*T, 197, 768) — CLS + 196 patches