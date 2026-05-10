# training/loss.py
"""Losses and text prototype construction for ALT contrastive training."""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
import clip


def build_label_embeddings(label_file: str, device: torch.device) -> torch.Tensor:
    """
    Build C ∈ R^{152 × 512}: CLIP text embeddings for all 152 action classes.
    label_file: path to all_labels_ordered.json — flat list of 152 strings,
                index == class id.
    """
    labels = json.load(open(label_file))   # list of 152 strings
    assert len(labels) == 152, f"Expected 152 labels, got {len(labels)}"

    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()

    templates = [
        "a video of a person {}.",
        "a photo of a person {}.",
        "a person is {}.",
        "{}.",
    ]

    all_c = []
    with torch.no_grad():
        for label in labels:
            texts  = [t.format(label) for t in templates]
            tokens = clip.tokenize(texts, truncate=True).to(device)
            embs   = clip_model.encode_text(tokens).float()
            embs   = F.normalize(embs, dim=-1)
            c      = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
            all_c.append(c)

    C = torch.cat(all_c, dim=0)   # (152, 512)
    del clip_model
    torch.cuda.empty_cache()
    return C


def contrastive_loss(
    z: torch.Tensor,
    C: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: float = 100.0,
) -> torch.Tensor:
    """Compute ALT classification loss with cosine logits.

    Args:
        z: video embeddings `(B, 512)` from ALT Eq. (9), L2-normalized.
        C: class text prototypes `(152, 512)`, L2-normalized.
        labels: class indices `(B,)`.
    """
    logits = logit_scale * (z @ C.T)   # (B, 152)
    return F.cross_entropy(logits, labels)
