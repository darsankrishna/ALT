# models/clip_encoder.py
"""Region-aware CLIP encoder with ToMe token merging for ALT Eq. (2)-(3)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

try:
    import clip
except ImportError as exc:
    raise ImportError("openai/CLIP not found — pip install git+https://github.com/openai/CLIP.git") from exc

try:
    from tome.merge import bipartite_soft_matching, merge_wavg

    _TOME_OK = True
except ImportError:
    _TOME_OK = False


class RegionAwareCLIPEncoder(nn.Module):
    """Extract CLIP patch tokens and merge regions with ToMe.

    Input:
    - frames flattened to `(B, 3, 224, 224)` for visual encoding.

    Output:
    - all tokens `(B, N'+1, 512)` where index 0 is CLS and `N'` is merged patch count.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/16",
        freeze: bool = True,
        tome_r: int = 8,
        tome_layers: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        clip_model_name = self._normalize_clip_model_name(clip_model_name)
        clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.visual: nn.Module = clip_model.visual
        self.d: int = int(clip_model.visual.output_dim)

        if freeze:
            for param in self.visual.parameters():
                param.requires_grad_(False)

        self.tome_r: int = max(0, int(tome_r))
        self.tome_layers: Optional[List[int]] = list(tome_layers) if tome_layers is not None else None
        self._token_size: Optional[torch.Tensor] = None

    def _normalize_clip_model_name(self, model_name: str) -> str:
        """Map HF-style CLIP names to OpenAI CLIP names used by `clip.load`."""
        name_map = {
            "openai/clip-vit-base-patch16": "ViT-B/16",
            "openai/clip-vit-base-patch32": "ViT-B/32",
        }
        return name_map.get(model_name, model_name)

    def _should_merge(self, block_index: int) -> bool:
        """Return whether ToMe should be applied at this transformer block."""
        if self.tome_r <= 0 or not _TOME_OK:
            return False
        if self.tome_layers is None:
            return True
        return block_index in self.tome_layers

    def _merge_tokens(self, patches: torch.Tensor, r: int) -> torch.Tensor:
        """Merge patch tokens `(B, N, d)` to `(B, N-r, d)` using ToMe."""
        if r <= 0 or patches.size(1) < 2:
            return patches
        if r % 2 != 0:
            r -= 1
        if r <= 0:
            return patches
        merge, _ = bipartite_soft_matching(patches, r, class_token=False, distill_token=False)
        merged, self._token_size = merge_wavg(merge, patches, self._token_size)
        return merged

    def _forward_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Forward CLIP visual transformer with optional per-block ToMe merging."""
        vit = self.visual
        batch_size = x.shape[0]
        dtype = next(iter(vit.parameters())).dtype
        x = x.to(dtype)

        x = vit.conv1(x)
        x = x.reshape(batch_size, x.shape[1], -1).permute(0, 2, 1)
        cls_token = vit.class_embedding.to(dtype).expand(batch_size, 1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + vit.positional_embedding.to(dtype)
        x = vit.ln_pre(x)

        self._token_size = None
        for block_idx, block in enumerate(vit.transformer.resblocks):
            x = x.permute(1, 0, 2)
            x = block(x)
            x = x.permute(1, 0, 2)

            if self._should_merge(block_idx):
                cls_part = x[:, 0:1, :]
                patch_part = x[:, 1:, :]
                patch_part = self._merge_tokens(patch_part, self.tome_r)
                x = torch.cat([cls_part, patch_part], dim=1)

        x = vit.ln_post(x)
        if vit.proj is not None:
            x = x @ vit.proj
        return x.float()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode `(B, T, 3, 224, 224)` to `(B, T, N'+1, 512)` token features."""
        batch_size, num_frames, channels, height, width = frames.shape
        flat_frames = frames.reshape(batch_size * num_frames, channels, height, width)
        feats = self._forward_patch_tokens(flat_frames)
        token_count = feats.shape[1]
        return feats.reshape(batch_size, num_frames, token_count, self.d)
