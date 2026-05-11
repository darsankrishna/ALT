# data/video_dataset.py
"""Video dataset utilities for loading sparse frame clips for ALT."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

try:
    import decord
    decord.bridge.set_bridge("torch")
    DECORD_OK = True
except ImportError:
    DECORD_OK = False
    import torchvision.io as tvio


class VideoDataset(Dataset):
    """Loads videos from JSON entries and returns frame tensors and labels.

    Each sample returns:
    - frames: `(T, 3, H, W)`
    - label: scalar class index
    """

    def __init__(
        self,
        json_path: Union[str, Path],
        data_roots: Union[str, Dict[str, str]],
        num_frames: int = 8,
        transform: Optional[Any] = None,
        mode: str = "train",
    ) -> None:
        with open(json_path, "r", encoding="utf-8") as handle:
            self.entries: List[Dict[str, Any]] = json.load(handle)
        # data_roots: dict like {"ucf": "/path/ucf", "hmdb": "/path/hmdb"}
        #             or str for single root
        self.roots: Dict[str, str] = data_roots if isinstance(data_roots, dict) else {"default": data_roots}
        self.num_frames: int = num_frames
        self.transform: Optional[Any] = transform
        self.mode: str = mode

    def _get_root(self, entry: Dict[str, Any]) -> Path:
        """Select dataset root based on the optional `entry['dataset']` tag."""
        tag = entry.get("dataset", "default")
        root = self.roots.get(tag, list(self.roots.values())[0])
        return Path(root)

    def _normalize_filename(self, filename: str) -> str:
        """Normalize filenames for HMDB lookup: alnum/underscore/dot only, lowercased."""
        return re.sub(r"[^A-Za-z0-9_.]", "", filename).lower()

    def _resolve_path(self, entry: Dict[str, Any]) -> Path:
        """Resolve a video path for UCF/HMDB/default layouts."""
        tag = entry.get("dataset", "default")
        rel_path = Path(entry["path"])

        if tag == "ucf":
            root = Path(self.roots["ucf"])
            for split in ("train", "val", "test"):
                candidate = root / split / rel_path
                if candidate.exists():
                    return candidate
            return root / "train" / rel_path

        if tag == "hmdb":
            root = Path(self.roots["hmdb"])
            label = str(entry.get("label_name", rel_path.parent.name))
            filename = rel_path.name
            target_norm = self._normalize_filename(filename)
            label_dir = root / label
            if label_dir.exists():
                for file_path in label_dir.iterdir():
                    if (
                        file_path.is_file()
                        and self._normalize_filename(file_path.name) == target_norm
                    ):
                        return file_path
            return label_dir / filename

        return self._get_root(entry) / rel_path

    def _sparse_sample(self, total_frames: int) -> List[int]:
        """Sample `num_frames` indices uniformly over the full temporal span."""
        seg = total_frames / self.num_frames
        if self.mode == "train":
            return [min(int(i * seg + torch.rand(1).item() * seg), total_frames - 1)
                    for i in range(self.num_frames)]
        return [min(int((i + 0.5) * seg), total_frames - 1)
                for i in range(self.num_frames)]

    def _load_with_decord(self, path: Path) -> torch.Tensor:
        """Load `(T, 3, H, W)` frames using decord."""
        vr     = decord.VideoReader(str(path), num_threads=1)
        fidxs  = self._sparse_sample(len(vr))
        frames = vr.get_batch(fidxs).permute(0, 3, 1, 2).float() / 255.0
        return frames   # (T, 3, H, W)

    def _load_with_torchvision(self, path: Path) -> torch.Tensor:
        """Load `(T, 3, H, W)` frames using torchvision fallback."""
        video, _, _ = tvio.read_video(str(path), pts_unit='sec')
        # video: (T, H, W, 3)
        total = len(video)
        if total == 0:
            raise RuntimeError(f"Empty video: {path}")
        fidxs  = self._sparse_sample(total)
        frames = video[fidxs].permute(0, 3, 1, 2).float() / 255.0
        return frames

    def __getitem__(self, idx: int) -> Any:
        """Return transformed clip tensor and class label for one sample."""
        entry = self.entries[idx]
        path = self._resolve_path(entry)

        try:
            if DECORD_OK:
                frames = self._load_with_decord(path)
            else:
                frames = self._load_with_torchvision(path)
        except Exception:
            # Fallback: return zero frames — training will still proceed
            frames = torch.zeros(self.num_frames, 3, 224, 224)

        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])

        return frames, entry["label"]

    def __len__(self) -> int:
        """Return total number of video samples."""
        return len(self.entries)
