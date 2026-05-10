# data/video_dataset.py
import json, torch
from pathlib import Path
from torch.utils.data import Dataset

try:
    import decord
    decord.bridge.set_bridge("torch")
    DECORD_OK = True
except ImportError:
    DECORD_OK = False
    import torchvision.io as tvio


class VideoDataset(Dataset):
    def __init__(self, json_path, data_roots, num_frames=8, transform=None, mode="train"):
        self.entries    = json.load(open(json_path))
        # data_roots: dict like {"ucf": "/path/ucf", "hmdb": "/path/hmdb"}
        #             or str for single root
        self.roots      = data_roots if isinstance(data_roots, dict) else {"default": data_roots}
        self.num_frames = num_frames
        self.transform  = transform
        self.mode       = mode

    def _get_root(self, entry):
        tag = entry.get("dataset", "default")
        root = self.roots.get(tag, list(self.roots.values())[0])
        return Path(root)

    def _sparse_sample(self, total_frames):
        seg = total_frames / self.num_frames
        if self.mode == "train":
            return [min(int(i * seg + torch.rand(1).item() * seg), total_frames - 1)
                    for i in range(self.num_frames)]
        return [min(int((i + 0.5) * seg), total_frames - 1)
                for i in range(self.num_frames)]

    def _load_with_decord(self, path):
        vr     = decord.VideoReader(str(path), num_threads=1)
        fidxs  = self._sparse_sample(len(vr))
        frames = vr.get_batch(fidxs).permute(0, 3, 1, 2).float() / 255.0
        return frames   # (T, 3, H, W)

    def _load_with_torchvision(self, path):
        video, _, _ = tvio.read_video(str(path), pts_unit='sec')
        # video: (T, H, W, 3)
        total = len(video)
        if total == 0:
            raise RuntimeError(f"Empty video: {path}")
        fidxs  = self._sparse_sample(total)
        frames = video[fidxs].permute(0, 3, 1, 2).float() / 255.0
        return frames

    def __getitem__(self, idx):
        entry = self.entries[idx]
        path  = self._get_root(entry) / entry["path"]

        try:
            if DECORD_OK:
                frames = self._load_with_decord(path)
            else:
                frames = self._load_with_torchvision(path)
        except Exception as e:
            # Fallback: return zero frames — training will still proceed
            frames = torch.zeros(self.num_frames, 3, 224, 224)

        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])

        return frames, entry["label"]

    def __len__(self):
        return len(self.entries)