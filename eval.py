# eval.py
"""Evaluation script for ALT zero-shot protocol on single or multi splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip

from models.alt         import ALT
from data.video_dataset import VideoDataset
from data.transforms    import get_val_transforms


def resolve_split_files(split_json: str) -> List[Path]:
    """Resolve one split file or all official split files in a directory."""
    path = Path(split_json)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"split_json path not found: {split_json}")

    split_files = sorted(
        [
            p for p in path.glob("*.json")
            if any(token in p.stem for token in ("split1", "split2", "split3"))
        ]
    )
    if not split_files:
        raise FileNotFoundError(f"No *_split1/2/3 JSON files found under: {split_json}")
    return split_files


@torch.no_grad()
def evaluate_split(
    model: ALT,
    cfg: dict,
    split_file: Path,
    C: torch.Tensor,
    device: torch.device,
    label_offset: int,
) -> Tuple[float, int, int]:
    """Evaluate one split and return `(acc, correct, total)`."""
    roots = cfg['data_roots']
    ds = VideoDataset(str(split_file), roots, cfg['num_frames'], get_val_transforms(cfg['frame_size']), mode='val')
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    use_amp = (device.type == "cuda")

    correct = total = 0
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels_mapped = labels - label_offset
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            z = model(frames)
            logits = 100.0 * (z @ C.T)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels_mapped.to(device)).sum().item()
        total += labels.size(0)
    return (correct / total if total > 0 else 0.0), correct, total


def _load_labels(label_file: str) -> List[str]:
    """Load labels from either flat list JSON or {'labels': [...]}."""
    payload = json.load(open(label_file))
    if isinstance(payload, list):
        labels = payload
    elif isinstance(payload, dict):
        labels = payload.get("labels")
    else:
        labels = None
    if not isinstance(labels, list):
        raise ValueError(f"Unsupported label file format: {label_file}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split_json', required=True,
                        help='e.g. corpus/hmdb_val_split1.json')
    parser.add_argument('--label_offset', type=int, default=101,
                        help='101 for HMDB zero-shot (labels 101-151 → map to 0-50)')
    parser.add_argument('--n_classes',    type=int, default=51)
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    S     = torch.load(cfg['corpus_path'], map_location='cpu').float()
    model = ALT(cfg, S)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, best_acc={ckpt.get('best_acc',0):.4f})")

    # Build C for target classes only
    all_labels = _load_labels(cfg['label_file'])   # 152 strings
    target_labels = all_labels[args.label_offset: args.label_offset + args.n_classes]

    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    all_c = []
    with torch.no_grad():
        for label in target_labels:
            tokens = clip.tokenize([f"a video of a person {label}."], truncate=True).to(device)
            emb    = clip_model.encode_text(tokens).float()
            all_c.append(F.normalize(emb, dim=-1))
    C = torch.cat(all_c, dim=0)   # (n_classes, 512)
    del clip_model; torch.cuda.empty_cache()

    split_files = resolve_split_files(args.split_json)
    accs: List[float] = []
    for split_file in split_files:
        acc, correct, total = evaluate_split(
            model=model,
            cfg=cfg,
            split_file=split_file,
            C=C,
            device=device,
            label_offset=args.label_offset,
        )
        accs.append(acc)
        print(f"{split_file.name}: {acc:.4f} ({correct}/{total})")

    if len(accs) == 1:
        print(f"\nZero-shot accuracy on split: {accs[0]:.4f}")
    else:
        print(f"\nZero-shot accuracy over {len(accs)} splits: {float(np.mean(accs)):.4f} ± {float(np.std(accs)):.4f}")


if __name__ == '__main__':
    main()
