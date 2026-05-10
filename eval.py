# eval.py
import argparse, yaml, torch, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import clip

from models.alt         import ALT
from data.video_dataset import VideoDataset
from data.transforms    import get_val_transforms


def main():
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
    all_labels = json.load(open(cfg['label_file']))   # 152 strings
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

    # Dataset
    roots  = cfg['data_roots']
    ds     = VideoDataset(args.split_json, roots, cfg['num_frames'],
                          get_val_transforms(cfg['frame_size']), mode='val')
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    correct = total = 0
    with torch.no_grad():
        for frames, labels in loader:
            frames         = frames.to(device, non_blocking=True)
            labels_mapped  = labels - args.label_offset   # remap to 0-based
            with autocast(dtype=torch.float16):
                z      = model(frames)
                logits = 100.0 * (z @ C.T)
            preds   = logits.argmax(dim=-1)
            correct += (preds == labels_mapped.to(device)).sum().item()
            total   += labels.size(0)

    print(f"\nZero-shot accuracy on split: {correct/total:.4f}  ({correct}/{total})")


if __name__ == '__main__':
    main()