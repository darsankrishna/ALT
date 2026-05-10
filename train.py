# train.py
import argparse, yaml, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.alt          import ALT
from data.video_dataset  import VideoDataset
from data.transforms     import get_train_transforms, get_val_transforms
from training.trainer    import Trainer
from training.loss       import build_label_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--output_dir', default='runs/alt_b16')
    parser.add_argument('--resume',     default=None, help='path to checkpoint .pt')
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load corpus
    S = torch.load(cfg['corpus_path'], map_location='cpu').float()
    print(f"Corpus S: {S.shape}")

    # Build model
    model = ALT(cfg, S)
    total_params   = sum(p.numel() for p in model.parameters())
    trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    # Datasets
    roots    = cfg['data_roots']
    train_ds = VideoDataset(cfg['train_json'], roots, cfg['num_frames'],
                            get_train_transforms(cfg['frame_size']), mode='train')
    val_ds   = VideoDataset(cfg['val_json'],   roots, cfg['num_frames'],
                            get_val_transforms(cfg['frame_size']),  mode='val')
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Label embeddings
    C = build_label_embeddings(cfg['label_file'], device)
    print(f"Label embeddings C: {C.shape}")

    # Trainer
    trainer = Trainer(model, cfg, train_ds, val_ds, C, device)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resumed from epoch {start_epoch}")

    trainer.run(start_epoch=start_epoch, output_dir=args.output_dir)


if __name__ == '__main__':
    main()