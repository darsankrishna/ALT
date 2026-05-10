# train.py
import argparse, yaml, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.alt          import ALT
from corpus.build_corpus import build_corpus
from data.video_dataset  import VideoDataset
from data.transforms     import get_train_transforms, get_val_transforms
from training.trainer    import Trainer
from training.loss       import build_label_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--output_dir', default='runs/alt_b16')
    parser.add_argument('--log_dir',    default=None, help='optional directory for CSV logs')
    parser.add_argument('--resume',     default=None, help='path to checkpoint .pt')
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    cfg_device = cfg.get("device")
    if cfg_device is not None:
        device = torch.device(cfg_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load/build corpus
    corpus_path = cfg['corpus_path']
    corpus_dir = os.path.dirname(corpus_path)
    corpus_rebuild = bool(cfg.get("corpus_rebuild", False))
    corpus_build_if_missing = bool(cfg.get("corpus_build_if_missing", True))
    corpus_use_lesk = bool(cfg.get("corpus_use_lesk", False))
    corpus_use_t5_filter = bool(cfg.get("corpus_use_t5_filter", False))
    corpus_t5_model = cfg.get("corpus_t5_model", "t5-small")

    if corpus_rebuild or (not os.path.exists(corpus_path) and corpus_build_if_missing):
        print("Building corpus embeddings...")
        build_corpus(
            label_file=cfg['label_file'],
            output_dir=corpus_dir,
            output_path=corpus_path,
            device=str(device),
            use_lesk=corpus_use_lesk,
            use_t5_filter=corpus_use_t5_filter,
            t5_model_name=corpus_t5_model,
        )

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Missing corpus embeddings at: {corpus_path}")

    S = torch.load(corpus_path, map_location='cpu').float()
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

    trainer.run(start_epoch=start_epoch, output_dir=args.output_dir, log_dir=args.log_dir)


if __name__ == '__main__':
    main()
