# ALT: Align-Before-Adapt for Video Action Recognition

This repository contains a PyTorch implementation of ALT (Align-Before-Adapt), a CVPR 2024 video action recognition model built around a frozen CLIP ViT-B/16 visual backbone, entity-to-region alignment, and a temporal adapter. This is a personal working implementation for the paper, since the authors did not publish an official code repository, and the repo is organized around UCF101/HMDB51 experiments plus Kaggle notebook workflows for corpus building, training, and HMDB51 evaluation.

## What the model does

At a high level, the pipeline is:

1. Encode each frame with CLIP ViT-B/16 and optionally apply ToMe token merging in the visual transformer.
2. Align patch tokens to a corpus of action-entity embeddings using cosine similarity and Gumbel-Softmax.
3. Aggregate aligned regions into per-frame queries and combine them with CLS-derived key/value features.
4. Pass the temporal sequence through stacked adapter blocks with temporal self-attention, cross-attention, and a depthwise temporal convolution.
5. Compare the final video embedding against CLIP text prototypes with cosine logits.

The implementation follows the paper structure described in the code comments around Eq. (2) to Eq. (9).

## Repository Layout

- `models/alt.py`: top-level ALT module that wires encoder, alignment, and adapter.
- `models/clip_encoder.py`: CLIP visual encoder wrapper with optional ToMe token merging.
- `models/alignment.py`: entity-to-region alignment module and straight-through Gumbel-Softmax assignment.
- `models/video_adapter.py`: temporal adapter stack used to produce the final video embedding.
- `training/loss.py`: CLIP text prototype construction and cosine classification loss.
- `training/trainer.py`: training loop, checkpointing, AMP, LR scheduling, and CSV logging.
- `data/video_dataset.py`: video loading, sparse frame sampling, and dataset path resolution.
- `data/transforms.py`: CLIP-normalized train/validation transforms.
- `corpus/build_action_labels.py`: builds the ordered 152-class label list and supporting label maps.
- `corpus/build_ucf_hmdb_index.py`: generates UCF/HMDB split JSONs.
- `corpus/build_corpus.py`: builds corpus embeddings from action labels and optional expansions.
- `train.py`: training entry point.
- `eval.py`: HMDB51 zero-shot / multi-split evaluation entry point.
- `configs/alt_b16.yaml`: Kaggle-oriented default configuration.
- `configs/alt_b16_local.yaml`: local-path variant of the same configuration.
- `(KAGGLE)ALT_(UCF101)_pretrained_zero_eval_on_HMDB51.ipynb`: Kaggle notebook for UCF-only training and HMDB51 zero-shot evaluation.
- `(KAGGLE)ALT_(UCF101+HMDB51)_pretrained_supervised_eval_on_HMDB51.ipynb`: Kaggle notebook for combined UCF+HMDB training and HMDB51 supervised evaluation.

## Installation

The code expects Python 3.10+ with PyTorch and torchvision already available. The implementation also depends on OpenAI CLIP and optionally ToMe for token merging.

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision pyyaml numpy decord wordninja nltk spacy transformers sentencepiece requests
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/facebookresearch/ToMe.git
```

If you plan to rebuild the corpus locally, also install the language resources used by `corpus/build_corpus.py`:

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4
```

## Data And Corpus Preparation

ALT uses three kinds of artifacts:

1. Video split JSONs, where each entry has `path`, `label`, and `dataset` fields.
2. A label file with the ordered 152-class list, where UCF101 occupies indices 0-100 and HMDB51 occupies indices 101-151.
3. A corpus embedding tensor saved as `corpus_embeddings.pt`.

The scripts in `corpus/` generate these artifacts from the official split files:

- `corpus/build_action_labels.py` creates `corpus/action_labels.json`, `corpus/all_labels_ordered.json`, `corpus/ucf_class_map.json`, and `corpus/hmdb_class_map.json`.
- `corpus/build_ucf_hmdb_index.py` creates `corpus/ucf_train_split1.json`, `corpus/ucf_val_split1.json`, `corpus/hmdb_train_split1.json`, and `corpus/hmdb_val_split{1,2,3}.json`.
- `corpus/build_corpus.py` builds `corpus/corpus_embeddings.pt` from the label file and optional LLM/WordNet-based noun expansions.

The Kaggle notebooks also patch `configs/alt_b16.yaml` at runtime so the workspace points at the mounted UCF101 and HMDB51 dataset roots.

## Running Training

Train with the main entry point:

```bash
python train.py --config configs/alt_b16.yaml --output_dir runs/alt_b16
```

Useful flags:

- `--resume PATH` resumes from a checkpoint saved by `training/trainer.py`.
- `--log_dir PATH` writes `metrics.csv` to a separate location.

The default config trains with a frozen CLIP backbone, separate learning rates for encoder and adapter parameters, gradient accumulation, AMP on CUDA, and early stopping.

## Running Evaluation

For HMDB51 zero-shot evaluation, use `eval.py` with a checkpoint and an HMDB split JSON:

```bash
python eval.py \
  --config configs/alt_b16.yaml \
  --checkpoint runs/alt_b16/best.pt \
  --split_json corpus/hmdb_val_split1.json \
  --label_offset 101 \
  --n_classes 51
```

If `--split_json` points to a directory, the script evaluates every `*_split1/2/3.json` file in that directory and reports the mean and standard deviation.

## Default Configuration

`configs/alt_b16.yaml` is the main runtime configuration used by the Kaggle notebooks. Key defaults include:

- `clip_model: ViT-B/16`
- `freeze_backbone: true`
- `tome_r: 8`
- `adapter_blocks: 4`
- `n_heads: 8`
- `num_frames: 8`
- `frame_size: 224`
- `batch_size: 16`
- `grad_accum_steps: 2`
- `epochs: 50`
- `warmup_epochs: 5`
- `early_stopping_patience: 5` in the zero-shot notebook, `0` in the supervised notebook

The local config file uses the same structure but points to local paths instead of Kaggle mount points.

## Kaggle Notebook Experiments

The repo includes two notebook-driven experiment tracks.

### 1. UCF101 Pretrained, HMDB51 Zero-Shot Evaluation

Notebook: `(KAGGLE)ALT_(UCF101)_pretrained_zero_eval_on_HMDB51.ipynb`

Observed workflow:

- Copy the ALT code bundle into `/kaggle/working/alt`.
- Build HMDB split JSONs from the official train/test split files.
- Install `CLIP`, `ToMe`, and `decord`.
- Patch `configs/alt_b16.yaml` to use UCF101 train/val JSONs and Kaggle dataset roots.
- Rebuild the corpus embeddings with Lesk and T5 filtering enabled.
- Verify that ToMe is active and that the dataset resolver works.
- Train on UCF101 split1, then evaluate HMDB51 split1/2/3.

Logged results in the notebook:

- HMDB51 split1 zero-shot accuracy: 0.2340
- HMDB51 split2 zero-shot accuracy: 0.2275
- HMDB51 split3 zero-shot accuracy: 0.2529
- Mean over the three HMDB splits: about 0.2381
- The checkpoint used for evaluation was reported as epoch 10 with `best_acc=0.8818` on the training side.

The notebook also includes a tiny HMDB sanity run that evaluates a 10-sample split and confirms the evaluation path executes end to end.

### 2. UCF101+HMDB51 Supervised Evaluation on HMDB51

Notebook: `(KAGGLE)ALT_(UCF101+HMDB51)_pretrained_supervised_eval_on_HMDB51.ipynb`

Observed workflow:

- Copy the ALT code bundle into `/kaggle/working/alt`.
- Generate HMDB51 val split JSONs.
- Install `CLIP`, `ToMe`, and `decord`.
- Patch the config to use the combined `train_labels.json` and `val_labels.json` corpus.
- Rebuild the corpus embeddings.
- Run full training and then evaluate all HMDB51 splits with the saved checkpoint.

Logged results in the notebook:

- HMDB51 split1 supervised accuracy: 0.6418
- HMDB51 split2 supervised accuracy: 0.8948
- HMDB51 split3 supervised accuracy: 0.8791
- Best checkpoint reported: epoch 31 with `best_acc=0.8726`

## Code vs. Paper Notes

The implementation matches the paper structure at the module level, but the repo adds a few practical pieces that are specific to this codebase:

- The corpus is built from the ordered 152-class label list and can optionally use Lesk and a T5 filter to refine corpus entries.
- The CLIP backbone is frozen by default to reduce memory use and training cost.
- ToMe is enabled by default in the Kaggle config to merge visual tokens inside the CLIP encoder.
- HMDB labels are offset by 101 so the shared 152-way label space stays aligned with the UCF/HMDB split convention used here.
- The evaluation script reports per-split accuracies and can average over multiple HMDB split files.

## Outputs And Checkpoints

Training writes checkpoints and logs to `runs/alt_b16/` by default, including:

- `best.pt`
- `ckpt_epoch*.pt`
- `metrics.csv`

The Kaggle notebooks also save a patched config file and, in some runs, a copied `eval.py` for notebook-local execution.

## Troubleshooting

- If `clip.load(...)` fails, confirm the OpenAI CLIP package is installed from GitHub and not a different CLIP wrapper.
- If video loading fails, check that `decord` is installed; otherwise the dataset falls back to `torchvision.io.read_video`.
- If corpus building fails, verify that `corpus/all_labels_ordered.json` exists and that spaCy/NLTK resources are available.
- If HMDB evaluation returns zero accuracy, confirm that `--label_offset 101` and `--n_classes 51` match the HMDB label layout used in this repo.

## Reproducibility

The training loop sets Python, NumPy, and PyTorch seeds from the config, and the Kaggle notebooks record the concrete data roots, checkpoint locations, and split files used for each experiment.