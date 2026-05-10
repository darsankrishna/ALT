# training/trainer.py
"""Training loop for ALT with reproducibility, early stop, and CSV logging."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.loss import contrastive_loss


class Trainer:
    """Train ALT end-to-end while keeping checkpoint and metric bookkeeping."""

    def __init__(self, model: nn.Module, cfg: Dict[str, Any], train_ds: Any, val_ds: Any, C: torch.Tensor, device: torch.device) -> None:
        self.cfg: Dict[str, Any] = cfg
        self.device: torch.device = device
        self._set_seed(int(cfg.get("seed", 42)))
        self.model: nn.Module = model.to(device)
        self.C: torch.Tensor = C.to(device)        # (152, 512) label embeddings

        # Optimizer: separate LRs for backbone vs adapter
        backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
        other_params    = [p for n, p in model.named_parameters()
                           if 'encoder' not in n and p.requires_grad]

        param_groups = [{'params': other_params, 'lr': cfg['adapter_lr']}]
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': cfg['backbone_lr']})

        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])

        steps_per_epoch  = len(train_ds) // cfg['batch_size']
        total_steps      = steps_per_epoch * cfg['epochs']
        warmup_steps     = steps_per_epoch * cfg['warmup_epochs']

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

        self.scheduler: torch.optim.lr_scheduler.LambdaLR = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Enable AMP only on CUDA.
        self.use_amp: bool = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.train_loader = DataLoader(
            train_ds, cfg['batch_size'], shuffle=True,
            num_workers=cfg.get('num_workers', 2),
            pin_memory=self.use_amp, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, cfg['batch_size'], shuffle=False,
            num_workers=cfg.get('num_workers', 2),
            pin_memory=self.use_amp,
        )
        self.grad_accum: int = int(cfg.get('grad_accum_steps', 4))
        self.best_acc: float = 0.0
        self.global_step: int = 0
        self.log_interval: int = int(cfg.get("log_interval", 50))
        self.early_stopping_patience: int = int(cfg.get("early_stopping_patience", 0))
        self.early_stop_triggered: bool = False
        self.no_improve_epochs: int = 0
        self._csv_path: Optional[Path] = None

    def _set_seed(self, seed: int) -> None:
        """Set Python/NumPy/PyTorch seeds for reproducible training behavior."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _init_csv_logger(self, log_dir: Path) -> None:
        """Create CSV log file with header if missing."""
        log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_dir / "metrics.csv"
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["epoch", "train_loss", "val_acc", "lr"])

    def _append_csv_log(self, epoch: int, train_loss: float, val_acc: float, lr: float) -> None:
        """Append one epoch worth of metrics to CSV."""
        if self._csv_path is None:
            return
        with open(self._csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([epoch, train_loss, val_acc, lr])

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch and return mean epoch loss."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, (frames, labels) in enumerate(self.train_loader):
            frames = frames.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                z    = self.model(frames)
                loss = contrastive_loss(z, self.C, labels)
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.grad_accum

            if step % self.log_interval == 0:
                print(f"  ep{epoch} step{step}/{len(self.train_loader)} "
                      f"loss={loss.item()*self.grad_accum:.4f} "
                      f"lr={self.scheduler.get_last_lr()[0]:.2e}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate top-1 validation accuracy."""
        self.model.eval()
        correct = total = 0
        for frames, labels in self.val_loader:
            frames = frames.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                z      = self.model(frames)
                logits = 100.0 * (z @ self.C.T)
            preds   = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        return correct / total if total > 0 else 0.0

    def save_checkpoint(self, epoch: int, path: str) -> None:
        """Save checkpoint including optimizer/scheduler/scaler and config."""
        torch.save({
            'epoch':           epoch,
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state':    self.scaler.state_dict(),
            'best_acc':        self.best_acc,
            'global_step':     self.global_step,
            'config':          self.cfg,
        }, path)
        print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load training state and return checkpoint epoch index."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.scaler.load_state_dict(ckpt['scaler_state'])
        self.best_acc    = ckpt.get('best_acc', 0.0)
        self.global_step = ckpt.get('global_step', 0)
        return ckpt['epoch']

    def run(self, start_epoch: int = 0, output_dir: str = "runs/", log_dir: Optional[str] = None) -> None:
        """Execute epoch loop with optional early stopping and CSV logging."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._init_csv_logger(Path(log_dir) if log_dir else output_path)
        for epoch in range(start_epoch, self.cfg['epochs']):
            loss = self.train_epoch(epoch)
            acc  = self.evaluate()
            current_lr = float(self.scheduler.get_last_lr()[0])
            self._append_csv_log(epoch, loss, acc, current_lr)
            print(f"Epoch {epoch}: loss={loss:.4f}  val_acc={acc:.4f}  best={self.best_acc:.4f}")

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, f"{output_dir}/ckpt_epoch{epoch}.pt")

            if acc > self.best_acc:
                self.best_acc = acc
                self.save_checkpoint(epoch, f"{output_dir}/best.pt")
                print(f"  ★ New best: {acc:.4f}")
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1
                if self.early_stopping_patience > 0 and self.no_improve_epochs >= self.early_stopping_patience:
                    self.early_stop_triggered = True
                    print(f"  Early stopping at epoch {epoch} (patience={self.early_stopping_patience})")
                    break
