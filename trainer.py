import os
from typing import Dict, List, Optional, Tuple

from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import seed_everything, char_level_voting, decode_with_confidence, SRMetricLoss


class Trainer:
    """Encapsulates training, validation, and inference logic."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str],
        mode: str = 'train'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.device
        
        seed_everything(config.seed, benchmark=getattr(config, 'use_cudnn_benchmark', False))

        if mode == 'train':
            self._init_training_components()
    def _init_training_components(self):
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        # self.sr_loss_func = SRMetricLoss(device=self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=getattr(self.config, 'weight_decay', 1e-4)
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            steps_per_epoch=len(self.train_loader),
            epochs=self.config.epochs,
            pct_start=0.2,
            anneal_strategy='cos'
        )

        self.scaler = GradScaler()

        self.best_acc = 0.0
        self.current_epoch = 0

    def _get_output_path(self, filename: str) -> str:
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _get_exp_name(self) -> str:
        return getattr(self.config, 'experiment_name', 'baseline')

    def train_one_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}")
        
        for images, targets, target_lengths, _, _, hr_targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            hr_targets = hr_targets.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                preds = self.model(images)
                # preds, sr_out = self.model(images, return_sr=True)
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(1),
                    dtype=torch.long
                )
                loss_ctc = self.criterion(preds_permuted, targets, input_lengths, target_lengths)
                # loss_sr = self.criterion_sr(sr_out, hr_targets)

                # mse, perceptual, edge = self.sr_loss_func(sr_out, hr_targets)
                # loss_sr_total = 1.0 * mse + 0.5 * perceptual + 0.3 * edge
                loss_sr_total = torch.tensor(0.0).to(self.device) 
                total_loss = loss_ctc + 0.5 * loss_sr_total


            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.config, 'grad_clip', 10.0))
            
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
            
            epoch_loss += total_loss.item() * images.size(0)  # better averaging
            pbar.set_postfix(
                ctc=f"{loss_ctc.item():.3f}", 
                sr=f"{loss_sr_total.item():.3f}", 
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}"
            )        
        return epoch_loss / len(self.train_loader.dataset)  # per sample

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        """Run validation and generate submission data."""
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0, 'cer': 0.0}, []
        
        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds: List[str] = []
        all_targets: List[str] = []
        submission_data: List[str] = []
        
        with torch.no_grad():
            for images, targets, target_lengths, labels_text, track_ids, hr_targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                hr_targets = hr_targets.to(self.device)
                # preds, sr_out = self.model(images, return_sr=True)
                preds = self.model(images)
                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long
                )
                loss = self.criterion(
                    preds.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths
                )
                # mse, perceptual, edge = self.sr_loss_func(sr_out, hr_targets)
                # loss_sr_total = 1.0 * mse + 0.1 * perceptual + 0.1 * edge
                loss_sr_total = torch.tensor(0.0).to(self.device)
                loss = loss + 0.5 * loss_sr_total
                val_loss += loss.item() * images.size(0)

                # Decode predictions
                decoded_list = decode_with_confidence(preds, self.idx2char)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    
                    all_preds.append(pred_text)
                    all_targets.append(gt_text)
                    
                    if pred_text == gt_text:
                        total_correct += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")
                    
                total_samples += len(labels_text)

        avg_val_loss = val_loss / total_samples if total_samples > 0 else 0.0
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        
        metrics = {
            'loss': avg_val_loss,
            'acc': val_acc,
        }
        
        return metrics, submission_data
    def fit(self, start_epoch: int = 0, best_acc: float = 0.0):
        self.current_epoch = start_epoch
        self.best_acc = best_acc
        
        print(f"🚀 Start training from epoch {start_epoch+1}/{self.config.epochs} | Best so far: {best_acc:.2f}%")
        
        for epoch in range(start_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_one_epoch()
            val_metrics, submission_data = self.validate()
            
            val_loss = val_metrics.get('loss', float('nan'))
            val_acc  = val_metrics.get('acc',  float('nan'))
            lr = self.scheduler.get_last_lr()[0]
            
            # Logging thông minh
            log_str = f"Epoch {epoch+1:03d}/{self.config.epochs}: Train Loss: {train_loss:.4f} | LR: {lr:.2e}"
            if not torch.isnan(torch.tensor(val_loss)):
                log_str += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            else:
                log_str += " | No validation set"
            print(log_str)
            
            # Save best
            if not torch.isnan(torch.tensor(val_acc)) and val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(is_best=True)
                print(f"  ⭐ New best: {val_acc:.2f}%")
                if submission_data:
                    self.save_submission(submission_data)
        
        # Save final model (đặc biệt hữu ích ở submission mode)
        self.save_checkpoint(is_best=False)
        print(f"\n✅ Training finished | Best Val Acc: {self.best_acc:.2f}%")

    def save_checkpoint(self, is_best: bool = False):
        exp_name = self._get_exp_name()
        filename = f"{exp_name}_best.pth" if is_best else f"{exp_name}_last.pth"
        path = self._get_output_path(filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'config': vars(self.config) if hasattr(self.config, '__dict__') else {},
        }
        torch.save(checkpoint, path)
        print(f"{'⭐ Best' if is_best else 'Last'} checkpoint saved: {path}")
    def save_submission(self, submission_data: List[str]) -> None:
        """Save submission file with experiment name."""
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, 'w') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filename}")
    # def predict_test(self, test_loader: DataLoader, output_filename: str = None) -> None:
    #     if output_filename is None:
    #         output_filename = f"submission_{self._get_exp_name()}_final.txt"

    #     print(f"🔮 Inference on test set → {output_filename}")

    #     self.model.eval()
    #     submission_lines = []

    #     with torch.no_grad(), tqdm(total=len(test_loader), desc="Predict") as pbar:
    #         for images, _, _, _, track_ids in test_loader:
    #             # images: [B, 5, C, H, W]
    #             images = images.to(self.device)

    #             B, T, C, H, W = images.shape
    #             assert T >= 3, "Need at least 3 frames"

    #             # store predictions per sample
    #             all_preds = [[] for _ in range(B)]

    #             # sliding windows: (0-2), (1-3), (2-4)
    #             for start in range(T - 2):
    #                 clip = images[:, start:start+3]  # [B, 3, C, H, W]

    #                 preds = self.model(clip)
    #                 decoded = decode_with_confidence(preds, self.idx2char)

    #                 for i, (text, conf) in enumerate(decoded):
    #                     all_preds[i].append((text, conf))

    #             # voting per track
    #             for tid, preds in zip(track_ids, all_preds):
    #                 best_text, best_conf = char_level_voting(preds)
    #                 submission_lines.append(f"{tid},{best_text};{best_conf:.4f}")


    #             pbar.update(1)

    #     path = self._get_output_path(output_filename)
    #     with open(path, 'w', encoding='utf-8') as f:
    #         f.write("\n".join(submission_lines))

    #     print(f"→ Saved {len(submission_lines)} predictions")
    def predict_test(self, test_loader: DataLoader, output_filename: str = None) -> None:
        if output_filename is None:
            output_filename = f"submission_{self._get_exp_name()}_final.txt"
        
        print(f"🔮 Inference on test set → {output_filename}")
        
        self.model.eval()
        submission_lines = []
        
        with torch.no_grad(), tqdm(total=len(test_loader), desc="Predict") as pbar:
            for images, _, _, _, track_ids, _ in test_loader:
                images = images.to(self.device)
                preds = self.model(images)
                decoded = decode_with_confidence(preds, self.idx2char)
                
                for tid, (text, conf) in zip(track_ids, decoded):
                    submission_lines.append(f"{tid},{text};{conf:.4f}")
                pbar.update(1)
        
        path = self._get_output_path(output_filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(submission_lines))
        
        print(f"→ Saved {len(submission_lines)} predictions")