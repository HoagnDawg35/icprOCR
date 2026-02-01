import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import seed_everything, decode_with_confidence
from src.models.model_distill import ResTranOCR_Distill

class TrainerDistill:
    """Trainer for Knowledge Distillation."""
    
    def __init__(
        self,
        student_model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str],
        mode: str = 'train'
    ):
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.device
        
        seed_everything(config.seed)

        if mode == 'train':
            self._init_training_components()
            self._init_teacher() # Initialize teacher
            
    def _init_training_components(self):
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.distill_loss_fn = nn.MSELoss() # L2 Loss
        
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
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
        
    def _init_teacher(self):
        """Load and freeze teacher model."""
        teacher_path = getattr(self.config, 'teacher_checkpoint', None)
        if not teacher_path or not os.path.exists(teacher_path):
            raise FileNotFoundError(f"Teacher checkpoint not found at: {teacher_path}")
            
        print(f"👨‍🏫 Loading Teacher from {teacher_path}...")
        
        # Initialize Teacher with SAME architecture as Student (assuming homogeneity)
        # Using Distill class to allow feature return
        self.teacher_model = ResTranOCR_Distill(
            num_classes=self.config.num_classes,
            num_frames=self.config.num_frames,
            transformer_heads=self.config.transformer_heads,
            transformer_layers=self.config.transformer_layers,
            transformer_ff_dim=self.config.transformer_ff_dim,
            dropout=self.config.transformer_dropout,
            use_stn=self.config.use_stn,
            ctc_mid_channels=self.config.ctc_head.mid_channels,
            ctc_dropout=self.config.ctc_head.dropout,
            ctc_return_feats=False # We handle features explicitly in forward
        )
        
        checkpoint = torch.load(teacher_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Load weights
        msg = self.teacher_model.load_state_dict(state_dict, strict=False)
        print(f"   Teacher loaded with msg: {msg}")
        
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        # Freeze Teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
    def _get_output_path(self, filename: str) -> str:
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _get_exp_name(self) -> str:
        return getattr(self.config, 'experiment_name', 'distill_exp')

    def train_one_epoch(self) -> float:
        self.student_model.train()
        self.teacher_model.eval() # Ensure teacher is eval
        
        epoch_loss = 0.0
        epoch_distill_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}")
        
        # Expecting 6 items from Distill Dataset
        for lr_images, hr_images, targets, target_lengths, _, _ in pbar:
            lr_images = lr_images.to(self.device)
            hr_images = hr_images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                # 1. Teacher Forward (HR)
                with torch.no_grad():
                    _, teacher_feats_dict = self.teacher_model(hr_images, return_feats=True)
                    teacher_feat = teacher_feats_dict['transformer_out'] # [B, T', C]
                
                # 2. Student Forward (LR)
                student_logits, student_feats_dict = self.student_model(lr_images, return_feats=True)
                student_feat = student_feats_dict['transformer_out'] # [B, T', C]
                
                # 3. Losses
                # CTC Loss
                preds_permuted = student_logits.permute(1, 0, 2)
                input_lengths = torch.full((lr_images.size(0),), student_logits.size(1), dtype=torch.long)
                loss_ctc = self.criterion(preds_permuted, targets, input_lengths, target_lengths)
                
                # Distillation Loss (L2)
                # Ensure shapes match
                loss_distill = self.distill_loss_fn(student_feat, teacher_feat)
                
                w_distill = getattr(self.config, 'distill_weight', 1.0)
                total_loss = loss_ctc + (w_distill * loss_distill)

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), getattr(self.config, 'grad_clip', 10.0))
            
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
            
            epoch_loss += total_loss.item()
            epoch_distill_loss += loss_distill.item()
            
            pbar.set_postfix(
                ctc=f"{loss_ctc.item():.3f}", 
                distill=f"{loss_distill.item():.4f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}"
            )        
            
        return epoch_loss / len(self.train_loader)

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        """Validation (Standard - on LR val set)."""
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0}, []
        
        self.student_model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle different batch structures (standard vs distill)
                if len(batch) == 6:
                     # If using distill dataset for val, ignore HR
                     images, _, targets, target_lengths, labels_text, track_ids = batch
                else:
                     images, targets, target_lengths, labels_text, track_ids = batch
                
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                preds = self.student_model(images) # standard forward
                input_lengths = torch.full((images.size(0),), preds.size(1), dtype=torch.long)
                
                loss = self.criterion(preds.permute(1, 0, 2), targets, input_lengths, target_lengths)
                val_loss += loss.item() * images.size(0)

                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    if pred_text == gt_text: total_correct += 1
                    submission_data.append(f"{track_ids[i]},{pred_text};{conf:.4f}")
                total_samples += len(labels_text)

        metrics = {
            'loss': val_loss / total_samples if total_samples > 0 else 0.0,
            'acc': (total_correct / total_samples) * 100 if total_samples > 0 else 0.0,
        }
        return metrics, submission_data

    def fit(self, start_epoch: int = 0, best_acc: float = 0.0):
        self.current_epoch = start_epoch
        self.best_acc = best_acc
        
        print(f"🚀 Start Distillation from epoch {start_epoch+1}/{self.config.epochs} | Best so far: {best_acc:.2f}%")
        
        for epoch in range(start_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            avg_train_loss = self.train_one_epoch()
            val_metrics, submission_data = self.validate()
            
            val_loss = val_metrics.get('loss', 0.0)
            val_acc = val_metrics.get('acc', 0.0)
            lr = self.scheduler.get_last_lr()[0]
            
            print(f"Epoch {epoch+1:03d}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr:.2e}")
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(is_best=True)
                print(f"  ⭐ New best: {val_acc:.2f}%")
                if submission_data:
                    self.save_submission(submission_data)
        
        self.save_checkpoint(is_best=False)
        print(f"\n✅ Training finished | Best Val Acc: {self.best_acc:.2f}%")

    def save_checkpoint(self, is_best: bool = False):
        exp_name = self._get_exp_name()
        filename = f"{exp_name}_best.pth" if is_best else f"{exp_name}_last.pth"
        path = self._get_output_path(filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'config': vars(self.config) if hasattr(self.config, '__dict__') else {},
        }
        torch.save(checkpoint, path)

    def save_submission(self, submission_data: List[str]) -> None:
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, 'w') as f:
            f.write("\n".join(submission_data))
