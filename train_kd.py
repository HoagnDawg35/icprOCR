import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

from configs import config, load_config_from_yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports for datasets
from src.dataloader.icpr_hr import ICPR_LPR_HR_Dataset
from src.dataloader.icpr_distill import ICPR_LPR_Distill_Dataset

# Import generic model (Distill version supports both modes efficiently)
from src.models.model_distill import ResTranOCR_Distill

# Imports for trainers
# from trainer import Trainer
from trainer_distill import TrainerDistill
from src.utils import seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Training Script for HR-Only (Teacher) and Distillation (Student)")
    parser.add_argument("--mode", type=str, required=True, choices=["teacher", "student"], help="Training mode: 'teacher' (HR-only) or 'student' (Distillation)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    
    # Common overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    
    # Distillation specific
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from (Student or Teacher)")
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="Teacher checkpoint for student distillation")
    
    return parser.parse_args()

def load_model(config):
    # We use ResTranOCR_Distill for BOTH modes to ensure architecture compatibility
    # It behaves like standard ResTranOCR when return_feats=False
    model = ResTranOCR_Distill(
        num_classes=config.num_classes,
        num_frames=config.num_frames,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        transformer_ff_dim=config.transformer_ff_dim,
        dropout=config.transformer_dropout,
        use_stn=config.use_stn,
        ctc_mid_channels=config.ctc_head.mid_channels,
        ctc_dropout=config.ctc_head.dropout,
        ctc_return_feats=False 
    )
    return model

def train_teacher(args, config):
    print("\n" + "="*50)
    print("👑 MODE: TEACHER (HR-ONLY TRAINING)")
    print("="*50)
    
    # 1. Dataset: HR Only
    # Standard overrides for HR dataset
    augmentation_level = "hr_clean" # Enforce HR clean transforms
    
    train_ds = ICPR_LPR_HR_Dataset(
        root_dir=config.data_root,
        mode='train',
        split_ratio=config.split_ratio,
        img_height=config.img_height,
        img_width=config.img_width,
        char2idx=config.char2idx,
        val_split_file=config.val_split_file,
        seed=config.seed,
        augmentation_level=augmentation_level,
        num_frames=config.num_frames,
    )
    
    val_ds = ICPR_LPR_HR_Dataset(
        root_dir=config.data_root,
        mode='val',
        augmentaion_level=augmentation_level,
        img_height=config.img_height,
        img_width=config.img_width,
        char2idx=config.char2idx,
        val_split_file=config.val_split_file,
        seed=config.seed,
        num_frames=config.num_frames,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ICPR_LPR_HR_Dataset.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=ICPR_LPR_HR_Dataset.collate_fn,
        num_workers=config.num_workers
    )
    
    print(f"📊 [Teacher] Train HR Samples: {len(train_ds)} | Val HR Samples: {len(val_ds)}")
    
    # 2. Model
    model = load_model(config)
    model = model.to(config.device)
    
    # Resume if checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.checkpoint:
        print(f"📥 Resuming Teacher from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        if 'epoch' in ckpt: start_epoch = ckpt['epoch'] + 1
        if 'best_acc' in ckpt: best_acc = ckpt['best_acc']

    # 3. Trainer (Standard)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.idx2char
    )
    
    trainer.fit(start_epoch=start_epoch, best_acc=best_acc)


def train_student(args, config):
    print("\n" + "="*50)
    print("🎓 MODE: STUDENT (KNOWLEDGE DISTILLATION)")
    print("="*50)
    
    if args.teacher_ckpt:
        config.teacher_checkpoint = args.teacher_ckpt
        
    print(f"   Teacher Checkpoint: {config.teacher_checkpoint}")
    
    # 1. Dataset: Distill (LR + HR)
    common_params = {
        'split_ratio': config.split_ratio,
        'img_height': config.img_height,
        'img_width': config.img_width,
        'char2idx': config.char2idx,
        'val_split_file': config.val_split_file,
        'seed': config.seed,
        'augmentation_level': config.augmentation_level,
        'num_frames': config.num_frames,
    }

    train_ds = ICPR_LPR_Distill_Dataset(
        root_dir=config.data_root,
        mode='train',
        clean_hr_guided=True, # Always True for distillation
        **common_params
    )
    
    # Validation on LR 
    val_ds = ICPR_LPR_Distill_Dataset(
        root_dir=config.data_root,
        mode='val',
        **common_params
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ICPR_LPR_Distill_Dataset.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=ICPR_LPR_Distill_Dataset.collate_fn,
        num_workers=config.num_workers
    )
    
    print(f"📊 [Student] Train Pairs: {len(train_ds)} | Val Samples: {len(val_ds)}")
    
    # 2. Model
    model = load_model(config)
    model = model.to(config.device)
    
    # Resume
    start_epoch = 0
    best_acc = 0.0
    if args.checkpoint:
        print(f"📥 Resuming Student from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        if 'epoch' in ckpt: start_epoch = ckpt['epoch'] + 1
        if 'best_acc' in ckpt: best_acc = ckpt['best_acc']

    # 3. Trainer (Distill)
    trainer = TrainerDistill(
        student_model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.idx2char
    )
    
    trainer.fit(start_epoch=start_epoch, best_acc=best_acc)


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    
    # Apply global args
    if args.epochs: config.epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    seed_everything(config.seed)
    
    if args.mode == "teacher":
        train_teacher(args, config)
    elif args.mode == "student":
        train_student(args, config)

if __name__ == "__main__":
    main()
