import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

from configs import config, load_config_from_yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataloader.icpr_distill import ICPR_LPR_Distill_Dataset
from src.models.model_distill import ResTranOCR_Distill
from src.utils import seed_everything
from trainer_distill import TrainerDistill

def parse_args():
    parser = argparse.ArgumentParser(description="Train Teacher-Student KD")
    parser.add_argument("--config", type=str, default="configs/icpr_distill.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume Student from checkpoint")
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="Override teacher checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()

def load_student_model(config):
    # Student is always ResTranOCR_Distill
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
    model = model.to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎓 Student Model: {total_params:,} parameters")
    return model

def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    
    # Overrides
    if args.teacher_ckpt: config.teacher_checkpoint = args.teacher_ckpt
    if args.epochs: config.epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    
    config.OUTPUT_DIR = "results"
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    seed_everything(config.seed)
    
    print("="*50)
    print("🧪 KNOWLEDGE DISTILLATION TRAINING")
    print(f"   Teacher: {config.teacher_checkpoint}")
    print(f"   Student Config: {args.config}")
    print("="*50)

    # 1. Dataset (Distill version)
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

    # Train w/ Clean HR Guided
    train_ds = ICPR_LPR_Distill_Dataset(
        root_dir=config.data_root,
        mode='train',
        clean_hr_guided=getattr(config, 'clean_hr_guided', True),
        **common_params
    )
    
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
    
    print(f"📊 Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")

    # 2. Student Model
    student_model = load_student_model(config)
    
    # Optional: Load Student Checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.checkpoint:
        print(f"📥 Loading Student Checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=config.device)
        student_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            best_acc = ckpt.get('best_acc', 0.0)

    # 3. Trainer
    trainer = TrainerDistill(
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.idx2char
    )
    
    # 4. Start
    trainer.fit(start_epoch=start_epoch, best_acc=best_acc)

if __name__ == "__main__":
    main()
