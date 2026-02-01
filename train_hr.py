import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

from configs import config, load_config_from_yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataloader.icpr_hr import ICPR_LPR_HR_Dataset
from src.models import ResTranOCR
from src.utils import seed_everything
from trainer import Trainer # Can reuse standard trainer, it works fine for standard supervision

def parse_args():
    parser = argparse.ArgumentParser(description="Train HR-Only (Teacher)")
    parser.add_argument("--config", type=str, default="configs/icpr_hr.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    
    # Overrides
    if args.epochs: config.epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    
    config.OUTPUT_DIR = "results"
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    seed_everything(config.seed)
    
    print("="*50)
    print("👑 HR-ONLY TRAINING (TEACHER)")
    print(f"   Config: {args.config}")
    print("="*50)

    # 1. Dataset (HR Only)
    train_ds = ICPR_LPR_HR_Dataset(
        root_dir=config.data_root,
        mode='train',
        split_ratio=config.split_ratio,
        img_height=config.img_height,
        img_width=config.img_width,
        char2idx=config.char2idx,
        val_split_file=config.val_split_file,
        seed=config.seed,
        augmentation_level="hr_clean", # Uses custom transform in icpr_hr.py
        num_frames=config.num_frames,
    )
    
    # Validation (Optional: Can reuse common Logic or HR, essentially same)
    val_ds = ICPR_LPR_HR_Dataset(
        root_dir=config.data_root,
        mode='val',
        split_ratio=config.split_ratio,
        img_height=config.img_height,
        img_width=config.img_width,
        char2idx=config.char2idx,
        val_split_file=config.val_split_file,
        seed=config.seed,
        num_frames=config.num_frames,
    )
    
    if len(train_ds) == 0:
        print("❌ Error: No HR samples found. Check data path.")
        sys.exit(1)
    
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
    
    print(f"📊 Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")

    # 2. Model (Standard ResTranOCR)
    model = ResTranOCR(
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
    
    # 3. Trainer (Standard)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.idx2char
    )
    
    # 4. Start
    trainer.fit()

if __name__ == "__main__":
    main()
