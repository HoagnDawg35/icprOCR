import os
import yaml
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src.models import ResTranOCR
from src.dataloader.icpr import ICPR_LPR_Datatset
from src.utils.post_proc import decode_with_confidence

def denormalize(tensor):
    """Reverses the normalization: img * 0.5 + 0.5"""
    return tensor * 0.5 + 0.5

def plot_error(track_id, frames, gt_text, pred_text, conf, save_path):
    """Plots 5 frames and labels for a misrecognized track."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    plt.suptitle(f"Track: {track_id} | GT: {gt_text} | Pred: {pred_text} ({conf:.2f})", color='red', fontsize=14)
    
    for i in range(5):
        img = denormalize(frames[i]).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Frame {i+1}")
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Scan OCR model and plot validation errors")
    parser.add_argument("--checkpoint", type=str, default="results/v2.pth", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/icpr.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots (defaults to results/validation_errors_<name>)")
    parser.add_argument("--max_plots", type=int, default=50, help="Maximum number of errors to plot")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"🚀 Starting error scanning with checkpoint: {args.checkpoint}")

    # 1. Load Config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            
    config = Config(**config_dict)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {config.device}")
    
    # 2. Setup Chars
    chars = config.chars
    idx2char = {i + 1: char for i, char in enumerate(chars)}
    idx2char[0] = "" # Blank
    char2idx = {char: i + 1 for i, char in enumerate(chars)}
    num_classes = len(chars) + 1

    # 3. Load Dataset
    print("📂 Loading dataset...")
    val_dataset = ICPR_LPR_Datatset(
        root_dir=config.data_root,
        mode='val',
        img_height=config.img_height,
        img_width=config.img_width,
        char2idx=char2idx,
        val_split_file=config.val_split_file,
        num_frames=config.num_frames
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"✅ Loaded {len(val_dataset)} validation samples.")

    # 4. Initialize Model (ResTranOCR)
    print("🏗️ Initializing ResTranOCR model...")
    model = ResTranOCR(
        num_classes=num_classes,
        num_frames=config.num_frames,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        transformer_ff_dim=config.transformer_ff_dim,
        dropout=config.transformer_dropout,
        use_stn=config.use_stn,
        ctc_mid_channels=config.ctc_head['mid_channels'],
        ctc_dropout=config.ctc_head['dropout'],
        ctc_return_feats=config.ctc_head['return_feats'],
    ).to(config.device)

    # 5. Load Weight
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found at {args.checkpoint}")
        return

    print(f"📥 Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Weights loaded.")

    # 6. Create Output Dir
    if args.output_dir is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output_dir = os.path.join("results", f"validation_errors_{ckpt_name}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📂 Plotting errors to {args.output_dir}...")

    # 7. Validation Loop
    count = 0
    total_samples = 0
    total_errors = 0
    
    with torch.no_grad():
        for images, targets, target_lengths, labels_text, track_ids in tqdm(val_loader, desc="Scanning Errors"):
            images = images.to(config.device)
            preds = model(images)
            
            decoded = decode_with_confidence(preds, idx2char)
            pred_text, conf = decoded[0]
            gt_text = labels_text[0]
            track_id = track_ids[0]
            
            total_samples += 1

            if pred_text != gt_text:
                total_errors += 1
                if count < args.max_plots:
                    save_path = os.path.join(args.output_dir, f"{track_id}_error.png")
                    plot_error(track_id, images[0], gt_text, pred_text, conf, save_path)
                    count += 1
                
    accuracy = (1 - total_errors / total_samples) * 100 if total_samples > 0 else 0
    print(f"\n📊 Summary:")
    print(f"   - Total Samples: {total_samples}")
    print(f"   - Total Errors : {total_errors}")
    print(f"   - Val Accuracy : {accuracy:.2f}%")
    print(f"   - Plotted      : {count} errors to {args.output_dir}")
    print(f"🏁 Finished.")

if __name__ == "__main__":
    main()
