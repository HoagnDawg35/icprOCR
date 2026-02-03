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
import json
import Levenshtein
import seaborn as sns

from src.models import ResTranOCR
from src.dataloader.icpr import ICPR_LPR_Datatset
from src.utils.post_proc import decode_with_confidence
from configs.config import load_config_from_yaml

def denormalize(tensor):
    """Reverses the normalization: img * 0.5 + 0.5"""
    return tensor * 0.5 + 0.5

def calculate_psnr(pred, gt):
    """Calculates PSNR between two tensors [C, H, W] or [B, C, H, W]."""
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    # Assuming pixels are in range [0, 1] after denormalization
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def plot_error(track_id, frames, gt_text, pred_text, conf, save_path, sr_frames=None):
    """Plots original frames and optionally SR frames for a misrecognized track."""
    num_rows = 2 if sr_frames is not None else 1
    fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3 * num_rows))
    plt.suptitle(f"Track: {track_id} | GT: {gt_text} | Pred: {pred_text} ({conf:.2f})", color='red', fontsize=14)
    
    # Ensure axes is always 2D
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(5):
        # Plot Original
        img = denormalize(frames[i]).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Original F{i+1}")

        # Plot SR if available
        if sr_frames is not None:
            sr_img = denormalize(sr_frames[i]).permute(1, 2, 0).cpu().numpy()
            sr_img = np.clip(sr_img, 0, 1)
            axes[1, i].imshow(sr_img)
            axes[1, i].axis('off')
            axes[1, i].set_title(f"SR F{i+1}")
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_alignment_ops(gt, pred):
    """Returns Levenshtein edit operations for GT and Pred."""
    return Levenshtein.editops(gt, pred)

def update_confusion_matrix(cm, gt, pred, chars):
    """Updates the character confusion matrix based on alignment."""
    ops = get_alignment_ops(gt, pred)
    
    # We use a special character '-' for insertions/deletions (epsilon)
    # cm is a dict of dicts: cm[gt_char][pred_char] = count
    
    # To track matches and substitutions easily, we can use a simpler approach:
    # 1. Start with full GT matched to nothing.
    # 2. Apply ops to find substitutions, deletions, and insertions.
    
    gt_list = list(gt)
    pred_list = list(pred)
    
    # Track used indices to identify matches
    gt_used = [False] * len(gt)
    pred_used = [False] * len(pred)
    
    for op, i, j in ops:
        if op == 'replace':
            char_gt = gt[i]
            char_pred = pred[j]
            cm[char_gt][char_pred] += 1
            gt_used[i] = True
            pred_used[j] = True
        elif op == 'delete':
            char_gt = gt[i]
            cm[char_gt]['<DEL>'] += 1
            gt_used[i] = True
        elif op == 'insert':
            char_pred = pred[j]
            cm['<INS>'][char_pred] += 1
            pred_used[j] = True
            
    # Matches
    for i in range(len(gt)):
        if not gt_used[i]:
            char_gt = gt[i]
            cm[char_gt][char_gt] += 1

def plot_confusion_matrix(cm, chars, save_path):
    """Plots the confusion matrix as a heatmap."""
    # Add special tokens to chars list for plotting
    plot_chars = sorted(list(chars))
    all_chars = plot_chars + ['<DEL>', '<INS>']
    
    # Initialize matrix
    matrix = np.zeros((len(all_chars), len(all_chars)))
    
    for i, c1 in enumerate(all_chars):
        for j, c2 in enumerate(all_chars):
            if c1 in cm and c2 in cm[c1]:
                matrix[i, j] = cm[c1][c2]
                
    # Filter out empty rows/cols to make it readable
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=2 if len(matrix.shape)>2 else 0) # wait matrix is 2D
    col_sums = matrix.sum(axis=0)
    
    mask_rows = row_sums > 0
    mask_cols = col_sums > 0
    mask = mask_rows | mask_cols
    
    final_chars = [all_chars[i] for i in range(len(all_chars)) if mask[i]]
    final_matrix = matrix[mask][:, mask]
    
    if final_matrix.size == 0:
        print("⚠️ Confusion matrix is empty, skipping plot.")
        return

    plt.figure(figsize=(20, 16))
    sns.heatmap(final_matrix, annot=True, fmt='.0f', cmap='YlGnBu',
                xticklabels=final_chars, yticklabels=final_chars)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Character-wise Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")

def plot_enhanced_heatmaps(cm, chars, output_dir):
    """Plots normalized and error-only heatmaps."""
    plot_chars = sorted(list(chars))
    all_chars = plot_chars + ['<DEL>', '<INS>']
    
    # Initialize matrix
    matrix = np.zeros((len(all_chars), len(all_chars)))
    for i, c1 in enumerate(all_chars):
        for j, c2 in enumerate(all_chars):
            if c1 in cm and c2 in cm[c1]:
                matrix[i, j] = cm[c1][c2]
                
    # Filter empty rows/cols
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    mask = (row_sums > 0) | (col_sums > 0)
    
    final_chars = [all_chars[i] for i in range(len(all_chars)) if mask[i]]
    final_matrix = matrix[mask][:, mask]
    
    if final_matrix.size == 0:
        return

    # 1. Normalized Heatmap
    # Normalize by row (Ground Truth totals)
    row_totals = final_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_totals[row_totals == 0] = 1
    norm_matrix = (final_matrix / row_totals) * 100
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(norm_matrix, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=final_chars, yticklabels=final_chars)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Normalized Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "char_confusion_matrix_norm.png"))
    plt.close()
    print(f"✅ Normalized confusion matrix saved.")

    # 2. Errors-only Heatmap (Mask Diagonal)
    error_matrix = final_matrix.copy()
    np.fill_diagonal(error_matrix, 0)
    
    # Filter again for entries that actually have errors
    err_row_sums = error_matrix.sum(axis=1)
    err_col_sums = error_matrix.sum(axis=0)
    err_mask = (err_row_sums > 0) | (err_col_sums > 0)
    
    err_chars = [final_chars[i] for i in range(len(final_chars)) if err_mask[i]]
    err_matrix_final = error_matrix[err_mask][:, err_mask]
    
    if err_matrix_final.size > 0:
        plt.figure(figsize=(20, 16))
        sns.heatmap(err_matrix_final, annot=True, fmt='.0f', cmap='OrRd',
                    xticklabels=err_chars, yticklabels=err_chars)
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('Character-wise Error Confusions (Diagonal Hidden)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "char_error_heatmap.png"))
        plt.close()
        print(f"✅ Error-only heatmap saved.")

def print_top_errors(cm, top_n=10):
    """Prints a summary of the most frequent character substitutions."""
    errors = []
    for gt_char, preds in cm.items():
        for pred_char, count in preds.items():
            if gt_char != pred_char and count > 0:
                errors.append((gt_char, pred_char, count))
    
    # Sort by count
    errors.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n📢 Top {top_n} Character Errors:")
    print(f"{'GT':<8} -> {'Pred':<8} : {'Count':<6}")
    print("-" * 30)
    for gt, pred, count in errors[:top_n]:
        print(f"{gt:<8} -> {pred:<8} : {count:<6}")
    print("")

def print_character_stats(cm, chars, output_dir=None):
    """Prints a detailed table of character-wise statistics and saves to CSV."""
    plot_chars = sorted(list(chars))
    
    # Prepare CSV data
    csv_rows = ["Char,Total,Correct,Sub,Del,Ins,Err%"]
    
    print("\n📊 Character-wise Error Statistics:")
    header = f"{'Char':<6} | {'Total':<6} | {'Correct':<8} | {'Sub':<6} | {'Del':<6} | {'Ins':<6} | {'Err%':<6}"
    print(header)
    print("-" * len(header))
    
    total_gt_all = 0
    total_correct_all = 0
    total_sub_all = 0
    total_del_all = 0
    total_ins_all = 0
    
    for c in plot_chars:
        # Total GT occurrences = Matches + Substitutions + Deletions
        correct = cm[c][c]
        deleted = cm[c]['<DEL>']
        substituted = sum(count for pred, count in cm[c].items() if pred != c and pred != '<DEL>' and pred != '<INS>')
        total_gt = correct + deleted + substituted
        
        inserted = cm['<INS>'][c]
        
        error_rate = ((total_gt - correct + inserted) / total_gt * 100) if total_gt > 0 else 0
        
        if total_gt > 0 or inserted > 0:
            print(f"{c:<6} | {total_gt:<6} | {correct:<8} | {substituted:<6} | {deleted:<6} | {inserted:<6} | {error_rate:>5.1f}%")
            csv_rows.append(f"{c},{total_gt},{correct},{substituted},{deleted},{inserted},{error_rate:.1f}")
            
            total_gt_all += total_gt
            total_correct_all += correct
            total_sub_all += substituted
            total_del_all += deleted
            total_ins_all += inserted

    # Print Insertions that don't map to a specific character if any (though cm['<INS>'] handles it)
    
    print("-" * len(header))
    total_err_rate = ((total_gt_all - total_correct_all + total_ins_all) / total_gt_all * 100) if total_gt_all > 0 else 0
    print(f"{'TOTAL':<6} | {total_gt_all:<6} | {total_correct_all:<8} | {total_sub_all:<6} | {total_del_all:<6} | {total_ins_all:<6} | {total_err_rate:>5.1f}%")
    csv_rows.append(f"TOTAL,{total_gt_all},{total_correct_all},{total_sub_all},{total_del_all},{total_ins_all},{total_err_rate:.1f}")
    print("")

    if output_dir:
        csv_path = os.path.join(output_dir, "char_stats.csv")
        with open(csv_path, 'w') as f:
            f.write("\n".join(csv_rows))
        print(f"✅ Character statistics saved to {csv_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Scan OCR model and plot validation errors")
    parser.add_argument("--checkpoint", type=str, default="results/v2.pth", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/icpr.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots (defaults to results/validation_errors_<name>)")
    parser.add_argument("--max_plots", type=int, default=50, help="Maximum number of errors to plot")
    parser.add_argument("--plot_cm", type=bool, default=True, help="Whether to plot confusion matrix")
    parser.add_argument("--gt_json", type=str, default="data/ground_truth.json", help="Path to ground truth JSON")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"🚀 Starting error scanning with checkpoint: {args.checkpoint}")

    # 1. Load Config
    config = load_config_from_yaml(args.config)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {config.device}")
    
    # 2. Setup Chars
    chars = config.chars
    idx2char = config.idx2char
    idx2char[0] = "" # Blank
    char2idx = config.char2idx
    num_classes = config.num_classes

    # 2.b Load Ground Truth JSON
    gt_data = {}
    if os.path.exists(args.gt_json):
        print(f"📂 Loading ground truth labels from {args.gt_json}...")
        with open(args.gt_json, 'r') as f:
            gt_data = json.load(f)
    
    # Initialize character confusion matrix
    # Format: cm[gt_char][pred_char] = count
    from collections import defaultdict
    cm = defaultdict(lambda: defaultdict(int))
    all_chars_with_special = list(chars) + ['<DEL>', '<INS>']
    for c1 in all_chars_with_special:
        for c2 in all_chars_with_special:
            cm[c1][c2] = 0

    # 3. Load Checkpoint first to detect architecture
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found at {args.checkpoint}")
        return

    print(f"📥 Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 4. Initialize Model (ResTranOCR)
    # Auto-detect SR from checkpoint
    use_tbsrn = getattr(config, 'use_tbsrn', False)
    has_sr_keys = any(k.startswith('tbsrn_neck') or k.startswith('sr_net') for k in state_dict.keys())
    
    if has_sr_keys and not use_tbsrn:
        print("💡 Auto-detected SR weights in checkpoint. Enabling SR support...")
        use_tbsrn = True
    elif not has_sr_keys and use_tbsrn:
        print("⚠️ SR enabled in config but not found in checkpoint. Disabling SR support...")
        use_tbsrn = False

    model = ResTranOCR(
        num_classes=num_classes,
        num_frames=config.num_frames,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        transformer_ff_dim=config.transformer_ff_dim,
        dropout=config.transformer_dropout,
        use_stn=config.use_stn,
        use_tbsrn=use_tbsrn,
        sr_config=config.sr_config,
        ctc_mid_channels=config.ctc_head.mid_channels,
        ctc_dropout=config.ctc_head.dropout,
        ctc_return_feats=config.ctc_head.return_feats,
    ).to(config.device)

    # 5. Load Dataset
    print("📂 Loading dataset...")
    val_dataset = ICPR_LPR_Datatset(
        root_dir=config.data_root,
        mode='val',
        img_height=config.img_height,
        img_width=config.img_width,
        char2idx=char2idx,
        val_split_file=config.val_split_file,
        num_frames=config.num_frames,
        hr_guided=use_tbsrn, # Now use_tbsrn is defined!
        hr_as_clean=True
    )
    
    collate_fn = val_dataset.collate_fn
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"✅ Loaded {len(val_dataset)} validation samples.")

    # 6. Load Weights
    print("📥 Loading weights into model...")
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        print(f"⚠️ Missing keys: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"⚠️ Unexpected keys: {len(msg.unexpected_keys)}")
    
    model.eval()
    print("✅ Model ready.")

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
    psnr_scores = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Scanning Errors"):
            # Unpack batch based on hr_guided mode
            if len(batch) == 6:
                images, hr_images, targets, target_lengths, labels_text, track_ids = batch
                hr_images = hr_images.to(config.device)
            else:
                images, targets, target_lengths, labels_text, track_ids = batch
                hr_images = None

            images = images.to(config.device)
            
            # Forward pass: request SR if enabled
            if use_tbsrn:
                preds, sr_out = model(images, return_sr=True)
            else:
                preds = model(images)
                sr_out = None
            
            decoded = decode_with_confidence(preds, idx2char)
            pred_text, conf = decoded[0]
            track_id = track_ids[0]

            # Calculate PSNR if SR available
            if sr_out is not None and hr_images is not None:
                # Denormalize both for PSNR [B*F, C, H, W]
                sr_denorm = denormalize(sr_out)
                # Reshape hr_images to match sr_out: [B, F, C, H_SR, W_SR] -> [B*F, C, H_SR, W_SR]
                b, f, c, h, w = hr_images.shape
                hr_flat = hr_images.view(b * f, c, h, w)
                hr_denorm = denormalize(hr_flat)
                
                score = calculate_psnr(sr_denorm, hr_denorm)
                if score != float('inf'):
                    psnr_scores.append(score)
            
            # Prefer ground_truth.json label if available
            gt_text = gt_data.get(track_id, labels_text[0])
            
            total_samples += 1

            # Update character confusion matrix
            update_confusion_matrix(cm, gt_text, pred_text, chars)

            if pred_text != gt_text:
                total_errors += 1
                if count < args.max_plots:
                    save_path = os.path.join(args.output_dir, f"{track_id}_error.png")
                    
                    # Prepare SR frames for plotting if available
                    current_sr_frames = None
                    if sr_out is not None:
                        # sr_out is [B*F, C, H*S, W*S]
                        _, c, h_sr, w_sr = sr_out.shape
                        current_sr_frames = sr_out.view(1, config.num_frames, c, h_sr, w_sr)[0]
                    
                    plot_error(track_id, images[0], gt_text, pred_text, conf, save_path, sr_frames=current_sr_frames)
                    count += 1
            
                
    accuracy = (1 - total_errors / total_samples) * 100 if total_samples > 0 else 0
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
    
    print(f"\n📊 Summary:")
    print(f"   - Total Samples: {total_samples}")
    print(f"   - Total Errors : {total_errors}")
    print(f"   - Val Accuracy : {accuracy:.2f}%")
    if psnr_scores:
        print(f"   - Avg SR PSNR  : {avg_psnr:.2f} dB")
    print(f"   - Plotted      : {count} errors to {args.output_dir}")

    # Plot Confusion Matrix
    if args.plot_cm:
        cm_save_path = os.path.join(args.output_dir, "char_confusion_matrix.png")
        plot_confusion_matrix(cm, chars, cm_save_path)
        plot_enhanced_heatmaps(cm, chars, args.output_dir)
        print_top_errors(cm)
        print_character_stats(cm, chars, args.output_dir)

    print("🏁 Finished.")

if __name__ == "__main__":
    main()
