import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from configs import config, load_config_from_yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataloader import ICPR_LPR_Datatset
from src.models import ResTranOCR
from src.utils import seed_everything

from trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train/Inference Multi-Frame OCR for License Plate Recognition"
    )
    
    # ========== MODE ==========
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer", "tune"],
        default="train",
        help="Mode: 'train' to train model, 'infer' to generate submission, 'tune' for hyperparameter tuning"
    )
    
    # ========== CHECKPOINT ==========
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint (for resuming training or inference)"
    )
    
    # ========== EXPERIMENT CONFIG ==========
    parser.add_argument(
        "-n", "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: from config)"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["crnn", "restran"],
        default=None,
        help="Model architecture (default: from config)"
    )
    
    # ========== TRAINING PARAMS ==========
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config)"
    )
    
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Learning rate (default: from config)"
    )
    
    # ========== DATA ==========
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for training data (default: from config)"
    )
    
    parser.add_argument(
        "--use-validation",
        action="store_true",
        help="Use validation split during training (default: train on full dataset)"
    )
    
    # ========== MODEL CONFIG ==========
    parser.add_argument(
        "--no-stn",
        action="store_true",
        help="Disable Spatial Transformer Network"
    )
    
    parser.add_argument(
        "--use-tbsrn",
        action="store_true",
        help="Enable TBSRN Super-Resolution"
    )
    
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="LSTM hidden size for CRNN (default: from config)"
    )
    
    parser.add_argument(
        "--transformer-heads",
        type=int,
        default=None,
        help="Number of transformer attention heads (default: from config)"
    )
    
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=None,
        help="Number of transformer encoder layers (default: from config)"
    )

    parser.add_argument(
        "--hr-guided",
        action="store_true",
        help="Use HR images as guidance/labels"
    )

    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="Path to teacher checkpoint for distillation"
    )

    parser.add_argument(
        "--hr-only",
        action="store_true",
        help="Train only on high-resolution images"
    )
    
    # ========== AUGMENTATION ==========
    parser.add_argument(
        "--aug-level",
        type=str,
        choices=["full", "light"],
        default=None,
        help="Augmentation level (default: from config)"
    )
    
    # ========== OTHERS ==========
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loader workers (default: from config)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save outputs (default: results/)"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use (e.g., 0, 1)"
    )
    
    return parser.parse_args()


def load_model(config):
    """Create model based on config."""
    if config.model_type == "restran":

        model = ResTranOCR(
            num_classes=config.num_classes,
            num_frames=config.num_frames,
            transformer_heads=config.transformer_heads,
            transformer_layers=config.transformer_layers,
            transformer_ff_dim=config.transformer_ff_dim,
            dropout=config.transformer_dropout,
            use_stn=config.use_stn,
            use_tbsrn=config.use_tbsrn,
            ctc_mid_channels=config.ctc_head.mid_channels,
            ctc_dropout=config.ctc_head.dropout,
            ctc_return_feats=config.ctc_head.return_feats,
            sr_config=config.sr_config,
        )
    else:  # crnn
        model = CRNN(
            num_classes=config.num_classes,
            hidden_size=config.hidden_size,
            rnn_dropout=config.rnn_dropout,
            use_stn=config.use_stn,
        )
    
    model = model.to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model ({config.model_type}): {total_params:,} total | {trainable_params:,} trainable")
    
    return model


def load_checkpoint(checkpoint_path, model, config, load_training_state=False):
    """Load checkpoint and optionally return training state."""
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("   ✅ Model weights loaded")
    
    # Return training state if needed
    if load_training_state and 'epoch' in checkpoint:
        return {
            'start_epoch': checkpoint.get('epoch', 0) + 1,
            'best_acc': checkpoint.get('best_acc', 0.0),
            'optimizer_state': checkpoint.get('optimizer_state_dict'),
            'scheduler_state': checkpoint.get('scheduler_state_dict')
        }
    
    return None


def mode_train(args, config):
    """Training mode."""
    print("\n" + "="*70)
    print("🚀 TRAINING MODE")
    print("="*70)
    
    # Validate data path
    if not os.path.exists(config.data_root):
        print(f"❌ ERROR: Data root not found: {config.data_root}")
        sys.exit(1)
    
    # Common dataset parameters
    common_ds_params = {
        'split_ratio': config.split_ratio,
        'img_height': config.img_height,
        'img_width': config.img_width,
        'char2idx': config.char2idx,
        'val_split_file': config.val_split_file,
        'seed': config.seed,
        'augmentation_level': config.augmentation_level,
        'num_frames': config.num_frames,
        'hr_guided': config.hr_guided,
        'hr_as_clean': config.use_tbsrn, # return clean HR if using TBSRN SR
    }
    
    # Create datasets
    val_loader = None
    
    if args.use_validation:
        print("📌 Using validation split")
        train_ds = ICPR_LPR_Datatset(
            root_dir=config.data_root,
            mode='train',
            full_train=False,
            **common_ds_params
        )
        
        val_ds = ICPR_LPR_Datatset(
            root_dir=config.data_root,
            mode='val',
            **common_ds_params
        )
        
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=ICPR_LPR_Datatset.collate_fn,
                num_workers=config.num_workers,
                pin_memory=True
            )
            print(f"   → Validation set: {len(val_ds)} samples")
    else:
        print("📌 Training on FULL dataset (no validation)")
        train_ds = ICPR_LPR_Datatset(
            root_dir=config.data_root,
            mode='train',
            full_train=True,
            **common_ds_params
        )
    
    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ICPR_LPR_Datatset.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"   → Training set: {len(train_ds)} samples")
    print(f"   → Batch size: {config.batch_size}")
    print(f"   → Total batches: {len(train_loader)}")
    
    # Create model
    model = load_model(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.idx2char
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_acc = 0.0
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        training_state = load_checkpoint(
            args.checkpoint, 
            model, 
            config, 
            load_training_state=True
        )
        
        if training_state:
            start_epoch = training_state['start_epoch']
            best_acc = training_state['best_acc']
            
            # Load optimizer and scheduler states
            if training_state['optimizer_state']:
                try:
                    trainer.optimizer.load_state_dict(training_state['optimizer_state'])
                    print("   ✅ Optimizer state loaded")
                except Exception as e:
                    print(f"   ⚠️  Could not load optimizer state: {e}")
            
            if training_state['scheduler_state']:
                try:
                    trainer.scheduler.load_state_dict(training_state['scheduler_state'])
                    print("   ✅ Scheduler state loaded")
                except Exception as e:
                    print(f"   ⚠️  Could not load scheduler state: {e}")
            
            print(f"   → Resuming from epoch {start_epoch}")
            print(f"   → Best accuracy so far: {best_acc:.2f}%")
    
    # Start training
    print("\n" + "="*70)
    print(f"Training: Epoch {start_epoch+1}/{config.epochs}")
    print("="*70)
    
    trainer.fit(start_epoch=start_epoch, best_acc=best_acc)
    
    print("\n✅ Training completed!")


def mode_infer(args, config):
    """Inference mode - generate submission file."""
    print("\n" + "="*70)
    print("🔮 INFERENCE MODE - Generating Submission")
    print("="*70)
    
    # Check checkpoint
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"❌ ERROR: Checkpoint required for inference")
        if args.checkpoint:
            print(f"   Not found: {args.checkpoint}")
        sys.exit(1)
    
    # Check test data
    if not os.path.exists(config.test_data_root):
        print(f"❌ ERROR: Test data not found: {config.test_data_root}")
        sys.exit(1)
    
    # Create test dataset
    test_ds = ICPR_LPR_Datatset(
        root_dir=config.test_data_root,
        mode='val',
        num_frames=config.num_frames,
        img_height=config.img_height,
        img_width=config.img_width,
        char2idx=config.char2idx,
        seed=config.seed,
        is_test=True,
    )
    
    if len(test_ds) == 0:
        print("❌ Test dataset is empty!")
        sys.exit(1)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=ICPR_LPR_Datatset.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"   → Test set: {len(test_ds)} samples")
    print(f"   → Batch size: {config.batch_size}")
    
    # Create model
    model = load_model(config)
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model, config, load_training_state=False)
    
    # Create trainer (without train_loader)
    trainer = Trainer(
        model=model,
        train_loader=None,  # Not needed for inference
        val_loader=None,
        config=config,
        idx2char=config.idx2char,
        mode='infer'
    )
    
    # Generate submission
    output_file = f"submission_{config.experiment_name}.txt"
    trainer.predict_test(test_loader, output_filename=output_file)
    
    print(f"\n✅ Submission file generated: {output_file}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load base config
    config = load_config_from_yaml("configs/icpr.yaml")
    
    # Override config from CLI arguments
    arg_to_config = {
        'experiment_name': 'experiment_name',
        'model': 'model_type',
        'epochs': 'epochs',
        'batch_size': 'batch_size',
        'learning_rate': 'learning_rate',
        'data_root': 'data_root',
        'seed': 'seed',
        'num_workers': 'num_workers',
        'hidden_size': 'hidden_size',
        'transformer_heads': 'transformer_heads',
        'transformer_layers': 'transformer_layers',
    }
    
    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)
    
    # Special overrides
    if args.aug_level is not None:
        config.augmentation_level = args.aug_level
    
    if args.no_stn:
        config.use_stn = False
    
    if args.use_tbsrn:
        config.use_tbsrn = True
        config.hr_guided = True # Auto-enable HR guidance for SR labels
    
    if args.hr_guided:
        config.hr_guided = True
        
    if args.teacher_checkpoint:
        config.teacher_checkpoint = args.teacher_checkpoint
        
    if args.gpu is not None:
        config.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Set seed
    seed_everything(config.seed)
    
    # Print configuration
    print("\n" + "="*70)
    print("⚙️  CONFIGURATION")
    print("="*70)
    print(f"MODE            : {args.mode.upper()}")
    print(f"EXPERIMENT      : {config.experiment_name}")
    print(f"MODEL           : {config.model_type}")
    print(f"USE_STN         : {config.use_stn}")
    print(f"USE_TBSRN       : {config.use_tbsrn}")
    print(f"HR_GUIDED       : {config.hr_guided}")
    print(f"DATA_ROOT       : {config.data_root}")
    print(f"EPOCHS          : {config.epochs}")
    print(f"BATCH_SIZE      : {config.batch_size}")
    print(f"LEARNING_RATE   : {config.learning_rate}")
    print(f"DEVICE          : {config.device}")
    print(f"CHECKPOINT      : {args.checkpoint if args.checkpoint else 'None'}")
    print(f"OUTPUT_DIR      : {config.OUTPUT_DIR}")
    
    # Route to appropriate mode
    if args.mode == "train":
        mode_train(args, config)
    elif args.mode == "infer":
        mode_infer(args, config)
    elif args.mode == "tune":
        from autotune import run_tuning
        run_tuning(config, args)
    else:
        print(f"❌ Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()