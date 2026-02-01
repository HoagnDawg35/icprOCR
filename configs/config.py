"""Configuration dataclass for the training pipeline."""
from dataclasses import dataclass, field
from typing import Dict, Optional
from sympy import false
import torch
import yaml
from pathlib import Path

@dataclass
class CTCHeadConfig:
    """Nested config cho CTCHead."""
    mid_channels: int = 1024
    dropout: float = 0.1
    return_feats: bool = False

@dataclass
class Config:
    """Training configuration with all hyperparameters."""
    
    # Experiment tracking
    model_type: str = "restran"
    experiment_name: str = "restran"
    augmentation_level: str = "full"
    use_stn: bool = True
    
    # Distillation Config
    student_model_type: Optional[str] = None
    teacher_checkpoint: Optional[str] = None
    distill_weight: float = 1.0
    distill_layer: str = "transformer"
    clean_hr_guided: bool = False
    
    # Data paths
    data_root: str = 'data/train'
    test_data_root: str = "data/test"
    val_split_file: str = "data/val_tracks.json"
    submission_file: str = "submission.txt"
    
    img_height: int = 32
    img_width: int = 128
    num_frames: int = 5
    # Character set
    chars: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Training hyperparameters
    batch_size: int = 64
    learning_rate: float = 5e-4
    epochs: int = 30
    seed: int = 42
    num_workers: int = 10
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    split_ratio: float = 0.9
    use_cudnn_benchmark: bool = False
    
    # CRNN model hyperparameters
    hidden_size: int = 256
    rnn_dropout: float = 0.25
    
    # ResTranOCR model hyperparameters
    transformer_heads: int = 8
    transformer_layers: int = 3
    transformer_ff_dim: int = 2048
    transformer_dropout: float = 0.1
    ctc_head: CTCHeadConfig = field(default_factory=CTCHeadConfig)
    
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    output_dir: str = "results"
    
    # Derived attributes (computed in __post_init__)
    char2idx: Dict[str, int] = field(default_factory=dict, init=False)
    idx2char: Dict[int, str] = field(default_factory=dict, init=False)
    num_classes: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Compute derived attributes after initialization."""
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx2char = {idx + 1: char for idx, char in enumerate(self.chars)}
        self.num_classes = len(self.chars) + 1  # +1 for blank
        
        # Set experiment_name to model_type if not specified
        if self.experiment_name == "restran" and self.model_type != "restran":
            self.experiment_name = self.model_type


def load_config_from_yaml(yaml_path: str) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        Config object with values from YAML file
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if 'ctc_head' in config_dict:
        config_dict['ctc_head'] = CTCHeadConfig(**config_dict['ctc_head'])

    # Handle device separately since it needs special treatment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Remove device from dict if present (we'll set it explicitly)
    config_dict.pop('device', None)
    
    # Create Config object with values from YAML
    config = Config(**config_dict)
    config.device = device
    
    return config


def get_default_config() -> Config:
    """Returns the default configuration."""
    return Config()


def save_config_to_yaml(config: Config, yaml_path: str):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Config object to save
        yaml_path: Path where to save the YAML file
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dict, excluding derived attributes and device
    config_dict = {
        'model_type': config.model_type,
        'experiment_name': config.experiment_name,
        'augmentation_level': config.augmentation_level,
        'use_stn': config.use_stn,
        'data_root': config.data_root,
        'test_data_root': config.test_data_root,
        'val_split_file': config.val_split_file,
        'submission_file': config.submission_file,
        'img_height': config.img_height,
        'img_width': config.img_width,
        'chars': config.chars,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'epochs': config.epochs,
        'seed': config.seed,
        'num_workers': config.num_workers,
        'weight_decay': config.weight_decay,
        'grad_clip': config.grad_clip,
        'split_ratio': config.split_ratio,
        'use_cudnn_benchmark': config.use_cudnn_benchmark,
        'hidden_size': config.hidden_size,
        'rnn_dropout': config.rnn_dropout,
        'transformer_heads': config.transformer_heads,
        'transformer_layers': config.transformer_layers,
        'transformer_ff_dim': config.transformer_ff_dim,
        'transformer_dropout': config.transformer_dropout,
        'output_dir': config.output_dir,
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


# Example usage
if __name__ == "__main__":
    # Load from YAML
    config = load_config_from_yaml("config.yaml")
    print(f"Model type: {config.model_type}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {config.device}")
    print(f"Number of classes: {config.num_classes}")
    
    # Or use default config
    default_config = get_default_config()
    
    # Save config to YAML
    save_config_to_yaml(default_config, "config_backup.yaml")