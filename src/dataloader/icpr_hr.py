import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataloader import get_val_transforms

def get_hr_train_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """
    Augmentation pipeline for Teacher (HR) training.
    Includes Geometric augmentations (Rotation, Perspective, Grid)
    but EXCLUDES Quality degradations (Blur, Noise, Compression) 
    to maintain 'Clean HR' status.
    """
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        
        # Geometric Distortions (Important for robustness)
        A.OneOf([
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.05, 0.05),
                rotate=(-10, 10),
                shear=(-5, 5),
            ),
            A.Perspective(scale=(0.02, 0.08)),
            A.GridDistortion(num_steps=5, distort_limit=0.3),
        ], p=0.7),
        
        # Light Color/Contrast adjustments (optional, but good for variance)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10),
        ], p=0.3),

        # Normalization
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

class ICPR_LPR_HR_Dataset(Dataset):
    """
    Dataset for HR-Only Training.
    Loads ONLY HR images and treats them as clean targets (no degradation).
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "hr_clean", # Used to select transform
        num_frames: int = 5,
        **kwargs # Ignore extra args
    ):
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.num_frames = num_frames
        
        # Set up Transform
        if mode == 'train':
            self.transform = get_hr_train_transforms(img_height, img_width)
        else:
            self.transform = get_val_transforms(img_height, img_width)
            
        # No degradation needed for HR training

        print(f"[{mode.upper()} - HR ONLY] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print("❌ ERROR: No data found.")
            return

        train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
        selected_tracks = train_tracks if mode == 'train' else val_tracks
        
        self._index_samples(selected_tracks)
        print(f"-> Total: {len(self.samples)} HR samples.")

    def _load_or_create_split(self, all_tracks, split_ratio):
        # Use existing val split logic to maintain consistency with other experiments
        train_tracks, val_tracks = [], []
        if os.path.exists(self.val_split_file):
            try:
                with open(self.val_split_file, 'r') as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()
            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            # Basic validation
            if not val_tracks and len(all_tracks) > 10: 
                 # Fallback logic omitted for brevity, assuming split file is good or simple split
                 pass
        
        if not val_tracks:
            # Simple split if file failed
            random.Random(self.seed).shuffle(all_tracks)
            split_idx = int(len(all_tracks) * split_ratio)
            train_tracks = all_tracks[:split_idx]
            val_tracks = all_tracks[split_idx:]
            
        return train_tracks, val_tracks

    def _index_samples(self, tracks: List[str]) -> None:
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path): continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list): data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label: continue
                
                hr_files = sorted(glob.glob(os.path.join(track_path, "hr-*.png")) + glob.glob(os.path.join(track_path, "hr-*.jpg")))
                
                # STRICTLY Use HR files
                if hr_files:
                    self.samples.append({
                        'paths': hr_files,
                        'label': label,
                        'track_id': os.path.basename(track_path)
                    })
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        item = self.samples[idx]
        img_paths = item['paths']
        label = item['label']
        track_id = item['track_id']
        
        # Load HR (no degradation)
        images_tensor = self._load_sequence(img_paths)
        
        target = [self.char2idx[c] for c in label if c in self.char2idx]
        if not target: target = [0]
        
        return images_tensor, torch.tensor(target, dtype=torch.long), len(target), label, track_id

    def _load_sequence(self, img_paths: List[str]) -> torch.Tensor:
        if len(img_paths) >= self.num_frames:
            indices = torch.linspace(0, len(img_paths)-1, self.num_frames).long()
            selected_paths = [img_paths[i] for i in indices]
        else:
            selected_paths = img_paths + [img_paths[-1]] * (self.num_frames - len(img_paths))
        
        images_list = []
        for p in selected_paths:
            image = cv2.imread(p, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply Transform (Geometric/Color Aug only, no degradation)
            image = self.transform(image=image)['image']
            images_list.append(image)

        return torch.stack(images_list, dim=0)

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple:
        images, targets, target_lengths, labels_text, track_ids = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets, dim=0)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return images, targets, target_lengths, labels_text, track_ids
