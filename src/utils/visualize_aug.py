import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from src.dataloader import get_train_transforms
import torch

def visualize_augmentations(image_path, num_samples=10, output_path="aug_visualization.png"):
    if not os.path.exists(image_path):
        print(f"File {image_path} not found.")
        return

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running on: {device}")

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get transforms
    h, w = 32, 128
    transforms = get_train_transforms(img_height=h, img_width=w)
    
    plt.figure(figsize=(15, 12))
    
    # Original
    plt.subplot(num_samples + 1, 1, 1)
    plt.imshow(cv2.resize(image, (w, h)))
    plt.title(f"Original (Device: {device})")
    plt.axis('off')
    
    for i in range(num_samples):
        # Apply transforms (CPU)
        augmented = transforms(image=image)['image']
        
        # Move to device (as requested)
        augmented = augmented.to(device)
        
        # Move back to CPU for visualization
        aug_np = augmented.permute(1, 2, 0).cpu().numpy()
        aug_np = (aug_np * 0.5 + 0.5) * 255
        aug_np = aug_np.astype(np.uint8)
        
        plt.subplot(num_samples + 1, 1, i + 2)
        plt.imshow(aug_np)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved visualization to {output_path}")

if __name__ == "__main__":
    # Find a sample image
    import glob
    samples = glob.glob("data/train/**/*.jpg", recursive=True) + glob.glob("data/train/**/*.png", recursive=True)
    if samples:
        visualize_augmentations(samples[0])
    else:
        print("No sample images found in data/train")
