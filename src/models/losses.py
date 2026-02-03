import torch
import torch.nn as nn
import torch.nn.functional as F

class SRMetricLoss(nn.Module):
    """
    Combined loss for Super-Resolution.
    Includes MSE, simple Edge loss, and a placeholder for Perceptual loss.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.mse = nn.MSELoss()
        
        # Sobel filters for edge loss
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)

    def forward(self, sr_out, hr_targets):
        """
        Args:
            sr_out: [B, 3, H, W]
            hr_targets: [B, 3, H, W]
        Returns:
            mse, perceptual, edge
        """
        # 1. MSE Loss
        mse_loss = self.mse(sr_out, hr_targets)
        
        # 2. Perceptual Loss (Placeholder - using scaled MSE or small constant if we don't have VGG)
        # In a real scenario, this would use a pretrained VGG19.
        # Here we use a slightly different feature-based comparison if possible, or just MSE for now.
        perceptual_loss = self.mse(F.avg_pool2d(sr_out, 2), F.avg_pool2d(hr_targets, 2))
        
        # 3. Edge Loss (Sobel based)
        sr_gray = torch.mean(sr_out, dim=1, keepdim=True)
        hr_gray = torch.mean(hr_targets, dim=1, keepdim=True)
        
        sr_grad_x = F.conv2d(sr_gray, self.sobel_x, padding=1)
        sr_grad_y = F.conv2d(sr_gray, self.sobel_y, padding=1)
        
        hr_grad_x = F.conv2d(hr_gray, self.sobel_x, padding=1)
        hr_grad_y = F.conv2d(hr_gray, self.sobel_y, padding=1)
        
        edge_loss = self.mse(sr_grad_x, hr_grad_x) + self.mse(sr_grad_y, hr_grad_y)
        
        return mse_loss, perceptual_loss, edge_loss
