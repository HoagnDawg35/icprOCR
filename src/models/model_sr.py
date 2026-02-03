import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.necks.components import TBSRNBlock, Upsampler, STNBlock

class StackedSRNet(nn.Module):
    """
    Stacked TBSRN Network for Super-Resolution.
    Uses multiple TBSRN blocks followed by an upsampler.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_blocks: int = 4,
        channels: int = 64,
        scale_factor: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_pam: bool = True,
        use_cam: bool = True,
        use_stn: bool = True
    ):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Spatial Transformer Network for rectification
        self.stn = STNBlock(in_channels) if use_stn else None
        
        # Initial feature extraction (Lightweight)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Stack of TBSRN blocks
        self.blocks = nn.ModuleList([
            TBSRNBlock(
                channels=channels,
                num_heads=num_heads,
                dropout=dropout,
                use_pam=use_pam,
                use_cam=use_cam
            ) for _ in range(num_blocks)
        ])
        
        # Refinement before upsampling (Depthwise Separable)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling module
        self.upsampler = Upsampler(in_channels=channels, scale_factor=scale_factor)
        
        # Final reconstruction to RGB
        self.tail = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Low-resolution input [B, 3, H, W]
        Returns:
            High-resolution output [B, 3, H*scale, W*scale]
        """
        identity = x
        
        # 1. Spatial alignment/rectification
        if self.stn is not None:
            theta = self.stn(x)
            grid = F.affine_grid(theta, x.size(), align_corners=True)
            x = F.grid_sample(x, grid, align_corners=True)

        # 2. Initial features
        feat = self.head(x)
        
        # 3. Stack of TBSRN blocks with global skip connection
        res_feat = feat
        for block in self.blocks:
            feat = block(feat)
        feat = feat + res_feat
            
        # 4. Refine and upsample
        feat = self.refine(feat)
        feat = self.upsampler(feat)
        
        # 5. Reconstruct RGB
        out = self.tail(feat)
        
        # 6. Global Residual Learning (Add upsampled input)
        # We upsample the potentially aligned input x
        identity_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        out = out + identity_up
        
        return out
