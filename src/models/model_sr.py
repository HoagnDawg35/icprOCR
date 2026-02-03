import torch
import torch.nn as nn
from src.models.necks.components import TBSRNBlock, Upsampler

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
        use_cam: bool = True
    ):
        super().__init__()
        
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
        # Initial features
        feat = self.head(x)
        
        # Process through TBSRN blocks
        for block in self.blocks:
            feat = block(feat)
            
        # Refine and upsample
        feat = self.refine(feat)
        feat = self.upsampler(feat)
        
        # Reconstruct RGB
        out = self.tail(feat)
        
        return out
