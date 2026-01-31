
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from typing import Optional

class ConvNeXtFeatureExtractor(nn.Module):
    """
    ConvNeXt-based backbone customized for OCR.
    ConvNeXt is a modern pure CNN that rivals Vision Transformers.
    
    Args:
        model_name: ConvNeXt variant
            - 'convnext_tiny' (recommended - 28M params)
            - 'convnext_small' (50M params)
            - 'convnext_base' (89M params)
            - 'convnextv2_tiny', 'convnextv2_base'
        pretrained: Whether to use ImageNet pretrained weights
        out_channels: Output feature channels (default 512)
    """
    def __init__(
        self,
        model_name: str = 'convnext_tiny',
        pretrained: bool = True,
        out_channels: int = 512
    ):
        super().__init__()
        
        # Create ConvNeXt backbone
        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[3]  # Get stage 4 output (last stage)
        )
        
        # Get the output channels from the backbone
        backbone_channels = self.backbone.feature_info.channels()[-1]
        
        # --- OCR Customization ---
        self._modify_strides_for_ocr()
        
        # Project to desired output channels
        # Use BatchNorm instead of LayerNorm for dynamic spatial dimensions
        self.channel_projection = nn.Sequential(
            nn.Conv2d(backbone_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.out_channels = out_channels
    
    def _modify_strides_for_ocr(self):
        """
        Modify downsampling layers in ConvNeXt to preserve width.
        ConvNeXt uses explicit downsample layers between stages.
        """
        try:
            stages = self.backbone.stages
            
            # Modify downsample layers in later stages (stage 2, 3)
            for stage_idx in [2, 3]:
                if stage_idx < len(stages):
                    stage = stages[stage_idx]
                    # ConvNeXt has a downsample layer at the start of each stage
                    if hasattr(stage, 'downsample') and stage.downsample is not None:
                        downsample = stage.downsample
                        # Modify the convolution stride from (2, 2) to (2, 1)
                        for module in downsample.modules():
                            if isinstance(module, nn.Conv2d):
                                if module.stride == (2, 2):
                                    module.stride = (2, 1)
                                    # Adjust padding
                                    module.padding = (1, 0)
        except AttributeError:
            # If structure is different, skip modification
            pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch, 3, H, W]
        Returns:
            Features [Batch, out_channels, 1, W']
            where W' is preserved for sequence modeling
        """
        # Extract features
        features = self.backbone(x)[0]  # Get last feature map
        
        # Project channels
        features = self.channel_projection(features)
        
        # Collapse height to 1 for sequence modeling
        # Output shape: [Batch, out_channels, 1, W']
        features = F.adaptive_avg_pool2d(features, (1, None))
        
        return features
