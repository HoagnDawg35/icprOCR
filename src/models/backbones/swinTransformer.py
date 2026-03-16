import torch
import torch.nn as nn
import timm


class SwinTransFeatureExtractor(nn.Module):
    """
    Swin Transformer feature extractor for OCR.
    Output shape: [B, 512, 1, W']
    """

    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        pretrained=True,
        out_channels=512
    ):
        super().__init__()

        # Load Swin Transformer
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )

        # Last stage channels
        in_channels = self.backbone.feature_info[-1]["num_chs"]

        # Channel projection
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Collapse height to 1 (OCR style)
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):

        # x: [B,3,H,W]

        features = self.backbone(x)

        # last feature map
        x = features[-1]

        # channel projection
        x = self.proj(x)

        # collapse height
        x = self.height_pool(x)

        return x