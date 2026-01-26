import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.heads import CTCHead
from src.models.backbones import ResNetFeatureExtractor
from src.models.necks import STNBlock, AttentionFusion, PositionalEncoding


class ResTranOCR(nn.Module):
    """
    Modern OCR architecture using optional STN, ResNet34 and Transformer.
    Pipeline: Input (5 frames) -> [Optional STN] -> ResNet34 -> Attention Fusion -> Transformer -> CTC Head
    """
    def __init__(
        self,
        num_classes: int,
        num_frames: int = 5,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        ctc_mid_channels: int = None,  
        ctc_return_feats: bool = False,
        ctc_dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.num_frames = num_frames
        # 1. Spatial Transformer Network
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone: ResNet34
        self.backbone = ResNetFeatureExtractor(pretrained=False)
        
        # 3. Attention Fusion
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
        # 4. Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 5. Prediction Head
        # self.head = nn.Linear(self.cnn_channels, num_classes)
        self.ctc_head = CTCHead(
            in_channels=self.cnn_channels,
            out_channels=num_classes,
            mid_channels=ctc_mid_channels,      
            return_feats=ctc_return_feats,
            dropout=ctc_dropout
        )

        # 6. Upsampler
        self.upsampler = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x: torch.Tensor, return_feats: bool = False) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
            return_feats: If True, return (features, logits)
        Returns:
            Logits: [Batch, Seq_Len, Num_Classes] (or (features, logits) if return_feats=True)
        """
        b, f, c, h, w = x.size()
        assert f == self.num_frames, f"Expected {self.num_frames} frames, got {f}"

        x = x.contiguous()
        x_flat = x.view(b * f, c, h, w)  # [B*F, C, H, W]
        
        if self.use_stn:
            theta = self.stn(x_flat)  # [B*F, 2, 3]
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat
        # x_aligned = self.upsampler(x_aligned)  # Upsample input frames

        features = self.backbone(x_aligned)  # [B*F, 512, 1, W']
        fused = self.fusion(features, self.num_frames)     # [B, 512, 1, W']
        
        # Prepare for Transformer: [B, C, 1, W'] -> [B, W', C]
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        # Add Positional Encoding and pass through Transformer
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) # [B, W', C]
        
        out = self.ctc_head(seq_out)              # [B, W', Num_Classes]
        return out.log_softmax(2)