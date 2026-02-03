import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.heads import CTCHead
from src.models.backbones import ResNetFeatureExtractor, ConvNeXtFeatureExtractor
from src.models.necks import STNBlock, AttentionFusion, PositionalEncoding, BiFPNFusion, TemporalConvFusion, MultiHeadCA, ResBlock, Upsampler, TBSRNBlock
from src.models.model_sr import StackedSRNet

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
        use_tbsrn: bool = False,
        ctc_mid_channels: int = None,  
        ctc_return_feats: bool = False,
        ctc_dropout: float = 0.1,
        sr_config: dict = None,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.use_tbsrn = use_tbsrn
        self.num_frames = num_frames
        # 1. Spatial Transformer Network
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone: ResNet34
        self.backbone = ResNetFeatureExtractor(pretrained=False)
        # self.backbone = ConvNeXtFeatureExtractor(
        #     model_name='convnext_tiny',
        #     pretrained=False,
        # )
        
        
        # 3. TBSRN (Transformer-Based Super-Resolution Network) [Optional]
        if self.use_tbsrn:
            # Feature-level TBSRN (as originally in code)
            self.tbsrn_neck = TBSRNBlock(channels=self.cnn_channels)
            
            # Image-level Super-Resolution Network
            sr_config = sr_config or {}
            self.sr_net = StackedSRNet(
                in_channels=3,
                num_blocks=sr_config.get('num_blocks', 4),
                channels=sr_config.get('channels', 64),
                scale_factor=sr_config.get('scale_factor', 2),
                num_heads=sr_config.get('num_heads', 4),
                dropout=sr_config.get('dropout', 0.1),
                use_pam=sr_config.get('use_pam', True),
                use_cam=sr_config.get('use_cam', True)
            )

        # 4. Attention Fusion
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        # self.fusion = BiFPNFusion(channels=self.cnn_channels)
        # self.fusion = TemporalConvFusion(channels=self.cnn_channels)
        # 5. Transformer Encoder
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
        
        # 6. Prediction Head
        # self.head = nn.Linear(self.cnn_channels, num_classes)
        self.ctc_head = CTCHead(
            in_channels=self.cnn_channels,
            out_channels=num_classes,
            mid_channels=ctc_mid_channels,      
            return_feats=ctc_return_feats,
            dropout=ctc_dropout
        )

        # 7. Feature Refiner
        self.feature_refiner = nn.Sequential(
            ResBlock(self.cnn_channels),
            ResBlock(self.cnn_channels),
            MultiHeadCA(self.cnn_channels)
        )

    def forward(self, x: torch.Tensor, return_sr: bool = False) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
        Returns:
            Logits: [Batch, Seq_Len, Num_Classes]
        """
        b, f, c, h, w = x.size()
        assert f == self.num_frames, f"Expected {self.num_frames} frames, got {f}"

        x = x.contiguous()

        
        x_flat = x.view(b * f, c, h, w)    # [B*F, C, H, W]
        if self.use_stn:
            theta = self.stn(x_flat)  # [B*F, 2, 3]
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat
        
        features = self.backbone(x_aligned)  # [B*F, 512, H', W']
        
        
        # 1. Spatial Refinement with TBSRN
        if self.use_tbsrn:
            features = self.tbsrn_neck(features)
            
        features = self.feature_refiner(features)
        fused = self.fusion(features, self.num_frames) # [B, 512, 1, W']

    
        # Prepare for Transformer: [B, C, 1, W'] -> [B, W', C]
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        # Add Positional Encoding and pass through Transformer
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) # [B, W', C]
        
        out = self.ctc_head(seq_out)  
        
        if return_sr and self.use_tbsrn:
            # Apply SR to input frames
            # Reshape x to [B*F, C, H, W] if not already
            sr_out = self.sr_net(x_flat) # [B*F, 3, H*S, W*S]
            return out.log_softmax(2), sr_out
            
        return out.log_softmax(2)
