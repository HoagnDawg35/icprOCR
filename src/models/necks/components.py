import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

class Upsampler(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(Upsampler, self).__init__()
        modules = []
        # Xử lý cho các scale factor là lũy thừa của 2 (2, 4, ...)
        if scale_factor == 2 or scale_factor == 4:
            for _ in range(int(math.log2(scale_factor))):
                modules.append(nn.Conv2d(in_channels, in_channels * 4, 3, padding=1))
                modules.append(nn.PixelShuffle(2))
                modules.append(nn.ReLU(inplace=True))
        # Xử lý cho scale factor là 3
        elif scale_factor == 3:
            modules.append(nn.Conv2d(in_channels, in_channels * 9, 3, padding=1))
            modules.append(nn.PixelShuffle(3))
            modules.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")
        
        self.upsampler = nn.Sequential(*modules)

    def forward(self, x):
        return self.upsampler(x)
    
class MultiHeadCA(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Tách channel thành nhiều head để học các nhóm đặc trưng khác nhau
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(
                channel, channel // reduction,
                kernel_size=1, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel // reduction, channel,
                kernel_size=1, padding=0, bias=True
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class ResBlock(nn.Module):
    def __init__(self, num_features, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.calayer = CALayer(num_features, reduction=16) # Có thể thay bằng Multi-head CA
        self.relu = nn.ReLU(inplace=True)
        self.res_scale = res_scale

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.calayer(out)
        out = out * self.res_scale + residual
        return out

class STNBlock(nn.Module):
    """
    Spatial Transformer Network (STN) for image alignment.
    Learns to crop and rectify images before feeding them to the backbone.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Localization network: Predicts transformation parameters
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8)) # Output fixed size for FC
        )
        
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch, C, H, W]
        Returns:
            theta: Affine transformation matrix [Batch, 2, 3]
        """
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module for combining multi-frame features.
    Computes a weighted sum of features from multiple frames based on their 'quality' scores.
    """
    def __init__(self, channels: int):
        super().__init__()
        # A small CNN to predict attention scores (quality map) from features
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
        Returns:
            Fused feature map. Shape: [Batch, C, H, W]
        """
        total_frames, c, h, w = x.size()
        batch_size = total_frames // num_frames

        # Reshape to [Batch, Frames, C, H, W]
        x_view = x.view(batch_size, num_frames, c, h, w)
        
        # Calculate attention scores: [Batch, Frames, 1, H, W]
        scores = self.score_net(x).view(batch_size, num_frames, 1, h, w)
        weights = F.softmax(scores, dim=1)  # Normalize scores across frames

        # Weighted sum fusion
        fused_features = torch.sum(x_view * weights, dim=1)
        return fused_features

class CrossFrameAttentionFusion(nn.Module):
    """
    Cross-frame attention fusion module that allows frames to attend to each other.
    This is particularly effective for low-resolution OCR where different frames 
    may capture complementary information.
    
    Pipeline: 
    1. Concat frames along channel dimension
    2. Multi-head attention across frames
    3. Fusion with residual connection
    """
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Multi-head attention for cross-frame interaction
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        
        # Frame importance scoring
        self.frame_scorer = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, 1)
        )

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: [B*F, C, H, W] - Features from all frames
            num_frames: Number of frames
        Returns:
            [B, C, H, W] - Fused features
        """
        bf, c, h, w = x.size()
        batch_size = bf // num_frames
        
        # Reshape: [B, F, C, H, W] -> [B, F, C, H*W] -> [B, F, H*W, C]
        x_reshaped = x.view(batch_size, num_frames, c, h * w)
        x_reshaped = x_reshaped.permute(0, 1, 3, 2)  # [B, F, H*W, C]
        
        # Flatten spatial dimensions: [B, F*H*W, C]
        x_flat = x_reshaped.reshape(batch_size, num_frames * h * w, c)
        
        # Self-attention across frames and spatial locations
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.multihead_attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # Feed-forward network
        x_flat = x_flat + self.ffn(self.norm2(x_flat))
        
        # Reshape back: [B, F*H*W, C] -> [B, F, H*W, C]
        x_reshaped = x_flat.view(batch_size, num_frames, h * w, c)
        
        # Global average pooling per frame: [B, F, C]
        frame_features = x_reshaped.mean(dim=2)
        
        # Score each frame's importance: [B, F, 1]
        frame_scores = self.frame_scorer(frame_features)
        frame_weights = F.softmax(frame_scores, dim=1)
        
        # Weighted combination of frames: [B, F, H*W, C] -> [B, H*W, C]
        x_weighted = (x_reshaped * frame_weights.unsqueeze(2)).sum(dim=1)
        
        # Reshape to [B, C, H, W]
        output = x_weighted.permute(0, 2, 1).view(batch_size, c, h, w)
        
        return output


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion that progressively combines frames in a tree-like structure.
    Good for capturing both local (adjacent frames) and global (all frames) dependencies.
    
    For 5 frames: (0,1) (2,3) -> merge -> (01, 23) + 4 -> final fusion
    """
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        
        # Pairwise fusion layers
        self.pair_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
        # Residual attention
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: [B*F, C, H, W]
            num_frames: Number of frames (expected 5)
        Returns:
            [B, C, H, W]
        """
        bf, c, h, w = x.size()
        batch_size = bf // num_frames
        
        # Reshape to [B, F, C, H, W]
        frames = x.view(batch_size, num_frames, c, h, w)
        
        # Level 1: Pairwise fusion
        pair1 = torch.cat([frames[:, 0], frames[:, 1]], dim=1)  # [B, 2C, H, W]
        pair1_fused = self.pair_fusion(pair1)  # [B, C, H, W]
        
        pair2 = torch.cat([frames[:, 2], frames[:, 3]], dim=1)
        pair2_fused = self.pair_fusion(pair2)
        
        middle_frame = frames[:, 4]  # [B, C, H, W]
        
        # Level 2: Combine all
        combined = torch.cat([pair1_fused, pair2_fused, middle_frame], dim=1)  # [B, 3C, H, W]
        fused = self.final_fusion(combined)  # [B, C, H, W]
        
        # Apply attention refinement
        attn_weights = self.attention(fused)
        output = fused * attn_weights + fused
        
        return output


class TemporalConvFusion(nn.Module):
    """
    Treats frames as temporal sequence and uses 1D convolutions along temporal dimension.
    Effective for capturing temporal patterns in multi-frame low-res OCR.
    """
    def __init__(self, channels: int, num_frames: int = 5, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        
        # Temporal convolution layers
        self.temporal_conv = nn.Sequential(
            # Conv along frame dimension
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
            
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )
        
        # Frame aggregation with learnable weights
        self.frame_weights = nn.Parameter(torch.ones(num_frames) / num_frames)
        
        # Spatial refinement
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: [B*F, C, H, W]
            num_frames: Number of frames
        Returns:
            [B, C, H, W]
        """
        bf, c, h, w = x.size()
        batch_size = bf // num_frames
        
        # Reshape to [B, C, F, H, W] for 3D conv
        x_3d = x.view(batch_size, num_frames, c, h, w)
        x_3d = x_3d.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        
        # Temporal convolution
        temporal_features = self.temporal_conv(x_3d)  # [B, C, F, H, W]
        
        # Weighted aggregation across frames
        temporal_features = temporal_features.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        
        # Normalize weights
        weights = F.softmax(self.frame_weights, dim=0)
        weights = weights.view(1, num_frames, 1, 1, 1)
        
        # Aggregate: [B, C, H, W]
        fused = (temporal_features * weights).sum(dim=1)
        
        # Spatial refinement
        output = self.spatial_refine(fused) + fused
        
        return output


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that dynamically selects fusion strategy based on input characteristics.
    Combines multiple fusion approaches and learns to weight them.
    """
    def __init__(self, channels: int, num_frames: int = 5, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        
        # Multiple fusion strategies
        self.avg_pool_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(32, channels),
            nn.GELU()
        )
        
        self.max_pool_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(32, channels),
            nn.GELU()
        )
        
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, num_frames, kernel_size=1)
        )
        
        # Strategy mixer
        self.strategy_mixer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # Final projection
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: [B*F, C, H, W]
            num_frames: Number of frames
        Returns:
            [B, C, H, W]
        """
        bf, c, h, w = x.size()
        batch_size = bf // num_frames
        
        frames = x.view(batch_size, num_frames, c, h, w)
        
        # Strategy 1: Average pooling
        avg_fused = self.avg_pool_fusion(frames.mean(dim=1))
        
        # Strategy 2: Max pooling
        max_fused = self.max_pool_fusion(frames.max(dim=1)[0])
        
        # Strategy 3: Attention-based
        attn_scores = self.attention_fusion(frames.mean(dim=1))  # [B, F, H, W]
        attn_scores = attn_scores.unsqueeze(2)  # [B, F, 1, H, W]
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_fused = (frames * attn_weights).sum(dim=1)
        
        # Learn strategy weights
        combined_features = torch.cat([avg_fused, max_fused, attn_fused], dim=1)
        strategy_weights = self.strategy_mixer(combined_features).unsqueeze(2).unsqueeze(3)
        
        # Mix strategies
        output = (
            strategy_weights[:, 0:1] * avg_fused +
            strategy_weights[:, 1:2] * max_fused +
            strategy_weights[:, 2:3] * attn_fused
        )
        
        output = self.output_proj(output)
        
        return output

class CNNBackbone(nn.Module):
    """A simple CNN backbone for CRNN baseline."""
    def __init__(self, out_channels=512):
        super().__init__()
        # Defined as a list of layers for clarity: Conv -> ReLU -> Pool
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 5 (Map to sequence height 1)
            nn.Conv2d(512, out_channels, 2, 1, 0), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

    def forward(self, x):
        return self.features(x)

class BiFPNFusion(nn.Module):
    """
    BiFPN-based fusion module for combining multi-frame features.
    Uses bidirectional feature propagation (top-down and bottom-up) with fast normalized fusion.
    Assumes frames are ordered temporally (e.g., frame 0 is earliest, frame 4 is latest).
    """
    def __init__(self, channels: int, num_frames: int = 5, eps: float = 1e-4):
        super().__init__()
        self.num_frames = num_frames
        self.eps = eps
        self.channels = channels

        # Learnable weights for top-down fusions (4 fusions, each with 2 inputs)
        self.w_td = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(4)])

        # Learnable weights for bottom-up fusions (4 fusions, each with 3 inputs)
        self.w_out = nn.ParameterList([nn.Parameter(torch.ones(3)) for _ in range(4)])

        # Learnable weights for final fusion of all output nodes
        self.w_final = nn.Parameter(torch.ones(num_frames))

        # Separable conv + BN + ReLU for each node (td0-4, out0-4: 10 in total)
        node_names = ['td0', 'td1', 'td2', 'td3', 'td4', 'out0', 'out1', 'out2', 'out3', 'out4']
        self.conv_bn_act = nn.ModuleDict()
        for name in node_names:
            self.conv_bn_act[name] = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),  # Depthwise
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # Pointwise
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )

    def _fast_fuse(self, inputs: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        w = F.relu(weights)
        norm_w = w / (w.sum() + self.eps)
        fused = sum(norm_w[i] * inputs[i] for i in range(len(inputs)))
        return fused

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
        Returns:
            Fused feature map. Shape: [Batch, C, H, W]
        """
        assert num_frames == self.num_frames, f"Expected {self.num_frames} frames, got {num_frames}"

        total_frames, c, h, w = x.size()
        batch_size = total_frames // num_frames

        # Reshape to [Batch, Frames, C, H, W]
        x_view = x.view(batch_size, num_frames, c, h, w)

        # Input features per frame
        ins = [x_view[:, i] for i in range(num_frames)]  # List of [B, C, H, W]

        # Top-down path
        td = [None] * num_frames
        td[4] = self.conv_bn_act['td4'](ins[4])  # Starting top node

        # td[3] = fuse(ins[3], td[4]) -> conv_bn_act
        fused = self._fast_fuse([ins[3], td[4]], self.w_td[0])
        td[3] = self.conv_bn_act['td3'](fused)

        # td[2]
        fused = self._fast_fuse([ins[2], td[3]], self.w_td[1])
        td[2] = self.conv_bn_act['td2'](fused)

        # td[1]
        fused = self._fast_fuse([ins[1], td[2]], self.w_td[2])
        td[1] = self.conv_bn_act['td1'](fused)

        # td[0]
        fused = self._fast_fuse([ins[0], td[1]], self.w_td[3])
        td[0] = self.conv_bn_act['td0'](fused)

        # Bottom-up path
        out = [None] * num_frames
        out[0] = self.conv_bn_act['out0'](td[0])  # Starting bottom node

        # out[1] = fuse(ins[1], td[1], out[0]) -> conv_bn_act
        fused = self._fast_fuse([ins[1], td[1], out[0]], self.w_out[0])
        out[1] = self.conv_bn_act['out1'](fused)

        # out[2]
        fused = self._fast_fuse([ins[2], td[2], out[1]], self.w_out[1])
        out[2] = self.conv_bn_act['out2'](fused)

        # out[3]
        fused = self._fast_fuse([ins[3], td[3], out[2]], self.w_out[2])
        out[3] = self.conv_bn_act['out3'](fused)

        # out[4]
        fused = self._fast_fuse([ins[4], td[4], out[3]], self.w_out[3])
        out[4] = self.conv_bn_act['out4'](fused)

        # Final fusion: fast normalized sum of all output nodes
        fused_final = self._fast_fuse(out, self.w_final)

        return fused_final
class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    Standard Sinusoidal implementation from 'Attention Is All You Need'.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [Batch, Seq_Len, Dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)