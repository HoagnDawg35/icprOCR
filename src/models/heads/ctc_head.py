
import torch.nn as nn
import torch.nn.functional as F
import torch
class CTCHead(nn.Module):
    """
    CTC prediction head with optional intermediate layer.
    PyTorch implementation with L2 regularization via weight_decay in optimizer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        return_feats: bool = False,
        dropout: float = 0.0
    ):
        super(CTCHead, self).__init__()
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats
        
        if mid_channels is None:
            # Single linear layer
            self.fc = nn.Linear(in_channels, out_channels)
            self._init_weights(self.fc, in_channels)
        else:
            # Two linear layers with intermediate dimension
            self.fc1 = nn.Linear(in_channels, mid_channels)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
            self.fc2 = nn.Linear(mid_channels, out_channels)
            
            self._init_weights(self.fc1, in_channels)
            self._init_weights(self.fc2, mid_channels)
    
    def _init_weights(self, layer: nn.Linear, fan_in: int):
        """Initialize weights with uniform distribution based on fan_in."""
        stdv = 1.0 / (fan_in ** 0.5)
        nn.init.uniform_(layer.weight, -stdv, stdv)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -stdv, stdv)
    
    def forward(self, x: torch.Tensor, targets=None):
        """
        Args:
            x: [Batch, Seq_Len, in_channels]
            targets: Not used in PyTorch version (for compatibility)
        
        Returns:
            If return_feats=True: (features, predictions)
            Else: predictions [Batch, Seq_Len, out_channels]
        """
        if self.mid_channels is None:
            predicts = self.fc(x)
            features = x
        else:
            features = self.fc1(x)
            if self.dropout is not None:
                features = self.dropout(features)
            predicts = self.fc2(features)
        
        if self.return_feats:
            result = (features, predicts)
        else:
            result = predicts
        
        # Apply softmax during inference
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            if self.return_feats:
                result = (features, predicts)
            else:
                result = predicts
        
        return result
