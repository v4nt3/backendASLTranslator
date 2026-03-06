import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging


logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Supports both learnable and sinusoidal encodings.
    Learnable encodings are recommended for sign language
    as they can adapt to motion patterns.
    
    Attributes:
        d_model: Model dimension
        max_len: Maximum sequence length
        learnable: Whether to use learnable encodings
        
    Example:
        >>> pe = PositionalEncoding(d_model=512, max_len=100)
        >>> x = torch.randn(32, 50, 512)
        >>> x = pe(x)
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 128,
        dropout: float = 0.1,
        learnable: bool = True
    ):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            learnable: Use learnable vs sinusoidal encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable
        
        if learnable:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            pe = self._create_sinusoidal_encoding(max_len, d_model)
            self.register_buffer('pe', pe)
            
    def _create_sinusoidal_encoding(
        self,
        max_len: int,
        d_model: int
    ) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            
        Returns:
            Positional encoding tensor [1, max_len, d_model]
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output with positional encoding added
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_len}"
            )
            
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class FeatureProjection(nn.Module):
    """
    Project input features to model dimension with normalization.
    
    Applies linear projection followed by layer normalization
    and dropout for regularization.
    
    Attributes:
        input_dim: Input feature dimension
        output_dim: Output dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize feature projection.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features.
        
        Args:
            x: Input [batch, seq_len, input_dim]
            
        Returns:
            Projected features [batch, seq_len, output_dim]
        """
        return self.projection(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing visual and pose features.
    
    Allows each modality to attend to the other, enabling
    richer multimodal representations.
    
    Architecture:
        - Visual attends to pose
        - Pose attends to visual
        - Outputs are concatenated or added
        
    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        
    Example:
        >>> cma = CrossModalAttention(d_model=512, num_heads=8)
        >>> visual = torch.randn(32, 50, 512)
        >>> pose = torch.randn(32, 50, 512)
        >>> fused = cma(visual, pose)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Visual attends to pose
        self.visual_to_pose = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Pose attends to visual
        self.pose_to_visual = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm_visual = nn.LayerNorm(d_model)
        self.norm_pose = nn.LayerNorm(d_model)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        visual: torch.Tensor,
        pose: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            visual: Visual features [batch, seq_len, d_model]
            pose: Pose features [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Fused features [batch, seq_len, d_model]
        """
        # Convert mask to attention format if provided
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # Invert for nn.MultiheadAttention
            
        # Visual attends to pose (query=visual, key/value=pose)
        visual_attended, _ = self.visual_to_pose(
            query=visual,
            key=pose,
            value=pose,
            key_padding_mask=key_padding_mask
        )
        visual_attended = self.norm_visual(visual + visual_attended)
        
        # Pose attends to visual
        pose_attended, _ = self.pose_to_visual(
            query=pose,
            key=visual,
            value=visual,
            key_padding_mask=key_padding_mask
        )
        pose_attended = self.norm_pose(pose + pose_attended)
        
        # Fuse modalities
        fused = torch.cat([visual_attended, pose_attended], dim=-1)
        fused = self.fusion(fused)
        
        return self.output_norm(fused)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence classification.
    
    Learns to weight different time steps based on their
    importance for classification. Superior to mean pooling
    for variable-length sequences.
    
    Attributes:
        d_model: Input dimension
        
    Example:
        >>> pool = AttentionPooling(d_model=512)
        >>> x = torch.randn(32, 50, 512)
        >>> pooled = pool(x)  # [32, 512]
    """
    
    def __init__(self, d_model: int):
        """
        Initialize attention pooling.
        
        Args:
            d_model: Input dimension
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            x: Input sequence [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len]
            
        Returns:
            Pooled representation [batch, d_model]
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # [batch, seq_len]
        
        # Mask invalid positions
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float('-inf'))
            
        # Softmax over sequence
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # [batch, d_model]
        
        return pooled


class TransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer with pre-normalization.
    
    Uses pre-layer normalization for more stable training,
    which is particularly beneficial for deeper models.
    
    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        """
        Initialize transformer layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
        """
        super().__init__()
        
        # Pre-normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer layer.
        
        Args:
            x: Input [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            
        Returns:
            Output [batch, seq_len, d_model]
        """
        # Self-attention with pre-norm
        normed = self.norm1(x)
        
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask
            
        attended, _ = self.self_attn(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attended)
        
        # Feed-forward with pre-norm
        normed = self.norm2(x)
        x = x + self.ff(normed)
        
        return x


class ClassificationHead(nn.Module):
    """
    Classification head for sequence classification.
    
    Combines attention pooling with MLP classifier.
    Includes regularization via dropout and layer normalization.
    
    Attributes:
        d_model: Input dimension
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        pooling: str = "attention"
    ):
        """
        Initialize classification head.
        
        Args:
            d_model: Input dimension
            num_classes: Number of classes
            hidden_dim: Hidden dimension
            dropout: Dropout probability
            pooling: Pooling type ("attention", "mean", "cls")
        """
        super().__init__()
        
        self.pooling_type = pooling
        
        if pooling == "attention":
            self.pooling = AttentionPooling(d_model)
        elif pooling == "mean":
            self.pooling = None
        elif pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.pooling = None
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
            
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequence [batch, seq_len, d_model]
            attention_mask: Optional mask
            
        Returns:
            Logits [batch, num_classes]
        """
        if self.pooling_type == "attention":
            pooled = self.pooling(x, attention_mask)
        elif self.pooling_type == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling_type == "cls":
            pooled = x[:, 0]  # First token is CLS
            
        return self.classifier(pooled)
