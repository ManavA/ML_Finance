"""
Helformer: Hierarchical Efficient Long-range Transformer for Financial Time Series

A specialized transformer architecture designed for cryptocurrency and financial markets
that handles multiple time scales, long-range dependencies, and market microstructure.

Key Features:
- Multi-scale temporal attention (minute, hour, day, week patterns)
- Efficient attention mechanism for long sequences
- Market-specific positional encoding
- Volatility-aware attention scaling
- Cross-asset attention for portfolio modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import math
from dataclasses import dataclass


@dataclass
class HelformerConfig:
    """Configuration for Helformer model"""
    # Model dimensions
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Sequence parameters
    max_seq_length: int = 4096  # ~2.8 days of minute data
    n_features: int = 7  # OHLCV + volume + indicators
    
    # Multi-scale parameters
    time_scales: List[int] = None  # [1, 5, 15, 60, 240, 1440] minutes
    scale_embedding_dim: int = 64
    
    # Financial specific
    use_volatility_scaling: bool = True
    use_volume_attention: bool = True
    use_cross_asset_attention: bool = False
    n_assets: int = 1
    
    # Efficient attention
    attention_type: str = "linformer"  # "full", "linformer", "performer"
    linformer_k: int = 256  # Projection dimension for Linformer
    
    # Output
    n_outputs: int = 3  # [return, volatility, direction]
    prediction_horizon: int = 1  # Steps ahead to predict
    
    def __post_init__(self):
        if self.time_scales is None:
            self.time_scales = [1, 5, 15, 60, 240, 1440]


class FinancialPositionalEncoding(nn.Module):
    """
    Positional encoding that incorporates financial time structure:
    - Intraday patterns (time of day)
    - Day of week effects
    - Month/quarter seasonality
    - Market regime indicators
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Time-based embeddings
        self.hour_embedding = nn.Embedding(24, d_model // 4)
        self.day_embedding = nn.Embedding(7, d_model // 4)
        self.month_embedding = nn.Embedding(12, d_model // 4)
        
        # Market regime embedding (learned)
        self.regime_embedding = nn.Linear(1, d_model // 4)
        
    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None,
                volatility_regime: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            timestamps: Unix timestamps (batch, seq_len)
            volatility_regime: Current volatility regime (batch, seq_len, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Base positional encoding
        pos_encoding = self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        
        if timestamps is not None:
            # Extract time components (simplified - in practice use proper datetime)
            hours = (timestamps // 3600) % 24
            days = (timestamps // 86400) % 7
            months = (timestamps // 2592000) % 12
            
            # Time-based encodings
            hour_enc = self.hour_embedding(hours.long())
            day_enc = self.day_embedding(days.long())
            month_enc = self.month_embedding(months.long())
            
            # Combine time encodings
            time_encoding = torch.cat([hour_enc, day_enc, month_enc], dim=-1)
            
            # Add regime encoding if provided
            if volatility_regime is not None:
                regime_enc = self.regime_embedding(volatility_regime)
                time_encoding = torch.cat([time_encoding, regime_enc], dim=-1)
            
            # Project to d_model if needed
            if time_encoding.shape[-1] != self.d_model:
                padding = torch.zeros(batch_size, seq_len, 
                                     self.d_model - time_encoding.shape[-1],
                                     device=x.device)
                time_encoding = torch.cat([time_encoding, padding], dim=-1)
            
            pos_encoding = pos_encoding + time_encoding
        
        return x + pos_encoding


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention that captures patterns at different time resolutions
    """
    
    def __init__(self, config: HelformerConfig):
        super().__init__()
        self.config = config
        self.n_scales = len(config.time_scales)
        
        # Attention for each time scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(config.d_model, config.n_heads, 
                                 dropout=config.dropout, batch_first=True)
            for _ in config.time_scales
        ])
        
        # Scale combination
        self.scale_weights = nn.Parameter(torch.ones(self.n_scales) / self.n_scales)
        self.scale_projection = nn.Linear(config.d_model * self.n_scales, config.d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention at multiple time scales and combine
        """
        batch_size, seq_len, d_model = x.shape
        scale_outputs = []
        
        for scale_idx, (scale, attn) in enumerate(zip(self.config.time_scales, 
                                                      self.scale_attentions)):
            if scale == 1:
                # Finest scale - use original sequence
                scale_out, _ = attn(x, x, x, attn_mask=mask)
            else:
                # Downsample for coarser scales
                pooled_len = seq_len // scale
                if pooled_len > 0:
                    # Average pooling for downsampling
                    x_pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=scale, 
                                           stride=scale).transpose(1, 2)
                    
                    # Apply attention at this scale
                    scale_out_pooled, _ = attn(x_pooled, x_pooled, x_pooled)
                    
                    # Upsample back to original resolution
                    scale_out = F.interpolate(scale_out_pooled.transpose(1, 2), 
                                             size=seq_len, mode='linear', 
                                             align_corners=False).transpose(1, 2)
                else:
                    scale_out = x
            
            scale_outputs.append(scale_out * self.scale_weights[scale_idx])
        
        # Combine scales
        combined = torch.cat(scale_outputs, dim=-1)
        output = self.scale_projection(combined)
        
        return output


class VolatilityAwareAttention(nn.Module):
    """
    Attention mechanism that adjusts based on market volatility
    """
    
    def __init__(self, config: HelformerConfig):
        super().__init__()
        self.config = config
        
        # Volatility processing
        self.volatility_net = nn.Sequential(
            nn.Linear(1, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, config.n_heads),
            nn.Sigmoid()
        )
        
        # Standard attention
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, 
            dropout=config.dropout, batch_first=True
        )
        
    def forward(self, x: torch.Tensor, volatility: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            volatility: Volatility estimates (batch, seq_len, 1)
            mask: Attention mask
        """
        # Compute volatility-based attention scaling
        vol_scales = self.volatility_net(volatility)  # (batch, seq_len, n_heads)
        
        # Apply attention with volatility scaling
        batch_size, seq_len, _ = x.shape
        
        # Reshape for multi-head attention
        q = k = v = x
        
        # Apply standard attention
        attn_output, attn_weights = self.attention(q, k, v, attn_mask=mask)
        
        # Scale by volatility (simplified - in practice modify attention computation)
        vol_scale_avg = vol_scales.mean(dim=1, keepdim=True)
        attn_output = attn_output * (1 + vol_scale_avg)
        
        return attn_output


class HelformerBlock(nn.Module):
    """
    Single Helformer block with multi-scale and volatility-aware attention
    """
    
    def __init__(self, config: HelformerConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale attention
        self.multi_scale_attn = MultiScaleAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        
        # Volatility-aware attention (optional)
        if config.use_volatility_scaling:
            self.vol_attn = VolatilityAwareAttention(config)
            self.norm_vol = nn.LayerNorm(config.d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor, volatility: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Helformer block
        """
        # Multi-scale attention
        attn_out = self.multi_scale_attn(x, mask)
        x = self.norm1(x + attn_out)
        
        # Volatility-aware attention
        if self.config.use_volatility_scaling and volatility is not None:
            vol_out = self.vol_attn(x, volatility, mask)
            x = self.norm_vol(x + vol_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class Helformer(nn.Module):
    """
    Main Helformer model for financial time series prediction
    """
    
    def __init__(self, config: HelformerConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.n_features, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        
        # Positional encoding
        self.pos_encoding = FinancialPositionalEncoding(config.d_model, 
                                                        config.max_seq_length)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HelformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output heads
        self.output_heads = nn.ModuleDict({
            'return': nn.Linear(config.d_model, config.prediction_horizon),
            'volatility': nn.Linear(config.d_model, config.prediction_horizon),
            'direction': nn.Linear(config.d_model, config.prediction_horizon * 3)  # 3 classes
        })
        
        # Cross-asset attention (optional)
        if config.use_cross_asset_attention and config.n_assets > 1:
            self.cross_asset_attn = nn.MultiheadAttention(
                config.d_model, config.n_heads, 
                dropout=config.dropout, batch_first=True
            )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def compute_volatility(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute rolling volatility from price data
        """
        # Simple rolling std as volatility proxy
        # In practice, use more sophisticated methods (GARCH, etc.)
        returns = torch.diff(x[:, :, 3], dim=1) / x[:, :-1, 3]  # Using close prices
        
        # Compute rolling volatility with window size 20
        window = 20
        volatilities = []
        for i in range(returns.shape[1]):
            if i < window:
                vol = returns[:, :i+1].std(dim=1, keepdim=True)
            else:
                vol = returns[:, i-window+1:i+1].std(dim=1, keepdim=True)
            volatilities.append(vol)
        
        volatility = torch.cat(volatilities, dim=1).unsqueeze(-1)
        
        # Pad to match input length
        volatility = F.pad(volatility, (0, 0, 1, 0), value=volatility[:, 0:1].squeeze())
        
        return volatility
    
    def forward(self, x: torch.Tensor, 
                timestamps: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Helformer
        
        Args:
            x: Input features (batch, seq_len, n_features)
                Expected format: [open, high, low, close, volume, ...]
            timestamps: Unix timestamps (batch, seq_len)
            mask: Attention mask (batch, seq_len)
        
        Returns:
            Dictionary with predictions:
                - 'return': Predicted returns
                - 'volatility': Predicted volatility
                - 'direction': Direction probabilities (down, neutral, up)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute volatility for volatility-aware attention
        volatility = self.compute_volatility(x)
        
        # Input projection
        x_proj = self.input_projection(x)
        x_proj = self.input_norm(x_proj)
        
        # Add positional encoding
        x_encoded = self.pos_encoding(x_proj, timestamps, volatility)
        x_encoded = self.dropout(x_encoded)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x_encoded = block(x_encoded, volatility, mask)
        
        # Cross-asset attention if applicable
        if self.config.use_cross_asset_attention and self.config.n_assets > 1:
            # Reshape for cross-asset attention
            # This would need proper implementation for multi-asset scenarios
            pass
        
        # Generate predictions from the last position
        # In practice, might want to use pooling or attention-based aggregation
        final_representation = x_encoded[:, -1, :]  # (batch, d_model)
        
        # Output predictions
        outputs = {}
        outputs['return'] = self.output_heads['return'](final_representation)
        outputs['volatility'] = F.softplus(self.output_heads['volatility'](final_representation))
        
        # Direction classification
        direction_logits = self.output_heads['direction'](final_representation)
        direction_logits = direction_logits.view(batch_size, self.config.prediction_horizon, 3)
        outputs['direction'] = F.softmax(direction_logits, dim=-1)
        
        return outputs


class HelformerTrainer:
    """
    Training utilities for Helformer model
    """
    
    def __init__(self, model: Helformer, config: HelformerConfig):
        self.model = model
        self.config = config
        
        # Optimizers
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )
        
        # Loss functions
        self.return_loss = nn.MSELoss()
        self.volatility_loss = nn.MSELoss()
        self.direction_loss = nn.CrossEntropyLoss()
        
    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss for all prediction tasks
        """
        losses = {}
        
        # Return prediction loss
        if 'return' in targets:
            losses['return'] = self.return_loss(predictions['return'], targets['return'])
        
        # Volatility prediction loss
        if 'volatility' in targets:
            losses['volatility'] = self.volatility_loss(predictions['volatility'], 
                                                       targets['volatility'])
        
        # Direction classification loss
        if 'direction' in targets:
            # Reshape for cross-entropy
            pred_dir = predictions['direction'].view(-1, 3)
            target_dir = targets['direction'].view(-1)
            losses['direction'] = self.direction_loss(pred_dir, target_dir)
        
        # Combined loss with weights
        total_loss = (losses.get('return', 0) * 1.0 +
                     losses.get('volatility', 0) * 0.5 +
                     losses.get('direction', 0) * 0.3)
        
        return total_loss, losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(batch['features'], batch.get('timestamps'))
        
        # Compute loss
        total_loss, losses = self.compute_loss(predictions, batch['targets'])
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Return losses for logging
        return {
            'total_loss': total_loss.item(),
            **{k: v.item() for k, v in losses.items()}
        }
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate model on validation/test data
        """
        self.model.eval()
        total_losses = {}
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                predictions = self.model(batch['features'], batch.get('timestamps'))
                _, losses = self.compute_loss(predictions, batch['targets'])
                
                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0) + v.item()
                n_batches += 1
        
        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses


def create_helformer(
    n_features: int = 7,
    max_seq_length: int = 1440,  # 1 day of minute data
    prediction_horizon: int = 60,  # Predict next hour
    **kwargs
) -> Helformer:
    """
    Factory function to create Helformer model with sensible defaults
    """
    config = HelformerConfig(
        n_features=n_features,
        max_seq_length=max_seq_length,
        prediction_horizon=prediction_horizon,
        **kwargs
    )
    
    model = Helformer(config)
    
    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Creating Helformer model...")
    
    # Create model
    model = create_helformer(
        n_features=7,  # OHLCV + 2 indicators
        max_seq_length=1440,  # 1 day of minute data
        prediction_horizon=60,  # Predict next hour
        d_model=256,
        n_layers=4,
        n_heads=8
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 1440
    n_features = 7
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, n_features)
    timestamps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1) * 60  # Minute timestamps
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, timestamps)
    
    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print("\nHelformer model created successfully!")