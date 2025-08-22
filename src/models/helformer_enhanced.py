import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
import math
from dataclasses import dataclass


@dataclass
class HelformerConfig:
    input_dim: int = 32
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 8
    d_ff: int = 2048
    max_seq_length: int = 1000
    dropout: float = 0.1
    use_holt_winters: bool = True
    alpha: float = 0.3
    beta: float = 0.1
    gamma: float = 0.1
    season_length: int = 24
    use_multi_scale: bool = True
    scales: List[int] = None
    use_uncertainty: bool = True
    n_quantiles: int = 3
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [1, 4, 24, 168]


class HoltWintersDecomposition(nn.Module):
    
    def __init__(self, config: HelformerConfig):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(config.alpha))
        self.beta = nn.Parameter(torch.tensor(config.beta))
        self.gamma = nn.Parameter(torch.tensor(config.gamma))
        self.season_length = config.season_length
        
        self.level_proj = nn.Linear(config.input_dim, config.d_model // 3)
        self.trend_proj = nn.Linear(config.input_dim, config.d_model // 3)
        self.season_proj = nn.Linear(config.input_dim, config.d_model // 3)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        gamma = torch.sigmoid(self.gamma)
        
        level = torch.zeros_like(x)
        trend = torch.zeros_like(x)
        season = torch.zeros_like(x)
        
        level[:, 0] = x[:, 0]
        
        for t in range(1, seq_len):
            if t >= self.season_length:
                prev_season = season[:, t - self.season_length]
            else:
                prev_season = torch.zeros_like(x[:, 0])
            
            level[:, t] = alpha * (x[:, t] - prev_season) + (1 - alpha) * (level[:, t-1] + trend[:, t-1])
            trend[:, t] = beta * (level[:, t] - level[:, t-1]) + (1 - beta) * trend[:, t-1]
            season[:, t] = gamma * (x[:, t] - level[:, t]) + (1 - gamma) * prev_season
        
        level_encoded = self.level_proj(level)
        trend_encoded = self.trend_proj(trend)
        season_encoded = self.season_proj(season)
        
        return level_encoded, trend_encoded, season_encoded


class MultiScaleAttention(nn.Module):
    
    def __init__(self, config: HelformerConfig):
        super().__init__()
        self.scales = config.scales
        self.d_model = config.d_model
        
        self.scale_projections = nn.ModuleList([
            nn.Conv1d(config.d_model, config.d_model, kernel_size=scale, stride=1, padding=scale//2)
            for scale in self.scales
        ])
        
        self.fusion = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        x_transposed = x.transpose(1, 2)
        
        multi_scale_features = []
        for conv in self.scale_projections:
            scale_feat = conv(x_transposed).transpose(1, 2)
            if scale_feat.size(1) != seq_len:
                scale_feat = F.interpolate(
                    scale_feat.transpose(1, 2), 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            multi_scale_features.append(scale_feat)
        
        multi_scale_tensor = torch.stack(multi_scale_features, dim=1)
        multi_scale_tensor = multi_scale_tensor.view(batch_size * len(self.scales), seq_len, d_model)
        
        fused, _ = self.fusion(x.repeat(len(self.scales), 1, 1), multi_scale_tensor, multi_scale_tensor)
        fused = fused.view(batch_size, len(self.scales), seq_len, d_model).mean(dim=1)
        
        return fused


class CryptoSpecificEmbedding(nn.Module):
    
    def __init__(self, config: HelformerConfig):
        super().__init__()
        
        self.price_embedding = nn.Linear(1, config.d_model // 4)
        self.volume_embedding = nn.Linear(1, config.d_model // 4)
        self.volatility_embedding = nn.Linear(1, config.d_model // 4)
        self.sentiment_embedding = nn.Linear(1, config.d_model // 4)
        
        self.time_embedding = nn.Embedding(168, config.d_model // 4)
        self.day_embedding = nn.Embedding(7, config.d_model // 4)
        
        self.fusion = nn.Linear(config.d_model + config.d_model // 2, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, 
                price: torch.Tensor,
                volume: torch.Tensor, 
                volatility: torch.Tensor,
                sentiment: Optional[torch.Tensor] = None,
                hour_of_week: Optional[torch.Tensor] = None,
                day_of_week: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        price_emb = self.price_embedding(price.unsqueeze(-1))
        volume_emb = self.volume_embedding(volume.unsqueeze(-1))
        volatility_emb = self.volatility_embedding(volatility.unsqueeze(-1))
        
        if sentiment is not None:
            sentiment_emb = self.sentiment_embedding(sentiment.unsqueeze(-1))
        else:
            sentiment_emb = torch.zeros_like(price_emb)
        
        embeddings = [price_emb, volume_emb, volatility_emb, sentiment_emb]
        
        if hour_of_week is not None:
            time_emb = self.time_embedding(hour_of_week.long())
            embeddings.append(time_emb)
        
        if day_of_week is not None:
            day_emb = self.day_embedding(day_of_week.long())
            embeddings.append(day_emb)
        
        combined = torch.cat(embeddings, dim=-1)
        fused = self.fusion(combined)
        
        return self.layer_norm(fused)


class EnhancedHelformerBlock(nn.Module):
    
    def __init__(self, config: HelformerConfig):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.multi_scale = MultiScaleAttention(config)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        multi_scale_out = self.multi_scale(x)
        x = self.norm2(x + self.dropout(multi_scale_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x


class UncertaintyHead(nn.Module):
    
    def __init__(self, config: HelformerConfig):
        super().__init__()
        self.n_quantiles = config.n_quantiles
        
        self.quantile_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 1)
            )
            for _ in range(config.n_quantiles)
        ])
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        quantiles = []
        for i, proj in enumerate(self.quantile_projections):
            q = proj(x)
            quantiles.append(q)
        
        quantiles = torch.cat(quantiles, dim=-1)
        quantiles = torch.sort(quantiles, dim=-1)[0]
        
        confidence = self.confidence_head(x)
        
        return {
            'quantiles': quantiles,
            'lower': quantiles[..., 0],
            'median': quantiles[..., self.n_quantiles // 2],
            'upper': quantiles[..., -1],
            'confidence': confidence
        }


class EnhancedHelformer(nn.Module):
    
    def __init__(self, config: HelformerConfig):
        super().__init__()
        self.config = config
        
        self.crypto_embedding = CryptoSpecificEmbedding(config)
        
        if config.use_holt_winters:
            self.holt_winters = HoltWintersDecomposition(config)
            self.component_fusion = nn.Linear(config.d_model, config.d_model)
        
        self.positional_encoding = self._create_positional_encoding(
            config.max_seq_length, config.d_model
        )
        
        self.encoder_blocks = nn.ModuleList([
            EnhancedHelformerBlock(config) 
            for _ in range(config.n_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveMaxPool1d(1)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 3)
        )
        
        if config.use_uncertainty:
            self.uncertainty_head = UncertaintyHead(config)
        
        self._init_weights()
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, 
                x: torch.Tensor,
                volume: Optional[torch.Tensor] = None,
                volatility: Optional[torch.Tensor] = None,
                sentiment: Optional[torch.Tensor] = None,
                hour_of_week: Optional[torch.Tensor] = None,
                day_of_week: Optional[torch.Tensor] = None,
                return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len, _ = x.shape
        
        if self.config.use_holt_winters:
            level, trend, season = self.holt_winters(x)
            decomposed = torch.cat([level, trend, season], dim=-1)
            x_embedded = self.component_fusion(decomposed)
        else:
            price = x[..., 0] if x.size(-1) > 1 else x.squeeze(-1)
            x_embedded = self.crypto_embedding(
                price, volume, volatility, sentiment, hour_of_week, day_of_week
            )
        
        x_embedded = x_embedded + self.positional_encoding[:, :seq_len, :]
        
        mask = self.create_attention_mask(seq_len, x.device)
        
        for block in self.encoder_blocks:
            x_embedded = block(x_embedded, mask)
        
        x_transposed = x_embedded.transpose(1, 2)
        global_features = self.global_pool(x_transposed).squeeze(-1)
        local_features = self.local_pool(x_transposed).squeeze(-1)
        
        combined = torch.cat([global_features, local_features], dim=-1)
        
        predictions = self.prediction_head(combined)
        
        output = {
            'predictions': predictions,
            'direction': torch.argmax(predictions, dim=-1),
            'probabilities': F.softmax(predictions, dim=-1),
            'features': x_embedded
        }
        
        if self.config.use_uncertainty and return_uncertainty:
            uncertainty = self.uncertainty_head(x_embedded[:, -1, :])
            output.update(uncertainty)
        
        return output


def create_helformer_for_crypto(
    input_dim: int = 32,
    use_sentiment: bool = True,
    use_uncertainty: bool = True,
    device: str = 'cuda'
) -> EnhancedHelformer:
    
    config = HelformerConfig(
        input_dim=input_dim,
        d_model=512,
        n_heads=16,
        n_layers=8,
        d_ff=2048,
        dropout=0.1,
        use_holt_winters=True,
        use_multi_scale=True,
        use_uncertainty=use_uncertainty,
        scales=[1, 4, 24, 168],
        season_length=24,
        n_quantiles=3
    )
    
    model = EnhancedHelformer(config)
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        model = torch.compile(model, mode='reduce-overhead')
    
    return model