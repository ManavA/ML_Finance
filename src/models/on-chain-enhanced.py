# src/models/on_chain_enhanced.py
"""
Models that incorporate on-chain metrics from Sanbase.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from src.models.base import BaseModel
from src.data.sanbase_collector import SanbaseCollector
import logging

logger = logging.getLogger(__name__)


class OnChainGRUModel(BaseModel):
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):

        super().__init__(input_size, output_size, config)
        
        self.hidden_size = config.get('hidden_size', 256)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.2)
        
        # Separate processing for on-chain vs price data
        self.on_chain_features = config.get('on_chain_features', 10)
        self.price_features = input_size - self.on_chain_features
        
        # On-chain feature processor
        self.on_chain_processor = nn.Sequential(
            nn.Linear(self.on_chain_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Price feature processor
        self.price_processor = nn.GRU(
            input_size=self.price_features,
            hidden_size=self.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Combined processor
        combined_size = 32 + self.hidden_size // 2
        
        self.combined_gru = nn.GRU(
            input_size=combined_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size * 2,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape
        
        # Split price and on-chain features
        price_features = x[:, :, :self.price_features]
        on_chain_features = x[:, :, self.price_features:]
        
        # Process price features with GRU
        price_out, _ = self.price_processor(price_features)
        
        # Process on-chain features
        on_chain_out = self.on_chain_processor(
            on_chain_features.reshape(-1, self.on_chain_features)
        )
        on_chain_out = on_chain_out.reshape(batch_size, seq_len, -1)
        
        # Combine features
        combined = torch.cat([price_out, on_chain_out], dim=-1)
        
        # Process with bidirectional GRU
        gru_out, _ = self.combined_gru(combined)
        
        # Apply attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Global pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Generate output
        output = self.output_layers(pooled)
        
        return output


class OnChainTransformer(BaseModel):
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):
        super().__init__(input_size, output_size, config)
        
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_encoder_layers = config.get('num_encoder_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 1024)
        self.dropout = config.get('dropout', 0.1)
        
        # Separate embeddings for different data types
        self.on_chain_features = config.get('on_chain_features', 10)
        self.price_features = input_size - self.on_chain_features
        
        # Feature projections
        self.price_projection = nn.Linear(self.price_features, self.d_model // 2)
        self.on_chain_projection = nn.Linear(self.on_chain_features, self.d_model // 2)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, self.d_model))
        
        # Cross-attention between price and on-chain
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_feedforward, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Split features
        price_features = x[:, :, :self.price_features]
        on_chain_features = x[:, :, self.price_features:]
        
        # Project features
        price_embed = self.price_projection(price_features)
        on_chain_embed = self.on_chain_projection(on_chain_features)
        
        # Combine embeddings
        combined_embed = torch.cat([price_embed, on_chain_embed], dim=-1)
        
        # Add positional encoding
        combined_embed = combined_embed + self.pos_encoder[:, :seq_len, :]
        
        # Cross-attention between price and on-chain
        attn_out, _ = self.cross_attention(
            combined_embed, combined_embed, combined_embed
        )
        
        # Transformer encoding
        transformer_out = self.transformer(attn_out)
        
        # Pool and output
        pooled = transformer_out.mean(dim=1)
        output = self.output_head(pooled)
        
        return output