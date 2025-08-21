# src/data/sanbase_collector.py
"""
Sanbase/Santiment API integration for on-chain and social metrics.
Provides unique alpha through on-chain analytics.
"""

import san
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class SanbaseCollector:
    """Collect on-chain and social metrics from Sanbase/Santiment."""
    
    def __init__(self, api_key: str):
        """
        Initialize Sanbase collector.
        
        Args:
            api_key: Sanbase API key
        """
        self.api_key = api_key
        san.ApiConfig.api_key = api_key
        self.cache_dir = Path('cache/sanbase')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Available metrics categories
        self.on_chain_metrics = [
            'daily_active_addresses',
            'network_growth',
            'transaction_volume',
            'circulation',
            'velocity',
            'nvt',  # Network Value to Transactions
            'nvt_transaction_volume',
            'mean_age',
            'mean_dollar_invested_age',
            'realized_value',
            'mvrv_usd',  # Market Value to Realized Value
            'mvrv_long_short_diff',
            'exchange_balance',
            'exchange_inflow',
            'exchange_outflow',
            'active_deposits',
            'withdrawal_transactions',
            'miners_balance',
            'dormant_circulation_365d',
            'age_consumed',
            'average_fees_usd',
            'median_fees_usd'
        ]
        
        self.social_metrics = [
            'social_volume_total',
            'social_dominance_total',
            'sentiment_positive_total',
            'sentiment_negative_total',
            'sentiment_balance_total',
            'social_active_users',
            'dev_activity',
            'github_activity',
            'telegram_discussion_overview',
            'twitter_followers'
        ]
        
        self.derivative_metrics = [
            'bitmex_perpetual_funding_rate',
            'bitmex_perpetual_open_interest',
            'makerdao_dai_savings_rate',
            'defi_total_value_locked_usd'
        ]
        
        logger.info(f"Sanbase collector initialized with {len(self.on_chain_metrics)} on-chain metrics")
    
    def fetch_metric(self, slug: str, metric: str,
                    start_date: str, end_date: str,
                    interval: str = '1d') -> pd.DataFrame:
        """
        Fetch a single metric from Sanbase.
        
        Args:
            slug: Asset slug (e.g., 'bitcoin', 'ethereum')
            metric: Metric name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with metric data
        """
        cache_key = f"{slug}_{metric}_{start_date}_{end_date}_{interval}.pkl"
        cache_path = self.cache_dir / cache_key
        
        # Check cache
        if cache_path.exists():
            logger.debug(f"Loading cached {metric} for {slug}")
            return pd.read_pickle(cache_path)
        
        try:
            # Fetch from Sanbase
            df = san.get(
                f"{metric}/{slug}",
                from_date=start_date,
                to_date=end_date,
                interval=interval
            )
            
            if df is not None and not df.empty:
                # Cache the data
                df.to_pickle(cache_path)
                logger.info(f"Fetched {metric} for {slug}: {len(df)} records")
                return df
            else:
                logger.warning(f"No data for {metric}/{slug}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching {metric} for {slug}: {e}")
            return pd.DataFrame()
    
    def fetch_all_metrics(self, slug: str,
                         start_date: str, end_date: str,
                         metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch all available metrics for an asset.
        
        Args:
            slug: Asset slug
            start_date: Start date
            end_date: End date
            metrics: List of metrics to fetch (None for all)
            
        Returns:
            DataFrame with all metrics
        """
        if metrics is None:
            metrics = self.on_chain_metrics + self.social_metrics
        
        all_data = {}
        
        for metric in metrics:
            df = self.fetch_metric(slug, metric, start_date, end_date)
            if not df.empty:
                # Rename value column to metric name
                if 'value' in df.columns:
                    df = df.rename(columns={'value': metric})
                all_data[metric] = df
        
        # Combine all metrics
        if all_data:
            combined = pd.concat(all_data.values(), axis=1, join='outer')
            # Remove duplicate columns
            combined = combined.loc[:, ~combined.columns.duplicated()]
            return combined
        else:
            return pd.DataFrame()
    
    def get_fear_greed_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Fear & Greed Index from Santiment.
        
        Returns:
            DataFrame with fear/greed values
        """
        try:
            df = san.get(
                "fear_and_greed/bitcoin",
                from_date=start_date,
                to_date=end_date,
                interval="1d"
            )
            return df
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return pd.DataFrame()
    
    def get_whale_transactions(self, slug: str, 
                              start_date: str, end_date: str,
                              threshold_usd: float = 100000) -> pd.DataFrame:
        """
        Get whale transaction data.
        
        Args:
            slug: Asset slug
            start_date: Start date
            end_date: End date
            threshold_usd: Minimum transaction value in USD
            
        Returns:
            DataFrame with whale transactions
        """
        try:
            df = san.get(
                f"whale_transaction_count_100k_usd/{slug}",
                from_date=start_date,
                to_date=end_date,
                interval="1d"
            )
            return df
        except Exception as e:
            logger.error(f"Error fetching whale transactions: {e}")
            return pd.DataFrame()
    
    def get_exchange_flows(self, slug: str,
                          start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get exchange inflow/outflow data.
        
        Returns:
            DataFrame with exchange flows
        """
        metrics = ['exchange_inflow', 'exchange_outflow', 'exchange_balance']
        flows = {}
        
        for metric in metrics:
            df = self.fetch_metric(slug, metric, start_date, end_date)
            if not df.empty:
                flows[metric] = df
        
        if flows:
            return pd.concat(flows.values(), axis=1)
        return pd.DataFrame()
    
    def get_social_sentiment(self, slug: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get comprehensive social sentiment data.
        
        Returns:
            DataFrame with social metrics
        """
        social_metrics = [
            'social_volume_total',
            'sentiment_positive_total',
            'sentiment_negative_total',
            'sentiment_balance_total'
        ]
        
        return self.fetch_all_metrics(slug, start_date, end_date, social_metrics)
    
    def get_holder_distribution(self, slug: str,
                               start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get holder distribution metrics.
        
        Returns:
            DataFrame with holder metrics
        """
        holder_metrics = [
            'percent_of_total_supply_on_exchanges',
            'holders_distribution_0_to_0.1',
            'holders_distribution_0.1_to_1',
            'holders_distribution_1_to_10',
            'holders_distribution_10_to_100',
            'holders_distribution_100_to_1k',
            'holders_distribution_1k_to_10k',
            'holders_distribution_10k_to_100k',
            'holders_distribution_100k_to_1M',
            'holders_distribution_1M_to_10M',
            'holders_distribution_10M_to_inf'
        ]
        
        return self.fetch_all_metrics(slug, start_date, end_date, holder_metrics)
    
    def calculate_smart_money_indicators(self, slug: str,
                                        data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate smart money indicators from on-chain data.
        
        Args:
            slug: Asset slug
            data: DataFrame with on-chain metrics
            
        Returns:
            DataFrame with smart money indicators
        """
        indicators = pd.DataFrame(index=data.index)
        
        # Network Value to Transactions Signal
        if 'nvt' in data.columns:
            nvt_mean = data['nvt'].rolling(30).mean()
            nvt_std = data['nvt'].rolling(30).std()
            indicators['nvt_signal'] = (data['nvt'] - nvt_mean) / nvt_std
        
        # MVRV Z-Score
        if 'mvrv_usd' in data.columns:
            mvrv_mean = data['mvrv_usd'].rolling(365).mean()
            mvrv_std = data['mvrv_usd'].rolling(365).std()
            indicators['mvrv_zscore'] = (data['mvrv_usd'] - mvrv_mean) / mvrv_std
        
        # Exchange Flow Pulse
        if 'exchange_inflow' in data.columns and 'exchange_outflow' in data.columns:
            indicators['exchange_netflow'] = data['exchange_outflow'] - data['exchange_inflow']
            indicators['exchange_flow_ratio'] = data['exchange_outflow'] / (data['exchange_inflow'] + 1)
        
        # Age Consumed Signal (indicates old coins moving)
        if 'age_consumed' in data.columns:
            indicators['age_consumed_ma'] = data['age_consumed'].rolling(7).mean()
            indicators['age_consumed_spike'] = data['age_consumed'] > indicators['age_consumed_ma'] * 2
        
        # Social Sentiment Score
        if 'sentiment_positive_total' in data.columns and 'sentiment_negative_total' in data.columns:
            total_sentiment = data['sentiment_positive_total'] + data['sentiment_negative_total']
            indicators['sentiment_score'] = (
                (data['sentiment_positive_total'] - data['sentiment_negative_total']) / 
                (total_sentiment + 1)
            )
        
        # Developer Activity Trend
        if 'dev_activity' in data.columns:
            indicators['dev_activity_ma'] = data['dev_activity'].rolling(30).mean()
            indicators['dev_activity_trend'] = (
                data['dev_activity'] / (indicators['dev_activity_ma'] + 1) - 1
            )
        
        # Dormant Circulation (long-term holders selling)
        if 'dormant_circulation_365d' in data.columns:
            indicators['dormant_activation'] = (
                data['dormant_circulation_365d'] > 
                data['dormant_circulation_365d'].rolling(30).mean() * 1.5
            )
        
        return indicators


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
    """GRU model enhanced with on-chain metrics."""
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):
        """
        Initialize on-chain enhanced GRU model.
        
        Args:
            input_size: Number of input features (including on-chain)
            output_size: Number of output features
            config: Model configuration
        """
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
        """
        Forward pass with on-chain data processing.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
               Last on_chain_features dimensions are on-chain metrics
            
        Returns:
            Output predictions
        """
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
    """Transformer model with on-chain metric integration."""
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):
        """Initialize on-chain transformer."""
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
        """Forward pass with cross-attention between price and on-chain data."""
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