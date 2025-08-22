# src/data/sanbase_collector.py
"""
Sanbase/Santiment API integration for on-chain and social metrics.
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
    
    def __init__(self, api_key: str):
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
        social_metrics = [
            'social_volume_total',
            'sentiment_positive_total',
            'sentiment_negative_total',
            'sentiment_balance_total'
        ]
        
        return self.fetch_all_metrics(slug, start_date, end_date, social_metrics)
    
    def get_holder_distribution(self, slug: str,
                               start_date: str, end_date: str) -> pd.DataFrame:

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
