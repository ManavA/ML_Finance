# src/strategies/baseline_strategies.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .base_strategy import BaseStrategy


class DollarCostAveraging(BaseStrategy):
    
    def __init__(self, frequency: int = 30):
        super().__init__(f"DCA (every {frequency} days)")
        self.frequency = frequency
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        
        signals = pd.Series(index=data.index, data=0)
        
        # Buy every N days
        for i in range(0, len(signals), self.frequency):
            signals.iloc[i] = 1
            
        return signals


class SMACrossoverStrategy(BaseStrategy):
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__(f"SMA({fast_period}/{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate SMAs
        fast_sma = data['close'].rolling(window=self.fast_period).mean()
        slow_sma = data['close'].rolling(window=self.slow_period).mean()
        
        # Initialize signals
        signals = pd.Series(index=data.index, data=0)
        
        # Generate crossover signals
        for i in range(1, len(data)):
            # Golden cross (bullish)
            if fast_sma.iloc[i] > slow_sma.iloc[i] and fast_sma.iloc[i-1] <= slow_sma.iloc[i-1]:
                signals.iloc[i] = 1
            # Death cross (bearish)
            elif fast_sma.iloc[i] < slow_sma.iloc[i] and fast_sma.iloc[i-1] >= slow_sma.iloc[i-1]:
                signals.iloc[i] = -1
        
        return signals


class BollingerBandsStrategy(BaseStrategy):
    
    def __init__(self, period: int = 20, num_std: float = 2):
 
        super().__init__(f"BB({period}, {num_std}σ)")
        self.period = period
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:

        # Calculate Bollinger Bands if not present
        if 'bb_lower' not in data.columns or 'bb_upper' not in data.columns:
            middle = data['close'].rolling(window=self.period).mean()
            std = data['close'].rolling(window=self.period).std()
            upper = middle + (std * self.num_std)
            lower = middle - (std * self.num_std)
        else:
            upper = data['bb_upper']
            lower = data['bb_lower']
        
        # Initialize signals
        signals = pd.Series(index=data.index, data=0)
        
        # Generate signals
        signals[data['close'] <= lower] = 1   # Buy at lower band
        signals[data['close'] >= upper] = -1  # Sell at upper band
        
        return signals