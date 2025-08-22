# src/strategies/baseline_strategies.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any


class BaseStrategy:
    
    def __init__(self, name: str):
        self.name = name
        self.signals = None
        self.positions = None
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
    
    def get_positions(self, signals: pd.Series) -> pd.Series:
        positions = pd.Series(index=signals.index, data=0)
        positions[signals == 1] = 1
        
        # Forward fill positions (stay in position until sell signal)
        positions = positions.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Exit on sell signals
        positions[signals == -1] = 0
        
        return positions


class BuyAndHoldStrategy(BaseStrategy):
    
    def __init__(self):
        super().__init__("Buy and Hold")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        
        signals = pd.Series(index=data.index, data=0)
        signals.iloc[0] = 1  # Buy on first day
        return signals


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


class RSIMeanReversionStrategy(BaseStrategy):
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(f"RSI({period}, {oversold}/{overbought})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:

        # Use pre-calculated RSI if available, otherwise calculate
        if 'rsi' in data.columns:
            rsi = data['rsi']
        else:
            rsi = self.calculate_rsi(data['close'], self.period)
        
        # Initialize signals
        signals = pd.Series(index=data.index, data=0)
        
        # Generate signals
        signals[rsi < self.oversold] = 1   # Buy when oversold
        signals[rsi > self.overbought] = -1  # Sell when overbought
        
        return signals
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


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


class MACDStrategy(BaseStrategy):

    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):

        super().__init__(f"MACD({fast},{slow},{signal})")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:

        # Calculate MACD if not present
        if 'macd' not in data.columns or 'macd_signal' not in data.columns:
            ema_fast = data['close'].ewm(span=self.fast, adjust=False).mean()
            ema_slow = data['close'].ewm(span=self.slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        else:
            macd = data['macd']
            macd_signal = data['macd_signal']
        
        # Initialize signals
        signals = pd.Series(index=data.index, data=0)
        
        # Generate crossover signals
        for i in range(1, len(data)):
            # MACD crosses above signal (bullish)
            if macd.iloc[i] > macd_signal.iloc[i] and macd.iloc[i-1] <= macd_signal.iloc[i-1]:
                signals.iloc[i] = 1
            # MACD crosses below signal (bearish)
            elif macd.iloc[i] < macd_signal.iloc[i] and macd.iloc[i-1] >= macd_signal.iloc[i-1]:
                signals.iloc[i] = -1
        
        return signals


class MomentumStrategy(BaseStrategy):
    
    def __init__(self, lookback: int = 20, threshold: float = 0.02):
  
        super().__init__(f"Momentum({lookback})")
        self.lookback = lookback
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:

        # Calculate momentum (rate of change)
        momentum = (data['close'] / data['close'].shift(self.lookback) - 1)
        
        # Initialize signals
        signals = pd.Series(index=data.index, data=0)
        
        # Generate signals based on threshold
        signals[momentum > self.threshold] = 1   # Buy on strong positive momentum
        signals[momentum < -self.threshold] = -1  # Sell on strong negative momentum
        
        return signals