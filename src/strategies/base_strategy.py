# src/strategies/base_strategy.py


import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class BaseStrategy(ABC):
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.signals = None
        self.positions = None
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def get_positions(self, signals: pd.Series) -> pd.Series:
        positions = pd.Series(index=signals.index, data=0)
        positions[signals == 1] = 1
        
        # Forward fill positions (stay in position until sell signal)
        positions = positions.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Exit on sell signals
        positions[signals == -1] = 0
        
        return positions
    
    def validate_data(self, data: pd.DataFrame) -> None:
        required_columns = ['close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        return data.copy()