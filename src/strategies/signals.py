
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseSignalGenerator(ABC):
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class MLSignalGenerator(BaseSignalGenerator):
    
    def __init__(self, model: torch.nn.Module, 
                 preprocessor: Any,
                 config: Dict[str, Any]):

        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:

        # Preprocess data
        features = self.preprocessor.add_technical_indicators(data)
        feature_columns = self.preprocessor._select_features(features)
        scaled_features = self.preprocessor.scale_data(features[feature_columns], fit=False)
        
        # Create sequences
        sequences, _ = self.preprocessor.create_sequences(scaled_features)
        
        # Generate predictions
        predictions = []
        
        with torch.no_grad():
            for seq in sequences:
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                pred = self.model(seq_tensor).cpu().numpy()
                predictions.append(pred[0])
        
        predictions = np.array(predictions)
        
        # Inverse transform predictions
        predictions = self.preprocessor.inverse_transform_predictions(predictions)
        
        # Generate signals based on predictions
        signals = self._predictions_to_signals(predictions, data, sequences)
        
        return signals
    
    def _predictions_to_signals(self, predictions: np.ndarray, 
                               data: pd.DataFrame,
                               sequences: np.ndarray) -> pd.DataFrame:
        seq_length = self.preprocessor.sequence_length
        
        # Get current prices (aligned with predictions)
        current_prices = data['close'].values[seq_length:seq_length + len(predictions)]
        
        # Calculate expected returns
        if len(predictions.shape) > 1:
            # Multi-horizon predictions - use average
            pred_prices = np.mean(predictions, axis=1)
        else:
            pred_prices = predictions
        
        expected_returns = (pred_prices - current_prices) / current_prices
        
        # Generate signals based on thresholds
        buy_threshold = self.config.get('buy_threshold', 0.02)  # 2% expected gain
        sell_threshold = self.config.get('sell_threshold', -0.02)  # 2% expected loss
        
        signals = np.where(expected_returns > buy_threshold, 1,
                          np.where(expected_returns < sell_threshold, -1, 0))
        
        # Apply filters
        if self.config.get('volume_filter', True):
            signals = self._apply_volume_filter(signals, data, seq_length)
        
        if self.config.get('trend_filter', True):
            signals = self._apply_trend_filter(signals, data, seq_length)
        
        # Create signals DataFrame
        signal_index = data.index[seq_length:seq_length + len(signals)]
        signals_df = pd.DataFrame(signals, index=signal_index, columns=['signal'])
        
        return signals_df
    
    def _apply_volume_filter(self, signals: np.ndarray, 
                            data: pd.DataFrame,
                            offset: int) -> np.ndarray:
        volumes = data['volume'].values[offset:offset + len(signals)]
        avg_volume = np.mean(volumes)
        
        # Only trade when volume is above average
        volume_filter = volumes > avg_volume * 0.8
        signals = signals * volume_filter
        
        return signals
    
    def _apply_trend_filter(self, signals: np.ndarray,
                           data: pd.DataFrame,
                           offset: int) -> np.ndarray:
        prices = data['close'].values[offset:offset + len(signals)]
        
        # Calculate short-term trend
        if len(prices) > 20:
            sma_20 = pd.Series(prices).rolling(20).mean().values
            
            # Only buy in uptrend, only sell in downtrend
            uptrend = prices > sma_20
            downtrend = prices < sma_20
            
            # Filter buy signals to uptrends
            signals = np.where((signals == 1) & ~uptrend, 0, signals)
            # Filter sell signals to downtrends  
            signals = np.where((signals == -1) & ~downtrend, 0, signals)
        
        return signals