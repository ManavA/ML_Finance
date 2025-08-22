# src/strategies/deep_learning.py
import torch
import pickle
import numpy as np
import pandas as pd
from src.models.advanced_models import AdvancedGRU
from src.strategies.base_strategy import BaseStrategy

class AdvancedGRUStrategy(BaseStrategy):
    def __init__(self, model_path='../models/advanced_gru_model.pt', input_size=None):
        super().__init__("Advanced GRU")
        
        if input_size is None:
            input_size = 20  # Default feature size
            
        # Initialize the model architecture
        self.model = AdvancedGRU(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            output_size=1,
            use_attention=True
        )
        
        # Load trained weights if available
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
        except FileNotFoundError:
            print(f"Model weights not found at {model_path}, using untrained model")
        except Exception as e:
            print(f"Error loading model: {e}, using untrained model")
    
    def prepare_features(self, data):
        
        features_df = data.copy()
        
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features_df[f'ma_{window}'] = features_df['close'].rolling(window).mean()
            features_df[f'ma_ratio_{window}'] = features_df['close'] / features_df[f'ma_{window}']
        
        # RSI
        delta = features_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = features_df['close'].ewm(span=12, adjust=False).mean()
        exp2 = features_df['close'].ewm(span=26, adjust=False).mean()
        features_df['macd'] = exp1 - exp2
        features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # Bollinger Bands
        bb_window = 20
        bb_std = features_df['close'].rolling(bb_window).std()
        features_df['bb_middle'] = features_df['close'].rolling(bb_window).mean()
        features_df['bb_upper'] = features_df['bb_middle'] + (bb_std * 2)
        features_df['bb_lower'] = features_df['bb_middle'] - (bb_std * 2)
        features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # Volume features
        if 'volume' in features_df.columns:
            features_df['volume_ma'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma']
        
        # Select feature columns
        feature_cols = [
            'returns', 'log_returns', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50'
        ]
        
        if 'volume_ratio' in features_df.columns:
            feature_cols.append('volume_ratio')
        
        # Pad feature_cols to match expected input_size
        while len(feature_cols) < self.model.hidden_size:
            feature_cols.append(feature_cols[-1])  # Repeat last feature
        
        feature_cols = feature_cols[:20]  # Limit to input_size
        
        # Fill missing columns with zeros
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        return features_df[feature_cols].fillna(0)
    
    def generate_signals(self, data):
        # Prepare features
        features = self.prepare_features(data)
        
        # Ensure we have enough data
        if len(features) < 10:
            return pd.Series(0, index=data.index)
        
        # Convert to tensor for model input
        sequence_length = min(10, len(features))  # Use last 10 time steps
        
        # Get the last sequence_length samples
        feature_array = features.iloc[-sequence_length:].values
        
        # Reshape for model: (1, sequence_length, n_features)
        X = torch.FloatTensor(feature_array).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(X)
            prediction_value = prediction.item()
        
        # Convert prediction to signal
        signals = pd.Series(0, index=data.index)
        
        # Simple threshold-based signal generation
        if prediction_value > 0.6:
            signals.iloc[-1] = 1  # Buy signal
        elif prediction_value < 0.4:
            signals.iloc[-1] = -1  # Sell signal
        else:
            signals.iloc[-1] = 0   # Hold signal
            
        return signals

class InverseRLStrategy(BaseStrategy):
    def __init__(self, model_path='../models/irl_model.pkl'):
        super().__init__("Inverse RL")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def generate_signals(self, data):
        # Simple implementation using the model's reward function
        try:
            # Extract features for the model
            features = self._extract_features(data)
            
            # Get reward predictions from the IRL model
            rewards = self.model.predict_rewards(features)
            
            # Convert rewards to trading signals
            signals = pd.Series(0, index=data.index)
            
            # Use reward thresholds to generate signals
            high_threshold = np.percentile(rewards, 75)
            low_threshold = np.percentile(rewards, 25)
            
            signals[rewards > high_threshold] = 1   # Buy signal
            signals[rewards < low_threshold] = -1   # Sell signal
            
            return signals
            
        except Exception as e:
            print(f"Error in IRL signal generation: {e}")
            # Fallback to neutral signals
            return pd.Series(0, index=data.index)
    
    def _extract_features(self, data):
        features = []
        
        # Price features
        features.append(data['close'].pct_change().fillna(0))
        features.append(data['volume'].pct_change().fillna(0))
        
        # Simple technical indicators
        if len(data) >= 20:
            features.append((data['close'] - data['close'].rolling(20).mean()).fillna(0))
            features.append(data['close'].rolling(20).std().fillna(0))
        else:
            features.extend([pd.Series(0, index=data.index)] * 2)
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.fillna(50))
        
        return np.column_stack(features)