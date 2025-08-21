#!/usr/bin/env python3
"""
Advanced ML models including deep learning and ensemble methods
Comprehensive implementation of LSTM, GRU, Transformer, and Ensemble models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import pickle
import json

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

# ============================================================================
# PYTORCH DATASETS
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# ADVANCED LSTM MODEL
# ============================================================================

class AdvancedLSTM(nn.Module):
    """Advanced LSTM with attention mechanism and residual connections"""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 output_size: int = 1,
                 use_attention: bool = True,
                 use_residual: bool = True):
        super(AdvancedLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention layer
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
        
        # Residual connection projection
        if use_residual:
            self.residual_projection = nn.Linear(input_size, hidden_size * 2)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            # Apply attention mechanism
            attention_weights = F.softmax(self.attention(lstm_out), dim=1)
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)
            out = context_vector
        else:
            # Take the last output
            out = lstm_out[:, -1, :]
        
        # Add residual connection
        if self.use_residual:
            residual = self.residual_projection(x[:, -1, :])
            out = out + residual
        
        # Fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        
        # Batch normalization (handle both training and eval)
        if out.shape[0] > 1:
            out = self.batch_norm(out)
        
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out


# ============================================================================
# ADVANCED GRU MODEL
# ============================================================================

class AdvancedGRU(nn.Module):
    """Advanced GRU with attention and layer normalization"""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 output_size: int = 1,
                 use_attention: bool = True):
        super(AdvancedGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Layer normalization
        gru_out = self.layer_norm(gru_out)
        
        if self.use_attention:
            # Self-attention
            attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            out = attn_out[:, -1, :]
        else:
            out = gru_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out


# ============================================================================
# ADVANCED TRANSFORMER MODEL
# ============================================================================

class AdvancedTransformer(nn.Module):
    """Advanced Transformer with custom positional encoding and multi-scale attention"""
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 d_ff: int = 1024,
                 dropout: float = 0.3,
                 output_size: int = 1,
                 max_seq_length: int = 500):
        super(AdvancedTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model * 2, d_model)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc2 = nn.Linear(d_model, d_model // 2)
        self.fc3 = nn.Linear(d_model // 2, output_size)
        
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Global pooling (combine avg and max pooling)
        # Transpose for pooling
        transformer_out_t = transformer_out.transpose(1, 2)
        avg_pool = self.global_avg_pool(transformer_out_t).squeeze(-1)
        max_pool = self.global_max_pool(transformer_out_t).squeeze(-1)
        
        # Concatenate pooled features
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Output layers
        out = self.dropout(pooled)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out


# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

class DeepEnsembleModel:
    """Ensemble of deep learning and traditional ML models"""
    
    def __init__(self,
                 models: Optional[Dict[str, Any]] = None,
                 ensemble_method: str = 'weighted_average',
                 use_stacking: bool = False):
        """
        Initialize ensemble model
        
        Args:
            models: Dictionary of models to ensemble
            ensemble_method: 'weighted_average', 'voting', 'stacking'
            use_stacking: Whether to use meta-learner for stacking
        """
        self.ensemble_method = ensemble_method
        self.use_stacking = use_stacking
        
        if models is None:
            # Default ensemble
            self.models = {}
        else:
            self.models = models
        
        self.weights = None
        self.meta_learner = None
        self.scalers = {}
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = {
            'model': model,
            'weight': weight,
            'type': self._get_model_type(model)
        }
        logger.info(f"Added {name} to ensemble with weight {weight}")
    
    def _get_model_type(self, model):
        """Determine model type"""
        if isinstance(model, nn.Module):
            return 'pytorch'
        elif hasattr(model, 'predict'):
            return 'sklearn'
        else:
            return 'unknown'
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Fit all models in the ensemble
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        # Train each model
        for name, model_info in self.models.items():
            logger.info(f"Training {name}...")
            
            model = model_info['model']
            model_type = model_info['type']
            
            if model_type == 'pytorch':
                # Train PyTorch model
                self._train_pytorch_model(model, X_train, y_train, X_val, y_val, **kwargs)
            elif model_type == 'sklearn':
                # Train sklearn model
                model.fit(X_train, y_train)
            
            logger.info(f"Completed training {name}")
        
        # Optimize weights if using weighted average
        if self.ensemble_method == 'weighted_average' and X_val is not None:
            self._optimize_weights(X_val, y_val)
        
        # Train meta-learner if using stacking
        if self.use_stacking and X_val is not None:
            self._train_meta_learner(X_val, y_val)
    
    def _train_pytorch_model(self, model, X_train, y_train, X_val, y_val, 
                           epochs=50, batch_size=32, learning_rate=0.001):
        """Train a PyTorch model"""
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training setup
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if X_val is not None:
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
    
    def _optimize_weights(self, X_val, y_val):
        """Optimize ensemble weights using validation data"""
        from scipy.optimize import minimize
        
        # Get predictions from each model
        predictions = []
        for name, model_info in self.models.items():
            pred = self._predict_single(model_info['model'], model_info['type'], X_val)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Optimization objective
        def objective(weights):
            weighted_pred = np.average(predictions, axis=0, weights=weights)
            mse = np.mean((weighted_pred - y_val) ** 2)
            return mse
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        # Initial guess
        initial_weights = np.array([1/len(self.models)] * len(self.models))
        
        # Optimize
        result = minimize(objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.weights = result.x
        
        # Update model weights
        for i, (name, model_info) in enumerate(self.models.items()):
            model_info['weight'] = self.weights[i]
            logger.info(f"Optimized weight for {name}: {self.weights[i]:.3f}")
    
    def _train_meta_learner(self, X_val, y_val):
        """Train meta-learner for stacking"""
        # Get predictions from base models
        meta_features = []
        
        for name, model_info in self.models.items():
            pred = self._predict_single(model_info['model'], model_info['type'], X_val)
            meta_features.append(pred)
        
        meta_features = np.column_stack(meta_features)
        
        # Train meta-learner (simple linear model)
        from sklearn.linear_model import Ridge
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(meta_features, y_val)
        
        logger.info("Trained meta-learner for stacking ensemble")
    
    def predict(self, X):
        """Generate ensemble predictions"""
        predictions = []
        
        # Get predictions from each model
        for name, model_info in self.models.items():
            pred = self._predict_single(model_info['model'], model_info['type'], X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Combine predictions based on method
        if self.use_stacking and self.meta_learner is not None:
            # Stacking
            meta_features = np.column_stack(predictions)
            ensemble_pred = self.meta_learner.predict(meta_features)
            
        elif self.ensemble_method == 'weighted_average':
            # Weighted average
            if self.weights is not None:
                weights = self.weights
            else:
                weights = [model_info['weight'] for model_info in self.models.values()]
                weights = np.array(weights) / np.sum(weights)
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
        elif self.ensemble_method == 'voting':
            # Voting (for classification-style predictions)
            ensemble_pred = np.sign(np.mean(np.sign(predictions), axis=0))
            
        else:
            # Simple average
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def _predict_single(self, model, model_type, X):
        """Get predictions from a single model"""
        if model_type == 'pytorch':
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                predictions = model(X_tensor).cpu().numpy().squeeze()
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def save(self, filepath: str):
        """Save ensemble model"""
        save_dict = {
            'ensemble_method': self.ensemble_method,
            'use_stacking': self.use_stacking,
            'weights': self.weights,
            'models': {}
        }
        
        # Save each model
        for name, model_info in self.models.items():
            if model_info['type'] == 'pytorch':
                model_path = f"{filepath}_{name}.pt"
                torch.save(model_info['model'].state_dict(), model_path)
                save_dict['models'][name] = {
                    'type': 'pytorch',
                    'path': model_path,
                    'weight': model_info['weight']
                }
            else:
                model_path = f"{filepath}_{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info['model'], f)
                save_dict['models'][name] = {
                    'type': 'sklearn',
                    'path': model_path,
                    'weight': model_info['weight']
                }
        
        # Save meta-learner if exists
        if self.meta_learner is not None:
            meta_path = f"{filepath}_meta.pkl"
            with open(meta_path, 'wb') as f:
                pickle.dump(self.meta_learner, f)
            save_dict['meta_learner_path'] = meta_path
        
        # Save main config
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        logger.info(f"Ensemble model saved to {filepath}")


# ============================================================================
# HYBRID CNN-LSTM MODEL
# ============================================================================

class CNNLSTMModel(nn.Module):
    """Hybrid CNN-LSTM model for capturing both local and temporal patterns"""
    
    def __init__(self,
                 input_size: int,
                 cnn_filters: List[int] = [64, 128, 256],
                 kernel_sizes: List[int] = [3, 5, 7],
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.3,
                 output_size: int = 1):
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_size
        
        for filters, kernel_size in zip(cnn_filters, kernel_sizes):
            conv = nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv)
            in_channels = filters
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_hidden, output_size)
        
    def forward(self, x):
        # Reshape for CNN (batch, channels, length)
        x = x.transpose(1, 2)
        
        # CNN forward pass
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape for LSTM (batch, length, channels)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def create_default_ensemble():
    """Create a default ensemble with multiple models"""
    
    # Initialize ensemble
    ensemble = DeepEnsembleModel(ensemble_method='weighted_average')
    
    # Add traditional ML models
    ensemble.add_model('rf', RandomForestRegressor(n_estimators=100, max_depth=10), weight=1.0)
    ensemble.add_model('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1), weight=1.2)
    ensemble.add_model('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1), weight=1.3)
    
    # Note: Deep learning models need proper input dimensions
    # They would be added after knowing the feature dimensions
    
    logger.info("Created default ensemble with traditional ML models")
    return ensemble


def main():
    """Test advanced models"""
    print("Advanced Models Module Loaded")
    print("\nAvailable Models:")
    print("  - AdvancedLSTM: Bidirectional LSTM with attention and residual connections")
    print("  - AdvancedGRU: Bidirectional GRU with multi-head attention")
    print("  - AdvancedTransformer: Transformer with positional encoding and pooling")
    print("  - CNNLSTMModel: Hybrid CNN-LSTM for local and temporal patterns")
    print("  - DeepEnsembleModel: Ensemble with stacking and weight optimization")
    print("\nDevice:", device)
    
if __name__ == "__main__":
    main()