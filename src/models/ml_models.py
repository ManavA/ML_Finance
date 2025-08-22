#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("Using CPU")

class TimeSeriesDataset(Dataset):
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class GRUModel(nn.Module):
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Take the last output
        last_out = gru_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class TransformerModel(nn.Module):
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model // 2, output_size)
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Take the last output
        last_out = transformer_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class MLModelTrainer:
    
    def __init__(self, 
                 model_type: str = 'lstm',
                 sequence_length: int = 48,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 epochs: int = 50,
                 early_stopping_patience: int = 10):
   
        self.model_type = model_type.lower()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def prepare_sequences(self, data: pd.DataFrame, 
                         feature_cols: List[str],
                         target_col: str) -> Tuple[np.ndarray, np.ndarray]:
  
        X = data[feature_cols].values
        y = data[target_col].values
        
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.sequence_length):
            X_sequences.append(X[i:i + self.sequence_length])
            y_sequences.append(y[i + self.sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_deep_model(self, 
                        train_data: pd.DataFrame,
                        val_data: pd.DataFrame,
                        feature_cols: List[str],
                        target_col: str) -> nn.Module:

        # Prepare sequences
        X_train, y_train = self.prepare_sequences(train_data, feature_cols, target_col)
        X_val, y_val = self.prepare_sequences(val_data, feature_cols, target_col)
        
        # Create datasets and loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        input_size = len(feature_cols)
        
        if self.model_type == 'lstm':
            model = LSTMModel(input_size=input_size).to(device)
        elif self.model_type == 'gru':
            model = GRUModel(input_size=input_size).to(device)
        elif self.model_type == 'transformer':
            model = TransformerModel(input_size=input_size).to(device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
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
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        self.model = model
        
        return model
    
    def train_traditional_model(self,
                              train_data: pd.DataFrame,
                              val_data: pd.DataFrame,
                              feature_cols: List[str],
                              target_col: str):

        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_val = val_data[feature_cols].values
        y_val = val_data[target_col].values
        
        if self.model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gb':
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            model = Lasso(alpha=0.001)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Validate
        val_score = model.score(X_val, y_val)
        logger.info(f"{self.model_type.upper()} validation R² score: {val_score:.4f}")
        
        self.model = model
        return model
    
    def predict(self, test_data: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:

        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type in ['lstm', 'gru', 'transformer']:
            # Deep learning models
            X_test, _ = self.prepare_sequences(
                test_data, 
                feature_cols, 
                'close'  # Dummy target for sequence preparation
            )
            
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test).to(device)
                predictions = self.model(X_tensor).cpu().numpy().squeeze()
            
            return predictions
        else:
            # Traditional models
            X_test = test_data[feature_cols].values
            predictions = self.model.predict(X_test)
            return predictions