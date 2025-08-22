#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class LSTMGRUHybrid(nn.Module):

    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 architecture: str = 'parallel'):
      
        super(LSTMGRUHybrid, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.architecture = architecture
        
        if architecture == 'parallel':
            # LSTM and GRU process in parallel
            self.lstm = nn.LSTM(
                input_size, 
                hidden_size, 
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            self.gru = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            # Combine outputs
            self.fc = nn.Linear(hidden_size * 2, output_size)
            
        elif architecture == 'sequential':
            # LSTM feeds into GRU
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            self.gru = nn.GRU(
                hidden_size,  # Takes LSTM output
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            self.fc = nn.Linear(hidden_size, output_size)
            
        elif architecture == 'stacked':
            # Alternating LSTM and GRU layers
            self.layers = nn.ModuleList()
            
            # First layer
            self.layers.append(nn.LSTM(
                input_size,
                hidden_size,
                1,
                batch_first=True
            ))
            
            # Alternating layers
            for i in range(1, num_layers * 2):
                if i % 2 == 1:
                    # GRU layer
                    self.layers.append(nn.GRU(
                        hidden_size,
                        hidden_size,
                        1,
                        batch_first=True
                    ))
                else:
                    # LSTM layer
                    self.layers.append(nn.LSTM(
                        hidden_size,
                        hidden_size,
                        1,
                        batch_first=True
                    ))
            
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)
            
        elif architecture == 'ensemble':
            # Multiple models with different configs
            self.models = nn.ModuleList()
            
            # Model 1: 2-layer LSTM + 1-layer GRU
            self.models.append(nn.Sequential(
                nn.LSTM(input_size, hidden_size, 2, batch_first=True),
            ))
            
            # Model 2: 1-layer LSTM + 2-layer GRU  
            self.models.append(nn.Sequential(
                nn.GRU(input_size, hidden_size, 2, batch_first=True),
            ))
            
            # Model 3: Mixed shallow
            self.models.append(nn.Sequential(
                nn.LSTM(input_size, hidden_size//2, 1, batch_first=True),
            ))
            
            self.models.append(nn.Sequential(
                nn.GRU(input_size, hidden_size//2, 1, batch_first=True),
            ))
            
            # Ensemble combination
            self.fc = nn.Linear(hidden_size * 2 + hidden_size, output_size)
            
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        
        if self.architecture == 'parallel':
            # Process through both LSTM and GRU
            lstm_out, _ = self.lstm(x)
            gru_out, _ = self.gru(x)
            
            # Take last timestep
            lstm_out = lstm_out[:, -1, :]
            gru_out = gru_out[:, -1, :]
            
            # Concatenate
            combined = torch.cat([lstm_out, gru_out], dim=1)
            combined = self.dropout(combined)
            
            # Final prediction
            output = self.fc(combined)
            
        elif self.architecture == 'sequential':
            # LSTM -> GRU
            lstm_out, _ = self.lstm(x)
            gru_out, _ = self.gru(lstm_out)
            
            # Take last timestep
            gru_out = gru_out[:, -1, :]
            gru_out = self.dropout(gru_out)
            
            # Final prediction
            output = self.fc(gru_out)
            
        elif self.architecture == 'stacked':
            # Process through alternating layers
            out = x
            for layer in self.layers:
                if isinstance(layer, nn.LSTM):
                    out, _ = layer(out)
                else:  # GRU
                    out, _ = layer(out)
                out = self.dropout(out)
            
            # Take last timestep
            out = out[:, -1, :]
            output = self.fc(out)
            
        elif self.architecture == 'ensemble':
            outputs = []
            
            # Get predictions from each model
            for model in self.models:
                if isinstance(model[0], nn.LSTM):
                    model_out, _ = model[0](x)
                else:  # GRU
                    model_out, _ = model[0](x) 
                    
                # Take last timestep
                model_out = model_out[:, -1, :]
                outputs.append(model_out)
            
            # Combine all outputs
            combined = torch.cat(outputs, dim=1)
            combined = self.dropout(combined)
            
            # Final prediction
            output = self.fc(combined)
        
        return output


class MultiArchitectureHybrid(nn.Module):

    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 output_size: int = 1,
                 dropout: float = 0.2):
        super(MultiArchitectureHybrid, self).__init__()
        
        # Create different architectures
        self.parallel_model = LSTMGRUHybrid(
            input_size, hidden_size, 2, output_size, dropout, 'parallel'
        )
        
        self.sequential_model = LSTMGRUHybrid(
            input_size, hidden_size, 2, output_size, dropout, 'sequential'
        )
        
        self.stacked_model = LSTMGRUHybrid(
            input_size, hidden_size, 2, output_size, dropout, 'stacked'
        )
        
        # Attention mechanism to weight models
        self.attention = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 models
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get predictions from each architecture
        parallel_out = self.parallel_model(x)
        sequential_out = self.sequential_model(x)
        stacked_out = self.stacked_model(x)
        
        # Stack predictions
        predictions = torch.stack([parallel_out, sequential_out, stacked_out], dim=1)
        
        # Get attention weights based on input
        # Use mean of input features across time
        x_summary = x.mean(dim=1)  # (batch, features)
        weights = self.attention(x_summary)  # (batch, 3)
        
        # Apply attention weights
        weights = weights.unsqueeze(2)  # (batch, 3, 1)
        weighted_pred = (predictions * weights).sum(dim=1)
        
        return weighted_pred


class LSTMGRUTrainer:
    
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, 
                   train_loader: torch.utils.data.DataLoader,
                   epoch: int) -> float:
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                predictions.extend(output.cpu().numpy())
                actuals.extend(target.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        # Directional accuracy
        direction_correct = np.mean((predictions > 0) == (actuals > 0))
        
        return {
            'loss': total_loss / len(val_loader),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_correct
        }


def create_hybrid_models() -> Dict[str, nn.Module]:
    models = {
        'parallel': LSTMGRUHybrid(
            input_size=100,  # Adjust based on features
            hidden_size=128,
            num_layers=2,
            architecture='parallel'
        ),
        'sequential': LSTMGRUHybrid(
            input_size=100,
            hidden_size=128,
            num_layers=2,
            architecture='sequential'
        ),
        'stacked': LSTMGRUHybrid(
            input_size=100,
            hidden_size=128,
            num_layers=2,
            architecture='stacked'
        ),
        'multi_arch': MultiArchitectureHybrid(
            input_size=100,
            hidden_size=128
        )
    }
    
    return models


def demo_hybrid_models():
    print("="*60)
    print("LSTM-GRU HYBRID MODELS")
    print("="*60)
    
    # Create sample data
    batch_size = 32
    seq_len = 50
    input_size = 100
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Test each architecture
    architectures = ['parallel', 'sequential', 'stacked']
    
    for arch in architectures:
        print(f"\n{arch.upper()} Architecture:")
        print("-"*40)
        
        model = LSTMGRUHybrid(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            architecture=arch
        )
        
        # Forward pass
        output = model(x)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Output shape: {output.shape}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Test multi-architecture ensemble
    print(f"\nMULTI-ARCHITECTURE ENSEMBLE:")
    print("-"*40)
    
    multi_model = MultiArchitectureHybrid(
        input_size=input_size,
        hidden_size=64
    )
    
    output = multi_model(x)
    
    total_params = sum(p.numel() for p in multi_model.parameters())
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print("\nEnsemble combines 3 architectures with attention weighting")
    


if __name__ == "__main__":
    demo_hybrid_models()