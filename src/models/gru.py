# src/models/gru.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class AttentionLayer(nn.Module):
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.
        
        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Calculate attention scores
        scores = self.attention(hidden_states)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights
        context = torch.sum(weights * hidden_states, dim=1)  # (batch, hidden_size)
        
        return context, weights.squeeze(-1)


class GRUModel(BaseModel):
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):

        super().__init__(input_size, output_size, config)
        
        self.hidden_size = config.get('hidden_size', 256)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.2)
        self.bidirectional = config.get('bidirectional', True)
        self.use_attention = config.get('attention', True)
        self.normalization = config.get('normalization', 'layer')
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Calculate the actual hidden size after GRU
        gru_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Normalization layers
        if self.normalization == 'layer':
            self.norm = nn.LayerNorm(gru_output_size)
        elif self.normalization == 'batch':
            self.norm = nn.BatchNorm1d(gru_output_size)
        else:
            self.norm = nn.Identity()
        
        # Attention layer
        if self.use_attention:
            self.attention = AttentionLayer(gru_output_size)
        
        # Output layers
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(gru_output_size, gru_output_size // 2)
        self.fc2 = nn.Linear(gru_output_size // 2, output_size)
        
        # Activation
        self.activation = nn.GELU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # GRU forward pass
        gru_out, hidden = self.gru(x)  # (batch, seq_len, hidden_size * num_directions)
        
        # Apply normalization
        if self.normalization == 'batch':
            # Reshape for batch norm
            batch_size, seq_len, features = gru_out.shape
            gru_out = gru_out.reshape(-1, features)
            gru_out = self.norm(gru_out)
            gru_out = gru_out.reshape(batch_size, seq_len, features)
        else:
            gru_out = self.norm(gru_out)
        
        # Apply attention if enabled
        if self.use_attention:
            context, _ = self.attention(gru_out)
            out = context
        else:
            # Use last hidden state
            out = gru_out[:, -1, :]
        
        # Pass through output layers
        out = self.dropout_layer(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out


# src/models/lstm.py
class LSTMModel(BaseModel):
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):

        super().__init__(input_size, output_size, config)
        
        self.hidden_size = config.get('hidden_size', 256)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.3)
        self.bidirectional = config.get('bidirectional', True)
        self.use_attention = config.get('attention', True)
        self.normalization = config.get('normalization', 'layer')
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Calculate the actual hidden size after LSTM
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Normalization layers
        if self.normalization == 'layer':
            self.norm = nn.LayerNorm(lstm_output_size)
        elif self.normalization == 'batch':
            self.norm = nn.BatchNorm1d(lstm_output_size)
        else:
            self.norm = nn.Identity()
        
        # Attention layer
        if self.use_attention:
            self.attention = AttentionLayer(lstm_output_size)
        
        # Output layers
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.fc2 = nn.Linear(lstm_output_size // 2, output_size)
        
        # Activation
        self.activation = nn.GELU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply normalization
        if self.normalization == 'batch':
            batch_size, seq_len, features = lstm_out.shape
            lstm_out = lstm_out.reshape(-1, features)
            lstm_out = self.norm(lstm_out)
            lstm_out = lstm_out.reshape(batch_size, seq_len, features)
        else:
            lstm_out = self.norm(lstm_out)
        
        # Apply attention if enabled
        if self.use_attention:
            context, _ = self.attention(lstm_out)
            out = context
        else:
            # Use last hidden state
            out = lstm_out[:, -1, :]
        
        # Pass through output layers
        out = self.dropout_layer(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out
