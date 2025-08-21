# src/models/base.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """Base class for all models."""
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):
        """
        Initialize base model.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            config: Model configuration
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# src/models/gru.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence models."""
    
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
    """GRU model with attention for cryptocurrency prediction."""
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):
        """
        Initialize GRU model.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            config: Model configuration
        """
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
        """Initialize model weights."""
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
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output predictions
        """
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
    """LSTM model with attention for cryptocurrency prediction."""
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            config: Model configuration
        """
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
        """Initialize model weights."""
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
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output predictions
        """
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


# src/models/transformer.py
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(BaseModel):
    """Transformer model for cryptocurrency prediction."""
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):
        """
        Initialize Transformer model.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            config: Model configuration
        """
        super().__init__(input_size, output_size, config)
        
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_encoder_layers = config.get('num_encoder_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 1024)
        self.dropout = config.get('dropout', 0.1)
        self.activation = config.get('activation', 'gelu')
        
        # Input projection
        self.input_projection = nn.Linear(input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=self.num_encoder_layers
        )
        
        # Output layers
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.fc2 = nn.Linear(self.dim_feedforward, output_size)
        
        # Activation
        self.output_activation = nn.GELU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output predictions
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch, d_model)
        
        # Pass through output layers
        x = self.dropout_layer(x)
        x = self.fc1(x)
        x = self.output_activation(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x


# Model factory
def create_model(model_type: str, input_size: int, output_size: int, 
                config: Dict[str, Any]) -> BaseModel:
    """
    Create a model based on type.
    
    Args:
        model_type: Type of model ('gru', 'lstm', 'transformer')
        input_size: Number of input features
        output_size: Number of output features
        config: Model configuration
        
    Returns:
        Model instance
    """
    models = {
        'gru': GRUModel,
        'lstm': LSTMModel,
        'transformer': TransformerModel
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = models[model_type.lower()]
    return model_class(input_size, output_size, config)