# src/models/transformer.py
import math


class PositionalEncoding(nn.Module):
    
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
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):

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
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

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


def create_model(model_type: str, input_size: int, output_size: int, 
                config: Dict[str, Any]) -> BaseModel:
    models = {
        'gru': GRUModel,
        'lstm': LSTMModel,
        'transformer': TransformerModel
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = models[model_type.lower()]
    return model_class(input_size, output_size, config)