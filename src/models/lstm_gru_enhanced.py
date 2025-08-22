import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
import math


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalAttention(nn.Module):
    
    def __init__(self, hidden_dim: int, n_heads: int = 8):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)


class WaveletTransform(nn.Module):
    
    def __init__(self, input_dim: int, wavelet_scales: List[int] = None):
        super().__init__()
        
        if wavelet_scales is None:
            wavelet_scales = [2, 4, 8, 16, 32]
        
        self.scales = wavelet_scales
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size=scale, stride=1, padding=scale//2)
            for scale in self.scales
        ])
        
        self.fusion = nn.Linear(input_dim * len(self.scales), input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = x.shape
        x_transposed = x.transpose(1, 2)
        
        multi_scale = []
        for conv in self.convs:
            scale_feat = conv(x_transposed)
            if scale_feat.size(2) != seq_len:
                scale_feat = F.interpolate(scale_feat, size=seq_len, mode='linear', align_corners=False)
            multi_scale.append(scale_feat.transpose(1, 2))
        
        concatenated = torch.cat(multi_scale, dim=-1)
        return self.fusion(concatenated)


class AdaptiveGating(nn.Module):
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.lstm_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.gru_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, lstm_out: torch.Tensor, gru_out: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([lstm_out, gru_out], dim=-1)
        
        lstm_weight = self.lstm_gate(combined)
        gru_weight = self.gru_gate(combined)
        
        total_weight = lstm_weight + gru_weight + 1e-8
        lstm_weight = lstm_weight / total_weight
        gru_weight = gru_weight / total_weight
        
        return lstm_weight * lstm_out + gru_weight * gru_out


class EnhancedLSTMGRU(nn.Module):
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 n_layers: int = 4,
                 dropout: float = 0.2,
                 use_attention: bool = True,
                 use_wavelet: bool = True,
                 use_positional: bool = True,
                 bidirectional: bool = True,
                 output_dim: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.direction_multiplier = 2 if bidirectional else 1
        
        if use_wavelet:
            self.wavelet = WaveletTransform(input_dim)
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        if use_positional:
            self.positional = PositionalEncoding(hidden_dim)
        
        self.lstm_layers = nn.ModuleList()
        self.gru_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(n_layers):
            input_size = hidden_dim if i == 0 else hidden_dim * self.direction_multiplier
            
            self.lstm_layers.append(
                nn.LSTM(input_size, hidden_dim, batch_first=True, 
                       bidirectional=bidirectional, dropout=dropout if i < n_layers - 1 else 0)
            )
            
            self.gru_layers.append(
                nn.GRU(input_size, hidden_dim, batch_first=True,
                      bidirectional=bidirectional, dropout=dropout if i < n_layers - 1 else 0)
            )
            
            self.layer_norms.append(nn.LayerNorm(hidden_dim * self.direction_multiplier))
            self.dropouts.append(nn.Dropout(dropout))
        
        self.adaptive_gates = nn.ModuleList([
            AdaptiveGating(hidden_dim * self.direction_multiplier) 
            for _ in range(n_layers)
        ])
        
        if use_attention:
            self.attention_layers = nn.ModuleList([
                TemporalAttention(hidden_dim * self.direction_multiplier)
                for _ in range(n_layers)
            ])
        else:
            self.attention_layers = None
        
        feature_dim = hidden_dim * self.direction_multiplier
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.output_network = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, output_dim)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
    
    def forward(self, 
                x: torch.Tensor,
                return_features: bool = False,
                return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len, _ = x.shape
        
        if hasattr(self, 'wavelet'):
            x = self.wavelet(x)
        
        x = self.input_projection(x)
        
        if hasattr(self, 'positional'):
            x = self.positional(x)
        
        hidden_states = []
        
        for i in range(self.n_layers):
            lstm_out, _ = self.lstm_layers[i](x)
            gru_out, _ = self.gru_layers[i](x)
            
            x = self.adaptive_gates[i](lstm_out, gru_out)
            
            x = self.layer_norms[i](x)
            x = self.dropouts[i](x)
            
            if self.attention_layers is not None:
                x = self.attention_layers[i](x)
            
            hidden_states.append(x)
        
        x_transposed = x.transpose(1, 2)
        max_pooled = self.global_max_pool(x_transposed).squeeze(-1)
        avg_pooled = self.global_avg_pool(x_transposed).squeeze(-1)
        last_hidden = x[:, -1, :]
        
        combined = torch.cat([max_pooled, avg_pooled, last_hidden], dim=-1)
        
        output = self.output_network(combined)
        
        result = {
            'predictions': output,
            'probabilities': F.softmax(output, dim=-1),
            'direction': torch.argmax(output, dim=-1)
        }
        
        if return_features:
            result['features'] = combined
            result['hidden_states'] = hidden_states
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(combined)
            result['uncertainty_mean'] = uncertainty[:, 0]
            result['uncertainty_std'] = F.softplus(uncertainty[:, 1])
        
        return result


class LSTMGRUEnsemble(nn.Module):
    
    def __init__(self,
                 input_dim: int,
                 n_models: int = 3,
                 hidden_dims: List[int] = None,
                 output_dim: int = 3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        
        self.models = nn.ModuleList([
            EnhancedLSTMGRU(
                input_dim=input_dim,
                hidden_dim=hidden_dims[i % len(hidden_dims)],
                n_layers=3 + i,
                dropout=0.2 + 0.05 * i,
                use_attention=(i % 2 == 0),
                use_wavelet=(i > 0),
                bidirectional=True,
                output_dim=output_dim
            )
            for i in range(n_models)
        ])
        
        self.ensemble_weights = nn.Parameter(torch.ones(n_models) / n_models)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_predictions = []
        all_uncertainties = []
        
        for model in self.models:
            output = model(x, return_uncertainty=True)
            all_predictions.append(output['predictions'])
            
            if 'uncertainty_mean' in output:
                all_uncertainties.append(output['uncertainty_std'])
        
        predictions_stack = torch.stack(all_predictions)
        
        weights = F.softmax(self.ensemble_weights, dim=0)
        weights = weights.view(-1, 1, 1)
        
        ensemble_pred = (predictions_stack * weights).sum(dim=0)
        
        result = {
            'predictions': ensemble_pred,
            'probabilities': F.softmax(ensemble_pred, dim=-1),
            'direction': torch.argmax(ensemble_pred, dim=-1),
            'individual_predictions': all_predictions
        }
        
        if all_uncertainties:
            uncertainties_stack = torch.stack(all_uncertainties)
            ensemble_uncertainty = (uncertainties_stack * weights.squeeze(-1)).sum(dim=0)
            result['uncertainty'] = ensemble_uncertainty
        
        return result


def create_lstm_gru_for_crypto(
    input_dim: int = 32,
    ensemble: bool = False,
    device: str = 'cuda'
) -> nn.Module:
    
    if ensemble:
        model = LSTMGRUEnsemble(
            input_dim=input_dim,
            n_models=3,
            hidden_dims=[256, 384, 512],
            output_dim=3
        )
    else:
        model = EnhancedLSTMGRU(
            input_dim=input_dim,
            hidden_dim=384,
            n_layers=4,
            dropout=0.2,
            use_attention=True,
            use_wavelet=True,
            use_positional=True,
            bidirectional=True,
            output_dim=3
        )
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
    
    return model