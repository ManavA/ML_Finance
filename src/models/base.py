# src/models/base.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    
    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
