# src/data/datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CryptoDataset(Dataset):
    
    
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 transform: Optional = None):
        
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx]
        targets = self.targets[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, targets


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    test_dataset = CryptoDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader