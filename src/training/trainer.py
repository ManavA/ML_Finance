# src/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for cryptocurrency models."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], 
                 device: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Device to use (cuda/cpu)
        """
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', True) and self.device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient clipping
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        ) if config['training']['early_stopping']['enabled'] else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Logging
        self.use_wandb = config['logging']['wandb']['enabled']
        if self.use_wandb:
            self._init_wandb()
        
        # Paths
        self.model_dir = Path(config['paths']['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_type = self.config['training'].get('optimizer', 'adam')
        lr = self.config['training'].get('learning_rate', 0.001)
        
        if opt_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        elif opt_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        elif opt_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif opt_type == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['epochs']
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config['logging']['wandb']['project'],
            entity=self.config['logging']['wandb']['entity'],
            config=self.config,
            name=f"{self.config['model']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb.watch(self.model)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (features, targets) in enumerate(progress_bar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc='Validation'):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(features)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate additional metrics
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        
        metrics = calculate_metrics(all_outputs, all_targets)
        metrics['loss'] = avg_loss
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            
        Returns:
            Training history and metrics
        """
        num_epochs = self.config['training']['epochs']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model has {self.model.get_num_parameters():,} parameters")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            logger.info(f"Train Loss: {train_loss:.6f}")
            logger.info(f"Val Loss: {val_loss:.6f}")
            logger.info(f"Val Metrics: {val_metrics}")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_loss):
                    logger.info("Early stopping triggered")
                    break
        
        # Test evaluation
        test_metrics = None
        if test_loader:
            logger.info("\nEvaluating on test set...")
            test_loss, test_metrics = self.validate(test_loader)
            logger.info(f"Test Loss: {test_loss:.6f}")
            logger.info(f"Test Metrics: {test_metrics}")
        
        # Load best model
        if self.best_model_path:
            self.load_checkpoint(self.best_model_path)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics
        }
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.model_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.model_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            logger.info(f"Saved best model with val_loss: {val_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


# src/training/callbacks.py
class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


# src/training/metrics.py
def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate various metrics for evaluation.
    
    Args:
        predictions: Model predictions
        targets: True values
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays if needed
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # MSE
    mse = np.mean((predictions - targets) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # MAPE
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    
    # R2 Score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Direction accuracy (for price movement)
    if len(predictions) > 1:
        pred_direction = np.diff(predictions) > 0
        true_direction = np.diff(targets) > 0
        direction_accuracy = np.mean(pred_direction == true_direction)
    else:
        direction_accuracy = 0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'direction_accuracy': float(direction_accuracy)
    }