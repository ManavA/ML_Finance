"""
Model training commands for the CLI
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import pickle
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.advanced_models import AdvancedLSTM, AdvancedGRU, AdvancedTransformer, DeepEnsembleModel

@click.group(name='train')
@click.pass_context
def train_group(ctx):
    """Model training and management commands"""
    pass

@train_group.command(name='model')
@click.option('--symbol', '-s', required=True, help='Symbol to train on')
@click.option('--model-type', '-m', required=True,
              type=click.Choice(['xgboost', 'lightgbm', 'catboost', 'lstm', 'gru', 'transformer', 'ensemble']),
              help='Model type to train')
@click.option('--features', default='all', help='Feature set to use (technical, fundamental, all)')
@click.option('--start', help='Training start date')
@click.option('--end', help='Training end date')
@click.option('--validation-split', default=0.2, type=float, help='Validation split ratio')
@click.option('--save-path', help='Path to save trained model')
@click.pass_context
def train_model(ctx, symbol, model_type, features, start, end, validation_split, save_path):
    """Train a machine learning model"""
    logger = ctx.obj['logger']
    
    click.echo(f"Training {model_type} model for {symbol}...")
    
    # Load data
    cache_dir = Path('data/cache')
    data_files = list(cache_dir.glob(f'*{symbol}*'))
    
    if not data_files:
        click.echo(f"Error: No data found for {symbol}")
        return
    
    file = data_files[0]
    if file.suffix == '.parquet':
        df = pd.read_parquet(file)
    else:
        df = pd.read_csv(file, index_col=0, parse_dates=True)
    
    # Filter by date range
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    
    click.echo(f"Loaded {len(df)} data points")
    
    # Feature engineering
    click.echo("Creating features...")
    
    # Technical indicators
    if features in ['technical', 'all']:
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Remove NaN values
    df = df.dropna()
    
    # Split data
    split_idx = int(len(df) * (1 - validation_split))
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    
    click.echo(f"Training set: {len(X_train)} samples")
    click.echo(f"Validation set: {len(X_val)} samples")
    
    # Train model based on type
    if model_type == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            objective='binary:logistic'
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        
    elif model_type == 'lightgbm':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            objective='binary'
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lambda x: None])
        
    elif model_type == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.01,
            verbose=False
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        
    elif model_type == 'gru':
        # Use the actual AdvancedGRU model
        input_size = len(feature_cols)
        model = AdvancedGRU(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            output_size=1,
            use_attention=True
        )
        
        # Reshape data for sequential model (samples, timesteps, features)
        # For simplicity, using window of 1 timestep
        X_train_seq = X_train.values.reshape(-1, 1, input_size)
        X_val_seq = X_val.values.reshape(-1, 1, input_size)
        
        # Train using ensemble framework
        ensemble = DeepEnsembleModel()
        ensemble.add_model('advanced_gru', model, weight=1.0)
        ensemble.fit(X_train_seq, y_train.values, X_val_seq, y_val.values, epochs=50, batch_size=32)
        model = ensemble  # Save the ensemble
        
    elif model_type == 'lstm':
        # Use the actual AdvancedLSTM model
        input_size = len(feature_cols)
        model = AdvancedLSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            output_size=1,
            use_attention=True,
            use_residual=True
        )
        
        # Reshape data for sequential model
        X_train_seq = X_train.values.reshape(-1, 1, input_size)
        X_val_seq = X_val.values.reshape(-1, 1, input_size)
        
        # Train using ensemble framework
        ensemble = DeepEnsembleModel()
        ensemble.add_model('advanced_lstm', model, weight=1.0)
        ensemble.fit(X_train_seq, y_train.values, X_val_seq, y_val.values, epochs=50, batch_size=32)
        model = ensemble
        
    elif model_type == 'transformer':
        # Use the actual AdvancedTransformer model
        input_size = len(feature_cols)
        model = AdvancedTransformer(
            input_size=input_size,
            d_model=128,
            nhead=8,
            num_layers=4,
            dropout=0.3,
            output_size=1
        )
        
        # Reshape data for sequential model
        X_train_seq = X_train.values.reshape(-1, 1, input_size)
        X_val_seq = X_val.values.reshape(-1, 1, input_size)
        
        # Train using ensemble framework
        ensemble = DeepEnsembleModel()
        ensemble.add_model('advanced_transformer', model, weight=1.0)
        ensemble.fit(X_train_seq, y_train.values, X_val_seq, y_val.values, epochs=50, batch_size=32)
        model = ensemble
        
    elif model_type == 'ensemble':
        # Create a full ensemble with multiple models
        ensemble = DeepEnsembleModel(ensemble_method='weighted_average', use_stacking=True)
        
        # Add traditional ML models
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        
        xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.01)
        lgb_model = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.01)
        
        ensemble.add_model('xgboost', xgb_model, weight=1.0)
        ensemble.add_model('lightgbm', lgb_model, weight=1.0)
        
        # Add deep learning model
        input_size = len(feature_cols)
        gru_model = AdvancedGRU(input_size=input_size, hidden_size=64, num_layers=2)
        ensemble.add_model('gru', gru_model, weight=1.2)
        
        # Prepare data for mixed training
        X_train_seq = X_train.values.reshape(-1, 1, input_size)
        X_val_seq = X_val.values.reshape(-1, 1, input_size)
        
        ensemble.fit(X_train.values, y_train.values, X_val.values, y_val.values)
        model = ensemble
        
    else:
        click.echo(f"Model type {model_type} not yet implemented")
        return
    
    # Evaluate model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    if model_type in ['gru', 'lstm', 'transformer']:
        # For deep learning models, use ensemble predict
        train_pred = (model.predict(X_train.values.reshape(-1, 1, len(feature_cols))) > 0.5).astype(int)
        val_pred = (model.predict(X_val.values.reshape(-1, 1, len(feature_cols))) > 0.5).astype(int)
    elif model_type == 'ensemble':
        train_pred = (model.predict(X_train.values) > 0.5).astype(int)
        val_pred = (model.predict(X_val.values) > 0.5).astype(int)
    else:
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    click.echo("\n" + "="*50)
    click.echo("MODEL PERFORMANCE")
    click.echo("="*50)
    click.echo(f"Training Accuracy: {train_accuracy:.4f}")
    click.echo(f"Validation Accuracy: {val_accuracy:.4f}")
    click.echo(f"Validation Precision: {precision_score(y_val, val_pred):.4f}")
    click.echo(f"Validation Recall: {recall_score(y_val, val_pred):.4f}")
    click.echo(f"Validation F1-Score: {f1_score(y_val, val_pred):.4f}")
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        click.echo("\nTop 10 Important Features:")
        for idx, row in importance.head(10).iterrows():
            click.echo(f"  {row['feature']:20} {row['importance']:.4f}")
    
    # Save model
    if not save_path:
        save_path = f"models/{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on model type
    if model_type in ['gru', 'lstm', 'transformer', 'ensemble']:
        # For ensemble models, use their save method
        if hasattr(model, 'save'):
            model.save(save_path.replace('.pkl', ''))
        else:
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'feature_cols': feature_cols,
                    'symbol': symbol,
                    'model_type': model_type,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'trained_at': datetime.now().isoformat()
                }, f)
    else:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_cols': feature_cols,
                'symbol': symbol,
                'model_type': model_type,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'trained_at': datetime.now().isoformat()
            }, f)
    
    click.echo(f"\nModel saved to {save_path}")

@train_group.command(name='list')
@click.pass_context
def list_models(ctx):
    """List all trained models"""
    logger = ctx.obj['logger']
    
    models_dir = Path('models')
    if not models_dir.exists():
        click.echo("No models found")
        return
    
    model_files = list(models_dir.glob('*.pkl'))
    
    if not model_files:
        click.echo("No trained models found")
        return
    
    click.echo(f"Found {len(model_files)} trained models:\n")
    
    for file in sorted(model_files):
        try:
            with open(file, 'rb') as f:
                model_info = pickle.load(f)
            
            click.echo(f"Model: {file.name}")
            click.echo(f"  Symbol: {model_info.get('symbol', 'Unknown')}")
            click.echo(f"  Type: {model_info.get('model_type', 'Unknown')}")
            click.echo(f"  Val Accuracy: {model_info.get('val_accuracy', 0):.4f}")
            click.echo(f"  Trained: {model_info.get('trained_at', 'Unknown')}")
            click.echo("")
        except Exception as e:
            click.echo(f"Error loading {file.name}: {e}")

@train_group.command(name='evaluate')
@click.argument('model_path')
@click.option('--symbol', help='Symbol to evaluate on (defaults to model symbol)')
@click.option('--start', help='Evaluation start date')
@click.option('--end', help='Evaluation end date')
@click.pass_context
def evaluate_model(ctx, model_path, symbol, start, end):
    """Evaluate a trained model on new data"""
    logger = ctx.obj['logger']
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        click.echo(f"Error loading model: {e}")
        return
    
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    if not symbol:
        symbol = model_data.get('symbol', 'BTC')
    
    click.echo(f"Evaluating {model_data.get('model_type', 'Unknown')} model on {symbol}...")
    
    # Load data
    cache_dir = Path('data/cache')
    data_files = list(cache_dir.glob(f'*{symbol}*'))
    
    if not data_files:
        click.echo(f"Error: No data found for {symbol}")
        return
    
    file = data_files[0]
    if file.suffix == '.parquet':
        df = pd.read_parquet(file)
    else:
        df = pd.read_csv(file, index_col=0, parse_dates=True)
    
    # Filter by date range
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    
    # Recreate features (simplified - should match training)
    # Technical indicators
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['ma_10'] = df['close'].rolling(10).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Prepare target
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    # Select features that exist
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features]
    y = df['target']
    
    # Make predictions
    model_type = model_data.get('model_type', 'unknown')
    if model_type in ['gru', 'lstm', 'transformer']:
        X_seq = X.values.reshape(-1, 1, len(available_features))
        predictions = (model.predict(X_seq) > 0.5).astype(int)
    elif model_type == 'ensemble':
        predictions = (model.predict(X.values) > 0.5).astype(int)
    else:
        predictions = model.predict(X)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    cm = confusion_matrix(y, predictions)
    
    click.echo("\n" + "="*50)
    click.echo("EVALUATION RESULTS")
    click.echo("="*50)
    click.echo(f"Period: {df.index[0]} to {df.index[-1]}")
    click.echo(f"Samples: {len(X)}")
    click.echo(f"Accuracy: {accuracy:.4f}")
    click.echo(f"Precision: {precision:.4f}")
    click.echo(f"Recall: {recall:.4f}")
    click.echo(f"F1-Score: {f1:.4f}")
    
    click.echo("\nConfusion Matrix:")
    click.echo(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    click.echo(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    # Trading simulation
    df['signal'] = predictions
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    
    total_return = (1 + df['strategy_returns']).prod() - 1
    sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
    
    click.echo("\nTrading Performance:")
    click.echo(f"Total Return: {total_return:.2%}")
    click.echo(f"Sharpe Ratio: {sharpe:.2f}")
    click.echo(f"Win Rate: {(df['strategy_returns'] > 0).mean():.2%}")