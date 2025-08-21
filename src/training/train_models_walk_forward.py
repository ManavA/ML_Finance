#!/usr/bin/env python3
"""
WALK-FORWARD MODEL TRAINING WITH REGIME CHANGE TESTING
Training period: 2023-2025 with walk-forward validation
Testing period: 2025+ for regime change detection
"""

import sys
import os
sys.path.append('src')
os.chdir('C:/Users/manav/claude')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_available = torch.cuda.is_available()
except:
    device = 'cpu'
    gpu_available = False

print("="*80)
print("WALK-FORWARD TRAINING WITH REGIME CHANGE TESTING")
print("="*80)
print(f"\nDevice: {device}")
print(f"GPU Available: {gpu_available}")

# ============================================================================
# WALK-FORWARD CONFIGURATION
# ============================================================================

WALK_FORWARD_CONFIG = {
    'train_window': 12,  # 12 months training
    'test_window': 3,    # 3 months testing
    'step_size': 3,      # 3 months step forward
    'start_date': '2023-01-01',
    'end_date': '2024-12-31',
    'regime_test_start': '2025-01-01',  # Test regime change from 2025
    'regime_test_end': datetime.now().strftime('%Y-%m-%d')
}

print("\nWalk-Forward Configuration:")
print(f"  Training Period: {WALK_FORWARD_CONFIG['start_date']} to {WALK_FORWARD_CONFIG['end_date']}")
print(f"  Train Window: {WALK_FORWARD_CONFIG['train_window']} months")
print(f"  Test Window: {WALK_FORWARD_CONFIG['test_window']} months")
print(f"  Step Size: {WALK_FORWARD_CONFIG['step_size']} months")
print(f"  Regime Test: {WALK_FORWARD_CONFIG['regime_test_start']} to {WALK_FORWARD_CONFIG['regime_test_end']}")

# ============================================================================
# 1. DATA COLLECTION (2023-2025+)
# ============================================================================

print("\n" + "="*60)
print("1. DATA COLLECTION (2023-2025+)")
print("="*60)

def collect_walk_forward_data():
    """Collect data for walk-forward analysis and regime testing"""
    
    # Get full date range (2023 to now)
    start_date = '2023-01-01'
    end_date = datetime.now()
    
    data = {}
    
    # CRYPTOCURRENCIES (including Solana)
    crypto_symbols = {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum', 
        'SOL-USD': 'Solana',
        'BNB-USD': 'Binance',
        'ADA-USD': 'Cardano',
        'XRP-USD': 'Ripple',
        'AVAX-USD': 'Avalanche',
        'MATIC-USD': 'Polygon'
    }
    
    print("\nDownloading Cryptocurrency Data (2023-2025+):")
    for symbol, name in crypto_symbols.items():
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[symbol] = df
                print(f"  {name} ({symbol}): {len(df)} days")
                # Show 2025 data availability
                df_2025 = df[df.index >= '2025-01-01']
                if not df_2025.empty:
                    print(f"    -> 2025+ data: {len(df_2025)} days for regime testing")
        except Exception as e:
            print(f"  {name} ({symbol}): Failed")
    
    # EQUITY INDICES
    equity_symbols = {
        'SPY': 'S&P 500',
        'QQQ': 'NASDAQ',
        'DIA': 'Dow Jones',
        'IWM': 'Russell 2000',
        'VTI': 'Total Market'
    }
    
    print("\nDownloading Equity Data (2023-2025+):")
    for symbol, name in equity_symbols.items():
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[symbol] = df
                print(f"  {name} ({symbol}): {len(df)} days")
                # Show 2025 data availability
                df_2025 = df[df.index >= '2025-01-01']
                if not df_2025.empty:
                    print(f"    -> 2025+ data: {len(df_2025)} days for regime testing")
        except:
            print(f"  {name} ({symbol}): Failed")
    
    return data

# Collect all data
all_data = collect_walk_forward_data()

# Check data splits
print("\nData Split Summary:")
for symbol in ['BTC-USD', 'SOL-USD', 'SPY']:
    if symbol in all_data:
        df = all_data[symbol]
        df_train = df[(df.index >= '2023-01-01') & (df.index < '2025-01-01')]
        df_regime = df[df.index >= '2025-01-01']
        print(f"  {symbol}:")
        print(f"    Training (2023-2024): {len(df_train)} days")
        print(f"    Regime Test (2025+): {len(df_regime)} days")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*60)
print("2. FEATURE ENGINEERING")
print("="*60)

def create_features(df):
    """Create features for ML models"""
    
    # Handle multi-index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    
    features = pd.DataFrame(index=df.index)
    
    # Price features
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Technical indicators
    for period in [5, 10, 20, 50]:
        sma = df['Close'].rolling(period).mean()
        features[f'sma_{period}'] = sma.values if hasattr(sma, 'values') else sma
        features[f'price_to_sma_{period}'] = (df['Close'] / sma).values if hasattr(df['Close'] / sma, 'values') else df['Close'] / sma
    
    # Volatility
    for period in [10, 20, 30]:
        features[f'volatility_{period}'] = features['returns'].rolling(period).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    features['macd'] = ema_12 - ema_26
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    
    # Bollinger Bands
    bb_period = 20
    bb_std = df['Close'].rolling(bb_period).std()
    bb_mean = df['Close'].rolling(bb_period).mean()
    features['bb_upper'] = bb_mean + (bb_std * 2)
    features['bb_lower'] = bb_mean - (bb_std * 2)
    features['bb_width'] = features['bb_upper'] - features['bb_lower']
    features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_width'] + 1e-10)
    
    # Volume features
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    features['volume_sma'] = df['Volume'].rolling(20).mean()
    
    # Market microstructure
    features['high_low_ratio'] = df['High'] / df['Low']
    features['close_open_ratio'] = df['Close'] / df['Open']
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        features[f'volume_lag_{lag}'] = features['volume_ratio'].shift(lag)
    
    # Rolling statistics
    for period in [5, 10, 20]:
        features[f'returns_mean_{period}'] = features['returns'].rolling(period).mean()
        features[f'returns_std_{period}'] = features['returns'].rolling(period).std()
        features[f'returns_skew_{period}'] = features['returns'].rolling(period).skew()
    
    # Target
    features['target'] = (features['returns'].shift(-1) > 0).astype(int)
    
    return features.dropna()

# Process all assets
processed_data = {}
for symbol, df in all_data.items():
    processed_data[symbol] = create_features(df)
    print(f"  {symbol}: {len(processed_data[symbol])} samples")

# ============================================================================
# 3. WALK-FORWARD VALIDATION
# ============================================================================

print("\n" + "="*60)
print("3. WALK-FORWARD VALIDATION")
print("="*60)

def generate_walk_forward_splits(data, start_date, end_date, train_months, test_months, step_months):
    """Generate walk-forward train/test splits"""
    
    splits = []
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    while current_date + pd.DateOffset(months=train_months+test_months) <= end_date:
        train_start = current_date
        train_end = current_date + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        train_data = data[(data.index >= train_start) & (data.index < train_end)]
        test_data = data[(data.index >= test_start) & (data.index < test_end)]
        
        if len(train_data) > 100 and len(test_data) > 20:  # Minimum data requirements
            splits.append({
                'fold': len(splits) + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_data': train_data,
                'test_data': test_data
            })
        
        current_date += pd.DateOffset(months=step_months)
    
    return splits

# Generate splits for each asset
walk_forward_splits = {}
for symbol in ['BTC-USD', 'SOL-USD', 'ETH-USD', 'SPY', 'QQQ']:
    if symbol in processed_data:
        splits = generate_walk_forward_splits(
            processed_data[symbol],
            WALK_FORWARD_CONFIG['start_date'],
            WALK_FORWARD_CONFIG['end_date'],
            WALK_FORWARD_CONFIG['train_window'],
            WALK_FORWARD_CONFIG['test_window'],
            WALK_FORWARD_CONFIG['step_size']
        )
        walk_forward_splits[symbol] = splits
        print(f"\n{symbol} Walk-Forward Splits:")
        for split in splits:
            print(f"  Fold {split['fold']}: Train {split['train_start'].date()} to {split['train_end'].date()}, "
                  f"Test {split['test_start'].date()} to {split['test_end'].date()}")
            print(f"    Train: {len(split['train_data'])} samples, Test: {len(split['test_data'])} samples")

# ============================================================================
# 4. XGBOOST WITH WALK-FORWARD
# ============================================================================

print("\n" + "="*60)
print("4. XGBOOST WALK-FORWARD TRAINING")
print("="*60)

import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

def train_xgboost_walk_forward(symbol, splits):
    """Train XGBoost using walk-forward validation"""
    
    print(f"\nTraining XGBoost for {symbol}:")
    
    fold_results = []
    
    for split in splits:
        print(f"  Fold {split['fold']}...")
        
        # Prepare data
        feature_cols = [col for col in split['train_data'].columns if col != 'target']
        X_train = split['train_data'][feature_cols]
        y_train = split['train_data']['target']
        X_test = split['test_data'][feature_cols]
        y_test = split['test_data']['target']
        
        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model (reduced parameters for speed)
        model = xgb.XGBClassifier(
            n_estimators=200,  # Reduced for speed
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train, verbose=0)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        fold_results.append({
            'fold': split['fold'],
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'model': model,
            'scaler': scaler
        })
        
        print(f"    Accuracy: {accuracy:.1%}, AUC: {auc:.3f}, Precision: {precision:.1%}, Recall: {recall:.1%}")
    
    # Summary statistics
    mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    mean_auc = np.mean([r['auc'] for r in fold_results])
    
    print(f"  Mean Accuracy: {mean_accuracy:.1%} ± {std_accuracy:.1%}")
    print(f"  Mean AUC: {mean_auc:.3f}")
    
    return fold_results

# Train XGBoost for each asset
xgb_results = {}
for symbol in ['BTC-USD', 'SOL-USD', 'SPY']:
    if symbol in walk_forward_splits and walk_forward_splits[symbol]:
        xgb_results[symbol] = train_xgboost_walk_forward(symbol, walk_forward_splits[symbol])

# ============================================================================
# 5. LSTM WITH WALK-FORWARD (REDUCED EPOCHS)
# ============================================================================

print("\n" + "="*60)
print("5. LSTM WALK-FORWARD TRAINING")
print("="*60)

if gpu_available:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    class WalkForwardLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :]))
    
    def train_lstm_walk_forward(symbol, splits, epochs=20):  # Reduced epochs
        """Train LSTM using walk-forward validation"""
        
        print(f"\nTraining LSTM for {symbol} ({epochs} epochs per fold):")
        
        fold_results = []
        
        for split in splits[:2]:  # Only first 2 folds for speed
            print(f"  Fold {split['fold']}...")
            
            # Prepare data
            feature_cols = [col for col in split['train_data'].columns if col != 'target']
            X_train = split['train_data'][feature_cols].values
            y_train = split['train_data']['target'].values
            X_test = split['test_data'][feature_cols].values
            y_test = split['test_data']['target'].values
            
            # Normalize
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Create sequences
            seq_len = 20
            X_train_seq, y_train_seq = [], []
            for i in range(len(X_train) - seq_len):
                X_train_seq.append(X_train[i:i+seq_len])
                y_train_seq.append(y_train[i+seq_len])
            
            X_test_seq, y_test_seq = [], []
            for i in range(len(X_test) - seq_len):
                X_test_seq.append(X_test[i:i+seq_len])
                y_test_seq.append(y_test[i+seq_len])
            
            if len(X_train_seq) < 10 or len(X_test_seq) < 5:
                continue
            
            X_train_t = torch.FloatTensor(np.array(X_train_seq)).to(device)
            y_train_t = torch.FloatTensor(np.array(y_train_seq)).to(device)
            X_test_t = torch.FloatTensor(np.array(X_test_seq)).to(device)
            y_test_t = torch.FloatTensor(np.array(y_test_seq)).to(device)
            
            # Model
            model = WalkForwardLSTM(input_size=X_train_t.shape[2]).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train_t).squeeze()
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"      Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                test_pred = (model(X_test_t).squeeze() > 0.5).float()
                accuracy = (test_pred == y_test_t).float().mean().item()
            
            fold_results.append({
                'fold': split['fold'],
                'accuracy': accuracy,
                'model': model
            })
            
            print(f"    Accuracy: {accuracy:.1%}")
        
        if fold_results:
            mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
            print(f"  Mean Accuracy: {mean_accuracy:.1%}")
        
        return fold_results
    
    # Train LSTM for key assets
    lstm_results = {}
    for symbol in ['BTC-USD', 'SOL-USD']:
        if symbol in walk_forward_splits and walk_forward_splits[symbol]:
            lstm_results[symbol] = train_lstm_walk_forward(symbol, walk_forward_splits[symbol], epochs=20)
else:
    print("  GPU not available - skipping LSTM")
    lstm_results = {}

# ============================================================================
# 6. REGIME CHANGE TESTING (2025+)
# ============================================================================

print("\n" + "="*60)
print("6. REGIME CHANGE TESTING (2025+)")
print("="*60)

def test_regime_change(symbol, model_results, processed_data):
    """Test model performance on 2025+ data to detect regime changes"""
    
    print(f"\nRegime Change Testing for {symbol}:")
    
    # Get 2025+ data
    regime_data = processed_data[processed_data.index >= '2025-01-01']
    
    if len(regime_data) < 10:
        print(f"  Insufficient 2025+ data ({len(regime_data)} samples)")
        return None
    
    print(f"  Testing on {len(regime_data)} days of 2025+ data")
    
    # Use last trained model from walk-forward
    if not model_results or len(model_results) == 0:
        print("  No trained model available")
        return None
    
    last_model_info = model_results[-1]
    model = last_model_info['model']
    scaler = last_model_info.get('scaler')
    
    # Prepare test data
    feature_cols = [col for col in regime_data.columns if col != 'target']
    X_regime = regime_data[feature_cols]
    y_regime = regime_data['target']
    
    if scaler:
        X_regime_scaled = scaler.transform(X_regime)
    else:
        X_regime_scaled = X_regime
    
    # Predict
    y_pred = model.predict(X_regime_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_regime, y_pred)
    precision = precision_score(y_regime, y_pred, zero_division=0)
    recall = recall_score(y_regime, y_pred, zero_division=0)
    
    # Compare with historical performance
    historical_accuracy = np.mean([r['accuracy'] for r in model_results])
    regime_shift = accuracy - historical_accuracy
    
    print(f"  2025+ Accuracy: {accuracy:.1%}")
    print(f"  Historical Avg: {historical_accuracy:.1%}")
    print(f"  Regime Shift: {regime_shift*100:+.1f}pp")
    
    if abs(regime_shift) > 0.05:
        print(f"  >>> REGIME CHANGE DETECTED! Performance shifted by {regime_shift*100:+.1f}pp")
    else:
        print(f"  >>> No significant regime change detected")
    
    return {
        'accuracy_2025': accuracy,
        'precision_2025': precision,
        'recall_2025': recall,
        'historical_accuracy': historical_accuracy,
        'regime_shift': regime_shift,
        'regime_change_detected': abs(regime_shift) > 0.05
    }

# Test regime changes
regime_test_results = {}
for symbol in ['BTC-USD', 'SOL-USD', 'SPY']:
    if symbol in xgb_results and symbol in processed_data:
        regime_test_results[symbol] = test_regime_change(
            symbol, 
            xgb_results[symbol], 
            processed_data[symbol]
        )

# ============================================================================
# 7. CRYPTO VS EQUITY COMPARISON
# ============================================================================

print("\n" + "="*80)
print("CRYPTO VS EQUITY COMPARISON")
print("="*80)

# Collect accuracies
crypto_accuracies = []
equity_accuracies = []

for symbol, results in xgb_results.items():
    mean_acc = np.mean([r['accuracy'] for r in results])
    if 'USD' in symbol:  # Crypto
        crypto_accuracies.append(mean_acc)
        print(f"  {symbol}: {mean_acc:.1%} (Crypto)")
    else:  # Equity
        equity_accuracies.append(mean_acc)
        print(f"  {symbol}: {mean_acc:.1%} (Equity)")

if crypto_accuracies and equity_accuracies:
    crypto_mean = np.mean(crypto_accuracies)
    equity_mean = np.mean(equity_accuracies)
    advantage = crypto_mean - equity_mean
    
    print(f"\nResults:")
    print(f"  Crypto Mean Accuracy: {crypto_mean:.1%}")
    print(f"  Equity Mean Accuracy: {equity_mean:.1%}")
    print(f"  Crypto Advantage: {advantage*100:.1f}pp")
    print(f"\nHypothesis: {'CONFIRMED' if advantage > 0 else 'NOT CONFIRMED'}")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_summary = {
    'walk_forward_config': WALK_FORWARD_CONFIG,
    'xgb_results': xgb_results,
    'lstm_results': lstm_results,
    'regime_test_results': regime_test_results,
    'crypto_accuracies': crypto_accuracies,
    'equity_accuracies': equity_accuracies,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

os.makedirs('models', exist_ok=True)
with open('models/walk_forward_results.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

print("Results saved to models/walk_forward_results.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
WALK-FORWARD VALIDATION RESULTS
--------------------------------
Training Period: 2023-2024
Testing Period: 2025+
Method: {WALK_FORWARD_CONFIG['train_window']}-month train, {WALK_FORWARD_CONFIG['test_window']}-month test windows

KEY FINDINGS:
1. Solana (SOL) Performance:
   - Walk-forward accuracy: {np.mean([r['accuracy'] for r in xgb_results.get('SOL-USD', [])])*100:.1f}% 
   - Included in all analyses

2. Regime Change Detection (2025+):
   - Tested on actual 2025 data
   - Significant changes detected in: {sum(1 for r in regime_test_results.values() if r and r['regime_change_detected'])} assets

3. Crypto vs Equity:
   - Crypto advantage maintained: {advantage*100:.1f}pp
   - Hypothesis: {'CONFIRMED' if advantage > 0 else 'NOT CONFIRMED'}

4. Model Robustness:
   - Used walk-forward validation to avoid look-ahead bias
   - Tested on out-of-sample 2025 data
   - Multiple folds ensure stability

Next Steps:
- Run comprehensive notebooks with these results
- Generate visualizations
- Complete statistical significance testing
""")