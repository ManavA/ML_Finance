#!/usr/bin/env python3


import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Check GPU
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_available = torch.cuda.is_available()
except:
    device = 'cpu'
    gpu_available = False

print("COMPREHENSIVE WALK-FORWARD TRAINING - ALL ASSETS")
print("="*80)
print(f"Device: {device}")

ASSET_COLORS = {
    'BTC': '#FF6B35',  # Orange
    'ETH': '#F77B71',  # Light red
    'SOL': '#FF9F1C',  # Gold
    'BNB': '#FFBF69',  # Light orange
    'ADA': '#FFD93D',  # Yellow
    'XRP': '#FCB1A6',  # Peach
    
    'SPY': '#2E86AB',  # Blue
    'QQQ': '#42B883',  # Green
    'DIA': '#264653',  # Dark blue
    'IWM': '#2A9D8F',  # Teal
    'VTI': '#457B9D',  # Light blue
}

# Asset groupings for aggregation
ASSET_GROUPS = {
    'Major Crypto': ['BTC-USD', 'ETH-USD'],
    'Alt Crypto': ['SOL-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD'],
    'Large Cap Equity': ['SPY', 'DIA'],
    'Tech/Growth Equity': ['QQQ'],
    'Small Cap Equity': ['IWM'],
    'Total Market': ['VTI']
}

WALK_FORWARD_CONFIG = {
    'train_window': 12,  # months
    'test_window': 3,    # months
    'step_size': 3,      # months
    'start_date': '2023-01-01',
    'end_date': '2024-12-31',
    'regime_test_start': '2025-01-01',
    'regime_test_end': datetime.now().strftime('%Y-%m-%d')
}

print(f"\nConfiguration:")
print(f"  Training: {WALK_FORWARD_CONFIG['start_date']} to {WALK_FORWARD_CONFIG['end_date']}")
print(f"  Walk-Forward: {WALK_FORWARD_CONFIG['train_window']}mo train, {WALK_FORWARD_CONFIG['test_window']}mo test")
print(f"  Regime Test: {WALK_FORWARD_CONFIG['regime_test_start']}+")

print("\n" + "="*60)
print("1. COMPREHENSIVE DATA COLLECTION")
print("="*60)

def collect_all_assets():
    
    start_date = '2023-01-01'
    end_date = datetime.now()
    
    data = {}
    
    # ALL CRYPTOCURRENCIES
    crypto_symbols = {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'SOL-USD': 'Solana',
        'BNB-USD': 'Binance Coin',
        'ADA-USD': 'Cardano',
        'XRP-USD': 'Ripple'
    }
    
    print("\nCryptocurrencies:")
    for symbol, name in crypto_symbols.items():
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[symbol] = df
                df_2025 = df[df.index >= '2025-01-01']
                print(f"  {name:15} ({symbol:8}): {len(df):4} days | 2025+: {len(df_2025):3} days")
        except:
            print(f"  {name:15} ({symbol:8}): FAILED")
    
    # ALL EQUITY INDICES
    equity_symbols = {
        'SPY': 'S&P 500',
        'QQQ': 'NASDAQ 100',
        'DIA': 'Dow Jones',
        'IWM': 'Russell 2000',
        'VTI': 'Total Market'
    }
    
    print("\nEquity Indices:")
    for symbol, name in equity_symbols.items():
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[symbol] = df
                df_2025 = df[df.index >= '2025-01-01']
                print(f"  {name:15} ({symbol:8}): {len(df):4} days | 2025+: {len(df_2025):3} days")
        except:
            print(f"  {name:15} ({symbol:8}): FAILED")
    
    return data

# Collect data
all_data = collect_all_assets()

print(f"\nTotal assets collected: {len(all_data)}")
print(f"  Cryptocurrencies: {sum(1 for s in all_data.keys() if 'USD' in s)}")
print(f"  Equity Indices: {sum(1 for s in all_data.keys() if 'USD' not in s)}")


print("\n" + "="*60)
print("2. FEATURE ENGINEERING")
print("="*60)

def create_features(df):
    
    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    
    features = pd.DataFrame(index=df.index)
    
    # Core features
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        sma = df['Close'].rolling(period).mean()
        features[f'sma_{period}'] = sma
        features[f'price_to_sma_{period}'] = df['Close'] / sma
    
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
    
    # Bollinger Bands
    bb_mean = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    features['bb_upper'] = bb_mean + (bb_std * 2)
    features['bb_lower'] = bb_mean - (bb_std * 2)
    features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
    
    # Volume
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Market microstructure
    features['high_low_ratio'] = df['High'] / df['Low']
    features['close_open_ratio'] = df['Close'] / df['Open']
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
    
    # Target
    features['target'] = (features['returns'].shift(-1) > 0).astype(int)
    
    return features.dropna()

# Process all assets
processed_data = {}
for symbol, df in all_data.items():
    processed_data[symbol] = create_features(df)
    print(f"  {symbol:10}: {len(processed_data[symbol]):5} samples")

print("\n" + "="*60)
print("3. GENERATING WALK-FORWARD SPLITS")
print("="*60)

def generate_walk_forward_splits(data, config):
    
    splits = []
    current_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])
    
    fold = 1
    while current_date + pd.DateOffset(months=config['train_window']+config['test_window']) <= end_date:
        train_start = current_date
        train_end = current_date + pd.DateOffset(months=config['train_window'])
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=config['test_window'])
        
        train_data = data[(data.index >= train_start) & (data.index < train_end)]
        test_data = data[(data.index >= test_start) & (data.index < test_end)]
        
        if len(train_data) > 100 and len(test_data) > 20:
            splits.append({
                'fold': fold,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_data': train_data,
                'test_data': test_data
            })
            fold += 1
        
        current_date += pd.DateOffset(months=config['step_size'])
    
    return splits

# Generate splits for all assets
walk_forward_splits = {}
for symbol in processed_data.keys():
    splits = generate_walk_forward_splits(processed_data[symbol], WALK_FORWARD_CONFIG)
    if splits:
        walk_forward_splits[symbol] = splits
        print(f"  {symbol:10}: {len(splits)} folds")

print("\n" + "="*60)
print("4. XGBOOST TRAINING - ALL ASSETS")
print("="*60)

import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

def train_xgboost_walk_forward(symbol, splits):
    
    fold_results = []
    
    for split in splits:
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
        
        # Train
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        
        fold_results.append({
            'fold': split['fold'],
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'model': model,
            'scaler': scaler
        })
    
    return fold_results

# Train all assets
xgb_results = {}
print("\nTraining XGBoost models:")

for symbol in walk_forward_splits.keys():
    results = train_xgboost_walk_forward(symbol, walk_forward_splits[symbol])
    xgb_results[symbol] = results
    
    mean_acc = np.mean([r['accuracy'] for r in results])
    std_acc = np.std([r['accuracy'] for r in results])
    asset_type = "Crypto" if 'USD' in symbol else "Equity"
    
    print(f"  {symbol:10} ({asset_type:6}): {mean_acc:.1%} ± {std_acc:.1%}")

print("\n" + "="*60)
print("5. AGGREGATED RESULTS BY GROUP")
print("="*60)

def aggregate_results_by_group():
    
    group_results = {}
    
    for group_name, symbols in ASSET_GROUPS.items():
        group_accs = []
        for symbol in symbols:
            if symbol in xgb_results:
                mean_acc = np.mean([r['accuracy'] for r in xgb_results[symbol]])
                group_accs.append(mean_acc)
        
        if group_accs:
            group_results[group_name] = {
                'mean': np.mean(group_accs),
                'std': np.std(group_accs),
                'n_assets': len(group_accs)
            }
    
    return group_results

group_results = aggregate_results_by_group()

print("\nPerformance by Asset Group:")
for group, stats in group_results.items():
    print(f"  {group:20}: {stats['mean']:.1%} ± {stats['std']:.1%} (n={stats['n_assets']})")

# Overall crypto vs equity
crypto_accs = []
equity_accs = []

for symbol, results in xgb_results.items():
    mean_acc = np.mean([r['accuracy'] for r in results])
    if 'USD' in symbol:
        crypto_accs.append(mean_acc)
    else:
        equity_accs.append(mean_acc)

print(f"\nOverall Performance:")
print(f"  All Crypto: {np.mean(crypto_accs):.1%} ± {np.std(crypto_accs):.1%}")
print(f"  All Equity: {np.mean(equity_accs):.1%} ± {np.std(equity_accs):.1%}")
print(f"  Advantage: {(np.mean(crypto_accs) - np.mean(equity_accs))*100:+.1f}pp")

print("\n" + "="*60)
print("6. REGIME CHANGE TESTING (2025+)")
print("="*60)

def test_regime_change(symbol, model_results, data):
    
    
    regime_data = data[data.index >= '2025-01-01']
    
    if len(regime_data) < 10:
        return None
    
    # Use last model
    last_model_info = model_results[-1]
    model = last_model_info['model']
    scaler = last_model_info['scaler']
    
    # Prepare data
    feature_cols = [col for col in regime_data.columns if col != 'target']
    X_regime = regime_data[feature_cols]
    y_regime = regime_data['target']
    
    X_regime_scaled = scaler.transform(X_regime)
    
    # Predict
    y_pred = model.predict(X_regime_scaled)
    accuracy = accuracy_score(y_regime, y_pred)
    
    # Compare with historical
    historical_accuracy = np.mean([r['accuracy'] for r in model_results])
    regime_shift = accuracy - historical_accuracy
    
    return {
        'accuracy_2025': accuracy,
        'historical_accuracy': historical_accuracy,
        'regime_shift': regime_shift,
        'regime_change_detected': abs(regime_shift) > 0.05,
        'n_samples': len(regime_data)
    }

# Test all assets
regime_results = {}
print("\nRegime Change Detection:")

for symbol in xgb_results.keys():
    if symbol in processed_data:
        result = test_regime_change(symbol, xgb_results[symbol], processed_data[symbol])
        if result:
            regime_results[symbol] = result
            shift = result['regime_shift'] * 100
            detected = "DETECTED" if result['regime_change_detected'] else "stable"
            print(f"  {symbol:10}: 2025 acc={result['accuracy_2025']:.1%}, shift={shift:+.1f}pp ({detected})")


print("\n" + "="*60)
print("7. SAVING RESULTS")
print("="*60)

# Prepare comprehensive results
comprehensive_results = {
    'walk_forward_config': WALK_FORWARD_CONFIG,
    'asset_colors': ASSET_COLORS,
    'asset_groups': ASSET_GROUPS,
    'xgb_results': xgb_results,
    'group_results': group_results,
    'regime_results': regime_results,
    'crypto_accuracies': crypto_accs,
    'equity_accuracies': equity_accs,
    'all_assets': list(all_data.keys()),
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

os.makedirs('models', exist_ok=True)
with open('models/comprehensive_results.pkl', 'wb') as f:
    pickle.dump(comprehensive_results, f)

print("  Saved: models/comprehensive_results.pkl")


print("\n" + "="*80)
print("COMPREHENSIVE TRAINING COMPLETE")
print("="*80)

print(f)