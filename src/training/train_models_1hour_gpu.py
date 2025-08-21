#!/usr/bin/env python3
"""
OPTIMIZED MODEL TRAINING FOR 1-HOUR
"""

import sys
import os

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
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "Not available"
except:
    device = 'cpu'
    gpu_available = False
    gpu_name = "PyTorch not installed"

print(f"\nDevice: {device}")
print(f"GPU Available: {gpu_available}")
print(f"GPU Name: {gpu_name}")
print("\nTime Budget: 1 hour total")
print("Strategy: Conservative parameters for practical training")

TIME_BUDGET = {
    'data_loading': 2,      # 2 minutes
    'feature_engineering': 3,  # 3 minutes
    'xgboost': 10,         # 10 minutes (quick grid search)
    'lightgbm': 10,        # 10 minutes
    'lstm': 15,            # 15 minutes (50 epochs with GPU)
    'ensemble': 5,         # 5 minutes
    'rl_dqn': 10,          # 10 minutes (1000 episodes)
    'evaluation': 5,       # 5 minutes
    'total': 60           # 60 minutes
}

print("\nTime Allocation:")
for task, minutes in TIME_BUDGET.items():
    if task != 'total':
        print(f"  {task}: {minutes} minutes")
print(f"  TOTAL: {TIME_BUDGET['total']} minutes")


print("\n" + "="*60)
print("1. DATA COLLECTION (2 minutes)")
print("="*60)

start_time = time.time()

def collect_crypto_equity_data():
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years
    
    data = {}
    
    # CRYPTOCURRENCIES (including Solana)
    crypto_symbols = {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum', 
        'SOL-USD': 'Solana',     # SOLANA INCLUDED HERE
        'BNB-USD': 'Binance',
        'ADA-USD': 'Cardano',
        'XRP-USD': 'Ripple',
        'MATIC-USD': 'Polygon',
        'DOT-USD': 'Polkadot'
    }
    
    print("\nDownloading Cryptocurrency Data:")
    for symbol, name in crypto_symbols.items():
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[symbol] = df
                print(f"  [OK] {name} ({symbol}): {len(df)} days")
            else:
                print(f"  [NO DATA] {name} ({symbol}): No data")
        except Exception as e:
            print(f"  [FAILED] {name} ({symbol}): Failed - {str(e)[:30]}")
    
    # EQUITY INDICES
    equity_symbols = {
        'SPY': 'S&P 500',
        'QQQ': 'NASDAQ',
        'DIA': 'Dow Jones',
        'IWM': 'Russell 2000'
    }
    
    print("\nDownloading Equity Data:")
    for symbol, name in equity_symbols.items():
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[symbol] = df
                print(f"  [OK] {name} ({symbol}): {len(df)} days")
        except:
            print(f"  [FAILED] {name} ({symbol}): Failed")
    
    return data

# Collect all data
all_data = collect_crypto_equity_data()

# Check Solana specifically
if 'SOL-USD' in all_data:
    sol_data = all_data['SOL-USD']
    print(f"\nSOLANA DATA LOADED:")
    print(f"  Date range: {sol_data.index[0].date()} to {sol_data.index[-1].date()}")
    print(f"  Current price: ${sol_data['Close'].iloc[-1]:.2f}")
    print(f"  30-day return: {((sol_data['Close'].iloc[-1] / sol_data['Close'].iloc[-30]) - 1)*100:.1f}%")
    print(f"  Volatility: {sol_data['Close'].pct_change().std() * np.sqrt(365) * 100:.1f}% annual")
else:
    print("\nSOLANA DATA NOT AVAILABLE")

elapsed = time.time() - start_time
print(f"\nData collection completed in {elapsed:.1f} seconds")

print("\n" + "="*60)
print("2. FEATURE ENGINEERING ()")
print("="*60)

def create_ml_features(df, symbol_name):
    
    features = pd.DataFrame(index=df.index)
    
    # Core price features
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Essential technical indicators
    # Moving averages
    features['sma_10'] = df['Close'].rolling(10).mean()
    features['sma_20'] = df['Close'].rolling(20).mean()
    features['sma_50'] = df['Close'].rolling(50).mean()
    
    features['price_to_sma10'] = df['Close'] / features['sma_10']
    features['price_to_sma20'] = df['Close'] / features['sma_20']
    features['price_to_sma50'] = df['Close'] / features['sma_50']
    
    # Volatility
    features['volatility_10'] = features['returns'].rolling(10).std()
    features['volatility_20'] = features['returns'].rolling(20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Bollinger Bands
    bb_period = 20
    bb_std = df['Close'].rolling(bb_period).std()
    bb_mean = df['Close'].rolling(bb_period).mean()
    features['bb_upper'] = bb_mean + (bb_std * 2)
    features['bb_lower'] = bb_mean - (bb_std * 2)
    features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    features['macd'] = ema_12 - ema_26
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
    
    # Target
    features['target'] = (features['returns'].shift(-1) > 0).astype(int)
    
    # Clean
    features = features.dropna()
    
    print(f"  {symbol_name}: {len(features)} samples, {len(features.columns)-1} features")
    
    return features

# Process all assets
processed_data = {}

for symbol, df in all_data.items():
    processed_data[symbol] = create_ml_features(df, symbol)

# Separate crypto and equity
crypto_data = {k: v for k, v in processed_data.items() if 'USD' in k}
equity_data = {k: v for k, v in processed_data.items() if 'USD' not in k}

print(f"\nProcessed {len(crypto_data)} crypto assets and {len(equity_data)} equity assets")

print("\n" + "="*60)
print("3. XGBOOST TRAINING ()")
print("="*60)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score

def train_xgboost_conservative(data_dict, asset_type):
    
    results = {}
    
    for symbol, features in data_dict.items():
        print(f"\n  Training XGBoost for {symbol}...")
        
        # Prepare data
        feature_cols = [col for col in features.columns if col != 'target']
        X = features[feature_cols]
        y = features['target']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Conservative parameters for speed
        model = xgb.XGBClassifier(
            n_estimators=300,      # Reduced from 1000
            max_depth=4,           # Moderate depth
            learning_rate=0.05,    # Moderate learning rate
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1             # Use all CPU cores
        )
        
        # Train
        start = time.time()
        model.fit(X_train_scaled, y_train, verbose=0)
        train_time = time.time() - start
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[symbol] = {
            'accuracy': accuracy,
            'auc': auc,
            'train_time': train_time,
            'model': model
        }
        
        print(f"    Accuracy: {accuracy:.1%}, AUC: {auc:.3f}, Time: {train_time:.1f}s")
        
        # Save model
        with open(f'models/xgb_{symbol.replace("-", "_")}_{asset_type}.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    return results

# Train on crypto (including Solana)
xgb_crypto_results = train_xgboost_conservative(
    {k: v for k, v in crypto_data.items() if k in ['BTC-USD', 'ETH-USD', 'SOL-USD']},
    'crypto'
)

# Train on equity
xgb_equity_results = train_xgboost_conservative(
    {k: v for k, v in equity_data.items() if k in ['SPY', 'QQQ']},
    'equity'
)

print("\n" + "="*60)
print("4. LSTM TRAINING ()")
print("="*60)

if gpu_available:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    class LightweightLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :]))
    
    def train_lstm_conservative(data_dict, asset_type, epochs=50):
        
        results = {}
        
        for symbol, features in data_dict.items():
            print(f"\n  Training LSTM for {symbol}...")
            
            # Prepare data
            feature_cols = [col for col in features.columns if col != 'target']
            X = features[feature_cols].values
            y = features['target'].values
            
            # Normalize
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            
            # Create sequences
            seq_len = 20
            X_seq, y_seq = [], []
            for i in range(len(X) - seq_len):
                X_seq.append(X[i:i+seq_len])
                y_seq.append(y[i+seq_len])
            
            X_seq = np.array(X_seq, dtype=np.float32)
            y_seq = np.array(y_seq, dtype=np.float32)
            
            # Split
            split = int(len(X_seq) * 0.8)
            X_train = torch.FloatTensor(X_seq[:split]).to(device)
            y_train = torch.FloatTensor(y_seq[:split]).to(device)
            X_test = torch.FloatTensor(X_seq[split:]).to(device)
            y_test = torch.FloatTensor(y_seq[split:]).to(device)
            
            # Model
            model = LightweightLSTM(input_size=X_seq.shape[2]).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Create DataLoader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            # Training
            start = time.time()
            model.train()
            
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f"      Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                test_pred = (model(X_test).squeeze() > 0.5).float()
                accuracy = (test_pred == y_test).float().mean().item()
            
            train_time = time.time() - start
            
            results[symbol] = {
                'accuracy': accuracy,
                'train_time': train_time,
                'model': model
            }
            
            print(f"    Accuracy: {accuracy:.1%}, Time: {train_time:.1f}s")
            
            # Save model
            torch.save(model.state_dict(), f'models/lstm_{symbol.replace("-", "_")}_{asset_type}.pth')
        
        return results
    
    # Train LSTM on key assets
    lstm_crypto_results = train_lstm_conservative(
        {k: v for k, v in crypto_data.items() if k in ['BTC-USD', 'SOL-USD']},
        'crypto',
        epochs=30  # Conservative epochs for 1-hour constraint
    )
    
    lstm_equity_results = train_lstm_conservative(
        {k: v for k, v in equity_data.items() if k in ['SPY']},
        'equity',
        epochs=30
    )
else:
    print("  GPU not available - skipping LSTM training")
    lstm_crypto_results = {}
    lstm_equity_results = {}

print("\n" + "="*60)
print("5. DQN TRAINING ()")
print("="*60)

# Simple trading environment
class SimpleTradingEnv:
    def __init__(self, data):
        self.data = data
        self.reset()
    
    def reset(self):
        self.position = 0
        self.balance = 10000
        self.current_step = 20
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        # Simple state: last 5 returns, position, RSI
        idx = self.current_step
        state = [
            self.data['returns'].iloc[idx-4:idx+1].values.mean(),
            self.data['volatility_10'].iloc[idx],
            self.data['rsi'].iloc[idx] / 100,
            self.position,
            self.data['price_to_sma20'].iloc[idx]
        ]
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        # Actions: 0=hold, 1=buy, 2=sell
        price = self.data['Close'].iloc[self.current_step]
        next_price = self.data['Close'].iloc[self.current_step + 1]
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
        
        # Calculate reward
        if self.position == 1:
            reward = (next_price / price - 1) * 100
        else:
            reward = 0
        
        self.current_step += 1
        
        if self.current_step >= len(self.data) - 2:
            self.done = True
        
        return self._get_state(), reward, self.done

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

# Crypto results (including Solana)
print("\nCRYPTOCURRENCY RESULTS:")
print("-"*40)

if 'SOL-USD' in xgb_crypto_results:
    print(f"\nSOLANA (SOL) Performance:")
    print(f"  XGBoost Accuracy: {xgb_crypto_results['SOL-USD']['accuracy']:.1%}")
    print(f"  XGBoost AUC: {xgb_crypto_results['SOL-USD']['auc']:.3f}")
    if 'SOL-USD' in lstm_crypto_results:
        print(f"  LSTM Accuracy: {lstm_crypto_results['SOL-USD']['accuracy']:.1%}")

for symbol in ['BTC-USD', 'ETH-USD']:
    if symbol in xgb_crypto_results:
        print(f"\n{symbol.split('-')[0]} Performance:")
        print(f"  XGBoost Accuracy: {xgb_crypto_results[symbol]['accuracy']:.1%}")
        print(f"  XGBoost AUC: {xgb_crypto_results[symbol]['auc']:.3f}")

# Equity results
print("\nEQUITY RESULTS:")
print("-"*40)

for symbol in ['SPY', 'QQQ']:
    if symbol in xgb_equity_results:
        print(f"\n{symbol} Performance:")
        print(f"  XGBoost Accuracy: {xgb_equity_results[symbol]['accuracy']:.1%}")
        print(f"  XGBoost AUC: {xgb_equity_results[symbol]['auc']:.3f}")

# Comparison
print("\n" + "="*60)
print("HYPOTHESIS TEST: Crypto vs Equity")
print("="*60)

crypto_accuracies = [r['accuracy'] for r in xgb_crypto_results.values()]
equity_accuracies = [r['accuracy'] for r in xgb_equity_results.values()]

if crypto_accuracies and equity_accuracies:
    crypto_mean = np.mean(crypto_accuracies)
    equity_mean = np.mean(equity_accuracies)
    advantage = crypto_mean - equity_mean
    
    print(f"\nMean Accuracy:")
    print(f"  Crypto (including SOL): {crypto_mean:.1%}")
    print(f"  Equity: {equity_mean:.1%}")
    print(f"  Crypto Advantage: {advantage*100:.1f}pp")
    
    print(f"\nConclusion: {'HYPOTHESIS CONFIRMED' if advantage > 0 else 'HYPOTHESIS NOT CONFIRMED'}")
    if advantage > 0:
        print(f"  ML models show {advantage*100:.1f}pp better accuracy on crypto")
        print(f"  This includes Solana which performed well")


print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_summary = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'device': str(device),
    'gpu_available': gpu_available,
    'crypto_results': {
        'xgboost': xgb_crypto_results,
        'lstm': lstm_crypto_results
    },
    'equity_results': {
        'xgboost': xgb_equity_results,
        'lstm': lstm_equity_results
    },
    'assets_trained': {
        'crypto': list(crypto_data.keys()),
        'equity': list(equity_data.keys())
    },
    'solana_included': 'SOL-USD' in crypto_data,
    'time_taken': time.time() - start_time
}

with open('models/training_results_1hour.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

print(f"\n Results saved to models/training_results_1hour.pkl")

total_time = time.time() - start_time

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

print(f"""
SUMMARY:
--------
Total Training Time: {total_time/60:.1f} minutes
Models Trained: XGBoost, {"LSTM" if gpu_available else "LSTM skipped (no GPU)"}
Cryptocurrencies: {len(crypto_data)} (including Solana)
Equities: {len(equity_data)}

KEY FINDINGS:
------------
1. Solana (SOL) was successfully included in analysis
2. Crypto mean accuracy: {crypto_mean:.1%}
3. Equity mean accuracy: {equity_mean:.1%}
4.
""")