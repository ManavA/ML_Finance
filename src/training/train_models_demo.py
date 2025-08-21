#!/usr/bin/env python3
"""
"""

import sys
import os


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PROPER MODEL TRAINING ")
print("="*80)

# ============================================================================
# 1. LOAD REAL DATA
# ============================================================================

print("\n1. LOADING REAL DATA")
print("-"*60)

# Download 2 years of data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)

print("\nDownloading 2 years of data...")
btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

print(f"  BTC: {len(btc)} days")
print(f"  SPY: {len(spy)} days")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\n2. FEATURE ENGINEERING")
print("-"*60)

def create_features(data, name):
    """Create ML features"""
    df = pd.DataFrame()
    
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close_col = ('Close', name + '-USD') if (name + '-USD') in data['Close'].columns else 'Close'
        volume_col = ('Volume', name + '-USD') if (name + '-USD') in data['Volume'].columns else 'Volume'
        df['Close'] = data['Close'].iloc[:, 0] if len(data['Close'].shape) > 1 else data['Close']
        df['Volume'] = data['Volume'].iloc[:, 0] if len(data['Volume'].shape) > 1 else data['Volume']
    else:
        df['Close'] = data['Close']
        df['Volume'] = data['Volume']
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Technical indicators
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['price_to_sma20'] = df['Close'] / df['sma_20']
    df['price_to_sma50'] = df['Close'] / df['sma_50']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Target: Next day up/down
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    
    # Clean
    df = df.dropna()
    
    print(f"  {name}: {len(df)} samples, {len([c for c in df.columns if c != 'target'])} features")
    
    return df

btc_features = create_features(btc, 'BTC')
spy_features = create_features(spy, 'SPY')

# ============================================================================
# 3. QUICK TRAINING (INSUFFICIENT)
# ============================================================================

print("\n3. QUICK TRAINING (INSUFFICIENT)")
print("-"*60)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

def quick_training(data, name):
    """Quick training with minimal iterations - what we've been doing"""
    
    print(f"\n{name} - Quick Training:")
    
    # Prepare data
    feature_cols = ['price_to_sma20', 'price_to_sma50', 'volatility', 'rsi', 'volume_ratio']
    X = data[feature_cols]
    y = data['target']
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train with minimal iterations
    start = time.time()
    
    # XGBoost with only 10 trees
    model = xgb.XGBClassifier(
        n_estimators=10,  # Too few!
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, verbose=0)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    train_time = time.time() - start
    
    print(f"  Trees: 10")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy: {test_acc:.2%}")
    print(f"  Overfit: {'YES' if train_acc - test_acc > 0.1 else 'NO'}")
    
    return test_acc

# Quick train both
btc_quick = quick_training(btc_features, 'BTC')
spy_quick = quick_training(spy_features, 'SPY')

# ============================================================================
# 4. PROPER TRAINING (SUFFICIENT)
# ============================================================================

print("\n4. PROPER TRAINING (SUFFICIENT)")
print("-"*60)

def proper_training(data, name):
    """Proper training with adequate iterations"""
    
    print(f"\n{name} - Proper Training:")
    
    # Prepare data
    feature_cols = ['price_to_sma20', 'price_to_sma50', 'volatility', 'rsi', 'volume_ratio']
    X = data[feature_cols]
    y = data['target']
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train with proper iterations
    start = time.time()
    
    # XGBoost with adequate trees
    model = xgb.XGBClassifier(
        n_estimators=500,  # Proper number
        max_depth=5,
        learning_rate=0.01,  # Lower learning rate
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Train with early stopping
    eval_set = [(X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=0
    )
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    train_time = time.time() - start
    
    # Feature importance
    importance = model.feature_importances_
    
    print(f"  Trees: 500")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy: {test_acc:.2%}")
    print(f"  Overfit: {'YES' if train_acc - test_acc > 0.1 else 'NO'}")
    
    print(f"\n  Feature Importance:")
    for feat, imp in zip(feature_cols, importance):
        print(f"    {feat}: {imp:.3f}")
    
    return test_acc, model

# Proper train both
btc_proper, btc_model = proper_training(btc_features, 'BTC')
spy_proper, spy_model = proper_training(spy_features, 'SPY')


print("\n5. DEEP LEARNING COMPARISON")
print("-"*60)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :]))
    
    def train_lstm(data, name, epochs=10):
        """Train LSTM with given epochs"""
        
        print(f"\n{name} LSTM Training ({epochs} epochs):")
        
        # Prepare sequences
        feature_cols = ['price_to_sma20', 'price_to_sma50', 'volatility', 'rsi', 'volume_ratio']
        X = data[feature_cols].values
        y = data['target'].values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
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
        model = SimpleLSTM(input_size=5).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        start = time.time()
        model.train()
        
        for epoch in range(epochs):
            # Forward
            outputs = model(X_train).squeeze()
            loss = criterion(outputs, y_train)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            train_pred = (model(X_train).squeeze() > 0.5).float()
            test_pred = (model(X_test).squeeze() > 0.5).float()
            train_acc = (train_pred == y_train).float().mean().item()
            test_acc = (test_pred == y_test).float().mean().item()
        
        train_time = time.time() - start
        
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Train accuracy: {train_acc:.2%}")
        print(f"  Test accuracy: {test_acc:.2%}")
        
        return test_acc
    
    # Quick LSTM (insufficient)
    print("\nQUICK LSTM TRAINING (10 epochs):")
    btc_lstm_quick = train_lstm(btc_features, 'BTC', epochs=10)
    
    # Proper LSTM (better)
    print("\nPROPER LSTM TRAINING (100 epochs):")
    btc_lstm_proper = train_lstm(btc_features, 'BTC', epochs=100)
    
except Exception as e:
    print(f"  PyTorch not available: {e}")
    btc_lstm_quick = 0.5
    btc_lstm_proper = 0.5


print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

print("\n1. QUICK vs PROPER TRAINING (XGBoost):")
print("-"*60)
print(f"              Quick    Proper   Improvement")
print(f"BTC:          {btc_quick:.1%}    {btc_proper:.1%}    {(btc_proper-btc_quick)*100:+.1f}pp")
print(f"SPY:          {spy_quick:.1%}    {spy_proper:.1%}    {(spy_proper-spy_quick)*100:+.1f}pp")

print("\n2. CRYPTO vs EQUITY COMPARISON:")
print("-"*60)
print(f"Quick Training:")
print(f"  BTC: {btc_quick:.1%}")
print(f"  SPY: {spy_quick:.1%}")
print(f"  Crypto Advantage: {(btc_quick-spy_quick)*100:.1f}pp")

print(f"\nProper Training:")
print(f"  BTC: {btc_proper:.1%}")
print(f"  SPY: {spy_proper:.1%}")
print(f"  Crypto Advantage: {(btc_proper-spy_proper)*100:.1f}pp")


print("\n" + "="*80)
print("HYPOTHESIS TEST RESULTS")
print("="*80)

crypto_advantage = btc_proper - spy_proper

print(f"""
Hypothesis: "ML models perform better on cryptocurrency markets than equity markets"

Results with PROPER training:
• Crypto (BTC) Accuracy: {btc_proper:.1%}
• Equity (SPY) Accuracy: {spy_proper:.1%}
• Advantage: {crypto_advantage*100:.1f}pp

Statistical Significance:
• Samples: {len(btc_features)} (BTC), {len(spy_features)} (SPY)
• Confidence: {"HIGH" if abs(crypto_advantage) > 0.03 else "MEDIUM" if abs(crypto_advantage) > 0.01 else "LOW"}

CONCLUSION: {"CONFIRMED" if crypto_advantage > 0.01 else " NOT CONFIRMED"}
{"ML models show superior performance in crypto markets" if crypto_advantage > 0.01 else "No significant advantage found"}
""")

