
import sys
import os
sys.path.append('src')

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import torch
from typing import Dict, Any

# Quick test configurations - minimal parameters for speed
QUICK_TEST_CONFIG = {
    'smoke': {  # 5 minutes total
        'data_days': 100,
        'train_split': 0.7,
        'models': {
            'xgboost': {'n_estimators': 10, 'max_depth': 3, 'learning_rate': 0.3},
            'lightgbm': {'n_estimators': 10, 'num_leaves': 7, 'learning_rate': 0.3},
            'lstm': {'epochs': 2, 'batch_size': 64, 'hidden_size': 32},
            'dqn': {'episodes': 10, 'batch_size': 32}
        }
    },
    'quick': {  # 30 minutes
        'data_days': 365,
        'train_split': 0.7,
        'models': {
            'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            'lightgbm': {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': 0.1},
            'lstm': {'epochs': 20, 'batch_size': 32, 'hidden_size': 64},
            'dqn': {'episodes': 100, 'batch_size': 32}
        }
    },
    'standard': {  # 2 hours
        'data_days': 730,
        'train_split': 0.7,
        'models': {
            'xgboost': {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05},
            'lightgbm': {'n_estimators': 300, 'num_leaves': 63, 'learning_rate': 0.05},
            'lstm': {'epochs': 50, 'batch_size': 32, 'hidden_size': 128},
            'dqn': {'episodes': 500, 'batch_size': 32}
        }
    }
}

class QuickValidator:
    def __init__(self, level='smoke'):
        self.level = level
        self.config = QUICK_TEST_CONFIG[level]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def fetch_quick_data(self, symbol='BTC-USD'):
        print(f"\nFetching {self.config['data_days']} days of {symbol} data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['data_days'])
        
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # Add basic features
        data['returns'] = data['Close'].pct_change()
        data['sma_10'] = data['Close'].rolling(10).mean()
        data['rsi'] = self._calculate_rsi(data['Close'])
        data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(10).mean()
        
        return data.dropna()
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def test_xgboost(self, data):
        try:
            import xgboost as xgb
            start = time.time()
            
            # Prepare data
            features = ['returns', 'sma_10', 'rsi', 'volume_ratio']
            X = data[features].values[:-1]
            y = (data['returns'].shift(-1) > 0).astype(int).values[:-1]
            
            split = int(len(X) * self.config['train_split'])
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Train with GPU if available
            params = self.config['models']['xgboost'].copy()
            params['tree_method'] = 'gpu_hist' if torch.cuda.is_available() else 'auto'
            params['predictor'] = 'gpu_predictor' if torch.cuda.is_available() else 'auto'
            
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = model.score(X_test, y_test)
            duration = time.time() - start
            
            self.results['xgboost'] = {
                'accuracy': accuracy,
                'duration': duration,
                'status': 'PASS' if accuracy > 0.45 else 'FAIL'
            }
            
            print(f"XGBoost: {accuracy:.2%} accuracy in {duration:.1f}s")
            return True
            
        except Exception as e:
            print(f"XGBoost failed: {str(e)[:50]}")
            self.results['xgboost'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def test_lightgbm(self, data):
        try:
            import lightgbm as lgb
            start = time.time()
            
            # Prepare data
            features = ['returns', 'sma_10', 'rsi', 'volume_ratio']
            X = data[features].values[:-1]
            y = (data['returns'].shift(-1) > 0).astype(int).values[:-1]
            
            split = int(len(X) * self.config['train_split'])
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Train with GPU if available
            params = self.config['models']['lightgbm'].copy()
            params['device'] = 'gpu' if torch.cuda.is_available() else 'cpu'
            params['gpu_use_dp'] = False  # Use float32 for speed
            
            model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = model.score(X_test, y_test)
            duration = time.time() - start
            
            self.results['lightgbm'] = {
                'accuracy': accuracy,
                'duration': duration,
                'status': 'PASS' if accuracy > 0.45 else 'FAIL'
            }
            
            print(f"LightGBM: {accuracy:.2%} accuracy in {duration:.1f}s")
            return True
            
        except Exception as e:
            print(f"LightGBM failed: {str(e)[:50]}")
            self.results['lightgbm'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def test_lstm(self, data):
        try:
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            start = time.time()
            
            # Prepare sequences
            features = ['returns', 'sma_10', 'rsi', 'volume_ratio']
            X_data = data[features].values
            y_data = (data['returns'].shift(-1) > 0).astype(int).values
            
            # Create sequences (window of 10)
            window = 10
            X, y = [], []
            for i in range(window, len(X_data)-1):
                X.append(X_data[i-window:i])
                y.append(y_data[i])
            
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            split = int(len(X) * self.config['train_split'])
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_test = torch.FloatTensor(X_test).to(self.device)
            y_test = torch.FloatTensor(y_test).to(self.device)
            
            # Simple LSTM model
            class SimpleLSTM(nn.Module):
                def __init__(self, input_size, hidden_size):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                    self.fc = nn.Linear(hidden_size, 1)
                    self.sigmoid = nn.Sigmoid()
                    
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    out = self.fc(lstm_out[:, -1, :])
                    return self.sigmoid(out).squeeze()
            
            # Train model
            model = SimpleLSTM(4, self.config['models']['lstm']['hidden_size']).to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            batch_size = self.config['models']['lstm']['batch_size']
            epochs = self.config['models']['lstm']['epochs']
            
            for epoch in range(epochs):
                model.train()
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                predictions = model(X_test)
                accuracy = ((predictions > 0.5) == y_test).float().mean().item()
            
            duration = time.time() - start
            
            self.results['lstm'] = {
                'accuracy': accuracy,
                'duration': duration,
                'status': 'PASS' if accuracy > 0.45 else 'FAIL'
            }
            
            print(f"LSTM: {accuracy:.2%} accuracy in {duration:.1f}s")
            return True
            
        except Exception as e:
            print(f"LSTM failed: {str(e)[:50]}")
            self.results['lstm'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def run_all_tests(self, symbol='BTC-USD'):
        print(f"\n{'='*60}")
        print(f"QUICK VALIDATION - {self.level.upper()} MODE")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Target Time: {5 if self.level == 'smoke' else 30} minutes")
        
        # Fetch data
        data = self.fetch_quick_data(symbol)
        print(f"Data shape: {data.shape}")
        
        # Run tests
        print(f"\nRunning Model Tests...")
        self.test_xgboost(data)
        self.test_lightgbm(data)
        self.test_lstm(data)
        
        # Summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        total_time = sum(r.get('duration', 0) for r in self.results.values())
        passed = sum(1 for r in self.results.values() if r.get('status') == 'PASS')
        failed = sum(1 for r in self.results.values() if r.get('status') == 'FAIL')
        errors = sum(1 for r in self.results.values() if r.get('status') == 'ERROR')
        
        print(f"\nResults:")
        for model, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            prefix = '[PASS]' if status == 'PASS' else '[FAIL]'
            acc = result.get('accuracy', 0)
            dur = result.get('duration', 0)
            print(f"  {prefix} {model:10s}: {status:5s} | Acc: {acc:.1%} | Time: {dur:.1f}s")
        
        print(f"\nStatistics:")
        print(f"  Total Time: {total_time:.1f} seconds")
        print(f"  Passed: {passed}/{len(self.results)}")
        print(f"  Failed: {failed}/{len(self.results)}")
        print(f"  Errors: {errors}/{len(self.results)}")
        
        if total_time < 300:  # Under 5 minutes
            print(f"\nSMOKE TEST SUCCESSFUL - Ready for full training")
        else:
            print(f"\nTests took {total_time:.1f}s - Consider optimization")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Model Validation')
    parser.add_argument('--level', choices=['smoke', 'quick', 'standard'], 
                       default='smoke', help='Test level')
    parser.add_argument('--symbol', default='BTC-USD', help='Symbol to test')
    
    args = parser.parse_args()
    
    validator = QuickValidator(level=args.level)
    results = validator.run_all_tests(symbol=args.symbol)
    
    # Return non-zero exit code if any tests failed
    if any(r.get('status') != 'PASS' for r in results.values()):
        sys.exit(1)
    
    return results


if __name__ == '__main__':
    main()