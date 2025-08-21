# quickstart.py - Minimal working example
"""
Quick start script for cryptocurrency ML trading.
This is a simplified version for getting started quickly.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# Simple GRU Model
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, 
                         batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use last time step
        return out


def fetch_crypto_data(symbol='BTC/USDT', days=365):
    """Fetch cryptocurrency data from Binance."""
    print(f"Fetching {symbol} data...")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Calculate timestamps
    end = datetime.now()
    start = end - timedelta(days=days)
    since = exchange.parse8601(start.isoformat())
    
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Fetch more data if needed
    while len(df) < days * 24:
        last_timestamp = int(df.index[-1].timestamp() * 1000)
        more_data = exchange.fetch_ohlcv(symbol, '1h', since=last_timestamp + 1, limit=1000)
        if not more_data:
            break
        new_df = pd.DataFrame(more_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        new_df.set_index('timestamp', inplace=True)
        df = pd.concat([df, new_df])
    
    print(f"Fetched {len(df)} data points")
    return df


def add_features(df):
    """Add technical indicators as features."""
    print("Adding technical indicators...")
    
    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    df['sma_7'] = ta.trend.sma_indicator(df['close'], window=7)
    df['sma_21'] = ta.trend.sma_indicator(df['close'], window=21)
    df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_width'] = bb.bollinger_wband()
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Clean up
    df.dropna(inplace=True)
    
    return df


def prepare_sequences(data, seq_length=168, pred_horizon=24):
    """Prepare sequences for training."""
    print(f"Creating sequences (length={seq_length}, horizon={pred_horizon})...")
    
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length - pred_horizon):
        seq = data[i:i + seq_length]
        target = data[i + seq_length + pred_horizon - 1, 3]  # Close price
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def train_model(model, X_train, y_train, X_val, y_val, epochs=50):
    """Train the model."""
    print(f"Training model for {epochs} epochs...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
    return model, train_losses, val_losses


def backtest_strategy(df, model, scaler, seq_length=168):
    """Simple backtesting strategy."""
    print("Running backtest...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Prepare data for prediction
    features = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                'rsi', 'macd', 'bb_width', 'volume_ratio']
    data = scaler.transform(df[features].values)
    
    # Generate predictions
    predictions = []
    
    for i in range(seq_length, len(data) - 24):
        seq = data[i-seq_length:i]
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(seq_tensor).cpu().numpy()[0, 0]
            predictions.append(pred)
    
    # Inverse transform predictions
    pred_array = np.array(predictions).reshape(-1, 1)
    dummy = np.zeros((len(pred_array), len(features)))
    dummy[:, 3] = pred_array.flatten()  # Put in close price column
    pred_prices = scaler.inverse_transform(dummy)[:, 3]
    
    # Calculate signals
    current_prices = df['close'].values[seq_length:-24]
    signals = np.where(pred_prices > current_prices * 1.01, 1,  # Buy signal
                       np.where(pred_prices < current_prices * 0.99, -1, 0))  # Sell signal
    
    # Simple backtest
    position = 0
    portfolio_value = 10000
    cash = 10000
    trades = []
    
    for i in range(len(signals)):
        price = current_prices[i]
        signal = signals[i]
        
        if signal == 1 and position == 0:  # Buy
            position = cash / price
            cash = 0
            trades.append({'type': 'buy', 'price': price})
        elif signal == -1 and position > 0:  # Sell
            cash = position * price * 0.999  # 0.1% fee
            position = 0
            trades.append({'type': 'sell', 'price': price})
        
        # Update portfolio value
        portfolio_value = cash + position * price if position > 0 else cash
    
    # Final value
    if position > 0:
        cash = position * current_prices[-1] * 0.999
        portfolio_value = cash
    
    # Calculate metrics
    total_return = (portfolio_value - 10000) / 10000
    num_trades = len(trades)
    
    print(f"\nBacktest Results:")
    print(f"Initial Capital: $10,000")
    print(f"Final Portfolio Value: ${portfolio_value:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Number of Trades: {num_trades}")
    
    if trades:
        wins = sum(1 for i in range(0, len(trades)-1, 2) 
                  if i+1 < len(trades) and trades[i+1]['price'] > trades[i]['price'])
        win_rate = wins / (len(trades) // 2) if len(trades) >= 2 else 0
        print(f"Win Rate: {win_rate:.2%}")
    
    return portfolio_value, trades


def main():
    """Main function to run the complete pipeline."""
    print("=" * 50)
    print("Cryptocurrency ML Trading - Quick Start")
    print("=" * 50)
    
    # 1. Fetch data
    df = fetch_crypto_data('BTC/USDT', days=365)
    
    # 2. Add features
    df = add_features(df)
    
    # 3. Prepare data
    features = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                'rsi', 'macd', 'bb_width', 'volume_ratio']
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features].values)
    
    # Create sequences
    X, y = prepare_sequences(scaled_data, seq_length=168, pred_horizon=24)
    
    # Split data (70/15/15)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 4. Create and train model
    input_size = len(features)
    model = SimpleGRU(input_size=input_size, hidden_size=128, output_size=1)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model, train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val, epochs=50
    )
    
    # 5. Test evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = nn.MSELoss()(test_predictions, y_test_tensor)
        
    print(f"\nTest Loss: {test_loss.item():.6f}")
    
    # Calculate direction accuracy
    pred_np = test_predictions.cpu().numpy().flatten()
    true_np = y_test
    
    if len(pred_np) > 1:
        pred_direction = np.diff(pred_np) > 0
        true_direction = np.diff(true_np) > 0
        direction_accuracy = np.mean(pred_direction == true_direction)
        print(f"Direction Accuracy: {direction_accuracy:.2%}")
    
    # 6. Backtest
    test_df = df.iloc[train_size + val_size:]
    portfolio_value, trades = backtest_strategy(test_df, model, scaler)
    
    # 7. Save model
    print(f"\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'features': features,
        'seq_length': 168,
        'pred_horizon': 24
    }, 'quickstart_model.pt')
    
    print("\nComplete! Model saved as 'quickstart_model.pt'")
    print("=" * 50)


if __name__ == "__main__":
    main()