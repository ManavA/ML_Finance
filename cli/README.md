# Crypto ML Trading CLI

A command-line interface for the cryptocurrency machine learning trading system.

## Installation

Ensure you have the required dependencies:

```bash
pip install click pandas numpy torch xgboost lightgbm catboost scikit-learn
```

## Usage

The CLI provides several command groups for different operations:

### Data Management

```bash
# Fetch data for specific symbols
python cli.py data fetch --symbols BTC,ETH --start 2023-01-01 --end 2024-01-01 --source polygon

# List cached data
python cli.py data list

# Get info about specific symbol data
python cli.py data info BTC

# Validate data quality
python cli.py data validate BTC

# Clear cache
python cli.py data clear-cache --older-than 30
```

### Model Training

```bash
# Train a model
python cli.py train model --symbol BTC --model-type gru --features all --validation-split 0.2

# Train different model types
python cli.py train model --symbol ETH --model-type xgboost
python cli.py train model --symbol BTC --model-type ensemble

# List trained models
python cli.py train list

# Evaluate a model
python cli.py train evaluate models/BTC_gru_20240101_120000.pkl --symbol BTC
```

### Backtesting

```bash
# Run a single backtest
python cli.py backtest run --symbol BTC --strategy ma_crossover --capital 10000

# Compare multiple strategies
python cli.py backtest compare --symbol BTC --strategies all --capital 10000

# Optimize strategy parameters
python cli.py backtest optimize --symbol BTC --strategy rsi --metric sharpe_ratio
```

### Analysis

```bash
# Analyze performance from results file
python cli.py analyze performance --results-file backtest_results.json

# Compare multiple result files
python cli.py analyze compare --results-dir ./results --output comparison.csv

# Generate comprehensive report
python cli.py analyze report --symbol BTC --period 30d --output btc_report.json
```

### Trading (Paper Trading Only)

```bash
# Start paper trading with a strategy
python cli.py trade start --symbol BTC --strategy ma_crossover --capital 10000

# Start trading with a trained model
python cli.py trade start --symbol BTC --model-path models/BTC_gru_model.pkl --capital 10000

# Check trading status
python cli.py trade status

# View trading history
python cli.py trade history --limit 5

# Stop trading session
python cli.py trade stop
```

## Available Models

### Traditional ML Models
- **XGBoost**: Gradient boosting classifier
- **LightGBM**: Efficient gradient boosting
- **CatBoost**: Categorical boosting

### Deep Learning Models
- **GRU**: Advanced GRU with attention mechanism (from `src.models.advanced_models.AdvancedGRU`)
- **LSTM**: Advanced LSTM with attention and residual connections
- **Transformer**: Advanced transformer with positional encoding
- **Ensemble**: Combination of multiple models with optimized weights

### Available Strategies
- **ma_crossover**: Moving average crossover
- **rsi**: RSI-based signals
- **bollinger**: Bollinger Bands strategy
- **macd**: MACD-based signals
- **combined**: Multi-indicator strategy

## Configuration

The CLI uses a default configuration but can load custom configs:

```bash
# Use custom config file
python cli.py --config config.yaml data fetch --symbols BTC

# Show current configuration
python cli.py config
```

## Examples

### Complete Workflow

1. **Fetch Data**:
   ```bash
   python cli.py data fetch --symbols BTC,ETH --start 2023-01-01 --source polygon
   ```

2. **Train Model**:
   ```bash
   python cli.py train model --symbol BTC --model-type gru --save-path models/btc_gru.pkl
   ```

3. **Backtest**:
   ```bash
   python cli.py backtest run --symbol BTC --strategy ma_crossover --output results/btc_ma.json
   ```

4. **Analyze Results**:
   ```bash
   python cli.py analyze performance --results-file results/btc_ma.json
   ```

5. **Paper Trade**:
   ```bash
   python cli.py trade start --symbol BTC --model-path models/btc_gru.pkl --capital 10000
   ```

### Model Comparison

```bash
# Train multiple models
python cli.py train model --symbol BTC --model-type xgboost --save-path models/btc_xgb.pkl
python cli.py train model --symbol BTC --model-type gru --save-path models/btc_gru.pkl
python cli.py train model --symbol BTC --model-type ensemble --save-path models/btc_ensemble.pkl

# Compare their performance
python cli.py train evaluate models/btc_xgb.pkl --symbol BTC
python cli.py train evaluate models/btc_gru.pkl --symbol BTC  
python cli.py train evaluate models/btc_ensemble.pkl --symbol BTC
```

## Notes

- **Live Trading**: Currently disabled for safety. Only paper trading is available.
- **Data Sources**: Supports Polygon.io, CoinMarketCap, and S3. Requires API keys in `.env` file.
- **Model Persistence**: Models are saved as pickle files with metadata.
- **Caching**: Data is automatically cached to avoid repeated API calls.

## Troubleshooting

1. **Missing Dependencies**: Install required packages with pip
2. **Data Access Issues**: Check API keys in `.env` file
3. **Model Loading Errors**: Ensure model files exist and are compatible
4. **Memory Issues**: Use smaller datasets or reduce model complexity

## Advanced Features

### Deep Learning Models

The CLI integrates with the advanced deep learning models from `src.models.advanced_models`:

- **AdvancedGRU**: Bidirectional GRU with multi-head attention and layer normalization
- **AdvancedLSTM**: LSTM with attention mechanism and residual connections  
- **AdvancedTransformer**: Transformer with custom positional encoding
- **DeepEnsembleModel**: Ensemble with stacking and automatic weight optimization

### Custom Strategy Development

Create custom strategies by implementing the `BaseStrategy` interface and adding them to the CLI commands.

Example usage:
```bash
# Use the advanced GRU strategy
python cli.py trade start --symbol BTC --strategy advanced_gru --capital 10000
```