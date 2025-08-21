# ML Finance Trading System

A comprehensive machine learning system for cryptocurrency and stock trading, featuring advanced models, real-time data collection, and both CLI and notebook interfaces.

## 📊 Results & Analysis
**[View Jupyter Notebook Results](https://github.com/ManavA/ML_Finance/tree/main/Jupyter%20Notebooks%2C%20Results%2C%20Analysis%20-%20Final)** - Comprehensive analysis including performance metrics, visualizations, and model comparisons

## Features

### Data Collection
- **Multiple Data Sources**: Polygon.io, CoinMarketCap, Yahoo Finance, S3
- **Automatic Fallback**: Seamless switching between sources
- **Smart Caching**: Local cache with TTL management
- **Real-time & Historical**: Support for both data types

### Models
- **Baseline Strategies**: SMA, RSI, MACD, Bollinger Bands
- **Machine Learning**: XGBoost, LightGBM, CatBoost, Random Forest
- **Deep Learning**: LSTM, GRU, Transformer (Helformer)
- **Reinforcement Learning**: DQN, A2C, PPO implementations
- **Ensemble Methods**: Voting, Stacking, Risk Parity

### Analysis
- **Comprehensive Backtesting**: Walk-forward analysis, multiple metrics
- **Risk Management**: Stop-loss, take-profit, position sizing
- **Performance Metrics**: Sharpe, Sortino, Calmar ratios
- **Visualization**: Interactive dashboards and reports

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ManavA/ML_Finance.git
cd ML_Finance

# Install dependencies
pip install -r requirements.txt

# Or install with setup.py for CLI access
pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```
POLYGON_API_KEY=your_key_here
CMC_API_KEY=your_key_here
```

3. Customize `config.yaml` for your needs

### CLI Usage

```bash
# Fetch data
mlf data fetch --symbols BTC,ETH --start 2023-01-01

# Train a model
mlf train model --symbol BTC --model-type xgboost

# Run backtest
mlf backtest run --symbol BTC --strategy ma_crossover

# Start paper trading
mlf trade start --symbol BTC --capital 10000 --paper

# Generate analysis report
mlf analyze report --symbol BTC --period 30d
```

### Python Usage

```python
from src.data.unified_collector import UnifiedDataCollector
from src.models.baseline import XGBoostModel
from src.analysis.backtester import Backtester

# Collect data
collector = UnifiedDataCollector()
data = collector.fetch_data("BTCUSD", start="2023-01-01")

# Train model
model = XGBoostModel()
model.fit(data)

# Generate signals
signals = model.get_signals(data)

# Backtest
backtester = Backtester()
results = backtester.run(data, signals)
print(f"Total Return: {results['metrics']['total_return']:.2%}")
```

## Project Structure

```
ML_Finance/
├── src/                    # Source code
│   ├── data/              # Data collection modules
│   ├── models/            # ML/DL models
│   ├── strategies/        # Trading strategies
│   ├── analysis/          # Backtesting & metrics
│   ├── features/          # Feature engineering
│   └── utils/             # Utilities
├── cli/                   # Command-line interface
│   ├── commands/          # CLI commands
│   └── core/              # CLI core functionality
├── notebooks/             # Jupyter notebooks
├── data/                  # Data storage
│   └── cache/            # Cached data files
├── models/                # Saved models
├── configs/               # Configuration files
└── tests/                 # Test files
```

## Available Models

### Traditional ML
- **XGBoost**: Gradient boosting with high performance
- **LightGBM**: Fast gradient boosting
- **CatBoost**: Handles categorical features well
- **Random Forest**: Ensemble of decision trees

### Deep Learning
- **LSTM**: Long Short-Term Memory networks
- **GRU**: Gated Recurrent Units
- **Transformer**: Attention-based architecture
- **Helformer**: Hierarchical transformer for finance

### Reinforcement Learning
- **DQN**: Deep Q-Network
- **A2C**: Advantage Actor-Critic
- **PPO**: Proximal Policy Optimization

## Strategies

### Baseline
- Buy & Hold
- SMA Crossover
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

### Advanced
- Momentum
- Mean Reversion
- Pairs Trading
- Statistical Arbitrage
- Multi-factor

## Data Sources

### Primary
- **Polygon.io**: Real-time and historical market data
- **CoinMarketCap**: Cryptocurrency data and metrics

### Secondary
- **Yahoo Finance**: Free stock and crypto data
- **S3**: Bulk historical data storage

### Specialized
- **Sanbase**: On-chain metrics
- **Bybit**: Derivatives data

## Development

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_models.py
```

### Code Quality
```bash
# Format code
black src/ cli/

# Lint code
flake8 src/ cli/

# Type checking
mypy src/
```

## Performance Tips

1. **Use caching**: Enable cache in config.yaml
2. **Parallel processing**: Set `n_jobs=-1` for all cores
3. **GPU acceleration**: Enable CUDA for deep learning
4. **Optimize data loading**: Use parquet format
5. **Feature selection**: Remove redundant features

## Common Issues

### TA-Lib Installation
```bash
# Windows
pip install TA-Lib-0.4.24-cp39-cp39-win_amd64.whl

# Linux
sudo apt-get install ta-lib
pip install ta-lib

# Mac
brew install ta-lib
pip install ta-lib
```

### Missing Data
- Check API keys in `.env`
- Verify network connection
- Try alternative data sources

### Out of Memory
- Reduce batch size
- Use data sampling
- Enable chunked processing

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Disclaimer

This software is for educational and research purposes only. Do not use for actual trading without understanding the risks. Cryptocurrency and stock trading involves substantial risk of loss.

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/yourusername/ML_Finance/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/ML_Finance/discussions)

## Acknowledgments

- **Professor MarcAntonio Awada and team at Harvard** - For guidance and academic support
- Thanks to all contributors who helped develop and test this system
- Built with Python and modern ML frameworks
- Inspired by quantitative finance research

## References & Resources

### Academic Foundations
- **Walk-Forward Optimization**: Based on Pardo (1992), enhanced with modern ML techniques
- **Diebold-Mariano Test**: Statistical test for forecast comparison (Diebold & Mariano, 1995)
- **Time Series Cross-Validation**: Bergmeir & Benitez (2012) methodology

### Data Sources & APIs
- **Polygon.io**: [Documentation](https://polygon.io/docs/) - Primary historical data provider
- **CoinMarketCap**: [API Docs](https://coinmarketcap.com/api/documentation/v1/) - Real-time crypto data
- **Sanbase/Santiment**: [Academy](https://academy.santiment.net/sanapi/) - On-chain analytics
- **Bybit Exchange**: [API Docs](https://bybit-exchange.github.io/docs/) - Derivatives trading

### Key Libraries & Frameworks
- **PyTorch**: Deep learning research and experimentation
- **scikit-learn**: Classical ML models and preprocessing
- **XGBoost/LightGBM**: Gradient boosting for tabular data
- **TA-Lib**: Technical analysis indicators
- **vectorbt**: Backtesting and portfolio optimization

### Research Areas
- **Market Microstructure**: Liquidity, spread, and depth analysis
- **Portfolio Optimization**: Modern portfolio theory with crypto adaptations
- **Risk Management**: VaR, CVaR, maximum drawdown metrics
- **Feature Engineering**: Technical, fundamental, and alternative data features

### Books & Publications
- *Advances in Financial Machine Learning* - Marcos López de Prado
- *Machine Learning for Asset Managers* - Marcos López de Prado
- *Quantitative Trading* - Ernest P. Chan
- *Algorithmic Trading* - Jeffrey Bacidore

For complete references and external dependencies, see [APPENDIX_REFERENCES.md](APPENDIX_REFERENCES.md)