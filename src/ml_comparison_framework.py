#!/usr/bin/env python3
"""
Main ML Comparison Framework
Tests the hypothesis: ML models offer greater performance edge in crypto vs equities
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from features.feature_engineering import FeatureEngineer
from backtesting.walk_forward import WalkForwardBacktester, BacktestResults
from models.ml_models import MLModelTrainer
from strategies.traditional_strategies import create_strategy_suite
from data.batch_collect_data import BatchDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLComparisonFramework:
    """
    Main framework for comparing ML models vs traditional strategies
    across crypto and equity markets
    """
    
    def __init__(self, 
                 cache_dir: str = 'data/ml_comparison_cache',
                 results_dir: str = 'results/ml_comparison'):
        """
        Initialize comparison framework
        
        Args:
            cache_dir: Directory with cached data
            results_dir: Directory for saving results
        """
        self.cache_dir = Path(cache_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.backtester = WalkForwardBacktester(
            train_months=12,
            val_months=3,
            test_months=3,
            step_months=3
        )
        
        # Store results
        self.all_results = []
        self.comparison_metrics = {}
        
    def load_and_prepare_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load and prepare data for a symbol
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Prepared DataFrame with features
        """
        # Look for cached parquet files
        pattern = f"{symbol}_*.parquet"
        files = list(self.cache_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No data found for {symbol}")
            return None
        
        # Load most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading {symbol} from {latest_file}")
        
        data = pd.read_parquet(latest_file)
        
        # Create features
        logger.info(f"Engineering features for {symbol}")
        data = self.feature_engineer.create_features(data, symbol)
        
        # Create target variables
        data = self.feature_engineer.create_target_variables(data, horizons=[1, 5, 10])
        
        # Filter to training period only (through Dec 2024)
        if 'data_usage' in data.columns:
            data = data[data['data_usage'] == 'training'].copy()
        
        return data
    
    def create_ml_model_functions(self) -> Dict:
        """
        Create model training functions for walk-forward testing
        
        Returns:
            Dictionary of model_name -> training_function
        """
        models = {}
        
        # Feature columns (exclude targets and metadata)
        def get_feature_cols(data):
            exclude = ['timestamp', 'symbol', 'data_usage', 'open', 'high', 'low', 'close', 'volume']
            exclude += [col for col in data.columns if 'target' in col]
            return [col for col in data.columns if col not in exclude]
        
        # LSTM model
        def train_lstm(train_data, val_data, test_data):
            trainer = MLModelTrainer(model_type='lstm', epochs=30, batch_size=32)
            feature_cols = get_feature_cols(train_data)
            target_col = 'target_return_5'  # 5-period ahead prediction
            
            # Train model
            trainer.train_deep_model(train_data, val_data, feature_cols, target_col)
            
            # Predict on test data
            predictions = trainer.predict(test_data, feature_cols)
            return predictions
        
        models['LSTM'] = train_lstm
        
        # GRU model
        def train_gru(train_data, val_data, test_data):
            trainer = MLModelTrainer(model_type='gru', epochs=30, batch_size=32)
            feature_cols = get_feature_cols(train_data)
            target_col = 'target_return_5'
            
            trainer.train_deep_model(train_data, val_data, feature_cols, target_col)
            predictions = trainer.predict(test_data, feature_cols)
            return predictions
        
        models['GRU'] = train_gru
        
        # Transformer model
        def train_transformer(train_data, val_data, test_data):
            trainer = MLModelTrainer(model_type='transformer', epochs=30, batch_size=32)
            feature_cols = get_feature_cols(train_data)
            target_col = 'target_return_5'
            
            trainer.train_deep_model(train_data, val_data, feature_cols, target_col)
            predictions = trainer.predict(test_data, feature_cols)
            return predictions
        
        models['Transformer'] = train_transformer
        
        # Random Forest
        def train_rf(train_data, val_data, test_data):
            trainer = MLModelTrainer(model_type='rf')
            feature_cols = get_feature_cols(train_data)
            target_col = 'target_return_5'
            
            # Combine train and val for traditional ML
            combined_train = pd.concat([train_data, val_data])
            trainer.train_traditional_model(combined_train, test_data, feature_cols, target_col)
            predictions = trainer.predict(test_data, feature_cols)
            return predictions
        
        models['RandomForest'] = train_rf
        
        # Gradient Boosting
        def train_gb(train_data, val_data, test_data):
            trainer = MLModelTrainer(model_type='gb')
            feature_cols = get_feature_cols(train_data)
            target_col = 'target_return_5'
            
            combined_train = pd.concat([train_data, val_data])
            trainer.train_traditional_model(combined_train, test_data, feature_cols, target_col)
            predictions = trainer.predict(test_data, feature_cols)
            return predictions
        
        models['GradientBoosting'] = train_gb
        
        return models
    
    def create_traditional_model_functions(self) -> Dict:
        """
        Create traditional strategy functions for walk-forward testing
        
        Returns:
            Dictionary of strategy_name -> strategy_function
        """
        strategies = create_strategy_suite()
        models = {}
        
        for name, strategy in strategies.items():
            def make_strategy_fn(strat):
                def strategy_fn(train_data, val_data, test_data):
                    # Traditional strategies don't need training
                    # Generate signals on test data
                    signals = strat.generate_signals(test_data)
                    return signals
                return strategy_fn
            
            models[name] = make_strategy_fn(strategy)
        
        return models
    
    def run_comparison(self, symbols: Optional[List[str]] = None):
        """
        Run full comparison of ML vs traditional strategies
        
        Args:
            symbols: List of symbols to test (default: all available)
        """
        if symbols is None:
            # Default symbols
            symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'SPY', 'QQQ']
        
        # Separate crypto and equity symbols
        crypto_symbols = [s for s in symbols if 'USD' in s]
        equity_symbols = [s for s in symbols if 'USD' not in s]
        
        logger.info(f"Testing {len(crypto_symbols)} crypto and {len(equity_symbols)} equity symbols")
        
        # Get all models
        ml_models = self.create_ml_model_functions()
        traditional_models = self.create_traditional_model_functions()
        all_models = {**ml_models, **traditional_models}
        
        # Results storage
        results_by_market = {
            'crypto': [],
            'equity': []
        }
        
        # Test each symbol
        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {symbol}")
            logger.info(f"{'='*60}")
            
            # Load and prepare data
            data = self.load_and_prepare_data(symbol)
            if data is None or len(data) < 1000:
                logger.warning(f"Insufficient data for {symbol}, skipping")
                continue
            
            # Determine market type
            market_type = 'crypto' if 'USD' in symbol else 'equity'
            
            # Run comparison
            comparison_results = self.backtester.compare_models(
                all_models,
                data,
                symbol
            )
            
            if not comparison_results.empty:
                results_by_market[market_type].append(comparison_results)
                self.all_results.append({
                    'symbol': symbol,
                    'market': market_type,
                    'results': comparison_results
                })
        
        # Analyze results
        self.analyze_results(results_by_market)
    
    def analyze_results(self, results_by_market: Dict):
        """
        Analyze and compare results between markets
        
        Args:
            results_by_market: Dictionary of market_type -> results list
        """
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS RESULTS")
        logger.info("="*60)
        
        # Aggregate by market type
        for market_type, results_list in results_by_market.items():
            if not results_list:
                continue
            
            logger.info(f"\n{market_type.upper()} MARKET RESULTS")
            logger.info("-"*40)
            
            # Combine all results for this market
            combined = pd.concat(results_list)
            
            # Calculate average metrics by model type
            ml_models = ['LSTM', 'GRU', 'Transformer', 'RandomForest', 'GradientBoosting']
            traditional_models = ['momentum', 'mean_reversion', 'trend_following', 'macd', 'rsi_mean_reversion']
            
            # ML performance
            ml_results = combined[combined.index.get_level_values('model').isin(ml_models)]
            if not ml_results.empty:
                ml_sharpe = ml_results[('sharpe_ratio', 'mean')].mean()
                ml_return = ml_results[('total_return', 'mean')].mean()
                ml_drawdown = ml_results[('max_drawdown', 'mean')].mean()
                
                logger.info(f"ML Models Average:")
                logger.info(f"  Sharpe Ratio: {ml_sharpe:.3f}")
                logger.info(f"  Total Return: {ml_return:.2f}%")
                logger.info(f"  Max Drawdown: {ml_drawdown:.2f}%")
            
            # Traditional performance
            trad_results = combined[combined.index.get_level_values('model').isin(traditional_models)]
            if not trad_results.empty:
                trad_sharpe = trad_results[('sharpe_ratio', 'mean')].mean()
                trad_return = trad_results[('total_return', 'mean')].mean()
                trad_drawdown = trad_results[('max_drawdown', 'mean')].mean()
                
                logger.info(f"Traditional Models Average:")
                logger.info(f"  Sharpe Ratio: {trad_sharpe:.3f}")
                logger.info(f"  Total Return: {trad_return:.2f}%")
                logger.info(f"  Max Drawdown: {trad_drawdown:.2f}%")
            
            # Calculate ML advantage
            if not ml_results.empty and not trad_results.empty:
                sharpe_advantage = ml_sharpe - trad_sharpe
                return_advantage = ml_return - trad_return
                
                logger.info(f"ML ADVANTAGE in {market_type}:")
                logger.info(f"  Sharpe Ratio Improvement: {sharpe_advantage:.3f}")
                logger.info(f"  Return Improvement: {return_advantage:.2f}%")
                
                self.comparison_metrics[market_type] = {
                    'ml_sharpe': ml_sharpe,
                    'trad_sharpe': trad_sharpe,
                    'sharpe_advantage': sharpe_advantage,
                    'ml_return': ml_return,
                    'trad_return': trad_return,
                    'return_advantage': return_advantage
                }
        
        # Test main hypothesis
        self.test_hypothesis()
    
    def test_hypothesis(self):
        """
        Test the main hypothesis: ML advantage is greater in crypto than equities
        """
        logger.info("\n" + "="*60)
        logger.info("HYPOTHESIS TEST RESULTS")
        logger.info("="*60)
        
        if 'crypto' in self.comparison_metrics and 'equity' in self.comparison_metrics:
            crypto_advantage = self.comparison_metrics['crypto']['sharpe_advantage']
            equity_advantage = self.comparison_metrics['equity']['sharpe_advantage']
            
            logger.info("Hypothesis: ML models offer greater edge in crypto vs equities")
            logger.info(f"Crypto ML Advantage (Sharpe): {crypto_advantage:.3f}")
            logger.info(f"Equity ML Advantage (Sharpe): {equity_advantage:.3f}")
            logger.info(f"Difference: {crypto_advantage - equity_advantage:.3f}")
            
            if crypto_advantage > equity_advantage:
                logger.info("✓ HYPOTHESIS SUPPORTED: ML models show greater advantage in crypto markets")
            else:
                logger.info("✗ HYPOTHESIS NOT SUPPORTED: ML advantage similar or greater in equities")
            
            # Statistical significance test (simplified)
            # In practice, would use bootstrap or more sophisticated tests
            difference = crypto_advantage - equity_advantage
            if abs(difference) > 0.1:  # Threshold for practical significance
                logger.info(f"Result is practically significant (difference > 0.1)")
            
            # Save detailed results
            self.save_results()
    
    def save_results(self):
        """Save all results to files"""
        # Save comparison metrics
        metrics_file = self.results_dir / 'comparison_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.comparison_metrics, f, indent=2)
        
        # Save detailed results
        results_file = self.results_dir / 'detailed_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(self.all_results, f)
        
        # Save summary report
        self.generate_report()
        
        logger.info(f"\nResults saved to {self.results_dir}")
    
    def generate_report(self):
        """Generate summary report"""
        report_file = self.results_dir / 'summary_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("ML vs TRADITIONAL STRATEGY COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Report Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("HYPOTHESIS: ML models offer greater performance edge in volatile cryptocurrency\n")
            f.write("markets compared to more efficient equity markets.\n\n")
            
            if self.comparison_metrics:
                f.write("RESULTS BY MARKET:\n")
                f.write("-"*40 + "\n\n")
                
                for market, metrics in self.comparison_metrics.items():
                    f.write(f"{market.upper()} MARKET:\n")
                    f.write(f"  ML Average Sharpe: {metrics['ml_sharpe']:.3f}\n")
                    f.write(f"  Traditional Average Sharpe: {metrics['trad_sharpe']:.3f}\n")
                    f.write(f"  ML Advantage: {metrics['sharpe_advantage']:.3f}\n")
                    f.write(f"  ML Return: {metrics['ml_return']:.2f}%\n")
                    f.write(f"  Traditional Return: {metrics['trad_return']:.2f}%\n\n")
                
                if 'crypto' in self.comparison_metrics and 'equity' in self.comparison_metrics:
                    crypto_adv = self.comparison_metrics['crypto']['sharpe_advantage']
                    equity_adv = self.comparison_metrics['equity']['sharpe_advantage']
                    
                    f.write("HYPOTHESIS TEST:\n")
                    f.write("-"*40 + "\n")
                    f.write(f"Crypto ML Advantage: {crypto_adv:.3f}\n")
                    f.write(f"Equity ML Advantage: {equity_adv:.3f}\n")
                    f.write(f"Difference: {crypto_adv - equity_adv:.3f}\n\n")
                    
                    if crypto_adv > equity_adv:
                        f.write("CONCLUSION: Hypothesis SUPPORTED\n")
                        f.write("ML models demonstrate greater advantage in cryptocurrency markets.\n")
                    else:
                        f.write("CONCLUSION: Hypothesis NOT SUPPORTED\n")
                        f.write("ML advantage is similar or greater in equity markets.\n")

def main():
    """Main execution function"""
    logger.info("Starting ML Comparison Framework")
    logger.info("="*60)
    
    # Initialize framework
    framework = MLComparisonFramework()
    
    # Run comparison
    # Start with smaller test if needed
    test_symbols = ['BTCUSD', 'SPY']  # Quick test with one crypto and one equity
    # full_symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'SPY', 'QQQ', 'IWM']
    
    framework.run_comparison(symbols=test_symbols)
    
    logger.info("\nComparison complete!")

if __name__ == "__main__":
    main()