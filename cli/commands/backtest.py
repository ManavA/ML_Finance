"""
Backtesting commands for the CLI
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.backtester import Backtester
from src.strategies.baseline_strategies import BaselineStrategies

@click.group(name='backtest')
@click.pass_context
def backtest_group(ctx):
    """Backtesting and strategy evaluation commands"""
    pass

@backtest_group.command(name='run')
@click.option('--symbol', '-s', required=True, help='Symbol to backtest')
@click.option('--strategy', required=True, 
              type=click.Choice(['ma_crossover', 'rsi', 'bollinger', 'macd', 'combined']),
              help='Strategy to test')
@click.option('--start', help='Start date (YYYY-MM-DD)')
@click.option('--end', help='End date (YYYY-MM-DD)')
@click.option('--capital', default=10000, type=float, help='Initial capital')
@click.option('--commission', default=0.001, type=float, help='Commission rate')
@click.option('--output', '-o', help='Output file for results')
@click.pass_context
def run_backtest(ctx, symbol, strategy, start, end, capital, commission, output):
    """Run a backtest for a specific strategy"""
    logger = ctx.obj['logger']
    
    click.echo(f"Running backtest for {symbol} using {strategy} strategy...")
    
    # Load data
    cache_dir = Path('data/cache')
    data_files = list(cache_dir.glob(f'*{symbol}*'))
    
    if not data_files:
        click.echo(f"Error: No data found for {symbol}")
        return
    
    # Load the first matching file
    file = data_files[0]
    if file.suffix == '.parquet':
        df = pd.read_parquet(file)
    else:
        df = pd.read_csv(file, index_col=0, parse_dates=True)
    
    # Filter by date range if specified
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    
    click.echo(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    
    # Initialize strategy
    strategies = BaselineStrategies()
    
    # Generate signals based on strategy
    if strategy == 'ma_crossover':
        signals = strategies.ma_crossover_strategy(df)
    elif strategy == 'rsi':
        signals = strategies.rsi_strategy(df)
    elif strategy == 'bollinger':
        signals = strategies.bollinger_bands_strategy(df)
    elif strategy == 'macd':
        signals = strategies.macd_strategy(df)
    elif strategy == 'combined':
        signals = strategies.combined_strategy(df)
    else:
        click.echo(f"Error: Unknown strategy {strategy}")
        return
    
    # Run backtest
    backtester = Backtester(initial_capital=capital, commission=commission)
    results = backtester.run(df, signals)
    
    # Display results
    click.echo("\n" + "="*50)
    click.echo("BACKTEST RESULTS")
    click.echo("="*50)
    
    metrics = results['metrics']
    click.echo(f"Total Return: {metrics['total_return']:.2%}")
    click.echo(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
    click.echo(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    click.echo(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    click.echo(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    click.echo(f"Total Trades: {metrics.get('total_trades', 0)}")
    click.echo(f"Final Portfolio Value: ${metrics.get('final_value', capital):,.2f}")
    
    # Save results if output specified
    if output:
        with open(output, 'w') as f:
            json.dump({
                'symbol': symbol,
                'strategy': strategy,
                'start_date': str(df.index[0]),
                'end_date': str(df.index[-1]),
                'metrics': metrics,
                'parameters': {
                    'initial_capital': capital,
                    'commission': commission
                }
            }, f, indent=2, default=str)
        click.echo(f"\nResults saved to {output}")

@backtest_group.command(name='compare')
@click.option('--symbol', '-s', required=True, help='Symbol to test')
@click.option('--strategies', default='all', help='Comma-separated strategies or "all"')
@click.option('--start', help='Start date (YYYY-MM-DD)')
@click.option('--end', help='End date (YYYY-MM-DD)')
@click.option('--capital', default=10000, type=float, help='Initial capital')
@click.pass_context
def compare_strategies(ctx, symbol, strategies, start, end, capital):
    """Compare multiple strategies"""
    logger = ctx.obj['logger']
    
    # Parse strategies
    if strategies == 'all':
        strategy_list = ['ma_crossover', 'rsi', 'bollinger', 'macd', 'combined']
    else:
        strategy_list = [s.strip() for s in strategies.split(',')]
    
    click.echo(f"Comparing {len(strategy_list)} strategies for {symbol}...")
    
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
    
    # Run backtests
    results = {}
    strategies_obj = BaselineStrategies()
    backtester = Backtester(initial_capital=capital)
    
    with click.progressbar(strategy_list, label='Testing strategies') as strategies_progress:
        for strategy in strategies_progress:
            # Generate signals
            if strategy == 'ma_crossover':
                signals = strategies_obj.ma_crossover_strategy(df)
            elif strategy == 'rsi':
                signals = strategies_obj.rsi_strategy(df)
            elif strategy == 'bollinger':
                signals = strategies_obj.bollinger_bands_strategy(df)
            elif strategy == 'macd':
                signals = strategies_obj.macd_strategy(df)
            elif strategy == 'combined':
                signals = strategies_obj.combined_strategy(df)
            else:
                continue
            
            # Run backtest
            result = backtester.run(df, signals)
            results[strategy] = result['metrics']
    
    # Display comparison table
    click.echo("\n" + "="*80)
    click.echo("STRATEGY COMPARISON")
    click.echo("="*80)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results).T
    
    # Format and display key metrics
    metrics_to_show = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    for metric in metrics_to_show:
        if metric in comparison_df.columns:
            click.echo(f"\n{metric.replace('_', ' ').title()}:")
            for strategy in comparison_df.index:
                value = comparison_df.loc[strategy, metric]
                if metric in ['total_return', 'max_drawdown', 'win_rate']:
                    click.echo(f"  {strategy:15} {value:.2%}")
                else:
                    click.echo(f"  {strategy:15} {value:.2f}")
    
    # Find best strategy
    best_return = comparison_df['total_return'].idxmax()
    best_sharpe = comparison_df['sharpe_ratio'].idxmax()
    
    click.echo("\n" + "="*80)
    click.echo(f"Best Total Return: {best_return} ({comparison_df.loc[best_return, 'total_return']:.2%})")
    click.echo(f"Best Sharpe Ratio: {best_sharpe} ({comparison_df.loc[best_sharpe, 'sharpe_ratio']:.2f})")

@backtest_group.command(name='optimize')
@click.option('--symbol', '-s', required=True, help='Symbol to optimize')
@click.option('--strategy', required=True, help='Strategy to optimize')
@click.option('--metric', default='sharpe_ratio', 
              type=click.Choice(['total_return', 'sharpe_ratio', 'win_rate']),
              help='Metric to optimize')
@click.pass_context
def optimize_strategy(ctx, symbol, strategy, metric):
    """Optimize strategy parameters"""
    logger = ctx.obj['logger']
    
    click.echo(f"Optimizing {strategy} strategy for {symbol} to maximize {metric}...")
    
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
    
    # Define parameter ranges based on strategy
    if strategy == 'ma_crossover':
        param_ranges = {
            'fast_period': range(5, 20, 2),
            'slow_period': range(20, 50, 5)
        }
    elif strategy == 'rsi':
        param_ranges = {
            'period': range(7, 21, 2),
            'oversold': range(20, 35, 5),
            'overbought': range(65, 80, 5)
        }
    else:
        click.echo(f"Optimization not implemented for {strategy}")
        return
    
    # Grid search optimization
    best_params = None
    best_score = -float('inf')
    
    strategies_obj = BaselineStrategies()
    backtester = Backtester()
    
    # Count total combinations
    total = 1
    for values in param_ranges.values():
        total *= len(values)
    
    click.echo(f"Testing {total} parameter combinations...")
    
    with click.progressbar(length=total, label='Optimizing') as bar:
        # Generate all combinations
        import itertools
        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]
        
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            # Generate signals with current parameters
            if strategy == 'ma_crossover':
                df['ma_fast'] = df['close'].rolling(params['fast_period']).mean()
                df['ma_slow'] = df['close'].rolling(params['slow_period']).mean()
                signals = (df['ma_fast'] > df['ma_slow']).astype(int).diff().fillna(0)
            elif strategy == 'rsi':
                # Calculate RSI with custom parameters
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(params['period']).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(params['period']).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                signals = pd.Series(0, index=df.index)
                signals[rsi < params['oversold']] = 1
                signals[rsi > params['overbought']] = -1
            
            # Run backtest
            result = backtester.run(df, signals)
            score = result['metrics'].get(metric, -float('inf'))
            
            if score > best_score:
                best_score = score
                best_params = params
            
            bar.update(1)
    
    click.echo(f"\n\nOptimization complete!")
    click.echo(f"Best parameters: {best_params}")
    click.echo(f"Best {metric}: {best_score:.4f}")