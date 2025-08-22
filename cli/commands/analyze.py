"""
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@click.group(name='analyze')
@click.pass_context
def analyze_group(ctx):
    pass

@analyze_group.command(name='performance')
@click.option('--results-file', '-r', required=True, help='Path to backtest results file')
@click.option('--format', type=click.Choice(['text', 'json', 'html']), default='text')
@click.pass_context
def analyze_performance(ctx, results_file, format):
    logger = ctx.obj['logger']
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        click.echo(f"Error loading results: {e}")
        return
    
    metrics = results.get('metrics', {})
    
    if format == 'text':
        click.echo("\n" + "="*60)
        click.echo("PERFORMANCE ANALYSIS REPORT")
        click.echo("="*60)
        click.echo(f"Symbol: {results.get('symbol', 'Unknown')}")
        click.echo(f"Strategy: {results.get('strategy', 'Unknown')}")
        click.echo(f"Period: {results.get('start_date', '')} to {results.get('end_date', '')}")
        
        click.echo("\nReturns:")
        click.echo(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        click.echo(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
        click.echo(f"  Monthly Return: {metrics.get('monthly_return', 0):.2%}")
        
        click.echo("\nRisk Metrics:")
        click.echo(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        click.echo(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        click.echo(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        click.echo(f"  Volatility: {metrics.get('volatility', 0):.2%}")
        
        click.echo("\nTrading Statistics:")
        click.echo(f"  Total Trades: {metrics.get('total_trades', 0)}")
        click.echo(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        click.echo(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        click.echo(f"  Average Win: {metrics.get('avg_win', 0):.2%}")
        click.echo(f"  Average Loss: {metrics.get('avg_loss', 0):.2%}")
        
    elif format == 'json':
        click.echo(json.dumps(results, indent=2))
        
    elif format == 'html':
        html = generate_html_report(results)
        output_file = results_file.replace('.json', '.html')
        with open(output_file, 'w') as f:
            f.write(html)
        click.echo(f"HTML report saved to {output_file}")

@analyze_group.command(name='compare')
@click.option('--results-dir', '-d', required=True, help='Directory with result files')
@click.option('--output', '-o', help='Output file for comparison')
@click.pass_context
def compare_results(ctx, results_dir, output):
    logger = ctx.obj['logger']
    
    results_path = Path(results_dir)
    if not results_path.exists():
        click.echo(f"Directory {results_dir} not found")
        return
    
    # Load all JSON result files
    result_files = list(results_path.glob('*.json'))
    
    if not result_files:
        click.echo("No result files found")
        return
    
    click.echo(f"Comparing {len(result_files)} results...")
    
    comparisons = []
    for file in result_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                comparisons.append({
                    'file': file.name,
                    'symbol': data.get('symbol', 'Unknown'),
                    'strategy': data.get('strategy', 'Unknown'),
                    'total_return': data.get('metrics', {}).get('total_return', 0),
                    'sharpe_ratio': data.get('metrics', {}).get('sharpe_ratio', 0),
                    'max_drawdown': data.get('metrics', {}).get('max_drawdown', 0),
                    'win_rate': data.get('metrics', {}).get('win_rate', 0)
                })
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    # Create comparison dataframe
    df = pd.DataFrame(comparisons)
    
    # Sort by total return
    df = df.sort_values('total_return', ascending=False)
    
    click.echo("\n" + "="*100)
    click.echo("STRATEGY COMPARISON")
    click.echo("="*100)
    
    # Display table
    for idx, row in df.iterrows():
        click.echo(f"\n{row['strategy']} ({row['symbol']}):")
        click.echo(f"  Return: {row['total_return']:.2%}  Sharpe: {row['sharpe_ratio']:.2f}  "
                  f"Drawdown: {row['max_drawdown']:.2%}  Win Rate: {row['win_rate']:.2%}")
    
    # Best performers
    click.echo("\n" + "-"*100)
    click.echo("BEST PERFORMERS:")
    click.echo(f"  Highest Return: {df.iloc[0]['strategy']} ({df.iloc[0]['total_return']:.2%})")
    
    best_sharpe_idx = df['sharpe_ratio'].idxmax()
    click.echo(f"  Best Risk-Adjusted: {df.loc[best_sharpe_idx, 'strategy']} "
              f"(Sharpe: {df.loc[best_sharpe_idx, 'sharpe_ratio']:.2f})")
    
    best_dd_idx = df['max_drawdown'].idxmin()
    click.echo(f"  Lowest Drawdown: {df.loc[best_dd_idx, 'strategy']} "
              f"({df.loc[best_dd_idx, 'max_drawdown']:.2%})")
    
    # Save comparison if requested
    if output:
        df.to_csv(output, index=False)
        click.echo(f"\nComparison saved to {output}")

@analyze_group.command(name='report')
@click.option('--symbol', '-s', required=True, help='Symbol to analyze')
@click.option('--period', default='30d', help='Analysis period (e.g., 30d, 3m, 1y)')
@click.option('--output', '-o', help='Output file for report')
@click.pass_context
def generate_report(ctx, symbol, period, output):
    logger = ctx.obj['logger']
    
    click.echo(f"Generating report for {symbol} over {period}...")
    
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
    
    # Parse period
    if period.endswith('d'):
        days = int(period[:-1])
        df = df.tail(days * 24)  # Assuming hourly data
    elif period.endswith('m'):
        months = int(period[:-1])
        df = df.tail(months * 30 * 24)
    elif period.endswith('y'):
        years = int(period[:-1])
        df = df.tail(years * 365 * 24)
    
    # Calculate statistics
    report = {
        'symbol': symbol,
        'period': period,
        'start_date': str(df.index[0]),
        'end_date': str(df.index[-1]),
        'data_points': len(df),
        'price_statistics': {
            'current': float(df['close'].iloc[-1]),
            'high': float(df['close'].max()),
            'low': float(df['close'].min()),
            'mean': float(df['close'].mean()),
            'std': float(df['close'].std())
        },
        'returns': {
            'total': float((df['close'].iloc[-1] / df['close'].iloc[0] - 1)),
            'daily_avg': float(df['close'].pct_change().mean()),
            'daily_std': float(df['close'].pct_change().std()),
            'sharpe': float(df['close'].pct_change().mean() / df['close'].pct_change().std() * np.sqrt(365))
        },
        'volume': {
            'total': float(df['volume'].sum()),
            'daily_avg': float(df['volume'].mean()),
            'daily_std': float(df['volume'].std())
        },
        'volatility': {
            'daily': float(df['close'].pct_change().std()),
            'annualized': float(df['close'].pct_change().std() * np.sqrt(365))
        },
        'technical_indicators': {}
    }
    
    # Add technical indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    
    report['technical_indicators'] = {
        'rsi_current': float(df['rsi'].iloc[-1]) if not pd.isna(df['rsi'].iloc[-1]) else None,
        'ma_20_current': float(df['ma_20'].iloc[-1]) if not pd.isna(df['ma_20'].iloc[-1]) else None,
        'ma_50_current': float(df['ma_50'].iloc[-1]) if not pd.isna(df['ma_50'].iloc[-1]) else None,
        'price_vs_ma20': float((df['close'].iloc[-1] / df['ma_20'].iloc[-1] - 1)) if not pd.isna(df['ma_20'].iloc[-1]) else None,
        'price_vs_ma50': float((df['close'].iloc[-1] / df['ma_50'].iloc[-1] - 1)) if not pd.isna(df['ma_50'].iloc[-1]) else None
    }
    
    # Display report
    click.echo("\n" + "="*60)
    click.echo(f"ANALYSIS REPORT - {symbol}")
    click.echo("="*60)
    click.echo(f"Period: {report['start_date']} to {report['end_date']}")
    click.echo(f"Data Points: {report['data_points']}")
    
    click.echo("\nPrice Statistics:")
    click.echo(f"  Current: ${report['price_statistics']['current']:,.2f}")
    click.echo(f"  High: ${report['price_statistics']['high']:,.2f}")
    click.echo(f"  Low: ${report['price_statistics']['low']:,.2f}")
    click.echo(f"  Mean: ${report['price_statistics']['mean']:,.2f}")
    
    click.echo("\nReturns:")
    click.echo(f"  Total Return: {report['returns']['total']:.2%}")
    click.echo(f"  Sharpe Ratio: {report['returns']['sharpe']:.2f}")
    
    click.echo("\nVolatility:")
    click.echo(f"  Daily: {report['volatility']['daily']:.2%}")
    click.echo(f"  Annualized: {report['volatility']['annualized']:.2%}")
    
    click.echo("\nTechnical Indicators:")
    if report['technical_indicators']['rsi_current']:
        click.echo(f"  RSI: {report['technical_indicators']['rsi_current']:.2f}")
    if report['technical_indicators']['price_vs_ma20']:
        click.echo(f"  Price vs MA20: {report['technical_indicators']['price_vs_ma20']:.2%}")
    if report['technical_indicators']['price_vs_ma50']:
        click.echo(f"  Price vs MA50: {report['technical_indicators']['price_vs_ma50']:.2%}")
    
    # Save report if requested
    if output:
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        click.echo(f"\nReport saved to {output}")

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_html_report(results):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .metric {{ margin: 10px 0; }}
            .label {{ font-weight: bold; display: inline-block; width: 150px; }}
            .value {{ color: #0066cc; }}
            .section {{ margin: 20px 0; padding: 20px; background: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>Backtest Performance Report</h1>
        <div class="section">
            <h2>Overview</h2>
            <div class="metric">
                <span class="label">Symbol:</span>
                <span class="value">{results.get('symbol', 'Unknown')}</span>
            </div>
            <div class="metric">
                <span class="label">Strategy:</span>
                <span class="value">{results.get('strategy', 'Unknown')}</span>
            </div>
        </div>
        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="metric">
                <span class="label">Total Return:</span>
                <span class="value">{results.get('metrics', {}).get('total_return', 0):.2%}</span>
            </div>
            <div class="metric">
                <span class="label">Sharpe Ratio:</span>
                <span class="value">{results.get('metrics', {}).get('sharpe_ratio', 0):.2f}</span>
            </div>
        </div>
    </body>
    </html>
    """
    return html