"""
Data management commands for the CLI
"""

import click
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.unified_collector import UnifiedDataCollector

@click.group(name='data')
@click.pass_context
def data_group(ctx):
    pass

@data_group.command(name='fetch')
@click.option('--symbols', '-s', required=True, help='Comma-separated list of symbols (e.g., BTC,ETH)')
@click.option('--start', default=None, help='Start date (YYYY-MM-DD)')
@click.option('--end', default=None, help='End date (YYYY-MM-DD)')
@click.option('--source', default='polygon', type=click.Choice(['polygon', 'cmc', 's3']), help='Data source')
@click.option('--interval', default='1h', help='Data interval (1h, 1d, etc.)')
@click.pass_context
def fetch_data(ctx, symbols, start, end, source, interval):
    logger = ctx.obj['logger']
    
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    # Parse dates
    if not start:
        start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end:
        end = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data for {symbol_list} from {start} to {end} using {source}")
    
    # Initialize collector
    collector = UnifiedDataCollector()
    
    # Fetch data for each symbol
    results = {}
    with click.progressbar(symbol_list, label='Fetching data') as symbols_progress:
        for symbol in symbols_progress:
            try:
                data = collector.fetch_data(
                    symbol=symbol,
                    start_date=start,
                    end_date=end,
                    source=source,
                    interval=interval
                )
                results[symbol] = data
                logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                results[symbol] = None
    
    # Summary
    successful = sum(1 for df in results.values() if df is not None and not df.empty)
    click.echo(f"\nFetch complete: {successful}/{len(symbol_list)} symbols successful")
    
    for symbol, data in results.items():
        if data is not None and not data.empty:
            click.echo(f"  {symbol}: {len(data)} records from {data.index[0]} to {data.index[-1]}")

@data_group.command(name='list')
@click.pass_context
def list_data(ctx):
    logger = ctx.obj['logger']
    
    cache_dir = Path('data/cache')
    if not cache_dir.exists():
        click.echo("No cached data found")
        return
    
    # Find all parquet and CSV files
    files = list(cache_dir.glob('*.parquet')) + list(cache_dir.glob('*.csv'))
    
    if not files:
        click.echo("No cached data files found")
        return
    
    click.echo(f"Found {len(files)} cached data files:")
    for file in sorted(files):
        size_mb = file.stat().st_size / (1024 * 1024)
        click.echo(f"  {file.name} ({size_mb:.2f} MB)")

@data_group.command(name='clear-cache')
@click.option('--symbol', help='Clear cache for specific symbol only')
@click.option('--older-than', type=int, help='Clear cache older than N days')
@click.confirmation_option(prompt='Are you sure you want to clear the cache?')
@click.pass_context
def clear_cache(ctx, symbol, older_than):
    logger = ctx.obj['logger']
    
    collector = UnifiedDataCollector()
    
    if symbol:
        result = collector.clear_cache(symbol=symbol)
    elif older_than:
        result = collector.clear_cache(older_than_days=older_than)
    else:
        result = collector.clear_cache()
    
    if result:
        click.echo(f"Cache cleared successfully")
    else:
        click.echo("Failed to clear cache")

@data_group.command(name='info')
@click.argument('symbol')
@click.pass_context
def data_info(ctx, symbol):
    logger = ctx.obj['logger']
    
    # Try to load from cache first
    cache_dir = Path('data/cache')
    
    # Look for files containing the symbol
    matching_files = list(cache_dir.glob(f'*{symbol}*'))
    
    if not matching_files:
        click.echo(f"No cached data found for {symbol}")
        return
    
    for file in matching_files:
        click.echo(f"\nFile: {file.name}")
        
        # Load and analyze the data
        if file.suffix == '.parquet':
            df = pd.read_parquet(file)
        elif file.suffix == '.csv':
            df = pd.read_csv(file, index_col=0, parse_dates=True)
        else:
            continue
        
        click.echo(f"  Records: {len(df)}")
        click.echo(f"  Date range: {df.index[0]} to {df.index[-1]}")
        click.echo(f"  Columns: {', '.join(df.columns)}")
        click.echo(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Basic statistics
        if 'close' in df.columns:
            click.echo(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            click.echo(f"  Average price: ${df['close'].mean():.2f}")

@data_group.command(name='validate')
@click.argument('symbol')
@click.pass_context
def validate_data(ctx, symbol):
    logger = ctx.obj['logger']
    
    # Load data
    cache_dir = Path('data/cache')
    matching_files = list(cache_dir.glob(f'*{symbol}*'))
    
    if not matching_files:
        click.echo(f"No cached data found for {symbol}")
        return
    
    for file in matching_files:
        click.echo(f"\nValidating: {file.name}")
        
        # Load data
        if file.suffix == '.parquet':
            df = pd.read_parquet(file)
        elif file.suffix == '.csv':
            df = pd.read_csv(file, index_col=0, parse_dates=True)
        else:
            continue
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            click.echo("  Missing values:")
            for col, count in missing[missing > 0].items():
                click.echo(f"    {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            click.echo("  No missing values")
        
        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            click.echo(f"  {duplicates} duplicate timestamps found")
        else:
            click.echo("  No duplicate timestamps")
        
        # Check time gaps
        if hasattr(df.index, 'to_series'):
            time_diffs = df.index.to_series().diff()
            expected_freq = time_diffs.mode()[0]
            gaps = time_diffs[time_diffs > expected_freq * 2]
            if len(gaps) > 0:
                click.echo(f"  {len(gaps)} time gaps detected")
            else:
                click.echo(" No significant time gaps")
        
        # Check price validity
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                click.echo(" Invalid prices (<=0) detected")
            else:
                click.echo(" All prices valid")