"""
Main CLI entry point for the Crypto ML Trading System
"""

import click
import sys
import os
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.commands import data, backtest, train, analyze, trade
from cli.core.config import load_config, get_default_config
from cli.utils.logger import setup_logger

@click.group()
@click.option('--config', '-c', default=None, help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):

    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        ctx.obj['config'] = load_config(config)
    else:
        ctx.obj['config'] = get_default_config()
    
    # Setup logging
    log_level = 'DEBUG' if verbose else ctx.obj['config'].get('logging', {}).get('level', 'INFO')
    ctx.obj['logger'] = setup_logger(log_level)
    
    if verbose:
        ctx.obj['logger'].debug(f"Configuration loaded: {ctx.obj['config']}")

# Add command groups
cli.add_command(data.data_group)
cli.add_command(backtest.backtest_group)
cli.add_command(train.train_group)
cli.add_command(analyze.analyze_group)
cli.add_command(trade.trade_group)

@cli.command()
@click.pass_context
def version(ctx):
    from cli import __version__
    click.echo(f"Crypto ML Trader v{__version__}")

@cli.command()
@click.pass_context
def config(ctx):
    import json
    click.echo("Current Configuration:")
    click.echo(json.dumps(ctx.obj['config'], indent=2))

if __name__ == '__main__':
    cli()