"""
Live/paper trading commands for the CLI
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import pickle
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.baseline_strategies import BaselineStrategies

@click.group(name='trade')
@click.pass_context
def trade_group(ctx):
    """Live and paper trading commands"""
    pass

@trade_group.command(name='start')
@click.option('--symbol', '-s', required=True, help='Symbol to trade')
@click.option('--strategy', help='Strategy or model to use')
@click.option('--model-path', help='Path to trained model')
@click.option('--capital', default=10000, type=float, help='Trading capital')
@click.option('--paper/--live', default=True, help='Paper trading (default) or live')
@click.option('--interval', default=60, type=int, help='Update interval in seconds')
@click.pass_context
def start_trading(ctx, symbol, strategy, model_path, capital, paper, interval):
    """Start live or paper trading"""
    logger = ctx.obj['logger']
    
    mode = "Paper" if paper else "Live"
    click.echo(f"Starting {mode} Trading")
    click.echo(f"Symbol: {symbol}")
    click.echo(f"Capital: ${capital:,.2f}")
    
    if not paper:
        click.echo("\n⚠️  WARNING: Live trading is not yet implemented for safety.")
        click.echo("Using paper trading mode instead.")
        paper = True
    
    # Initialize trading state
    state = {
        'symbol': symbol,
        'capital': capital,
        'position': 0,
        'cash': capital,
        'trades': [],
        'start_time': datetime.now(),
        'last_price': None
    }
    
    # Load model if specified
    model = None
    if model_path:
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                feature_cols = model_data['feature_cols']
            click.echo(f"Loaded model from {model_path}")
        except Exception as e:
            click.echo(f"Error loading model: {e}")
            return
    
    click.echo(f"\nStarting trading loop (Ctrl+C to stop)...")
    click.echo("-" * 60)
    
    try:
        while True:
            # Fetch latest data
            from src.data.unified_collector import UnifiedDataCollector
            collector = UnifiedDataCollector()
            
            try:
                # Get latest quote
                latest = collector.fetch_latest_quote(symbol)
                
                if latest is not None:
                    current_price = latest['price']
                    state['last_price'] = current_price
                    
                    # Display current status
                    portfolio_value = state['cash'] + state['position'] * current_price
                    pnl = portfolio_value - capital
                    pnl_pct = (pnl / capital) * 100
                    
                    click.echo(f"\n[{datetime.now().strftime('%H:%M:%S')}] {symbol}: ${current_price:.2f}")
                    click.echo(f"Portfolio: ${portfolio_value:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                    
                    # Generate signal
                    signal = generate_signal(symbol, strategy, model, feature_cols if model else None)
                    
                    # Execute trades based on signal
                    if signal == 1 and state['position'] == 0:
                        # Buy signal
                        shares = int(state['cash'] * 0.95 / current_price)  # Use 95% of cash
                        if shares > 0:
                            cost = shares * current_price
                            state['cash'] -= cost
                            state['position'] = shares
                            state['trades'].append({
                                'time': datetime.now(),
                                'action': 'BUY',
                                'shares': shares,
                                'price': current_price,
                                'value': cost
                            })
                            click.echo(f"BUY {shares} shares at ${current_price:.2f}")
                    
                    elif signal == -1 and state['position'] > 0:
                        # Sell signal
                        proceeds = state['position'] * current_price
                        state['cash'] += proceeds
                        state['trades'].append({
                            'time': datetime.now(),
                            'action': 'SELL',
                            'shares': state['position'],
                            'price': current_price,
                            'value': proceeds
                        })
                        click.echo(f"SELL {state['position']} shares at ${current_price:.2f}")
                        state['position'] = 0
                    
                    # Save state
                    save_trading_state(state)
                    
                else:
                    click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] Unable to fetch price")
                    
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                click.echo(f"Error: {e}")
            
            # Wait for next update
            time.sleep(interval)
            
    except KeyboardInterrupt:
        click.echo("\n\nStopping trading...")
        
        # Final summary
        if state['last_price']:
            final_value = state['cash'] + state['position'] * state['last_price']
            total_return = (final_value / capital - 1) * 100
            
            click.echo("\n" + "="*60)
            click.echo("TRADING SESSION SUMMARY")
            click.echo("="*60)
            click.echo(f"Duration: {datetime.now() - state['start_time']}")
            click.echo(f"Total Trades: {len(state['trades'])}")
            click.echo(f"Final Value: ${final_value:.2f}")
            click.echo(f"Total Return: {total_return:+.2f}%")
            
            if state['trades']:
                click.echo("\nRecent Trades:")
                for trade in state['trades'][-5:]:
                    click.echo(f"  {trade['time'].strftime('%H:%M')} "
                              f"{trade['action']} {trade['shares']} @ ${trade['price']:.2f}")

@trade_group.command(name='status')
@click.pass_context
def trading_status(ctx):
    """Show current trading status"""
    logger = ctx.obj['logger']
    
    state_file = Path('.trading_state.json')
    
    if not state_file.exists():
        click.echo("No active trading session found")
        return
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        click.echo("\n" + "="*60)
        click.echo("TRADING STATUS")
        click.echo("="*60)
        click.echo(f"Symbol: {state['symbol']}")
        click.echo(f"Started: {state['start_time']}")
        click.echo(f"Position: {state['position']} shares")
        click.echo(f"Cash: ${state['cash']:.2f}")
        
        if state['last_price']:
            portfolio_value = state['cash'] + state['position'] * state['last_price']
            click.echo(f"Portfolio Value: ${portfolio_value:.2f}")
            click.echo(f"Return: {(portfolio_value/state['capital']-1)*100:+.2f}%")
        
        click.echo(f"\nTotal Trades: {len(state.get('trades', []))}")
        
    except Exception as e:
        click.echo(f"Error loading trading state: {e}")

@trade_group.command(name='stop')
@click.confirmation_option(prompt='Are you sure you want to stop trading?')
@click.pass_context
def stop_trading(ctx):
    """Stop current trading session"""
    logger = ctx.obj['logger']
    
    state_file = Path('.trading_state.json')
    
    if not state_file.exists():
        click.echo("No active trading session found")
        return
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        click.echo("Trading session stopped")
        click.echo(f"Symbol: {state['symbol']}")
        click.echo(f"Final position: {state['position']} shares")
        
        # Archive the state
        archive_name = f"trading_session_{state['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        state_file.rename(archive_name)
        click.echo(f"Session archived to {archive_name}")
        
    except Exception as e:
        click.echo(f"Error stopping trading: {e}")

@trade_group.command(name='history')
@click.option('--limit', default=10, type=int, help='Number of sessions to show')
@click.pass_context
def trading_history(ctx, limit):
    """Show trading history"""
    logger = ctx.obj['logger']
    
    # Find archived trading sessions
    sessions = list(Path('.').glob('trading_session_*.json'))
    
    if not sessions:
        click.echo("No trading history found")
        return
    
    sessions = sorted(sessions, key=lambda x: x.stat().st_mtime, reverse=True)[:limit]
    
    click.echo(f"Found {len(sessions)} trading sessions:\n")
    
    for session_file in sessions:
        try:
            with open(session_file, 'r') as f:
                state = json.load(f)
            
            # Calculate summary
            if state.get('last_price'):
                final_value = state['cash'] + state['position'] * state['last_price']
                total_return = (final_value / state['capital'] - 1) * 100
            else:
                total_return = 0
            
            click.echo(f"{session_file.name}:")
            click.echo(f"  Symbol: {state['symbol']}")
            click.echo(f"  Trades: {len(state.get('trades', []))}")
            click.echo(f"  Return: {total_return:+.2f}%")
            click.echo("")
            
        except Exception as e:
            logger.error(f"Error loading {session_file}: {e}")

def generate_signal(symbol, strategy, model, feature_cols):
    """Generate trading signal"""
    # This is a simplified signal generator
    # In production, this would use real-time data and sophisticated logic
    
    if model:
        # Use ML model for signal
        # Would need to create features from latest data
        # For now, return random signal for demonstration
        return np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
    
    elif strategy:
        # Use rule-based strategy
        # Get the baseline strategy implementation
        try:
            strategies = BaselineStrategies()
            
            # Load recent data for signal generation
            from src.data.unified_collector import UnifiedDataCollector
            collector = UnifiedDataCollector()
            
            # Get recent data (last 50 periods for technical indicators)
            recent_data = collector.fetch_data(
                symbol=symbol,
                start_date=(datetime.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                source='polygon'
            )
            
            if recent_data is not None and len(recent_data) > 20:
                # Generate signals based on strategy
                if strategy == 'ma_crossover':
                    signals = strategies.ma_crossover_strategy(recent_data)
                elif strategy == 'rsi':
                    signals = strategies.rsi_strategy(recent_data)
                elif strategy == 'bollinger':
                    signals = strategies.bollinger_bands_strategy(recent_data)
                elif strategy == 'macd':
                    signals = strategies.macd_strategy(recent_data)
                else:
                    return 0
                
                # Return the last signal
                return int(signals.iloc[-1]) if not pd.isna(signals.iloc[-1]) else 0
            
        except Exception as e:
            print(f"Error generating strategy signal: {e}")
        
        return 0
    
    else:
        # No signal
        return 0

def save_trading_state(state):
    """Save trading state to file"""
    state_file = Path('.trading_state.json')
    
    # Convert datetime objects to strings
    state_copy = state.copy()
    state_copy['start_time'] = state_copy['start_time'].isoformat()
    
    if 'trades' in state_copy:
        for trade in state_copy['trades']:
            if isinstance(trade['time'], datetime):
                trade['time'] = trade['time'].isoformat()
    
    with open(state_file, 'w') as f:
        json.dump(state_copy, f, indent=2)