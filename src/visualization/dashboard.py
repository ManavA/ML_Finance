# src/monitoring/dashboard.py
"""
Real-time monitoring dashboard using Dash/Plotly.
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import redis
import json
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class TradingDashboard:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = dash.Dash(__name__)
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Data storage
        self.portfolio_history = []
        self.trade_history = []
        self.signal_history = []
        self.metrics_history = []
        
        # Setup layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
    def _setup_layout(self):
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1('Crypto ML Trading Dashboard', style={'text-align': 'center'}),
                html.Div(id='last-update', style={'text-align': 'center'}),
            ], style={'backgroundColor': '#1e1e1e', 'color': 'white', 'padding': '20px'}),
            
            # Main metrics row
            html.Div([
                html.Div([
                    html.H3('Portfolio Value'),
                    html.H2(id='portfolio-value', children='$0.00'),
                    html.P(id='portfolio-change', children='0.00%'),
                ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3('Daily P&L'),
                    html.H2(id='daily-pnl', children='$0.00'),
                    html.P(id='daily-pnl-pct', children='0.00%'),
                ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3('Open Positions'),
                    html.H2(id='open-positions', children='0'),
                    html.P(id='position-exposure', children='$0.00'),
                ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3('Win Rate'),
                    html.H2(id='win-rate', children='0.0%'),
                    html.P(id='trades-today', children='0 trades'),
                ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
            ], style={'padding': '20px', 'backgroundColor': '#2e2e2e'}),
            
            # Charts row
            html.Div([
                # Portfolio value chart
                dcc.Graph(id='portfolio-chart', style={'width': '50%', 'display': 'inline-block'}),
                
                # P&L distribution
                dcc.Graph(id='pnl-distribution', style={'width': '50%', 'display': 'inline-block'}),
            ]),
            
            # Positions table
            html.Div([
                html.H3('Open Positions', style={'color': 'white'}),
                dash_table.DataTable(
                    id='positions-table',
                    columns=[
                        {'name': 'Symbol', 'id': 'symbol'},
                        {'name': 'Side', 'id': 'side'},
                        {'name': 'Entry Price', 'id': 'entry_price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Current Price', 'id': 'current_price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Quantity', 'id': 'quantity', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                        {'name': 'Unrealized P&L', 'id': 'unrealized_pnl', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'P&L %', 'id': 'pnl_pct', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                    ],
                    style_cell={'textAlign': 'center', 'backgroundColor': '#1e1e1e', 'color': 'white'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'unrealized_pnl', 'filter_query': '{unrealized_pnl} > 0'},
                            'color': 'green'
                        },
                        {
                            'if': {'column_id': 'unrealized_pnl', 'filter_query': '{unrealized_pnl} < 0'},
                            'color': 'red'
                        },
                    ],
                )
            ], style={'padding': '20px', 'backgroundColor': '#2e2e2e'}),
            
            # Recent trades table
            html.Div([
                html.H3('Recent Trades', style={'color': 'white'}),
                dash_table.DataTable(
                    id='trades-table',
                    columns=[
                        {'name': 'Time', 'id': 'timestamp'},
                        {'name': 'Symbol', 'id': 'symbol'},
                        {'name': 'Action', 'id': 'action'},
                        {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Quantity', 'id': 'quantity', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                        {'name': 'P&L', 'id': 'pnl', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    ],
                    style_cell={'textAlign': 'center', 'backgroundColor': '#1e1e1e', 'color': 'white'},
                    page_size=10,
                )
            ], style={'padding': '20px', 'backgroundColor': '#2e2e2e'}),
            
            # Risk metrics
            html.Div([
                html.H3('Risk Metrics', style={'color': 'white'}),
                html.Div([
                    html.Div([
                        html.P('VaR (95%)', style={'color': 'gray'}),
                        html.H4(id='var-95', children='$0.00'),
                    ], style={'width': '20%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.P('Sharpe Ratio', style={'color': 'gray'}),
                        html.H4(id='sharpe-ratio', children='0.00'),
                    ], style={'width': '20%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.P('Max Drawdown', style={'color': 'gray'}),
                        html.H4(id='max-drawdown', children='0.00%'),
                    ], style={'width': '20%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.P('Beta', style={'color': 'gray'}),
                        html.H4(id='beta', children='0.00'),
                    ], style={'width': '20%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.P('Kelly Fraction', style={'color': 'gray'}),
                        html.H4(id='kelly-fraction', children='0.00%'),
                    ], style={'width': '20%', 'display': 'inline-block'}),
                ])
            ], style={'padding': '20px', 'backgroundColor': '#2e2e2e', 'color': 'white'}),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=5000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):        
        @self.app.callback(
            [Output('portfolio-value', 'children'),
             Output('portfolio-change', 'children'),
             Output('daily-pnl', 'children'),
             Output('daily-pnl-pct', 'children'),
             Output('open-positions', 'children'),
             Output('position-exposure', 'children'),
             Output('win-rate', 'children'),
             Output('trades-today', 'children'),
             Output('last-update', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            try:
                # Fetch data from Redis
                portfolio_value = float(self.redis_client.get('portfolio_value') or 10000)
                daily_pnl = float(self.redis_client.get('daily_pnl') or 0)
                positions = json.loads(self.redis_client.get('positions') or '[]')
                trades_today = int(self.redis_client.get('trades_today') or 0)
                wins_today = int(self.redis_client.get('wins_today') or 0)
                
                # Calculate metrics
                portfolio_change = (daily_pnl / portfolio_value) * 100 if portfolio_value > 0 else 0
                position_exposure = sum(p.get('value', 0) for p in positions)
                win_rate = (wins_today / trades_today * 100) if trades_today > 0 else 0
                
                # Format outputs
                portfolio_str = f'${portfolio_value:,.2f}'
                change_str = f'{portfolio_change:+.2f}%'
                change_color = 'green' if portfolio_change >= 0 else 'red'
                
                daily_pnl_str = f'${daily_pnl:+,.2f}'
                daily_pnl_pct_str = f'{portfolio_change:+.2f}%'
                pnl_color = 'green' if daily_pnl >= 0 else 'red'
                
                positions_str = str(len(positions))
                exposure_str = f'${position_exposure:,.2f}'
                
                win_rate_str = f'{win_rate:.1f}%'
                trades_str = f'{trades_today} trades'
                
                last_update = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                return (
                    portfolio_str,
                    html.Span(change_str, style={'color': change_color}),
                    html.Span(daily_pnl_str, style={'color': pnl_color}),
                    html.Span(daily_pnl_pct_str, style={'color': pnl_color}),
                    positions_str,
                    exposure_str,
                    win_rate_str,
                    trades_str,
                    last_update
                )
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                return ('$0.00', '0.00%', '$0.00', '0.00%', '0', '$0.00', '0.0%', '0 trades', 'Error')
        
        @self.app.callback(
            Output('portfolio-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_portfolio_chart(n):
            try:
                # Fetch historical data
                history = json.loads(self.redis_client.get('portfolio_history') or '[]')
                
                if not history:
                    return {'data': [], 'layout': {}}
                
                df = pd.DataFrame(history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#00d4ff', width=2)
                ))
                
                # Add benchmark if available
                if 'benchmark' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['benchmark'],
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
                
                fig.update_layout(
                    title='Portfolio Value Over Time',
                    xaxis_title='Date',
                    yaxis_title='Value ($)',
                    template='plotly_dark',
                    height=400
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating portfolio chart: {e}")
                return {'data': [], 'layout': {}}
        
        @self.app.callback(
            Output('positions-table', 'data'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_positions_table(n):
            try:
                positions = json.loads(self.redis_client.get('positions') or '[]')
                
                # Format for display
                formatted_positions = []
                for pos in positions:
                    current_price = float(self.redis_client.get(f'price_{pos["symbol"]}') or pos['entry_price'])
                    unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
                    pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                    
                    formatted_positions.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'current_price': current_price,
                        'quantity': pos['quantity'],
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_pct': pnl_pct
                    })
                
                return formatted_positions
                
            except Exception as e:
                logger.error(f"Error updating positions table: {e}")
                return []
    
    def run(self, port: int = 8050, debug: bool = False):
        self.app.run_server(host='0.0.0.0', port=port, debug=debug)
