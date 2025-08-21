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
    """Real-time trading monitoring dashboard."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dashboard.
        
        Args:
            config: Dashboard configuration
        """
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
        """Setup dashboard layout."""
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
        """Setup dashboard callbacks for real-time updates."""
        
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
            """Update main metrics."""
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
            """Update portfolio value chart."""
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
            """Update positions table."""
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
        """Run the dashboard."""
        self.app.run_server(host='0.0.0.0', port=port, debug=debug)


# src/monitoring/alerts.py
"""
Alert and notification system.
"""

import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert message."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


class AlertSystem:
    """Multi-channel alert and notification system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert system.
        
        Args:
            config: Alert configuration
        """
        self.config = config
        
        # Email settings
        self.email_enabled = config.get('email', {}).get('enabled', False)
        self.smtp_server = config.get('email', {}).get('smtp_server')
        self.smtp_port = config.get('email', {}).get('smtp_port', 587)
        self.email_from = config.get('email', {}).get('from')
        self.email_to = config.get('email', {}).get('to', [])
        self.email_password = config.get('email', {}).get('password')
        
        # Telegram settings
        self.telegram_enabled = config.get('telegram', {}).get('enabled', False)
        self.telegram_token = config.get('telegram', {}).get('bot_token')
        self.telegram_chat_id = config.get('telegram', {}).get('chat_id')
        
        # Discord settings
        self.discord_enabled = config.get('discord', {}).get('enabled', False)
        self.discord_webhook = config.get('discord', {}).get('webhook_url')
        
        # Alert rules
        self.alert_rules = self._setup_alert_rules()
        
    def _setup_alert_rules(self) -> Dict[str, Any]:
        """Setup alert rules and thresholds."""
        return {
            'portfolio_drawdown': {
                'threshold': 0.1,  # 10% drawdown
                'level': AlertLevel.WARNING
            },
            'daily_loss': {
                'threshold': 0.05,  # 5% daily loss
                'level': AlertLevel.ERROR
            },
            'position_loss': {
                'threshold': 0.1,  # 10% position loss
                'level': AlertLevel.WARNING
            },
            'error_rate': {
                'threshold': 5,  # 5 errors
                'level': AlertLevel.ERROR
            },
            'low_balance': {
                'threshold': 1000,  # $1000
                'level': AlertLevel.WARNING
            },
            'trade_executed': {
                'level': AlertLevel.INFO
            },
            'circuit_breaker': {
                'level': AlertLevel.CRITICAL
            }
        }
    
    async def send_alert(self, alert: Alert):
        """Send alert through all configured channels."""
        tasks = []
        
        if self.email_enabled:
            tasks.append(self._send_email(alert))
        
        if self.telegram_enabled:
            tasks.append(self._send_telegram(alert))
        
        if self.discord_enabled:
            tasks.append(self._send_discord(alert))
        
        # Send all alerts concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log the alert
        logger.log(
            logging.ERROR if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else logging.INFO,
            f"Alert [{alert.level.value}]: {alert.title} - {alert.message}"
        )
    
    async def _send_email(self, alert: Alert):
        """Send email alert."""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            msg['From'] = self.email_from
            msg['To'] = ', '.join(self.email_to)
            
            # Create HTML content
            html = f"""
            <html>
            <body>
                <h2 style="color: {self._get_color(alert.level)}">
                    {alert.title}
                </h2>
                <p><strong>Level:</strong> {alert.level.value.upper()}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong></p>
                <p>{alert.message}</p>
                {self._format_metadata(alert.metadata)}
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                server.send_message(msg)
            
            logger.debug(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_telegram(self, alert: Alert):
        """Send Telegram alert."""
        try:
            # Format message
            icon = self._get_icon(alert.level)
            message = f"{icon} *{alert.title}*\n\n"
            message += f"Level: {alert.level.value.upper()}\n"
            message += f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            message += alert.message
            
            if alert.metadata:
                message += "\n\n*Details:*\n"
                for key, value in alert.metadata.items():
                    message += f"• {key}: {value}\n"
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.debug(f"Telegram alert sent: {alert.title}")
                    else:
                        logger.error(f"Failed to send Telegram alert: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    async def _send_discord(self, alert: Alert):
        """Send Discord alert."""
        try:
            # Format embed
            color = self._get_discord_color(alert.level)
            
            embed = {
                'title': alert.title,
                'description': alert.message,
                'color': color,
                'timestamp': alert.timestamp.isoformat(),
                'fields': []
            }
            
            if alert.metadata:
                for key, value in alert.metadata.items():
                    embed['fields'].append({
                        'name': key,
                        'value': str(value),
                        'inline': True
                    })
            
            payload = {
                'embeds': [embed]
            }
            
            # Send to Discord
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json=payload) as response:
                    if response.status == 204:
                        logger.debug(f"Discord alert sent: {alert.title}")
                    else:
                        logger.error(f"Failed to send Discord alert: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
    
    def _get_color(self, level: AlertLevel) -> str:
        """Get color for alert level."""
        colors = {
            AlertLevel.INFO: '#00d4ff',
            AlertLevel.WARNING: '#ffa500',
            AlertLevel.ERROR: '#ff0000',
            AlertLevel.CRITICAL: '#8b0000'
        }
        return colors.get(level, '#ffffff')
    
    def _get_discord_color(self, level: AlertLevel) -> int:
        """Get Discord color code for alert level."""
        colors = {
            AlertLevel.INFO: 0x00d4ff,
            AlertLevel.WARNING: 0xffa500,
            AlertLevel.ERROR: 0xff0000,
            AlertLevel.CRITICAL: 0x8b0000
        }
        return colors.get(level, 0xffffff)
    
    def _get_icon(self, level: AlertLevel) -> str:
        """Get icon for alert level."""
        icons = {
            AlertLevel.INFO: 'ℹ️',
            AlertLevel.WARNING: '⚠️',
            AlertLevel.ERROR: '❌',
            AlertLevel.CRITICAL: '🚨'
        }
        return icons.get(level, '📢')
    
    def _format_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Format metadata for HTML."""
        if not metadata:
            return ""
        
        html = "<h3>Details:</h3><ul>"
        for key, value in metadata.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        
        return html
    
    async def check_conditions(self, metrics: Dict[str, Any]):
        """Check alert conditions and send alerts if triggered."""
        # Check portfolio drawdown
        if 'max_drawdown' in metrics:
            if abs(metrics['max_drawdown']) > self.alert_rules['portfolio_drawdown']['threshold']:
                await self.send_alert(Alert(
                    level=AlertLevel.WARNING,
                    title='High Portfolio Drawdown',
                    message=f"Portfolio drawdown has reached {metrics['max_drawdown']:.2%}",
                    timestamp=datetime.now(),
                    metadata=metrics
                ))
        
        # Check daily loss
        if 'daily_pnl_pct' in metrics:
            if metrics['daily_pnl_pct'] < -self.alert_rules['daily_loss']['threshold']:
                await self.send_alert(Alert(
                    level=AlertLevel.ERROR,
                    title='Significant Daily Loss',
                    message=f"Daily loss of {metrics['daily_pnl_pct']:.2%} exceeds threshold",
                    timestamp=datetime.now(),
                    metadata=metrics
                ))
        
        # Check low balance
        if 'balance' in metrics:
            if metrics['balance'] < self.alert_rules['low_balance']['threshold']:
                await self.send_alert(Alert(
                    level=AlertLevel.WARNING,
                    title='Low Account Balance',
                    message=f"Account balance (${metrics['balance']:.2f}) is below threshold",
                    timestamp=datetime.now(),
                    metadata=metrics
                ))
    
    async def send_trade_alert(self, trade_info: Dict[str, Any]):
        """Send alert for executed trade."""
        await self.send_alert(Alert(
            level=AlertLevel.INFO,
            title='Trade Executed',
            message=f"{trade_info['action']} {trade_info['quantity']} {trade_info['symbol']} @ ${trade_info['price']:.2f}",
            timestamp=datetime.now(),
            metadata=trade_info
        ))
    
    async def send_error_alert(self, error: str, details: Optional[Dict[str, Any]] = None):
        """Send error alert."""
        await self.send_alert(Alert(
            level=AlertLevel.ERROR,
            title='Trading System Error',
            message=error,
            timestamp=datetime.now(),
            metadata=details
        ))
    
    async def send_circuit_breaker_alert(self, reason: str):
        """Send circuit breaker alert."""
        await self.send_alert(Alert(
            level=AlertLevel.CRITICAL,
            title='Circuit Breaker Activated',
            message=f"Trading halted: {reason}",
            timestamp=datetime.now()
        ))