# src/monitoring/alerts.py

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
    
    def __init__(self, config: Dict[str, Any]):
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
        colors = {
            AlertLevel.INFO: '#00d4ff',
            AlertLevel.WARNING: '#ffa500',
            AlertLevel.ERROR: '#ff0000',
            AlertLevel.CRITICAL: '#8b0000'
        }
        return colors.get(level, '#ffffff')
    
    def _get_discord_color(self, level: AlertLevel) -> int:
        colors = {
            AlertLevel.INFO: 0x00d4ff,
            AlertLevel.WARNING: 0xffa500,
            AlertLevel.ERROR: 0xff0000,
            AlertLevel.CRITICAL: 0x8b0000
        }
        return colors.get(level, 0xffffff)
    
    def _get_icon(self, level: AlertLevel) -> str:
        icons = {
            AlertLevel.INFO: 'ℹ️',
            AlertLevel.WARNING: '⚠️',
            AlertLevel.ERROR: '❌',
            AlertLevel.CRITICAL: '🚨'
        }
        return icons.get(level, '📢')
    
    def _format_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        if not metadata:
            return ""
        
        html = "<h3>Details:</h3><ul>"
        for key, value in metadata.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        
        return html
    
    async def check_conditions(self, metrics: Dict[str, Any]):
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
        await self.send_alert(Alert(
            level=AlertLevel.INFO,
            title='Trade Executed',
            message=f"{trade_info['action']} {trade_info['quantity']} {trade_info['symbol']} @ ${trade_info['price']:.2f}",
            timestamp=datetime.now(),
            metadata=trade_info
        ))
    
    async def send_error_alert(self, error: str, details: Optional[Dict[str, Any]] = None):
        await self.send_alert(Alert(
            level=AlertLevel.ERROR,
            title='Trading System Error',
            message=error,
            timestamp=datetime.now(),
            metadata=details
        ))
    
    async def send_circuit_breaker_alert(self, reason: str):
        await self.send_alert(Alert(
            level=AlertLevel.CRITICAL,
            title='Circuit Breaker Activated',
            message=f"Trading halted: {reason}",
            timestamp=datetime.now()
        ))