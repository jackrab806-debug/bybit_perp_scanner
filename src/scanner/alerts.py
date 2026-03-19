"""Alert system for console, webhook, and Telegram notifications."""

import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional


def format_alert_message(symbol: str, vol_ratio: float, oi_delta: float, 
                        funding: Optional[float], basis: Optional[float]) -> str:
    """
    Format alert message according to specification.
    
    Args:
        symbol: Trading pair symbol
        vol_ratio: Volume spike ratio
        oi_delta: Open interest delta percentage
        funding: Funding rate
        basis: Basis percentage
        
    Returns:
        Formatted alert message
    """
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Format funding rate
    funding_str = f"{funding:.6f}" if funding is not None else "n/a"
    
    # Format basis percentage
    if basis is not None:
        basis_str = f"{basis:.4f}%"
    else:
        basis_str = "n/a"
    
    # Format OI delta as percentage
    oi_delta_str = f"{oi_delta:.2f}%"
    
    return f"[{timestamp}] {symbol}  vol×={vol_ratio:.2f}  OIΔ1h={oi_delta_str}  funding={funding_str}  basis={basis_str}"


def print_console_alert(symbol: str, vol_ratio: float, oi_delta: float, 
                       funding: Optional[float], basis: Optional[float]) -> None:
    """
    Print alert to console.
    
    Args:
        symbol: Trading pair symbol
        vol_ratio: Volume spike ratio
        oi_delta: Open interest delta percentage
        funding: Funding rate
        basis: Basis percentage
    """
    message = format_alert_message(symbol, vol_ratio, oi_delta, funding, basis)
    print(message)


def print_cycle_summary(alert_count: int) -> None:
    """
    Print cycle summary message.
    
    Args:
        alert_count: Number of alerts in this cycle
    """
    if alert_count > 0:
        print(f"— {alert_count} alert(s) this cycle —")
        print()  # Empty line for readability


def send_webhook(url: str, payload: Dict[str, Any]) -> bool:
    """
    Send webhook notification.
    
    Args:
        url: Webhook URL
        payload: Data to send
        
    Returns:
        True if successful, False otherwise
    """
    if not url:
        return False
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        response.raise_for_status()
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Webhook error: {e}")
        return False


def telegram_send(token: str, chat_id: str, text: str) -> bool:
    """
    Send Telegram message.
    
    Args:
        token: Telegram bot token
        chat_id: Telegram chat ID
        text: Message text
        
    Returns:
        True if successful, False otherwise
    """
    if not token or not chat_id:
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if not result.get('ok'):
            print(f"Telegram API error: {result.get('description', 'Unknown error')}")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Telegram error: {e}")
        return False


class AlertManager:
    """Manages all alert channels (console, webhook, Telegram)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_count = 0
        self.first_telegram_sent = False
    
    def send_alert(self, symbol: str, vol_ratio: float, oi_delta: float, 
                   funding: Optional[float], basis: Optional[float]) -> None:
        """
        Send alert through all configured channels.
        
        Args:
            symbol: Trading pair symbol
            vol_ratio: Volume spike ratio
            oi_delta: Open interest delta percentage
            funding: Funding rate
            basis: Basis percentage
        """
        # Always print to console
        print_console_alert(symbol, vol_ratio, oi_delta, funding, basis)
        self.alert_count += 1
        
        # Prepare message for external notifications
        message = format_alert_message(symbol, vol_ratio, oi_delta, funding, basis)
        
        # Send webhook if configured
        if self.config.get("USE_WEBHOOK") and self.config.get("WEBHOOK_URL"):
            webhook_payload = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'symbol': symbol,
                'vol_ratio': vol_ratio,
                'oi_delta_pct': oi_delta,
                'funding_rate': funding,
                'basis_pct': basis,
                'message': message
            }
            send_webhook(self.config["WEBHOOK_URL"], webhook_payload)
        
        # Send Telegram if configured
        if self.config.get("USE_TELEGRAM"):
            token = self.config.get("TELEGRAM_BOT_TOKEN")
            chat_id = self.config.get("TELEGRAM_CHAT_ID")
            
            if token and chat_id:
                # Send test message on first alert
                if not self.first_telegram_sent:
                    test_message = "🚀 <b>Bybit Perp Scanner Started</b>\n\nFirst alert triggered. Scanner is working correctly!"
                    if telegram_send(token, chat_id, test_message):
                        print("✓ Telegram test message sent successfully")
                    self.first_telegram_sent = True
                
                # Format message for Telegram with HTML
                telegram_message = f"🔔 <b>Perp Alert</b>\n\n<code>{message}</code>"
                telegram_send(token, chat_id, telegram_message)
    
    def end_cycle(self) -> None:
        """End the current scanning cycle and print summary."""
        if self.alert_count > 0:
            print_cycle_summary(self.alert_count)
        self.alert_count = 0
    
    def send_startup_message(self, symbol_count: int) -> None:
        """
        Send startup notification.
        
        Args:
            symbol_count: Number of symbols being tracked
        """
        message = f"Tracking {symbol_count} symbols..."
        print(message)
        
        # Send Telegram startup message if configured
        if self.config.get("USE_TELEGRAM"):
            token = self.config.get("TELEGRAM_BOT_TOKEN")
            chat_id = self.config.get("TELEGRAM_CHAT_ID")
            
            if token and chat_id:
                telegram_message = f"🚀 <b>Bybit Perp Scanner Started</b>\n\n{message}"
                if telegram_send(token, chat_id, telegram_message):
                    print("✓ Telegram startup message sent")
    
    def send_error_notification(self, error_message: str) -> None:
        """
        Send error notification through configured channels.
        
        Args:
            error_message: Error description
        """
        print(f"ERROR: {error_message}")
        
        # Send Telegram error notification if configured
        if self.config.get("USE_TELEGRAM"):
            token = self.config.get("TELEGRAM_BOT_TOKEN")
            chat_id = self.config.get("TELEGRAM_CHAT_ID")
            
            if token and chat_id:
                telegram_message = f"⚠️ <b>Scanner Error</b>\n\n<code>{error_message}</code>"
                telegram_send(token, chat_id, telegram_message)
