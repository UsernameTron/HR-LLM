"""Alert management for the sentiment analysis pipeline."""
import logging
from typing import Dict, List, Optional
import smtplib
from email.message import EmailMessage
import json
import requests

from .config import alert_thresholds

logger = logging.getLogger(__name__)

class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.active_alerts: Dict[str, Dict] = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load alert configuration."""
        if not config_path:
            return {
                "notification_channels": {
                    "email": {
                        "enabled": True,
                        "recipients": ["alerts@yourcompany.com"]
                    },
                    "slack": {
                        "enabled": True,
                        "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
                    }
                },
                "alert_severity": {
                    "warning": ["email"],
                    "critical": ["email", "slack"],
                    "rollback": ["email", "slack"]
                }
            }
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    async def trigger_alert(self, alert_type: str, severity: str, message: str, metrics: Optional[Dict] = None):
        """Trigger a new alert."""
        alert_id = f"{alert_type}_{severity}_{int(time.time())}"
        
        if alert_type in self.active_alerts:
            logger.info(f"Alert already active for {alert_type}")
            return
        
        self.active_alerts[alert_type] = {
            "id": alert_id,
            "severity": severity,
            "message": message,
            "metrics": metrics or {},
            "timestamp": time.time()
        }
        
        await self._send_notifications(alert_id)
    
    async def resolve_alert(self, alert_type: str):
        """Resolve an active alert."""
        if alert_type not in self.active_alerts:
            return
        
        alert = self.active_alerts.pop(alert_type)
        logger.info(f"Resolved alert: {alert['id']}")
        
        resolution_message = f"RESOLVED: {alert['message']}"
        await self._send_notifications(alert['id'], resolution_message)
    
    async def _send_notifications(self, alert_id: str, message: Optional[str] = None):
        """Send notifications through configured channels."""
        alert = self.active_alerts.get(alert_id.split('_')[0])
        if not alert:
            return
            
        message = message or alert['message']
        channels = self.config['alert_severity'].get(alert['severity'], [])
        
        for channel in channels:
            if channel == "email" and self.config['notification_channels']['email']['enabled']:
                await self._send_email(alert, message)
            elif channel == "slack" and self.config['notification_channels']['slack']['enabled']:
                await self._send_slack(alert, message)
    
    async def _send_email(self, alert: Dict, message: str):
        """Send email notification."""
        try:
            msg = EmailMessage()
            msg.set_content(self._format_alert_message(alert, message))
            
            msg['Subject'] = f"[{alert['severity'].upper()}] Sentiment Analysis Alert"
            msg['From'] = "monitoring@yourcompany.com"
            msg['To'] = self.config['notification_channels']['email']['recipients']
            
            # Note: Configure your SMTP settings in production
            logger.info(f"Would send email: {msg['Subject']}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
    
    async def _send_slack(self, alert: Dict, message: str):
        """Send Slack notification."""
        try:
            webhook_url = self.config['notification_channels']['slack']['webhook_url']
            
            payload = {
                "text": self._format_alert_message(alert, message),
                "username": "Sentiment Analysis Monitor",
                "icon_emoji": ":warning:" if alert['severity'] == "warning" else ":rotating_light:"
            }
            
            # Note: Configure your Slack webhook in production
            logger.info(f"Would send Slack message: {payload['text']}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
    
    def _format_alert_message(self, alert: Dict, message: str) -> str:
        """Format alert message with metrics."""
        formatted_message = f"{message}\n\nMetrics:"
        
        for metric, value in alert['metrics'].items():
            formatted_message += f"\n- {metric}: {value}"
            
        return formatted_message
