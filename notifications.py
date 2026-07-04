"""
Email Notification System for Financial Alerts
Uses SendGrid (free tier) or SMTP (Gmail)
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
import asyncio
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

# Email configuration from environment variables
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "your_email@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "your_app_password")
DEFAULT_RECIPIENTS = os.getenv("DEFAULT_RECIPIENTS", "admin@example.com").split(",")

# SendGrid configuration (alternative)
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", "noreply@example.com")

# ==========================================
# EMAIL FUNCTIONS
# ==========================================

async def send_smtp_email(
    subject: str,
    body: str,
    recipients: List[str],
    html_body: Optional[str] = None
) -> bool:
    """
    Send email using SMTP (Gmail/Outlook/etc.)
    
    Args:
        subject: Email subject line
        body: Plain text body
        recipients: List of recipient emails
        html_body: Optional HTML version of the body
    
    Returns:
        bool: Success status
    """
    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = ", ".join(recipients)
        
        # Attach plain text version
        part1 = MIMEText(body, "plain")
        msg.attach(part1)
        
        # Attach HTML version if provided
        if html_body:
            part2 = MIMEText(html_body, "html")
            msg.attach(part2)
        
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email sent to {', '.join(recipients)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

async def send_sendgrid_email(
    subject: str,
    body: str,
    recipients: List[str],
    html_body: Optional[str] = None
) -> bool:
    """
    Send email using SendGrid API (Free tier: 100 emails/day)
    
    Args:
        subject: Email subject line
        body: Plain text body
        recipients: List of recipient emails
        html_body: Optional HTML version of the body
    
    Returns:
        bool: Success status
    """
    if not SENDGRID_API_KEY:
        logger.warning("SendGrid API key not configured")
        return False
    
    try:
        url = "https://api.sendgrid.com/v3/mail/send"
        
        payload = {
            "personalizations": [{"to": [{"email": r} for r in recipients]}],
            "from": {"email": SENDGRID_FROM_EMAIL},
            "subject": subject,
            "content": [
                {"type": "text/plain", "value": body}
            ]
        }
        
        if html_body:
            payload["content"].append({"type": "text/html", "value": html_body})
        
        headers = {
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code == 202:
                logger.info(f"SendGrid email sent to {', '.join(recipients)}")
                return True
            else:
                logger.error(f"SendGrid error: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to send SendGrid email: {e}")
        return False

async def send_notification(
    subject: str,
    body: str,
    recipients: Optional[List[str]] = None,
    html_body: Optional[str] = None
) -> bool:
    """
    Main notification function - tries SMTP then falls back to SendGrid
    
    Args:
        subject: Email subject
        body: Plain text body
        recipients: List of recipient emails (defaults to DEFAULT_RECIPIENTS)
        html_body: Optional HTML version
    
    Returns:
        bool: Success status
    """
    if recipients is None:
        recipients = DEFAULT_RECIPIENTS
    
    if not recipients or recipients == ["admin@example.com"]:
        logger.warning("No recipients configured. Please set DEFAULT_RECIPIENTS in .env")
        # Still log the alert
        logger.info(f"ALERT: {subject}\n{body}")
        return False
    
    # Try SMTP first
    success = await send_smtp_email(subject, body, recipients, html_body)
    
    # Fallback to SendGrid if SMTP fails
    if not success and SENDGRID_API_KEY:
        success = await send_sendgrid_email(subject, body, recipients, html_body)
    
    return success

async def send_bulk_notification(
    alerts: List[Dict[str, Any]],
    recipients: Optional[List[str]] = None
) -> bool:
    """
    Send a summary notification for multiple alerts
    
    Args:
        alerts: List of alert dictionaries
        recipients: List of recipient emails
    
    Returns:
        bool: Success status
    """
    if not alerts:
        return False
    
    subject = f"📊 SMT Alert: {len(alerts)} divergence(s) detected"
    
    # Build plain text body
    body = "SMT Divergence Detection Report\n"
    body += "=" * 50 + "\n\n"
    body += f"Total Detections: {len(alerts)}\n"
    body += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Group by timeframe
    by_timeframe = {}
    for alert in alerts:
        tf = alert.get("timeframe", "unknown")
        if tf not in by_timeframe:
            by_timeframe[tf] = []
        by_timeframe[tf].append(alert)
    
    for timeframe, items in by_timeframe.items():
        body += f"\n📊 {timeframe.upper()} TIMEFRAME\n"
        body += "-" * 30 + "\n"
        
        for item in items:
            group = item.get("group", "unknown")
            base = item.get("base", "unknown")
            corr = item.get("correlated", "unknown")
            div_type = item.get("type", "unknown")
            change1 = item.get("change1", 0)
            change2 = item.get("change2", 0)
            
            body += f"  • {group.upper()}: {base} vs {corr}\n"
            body += f"    Type: {div_type.upper()} divergence\n"
            body += f"    {base} change: {change1:+.2f}%\n"
            body += f"    {corr} change: {change2:+.2f}%\n\n"
    
    # Build HTML version
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #1a1a2e; }}
            .header {{ background: #1a1a2e; color: white; padding: 15px; border-radius: 5px; }}
            .timeframe {{ background: #16213e; color: white; padding: 10px; margin-top: 20px; border-radius: 5px; }}
            .alert {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 4px solid #e74c3c; border-radius: 3px; }}
            .bullish {{ border-left-color: #27ae60; }}
            .bearish {{ border-left-color: #e74c3c; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9em; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 SMT Divergence Detection Report</h1>
            <p>Total Detections: {len(alerts)}</p>
            <p>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """
    
    for timeframe, items in by_timeframe.items():
        html_body += f"""
        <div class="timeframe">
            <h2>📊 {timeframe.upper()} TIMEFRAME</h2>
        </div>
        """
        
        for item in items:
            group = item.get("group", "unknown")
            base = item.get("base", "unknown")
            corr = item.get("correlated", "unknown")
            div_type = item.get("type", "unknown")
            change1 = item.get("change1", 0)
            change2 = item.get("change2", 0)
            
            alert_class = "bullish" if div_type == "bullish" else "bearish" if div_type == "bearish" else ""
            
            html_body += f"""
            <div class="alert {alert_class}">
                <strong>{group.upper()}</strong>: {base} vs {corr}<br>
                Type: <strong>{div_type.upper()} divergence</strong><br>
                {base} change: <strong>{change1:+.2f}%</strong><br>
                {corr} change: <strong>{change2:+.2f}%</strong>
            </div>
            """
    
    html_body += f"""
        <div class="timestamp">
            <p>---<br>Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    return await send_notification(subject, body, recipients, html_body)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def format_alert_message(alert: Dict[str, Any]) -> str:
    """Format a single alert for display"""
    return f"""
🔔 SMT DETECTION
Group: {alert.get('group', 'unknown')}
Base: {alert.get('base', 'unknown')} vs Correlated: {alert.get('correlated', 'unknown')}
Type: {alert.get('type', 'unknown')} divergence
Timeframe: {alert.get('timeframe', 'unknown')}
Change: {alert.get('change1', 0):+.2f}% / {alert.get('change2', 0):+.2f}%
Time: {alert.get('timestamp', '')}
"""

# ==========================================
# TEST FUNCTION
# ==========================================

async def test_notification():
    """Test the notification system"""
    test_alerts = [
        {
            "group": "precious_metals",
            "base": "XAUUSD",
            "correlated": "XAGUSD",
            "timeframe": "1h",
            "type": "bullish",
            "change1": 2.5,
            "change2": -1.2,
            "price1": 2800.50,
            "price2": 32.40,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    success = await send_bulk_notification(test_alerts)
    if success:
        print("✅ Test notification sent successfully")
    else:
        print("❌ Test notification failed")

if __name__ == "__main__":
    asyncio.run(test_notification())