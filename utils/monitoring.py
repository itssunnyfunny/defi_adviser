import logging
import os
from typing import Dict, Any
from datetime import datetime
import telegram
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import json

logger = logging.getLogger(__name__)

class AgentMonitor:
    def __init__(self):
        # Initialize Telegram bot
        self.telegram_bot = telegram.Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        # Initialize Prometheus metrics
        self.transactions_total = Counter(
            'defi_advisor_transactions_total',
            'Total number of transactions executed'
        )
        self.portfolio_value = Gauge(
            'defi_advisor_portfolio_value',
            'Current portfolio value in USD'
        )
        self.rebalance_duration = Histogram(
            'defi_advisor_rebalance_duration_seconds',
            'Time taken for portfolio rebalancing'
        )
        self.risk_score = Gauge(
            'defi_advisor_risk_score',
            'Current portfolio risk score'
        )
        
        # Start Prometheus HTTP server
        start_http_server(int(os.getenv("PROMETHEUS_PORT", "9090")))
        
        logger.info("Agent monitoring initialized")

    async def log_transaction(self, tx_hash: str, tx_type: str, details: Dict[str, Any]):
        """Log transaction details and send notification"""
        try:
            # Increment transaction counter
            self.transactions_total.inc()
            
            # Log to file
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "tx_hash": tx_hash,
                "type": tx_type,
                "details": details
            }
            
            with open("transactions.log", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Send Telegram notification
            message = (
                f"🔄 Transaction Executed\n"
                f"Type: {tx_type}\n"
                f"Hash: {tx_hash}\n"
                f"Details: {json.dumps(details, indent=2)}"
            )
            await self.telegram_bot.send_message(
                chat_id=self.chat_id,
                text=message
            )
            
        except Exception as e:
            logger.error(f"Failed to log transaction: {str(e)}")

    async def update_portfolio_metrics(
        self,
        portfolio_value: float,
        risk_score: float,
        metrics: Dict[str, float]
    ):
        """Update portfolio monitoring metrics"""
        try:
            # Update Prometheus gauges
            self.portfolio_value.set(portfolio_value)
            self.risk_score.set(risk_score)
            
            # Log metrics
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "portfolio_value": portfolio_value,
                "risk_score": risk_score,
                "metrics": metrics
            }
            
            with open("portfolio_metrics.log", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Send daily summary if needed
            await self._check_and_send_daily_summary()
            
        except Exception as e:
            logger.error(f"Failed to update portfolio metrics: {str(e)}")

    async def alert_threshold_breach(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        comparison: str
    ):
        """Send alert when a metric crosses a threshold"""
        try:
            message = (
                f"⚠️ Threshold Alert\n"
                f"Metric: {metric_name}\n"
                f"Current Value: {current_value}\n"
                f"Threshold: {threshold} ({comparison})"
            )
            
            await self.telegram_bot.send_message(
                chat_id=self.chat_id,
                text=message
            )
            
            logger.warning(
                f"Threshold breach - {metric_name}: "
                f"{current_value} {comparison} {threshold}"
            )
            
        except Exception as e:
            logger.error(f"Failed to send threshold alert: {str(e)}")

    async def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error details and send alert"""
        try:
            # Log error
            logger.error(
                f"Error occurred: {str(error)}",
                exc_info=True,
                extra=context
            )
            
            # Send alert
            message = (
                f"❌ Error Alert\n"
                f"Error: {str(error)}\n"
                f"Context: {json.dumps(context, indent=2)}"
            )
            
            await self.telegram_bot.send_message(
                chat_id=self.chat_id,
                text=message
            )
            
        except Exception as e:
            logger.error(f"Failed to log error: {str(e)}")

    async def _check_and_send_daily_summary(self):
        """Send daily portfolio performance summary"""
        try:
            now = datetime.utcnow()
            
            # Check if it's time for daily summary (e.g., at 00:00 UTC)
            if now.hour == 0 and now.minute == 0:
                # Calculate daily metrics
                with open("portfolio_metrics.log", "r") as f:
                    today_metrics = [
                        json.loads(line)
                        for line in f
                        if datetime.fromisoformat(json.loads(line)["timestamp"]).date() == now.date()
                    ]
                
                if not today_metrics:
                    return
                
                # Calculate summary statistics
                start_value = today_metrics[0]["portfolio_value"]
                end_value = today_metrics[-1]["portfolio_value"]
                daily_return = (end_value - start_value) / start_value * 100
                
                message = (
                    f"📊 Daily Portfolio Summary\n"
                    f"Date: {now.date()}\n"
                    f"Portfolio Value: ${end_value:,.2f}\n"
                    f"Daily Return: {daily_return:+.2f}%\n"
                    f"Risk Score: {today_metrics[-1]['risk_score']:.2f}"
                )
                
                await self.telegram_bot.send_message(
                    chat_id=self.chat_id,
                    text=message
                )
                
        except Exception as e:
            logger.error(f"Failed to send daily summary: {str(e)}")

    def start_operation_timer(self) -> datetime:
        """Start timing an operation"""
        return datetime.utcnow()

    def end_operation_timer(self, start_time: datetime, operation: str):
        """End timing an operation and record duration"""
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        if operation == "rebalance":
            self.rebalance_duration.observe(duration)
            
        logger.info(f"{operation} operation took {duration:.2f} seconds") 