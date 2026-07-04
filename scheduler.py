"""
Background Scheduler for SMT Detection
Runs checks on all timeframes and triggers notifications
"""

import asyncio
import logging
from datetime import datetime, timedelta
import schedule
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import SMTMonitor
from notifications import send_bulk_notification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

# Timeframes and their check intervals (in minutes)
TIMEFRAME_SCHEDULE = {
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080
}

# ==========================================
# SCHEDULER FUNCTIONS
# ==========================================

monitor = SMTMonitor()

async def run_timeframe_check(timeframe: str):
    """Run SMT check for a specific timeframe across all groups"""
    logger.info(f"Running check for {timeframe} timeframe...")
    
    all_results = []
    
    for group_name, group_config in monitor.SMT_GROUPS.items():
        try:
            results = await monitor.check_timeframe(group_name, group_config, timeframe)
            if results:
                all_results.extend(results)
                logger.info(f"Found {len(results)} divergences in {group_name} on {timeframe}")
        except Exception as e:
            logger.error(f"Error checking {group_name} on {timeframe}: {e}")
    
    if all_results:
        logger.info(f"Total {len(all_results)} divergences found on {timeframe}")
        await send_bulk_notification(all_results)
    
    return all_results

def run_checks_for_timeframe(timeframe: str):
    """Wrapper to run async check from sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_timeframe_check(timeframe))
    finally:
        loop.close()

def run_all_checks():
    """Run checks for all timeframes"""
    logger.info("Starting full SMT scan across all timeframes...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(monitor.run_full_check())
    finally:
        loop.close()

# ==========================================
# SCHEDULE SETUP
# ==========================================

def setup_schedule():
    """Setup the schedule for all timeframes"""
    
    # Schedule 30-minute checks
    schedule.every(30).minutes.do(run_checks_for_timeframe, "30m")
    
    # Schedule 1-hour checks
    schedule.every(1).hours.do(run_checks_for_timeframe, "1h")
    
    # Schedule 4-hour checks
    schedule.every(4).hours.do(run_checks_for_timeframe, "4h")
    
    # Schedule daily checks at market open (9:30 AM ET)
    schedule.every().day.at("09:30").do(run_checks_for_timeframe, "1d")
    
    # Schedule weekly checks on Monday at 9:30 AM ET
    schedule.every().monday.at("09:30").do(run_checks_for_timeframe, "1w")
    
    logger.info("✅ Scheduler configured with all timeframes")
    logger.info("   - 30m: every 30 minutes")
    logger.info("   - 1h: every hour")
    logger.info("   - 4h: every 4 hours")
    logger.info("   - 1d: daily at 9:30 AM ET")
    logger.info("   - 1w: weekly on Monday at 9:30 AM ET")

# ==========================================
# MAIN SCHEDULER LOOP
# ==========================================

def run_scheduler():
    """Main scheduler loop"""
    logger.info("Starting SMT Detection Scheduler...")
    setup_schedule()
    
    # Run initial full check on startup
    logger.info("Running initial full check...")
    run_all_checks()
    
    logger.info("Scheduler running. Press Ctrl+C to stop.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(10)  # Check every 10 seconds
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")

if __name__ == "__main__":
    run_scheduler()