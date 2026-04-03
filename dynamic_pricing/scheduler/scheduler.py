"""
APScheduler-based automation for periodic competitor price fetching,
dataset updates, and model retraining.
"""
import os
import sys
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("scheduler")

# Global scheduler instance
_scheduler: BackgroundScheduler | None = None


def job_fetch_competitor_prices():
    """Scheduled job: fetch and store competitor prices."""
    try:
        from scraping.scraper import fetch_all_competitor_prices
        from scraping.price_store import store_prices_batch, get_price_count

        logger.info("⏰ [Scheduled] Fetching competitor prices...")
        prices = fetch_all_competitor_prices(use_scraper=False)
        store_prices_batch(prices)
        total = get_price_count()
        logger.info(f"✅ [Scheduled] Stored {len(prices)} prices (total in DB: {total})")
    except Exception as e:
        logger.error(f"❌ [Scheduled] Price fetch failed: {e}", exc_info=True)


def job_retrain_model():
    """Scheduled job: retrain model if enough new data."""
    try:
        from models.retrain import retrain
        from scraping.price_store import get_price_count

        count = get_price_count()
        logger.info(f"⏰ [Scheduled] Checking retraining — {count} records in DB")

        if count >= config.RETRAIN_THRESHOLD_ROWS:
            metrics = retrain()
            logger.info(f"✅ [Scheduled] Retraining complete — R²: {metrics['test_r2']:.4f}")
        else:
            logger.info(f"⏭️  [Scheduled] Not enough data for retraining ({count} < {config.RETRAIN_THRESHOLD_ROWS})")
    except Exception as e:
        logger.error(f"❌ [Scheduled] Retraining failed: {e}", exc_info=True)


def start_scheduler():
    """Initialize and start the background scheduler."""
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        logger.info("Scheduler is already running")
        return _scheduler

    _scheduler = BackgroundScheduler()

    # Job 1: Fetch competitor prices periodically
    _scheduler.add_job(
        job_fetch_competitor_prices,
        trigger=IntervalTrigger(minutes=config.SCRAPE_INTERVAL_MINUTES),
        id="fetch_prices",
        name="Fetch Competitor Prices",
        replace_existing=True,
        next_run_time=datetime.now(),  # Run immediately on start
    )

    # Job 2: Retrain model periodically
    _scheduler.add_job(
        job_retrain_model,
        trigger=IntervalTrigger(hours=config.RETRAIN_INTERVAL_HOURS),
        id="retrain_model",
        name="Retrain ML Model",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(f"✅ Scheduler started — price fetch every {config.SCRAPE_INTERVAL_MINUTES}min, retrain every {config.RETRAIN_INTERVAL_HOURS}h")

    return _scheduler


def stop_scheduler():
    """Stop the scheduler gracefully."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("🛑 Scheduler stopped")
        _scheduler = None


def get_scheduler_status() -> dict:
    """Get current scheduler status and job info."""
    if _scheduler is None or not _scheduler.running:
        return {"running": False, "jobs": []}

    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": str(job.next_run_time) if job.next_run_time else "N/A",
        })

    return {"running": True, "jobs": jobs}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    import time

    scheduler = start_scheduler()
    try:
        print("Scheduler running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_scheduler()
