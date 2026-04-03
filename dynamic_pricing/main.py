"""
Main orchestrator for the Dynamic AI Pricing Engine.
Handles: dataset generation → model training → scheduler start → API launch.
"""
import os
import sys
import argparse
import logging

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("main")


def step_generate_dataset():
    """Step 1: Generate synthetic training data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating synthetic dataset...")
    logger.info("=" * 60)
    from data.generate_dataset import generate_dataset
    df = generate_dataset()
    logger.info(f"Dataset shape: {df.shape}")
    return df


def step_run_eda():
    """Step 2: Run Exploratory Data Analysis."""
    logger.info("=" * 60)
    logger.info("STEP 2: Running EDA...")
    logger.info("=" * 60)
    from data.eda import run_eda
    run_eda()


def step_train_model():
    """Step 3: Train the demand prediction model."""
    logger.info("=" * 60)
    logger.info("STEP 3: Training XGBoost model...")
    logger.info("=" * 60)
    from models.train_model import train_model
    model, encoders, metrics = train_model()
    return metrics


def step_seed_competitor_prices():
    """Step 4: Seed initial competitor prices into database."""
    logger.info("=" * 60)
    logger.info("STEP 4: Seeding competitor prices...")
    logger.info("=" * 60)
    from scraping.scraper import fetch_all_competitor_prices
    from scraping.price_store import store_prices_batch
    prices = fetch_all_competitor_prices()
    store_prices_batch(prices)
    logger.info(f"Seeded {len(prices)} competitor prices")


def step_start_api():
    """Step 5: Start the FastAPI server (blocking)."""
    logger.info("=" * 60)
    logger.info("STEP 5: Starting FastAPI server...")
    logger.info("=" * 60)
    import uvicorn
    from backend.app import app
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="info")


def run_full_pipeline():
    """Run the complete pipeline: generate → train → seed → serve."""
    logger.info("🚀 Dynamic AI Pricing Engine — Full Pipeline")
    logger.info(f"   Project root: {PROJECT_ROOT}")

    # Step 1: Generate dataset (skip if exists)
    if not os.path.exists(config.TRAINING_DATA_PATH):
        step_generate_dataset()
    else:
        logger.info("✅ Training data already exists, skipping generation")

    # Step 2: EDA
    plots_exist = os.path.exists(config.PLOTS_DIR) and len(os.listdir(config.PLOTS_DIR)) > 0
    if not plots_exist:
        step_run_eda()
    else:
        logger.info("✅ EDA plots already exist, skipping")

    # Step 3: Train model (skip if exists)
    if not os.path.exists(config.MODEL_PATH):
        step_train_model()
    else:
        logger.info("✅ Trained model already exists, skipping")

    # Step 4: Seed competitor prices
    step_seed_competitor_prices()

    # Step 5: Start API (includes scheduler via lifespan)
    step_start_api()


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic AI Pricing Engine — Main Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline (generate → train → serve)
  python main.py --generate         # Only generate dataset
  python main.py --eda              # Only run EDA
  python main.py --train            # Only train model
  python main.py --seed             # Only seed competitor prices
  python main.py --serve            # Only start API server
  python main.py --retrain          # Retrain model with latest data
  python main.py --generate --train # Generate + train only
        """,
    )
    parser.add_argument("--generate", action="store_true", help="Generate synthetic dataset")
    parser.add_argument("--eda", action="store_true", help="Run exploratory data analysis")
    parser.add_argument("--train", action="store_true", help="Train the ML model")
    parser.add_argument("--seed", action="store_true", help="Seed initial competitor prices")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--retrain", action="store_true", help="Retrain model with latest data")

    args = parser.parse_args()

    # If no specific flag is given, run the full pipeline
    any_flag = args.generate or args.eda or args.train or args.seed or args.serve or args.retrain

    if not any_flag:
        run_full_pipeline()
        return

    if args.generate:
        step_generate_dataset()
    if args.eda:
        step_run_eda()
    if args.train:
        step_train_model()
    if args.retrain:
        from models.retrain import retrain
        retrain()
    if args.seed:
        step_seed_competitor_prices()
    if args.serve:
        step_start_api()


if __name__ == "__main__":
    main()
