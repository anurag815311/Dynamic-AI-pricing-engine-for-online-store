"""
Model retraining pipeline — append new data, retrain, save updated model.
"""
import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("retrain")


def append_new_data(new_data: pd.DataFrame):
    """Append new observations to the training CSV."""
    if os.path.exists(config.TRAINING_DATA_PATH):
        existing = pd.read_csv(config.TRAINING_DATA_PATH)
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    combined.to_csv(config.TRAINING_DATA_PATH, index=False)
    logger.info(f"Appended {len(new_data)} rows → total {len(combined)} rows")
    return combined


def retrain():
    """Full retraining pipeline: load latest data → train → save."""
    from models.train_model import train_model

    logger.info("🔄 Starting model retraining...")
    model, encoders, metrics = train_model(save=True)
    logger.info(f"✅ Retraining complete — Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.3f}")

    # Reload cached model in predict module
    from models.predict import reload_model
    reload_model()
    logger.info("✅ Prediction module reloaded with new model")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    retrain()
