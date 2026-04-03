"""
Prediction module — load saved model and predict demand.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.preprocessing import (
    engineer_features,
    encode_categoricals,
    get_feature_columns,
    handle_missing_values,
)

# ── Module-level cache ────────────────────────────────────────────────────────
_model = None
_encoders = None


def _load_model():
    """Load model and encoders (cached)."""
    global _model, _encoders
    if _model is None:
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {config.MODEL_PATH}. Run train_model.py first."
            )
        _model = joblib.load(config.MODEL_PATH)
        _encoders = joblib.load(config.ENCODERS_PATH)
    return _model, _encoders


def reload_model():
    """Force-reload model from disk (after retraining)."""
    global _model, _encoders
    _model = None
    _encoders = None
    return _load_model()


def predict_demand(
    price: float,
    competitor_price: float,
    discount: float,
    category: str,
    season: str,
    day_of_week: str,
    stock_available: int,
    marketing_spend: float,
    customer_rating: float,
) -> float:
    """Predict units sold for a single set of inputs."""
    model, encoders = _load_model()

    # Build a single-row DataFrame
    row = pd.DataFrame([{
        "product_id": "PRED",
        "price": price,
        "competitor_price": competitor_price,
        "discount": discount,
        "category": category,
        "season": season,
        "day_of_week": day_of_week,
        "stock_available": stock_available,
        "marketing_spend": marketing_spend,
        "customer_rating": customer_rating,
        "units_sold": 0,  # placeholder for feature engineering
        "date": pd.Timestamp.now(),
    }])

    row = handle_missing_values(row)
    row = engineer_features(row)
    row, _ = encode_categoricals(row, encoders=encoders, fit=False)

    feature_cols = get_feature_columns()
    X = row[feature_cols]

    prediction = model.predict(X)[0]
    return max(0, float(prediction))


def predict_demand_batch(df: pd.DataFrame) -> np.ndarray:
    """Predict demand for a batch DataFrame (must have all required columns)."""
    model, encoders = _load_model()

    df = handle_missing_values(df.copy())
    df = engineer_features(df)
    df, _ = encode_categoricals(df, encoders=encoders, fit=False)

    feature_cols = get_feature_columns()
    X = df[feature_cols]

    predictions = model.predict(X)
    return np.maximum(0, predictions)
