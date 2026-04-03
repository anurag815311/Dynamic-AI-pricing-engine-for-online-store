"""
XGBoost demand prediction model — training, evaluation, and saving.
"""
import os
import sys
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.preprocessing import preprocess_pipeline


def train_model(save: bool = True):
    """Train XGBoost model and return metrics."""
    print("🔄 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, encoders, full_df = preprocess_pipeline()

    print(f"   Training set:  {X_train.shape}")
    print(f"   Test set:      {X_test.shape}")

    print("🔄 Training XGBoost model...")
    model = XGBRegressor(**config.XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "train_r2":   float(r2_score(y_train, y_pred_train)),
        "test_r2":    float(r2_score(y_test, y_pred_test)),
    }

    print(f"\n📊 Model Performance:")
    print(f"   Train RMSE: {metrics['train_rmse']:.3f}")
    print(f"   Test  RMSE: {metrics['test_rmse']:.3f}")
    print(f"   Train R²:   {metrics['train_r2']:.4f}")
    print(f"   Test  R²:   {metrics['test_r2']:.4f}")

    # ── Feature importance ────────────────────────────────────────────────
    importances = model.feature_importances_
    feature_names = X_train.columns
    importance_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print(f"\n🏆 Top Feature Importances:")
    for name, imp in importance_pairs[:10]:
        print(f"   {name:30s} → {imp:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    if save:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        joblib.dump(encoders, config.ENCODERS_PATH)
        print(f"\n✅ Model saved to {config.MODEL_PATH}")
        print(f"✅ Encoders saved to {config.ENCODERS_PATH}")

    return model, encoders, metrics


if __name__ == "__main__":
    model, encoders, metrics = train_model()
