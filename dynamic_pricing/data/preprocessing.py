"""
Data preprocessing and feature engineering for the Dynamic Pricing Engine.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(path: str = None) -> pd.DataFrame:
    """Load training data from CSV."""
    path = path or config.TRAINING_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}. Run generate_dataset.py first.")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features."""
    # Price difference
    df["price_diff"] = df["price"] - df["competitor_price"]

    # Price ratio
    df["price_ratio"] = df["price"] / df["competitor_price"].clip(lower=1)

    # Effective price after discount
    df["effective_price"] = df["price"] * (1 - df["discount"] / 100)

    # Sort by product and date for rolling features
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    # Rolling average demand (past 7 entries per product)
    df["rolling_demand_7"] = (
        df.groupby("product_id")["units_sold"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # Month and quarter from date
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
    else:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter

    # Is weekend flag
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    # Log of marketing spend (diminishing returns)
    df["log_marketing_spend"] = np.log1p(df["marketing_spend"])

    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """Label-encode categorical columns. Returns df and encoders dict."""
    categorical_cols = ["category", "season", "day_of_week"]
    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                raise ValueError(f"No encoder found for column '{col}'")
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[f"{col}_encoded"] = df[col].apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )

    return df, encoders


def get_feature_columns() -> list:
    """Return the list of feature columns used for training."""
    return [
        "price",
        "competitor_price",
        "discount",
        "stock_available",
        "marketing_spend",
        "customer_rating",
        "price_diff",
        "price_ratio",
        "effective_price",
        "rolling_demand_7",
        "month",
        "quarter",
        "is_weekend",
        "log_marketing_spend",
        "category_encoded",
        "season_encoded",
        "day_of_week_encoded",
    ]


TARGET_COLUMN = "units_sold"


def preprocess_pipeline(
    df: pd.DataFrame = None,
    encoders: dict = None,
    fit: bool = True,
    test_size: float = config.TEST_SIZE,
):
    """
    Full preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, encoders, full_df
    """
    if df is None:
        df = load_data()

    df = handle_missing_values(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit)

    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.RANDOM_STATE
        )
        return X_train, X_test, y_train, y_test, encoders, df
    else:
        return X, None, y, None, encoders, df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoders, df = preprocess_pipeline()
    print(f"✅ Preprocessing complete")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples:     {len(X_test)}")
    print(f"   Features:         {list(X_train.columns)}")
    print(f"   Target:           {TARGET_COLUMN}")
    print(f"\nFeature stats:\n{X_train.describe()}")
