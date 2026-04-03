"""
Central configuration for the Dynamic AI Pricing Engine.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
DB_PATH = os.path.join(DATA_DIR, "competitor_prices.db")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "demand_model.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")

# ─── Dataset Generation ──────────────────────────────────────────────────────
NUM_ROWS = 10000
PRODUCT_IDS = [f"P{str(i).zfill(3)}" for i in range(1, 51)]  # P001 - P050
CATEGORIES = ["Electronics", "Clothing", "Home", "Sports", "Beauty"]
SEASONS = ["Winter", "Spring", "Summer", "Fall"]
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ─── Price Optimization ──────────────────────────────────────────────────────
PRICE_RANGE_PERCENT = 0.20        # ±20% around current price
PRICE_STEPS = 50                  # Number of candidate prices to evaluate
MIN_PRICE = 50
MAX_PRICE = 5000

# ─── Scheduler ────────────────────────────────────────────────────────────────
SCRAPE_INTERVAL_MINUTES = 60      # Fetch competitor prices every N minutes
RETRAIN_THRESHOLD_ROWS = 500      # Retrain model after N new data rows
RETRAIN_INTERVAL_HOURS = 24       # Or retrain every N hours

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Streamlit ────────────────────────────────────────────────────────────────
STREAMLIT_PORT = 8501

# ─── Model Hyperparameters ────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL = "INFO"

# ─── Ensure directories exist ────────────────────────────────────────────────
for d in [DATA_DIR, MODEL_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)
