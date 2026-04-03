"""
Synthetic dataset generator for the Dynamic Pricing Engine.
Generates 10,000+ rows with realistic demand-price relationships.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_dataset(num_rows: int = config.NUM_ROWS, save: bool = True) -> pd.DataFrame:
    """Generate a synthetic pricing/demand dataset."""
    np.random.seed(config.RANDOM_STATE)

    # ── Base price ranges per category ────────────────────────────────────
    category_price_ranges = {
        "Electronics": (200, 3000),
        "Clothing":    (100, 1500),
        "Home":        (150, 2000),
        "Sports":      (100, 1800),
        "Beauty":      (50,  800),
    }

    rows = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range = (end_date - start_date).days

    for _ in range(num_rows):
        # Product
        product_id = np.random.choice(config.PRODUCT_IDS)
        category = np.random.choice(config.CATEGORIES)
        low, high = category_price_ranges[category]

        # Price
        price = round(np.random.uniform(low, high), 2)

        # Competitor price: usually within ±15% of our price
        competitor_offset = np.random.normal(0, 0.08)  # mean 0, std 8%
        competitor_price = round(price * (1 + competitor_offset), 2)
        competitor_price = max(competitor_price, low * 0.5)

        # Discount (0–40%, most products have small discounts)
        discount = round(np.random.beta(2, 8) * 40, 1)  # skewed toward low discounts

        # Date
        random_day = np.random.randint(0, date_range)
        date = start_date + timedelta(days=random_day)
        month = date.month

        # Season
        if month in [12, 1, 2]:
            season = "Winter"
        elif month in [3, 4, 5]:
            season = "Spring"
        elif month in [6, 7, 8]:
            season = "Summer"
        else:
            season = "Fall"

        day_of_week = date.strftime("%A")

        # Stock
        stock_available = np.random.randint(5, 500)

        # Marketing spend
        marketing_spend = round(np.random.exponential(500), 2)

        # Customer rating
        customer_rating = round(np.clip(np.random.normal(3.8, 0.7), 1.0, 5.0), 1)

        # ── Demand simulation (the core logic) ───────────────────────────
        # Base demand depends on category
        category_base = {
            "Electronics": 40, "Clothing": 55, "Home": 35, "Sports": 45, "Beauty": 60
        }
        base_demand = category_base[category]

        # Price effect: higher price → lower demand (inverse relationship)
        price_normalized = (price - low) / (high - low)  # 0 to 1
        price_effect = -30 * price_normalized  # up to -30 units

        # Competitor price effect: if our price < competitor, demand increases
        price_diff_ratio = (competitor_price - price) / price if price > 0 else 0
        competitor_effect = 20 * np.clip(price_diff_ratio, -0.3, 0.3)

        # Discount effect: more discount → more demand
        discount_effect = 0.5 * discount

        # Seasonal effect
        season_effect = {"Winter": 10, "Spring": 2, "Summer": 5, "Fall": -3}[season]

        # Weekend boost
        weekend_effect = 8 if day_of_week in ["Saturday", "Sunday"] else 0

        # Marketing effect (diminishing returns)
        marketing_effect = 5 * np.log1p(marketing_spend / 200)

        # Rating effect
        rating_effect = 3 * (customer_rating - 3.0)

        # Stock scarcity effect (low stock can signal urgency)
        stock_effect = 2 if stock_available < 30 else 0

        # Combine
        demand = (
            base_demand
            + price_effect
            + competitor_effect
            + discount_effect
            + season_effect
            + weekend_effect
            + marketing_effect
            + rating_effect
            + stock_effect
            + np.random.normal(0, 5)  # noise
        )
        units_sold = max(0, int(round(demand)))

        rows.append({
            "product_id": product_id,
            "category": category,
            "price": price,
            "competitor_price": competitor_price,
            "discount": discount,
            "units_sold": units_sold,
            "date": date.strftime("%Y-%m-%d"),
            "season": season,
            "day_of_week": day_of_week,
            "stock_available": stock_available,
            "marketing_spend": marketing_spend,
            "customer_rating": customer_rating,
        })

    df = pd.DataFrame(rows)

    if save:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(config.TRAINING_DATA_PATH, index=False)
        print(f"✅ Dataset generated: {len(df)} rows → {config.TRAINING_DATA_PATH}")

    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.describe())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
