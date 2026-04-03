"""
Mock competitor pricing API — simulates realistic price fluctuations.
"""
import os
import sys
import random
import hashlib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Base prices per product (deterministic from product_id)
def _base_price(product_id: str) -> float:
    """Generate a stable base price from product_id hash."""
    h = int(hashlib.md5(product_id.encode()).hexdigest(), 16)
    return 100 + (h % 2900)  # Range: 100 – 3000


def _get_category(product_id: str) -> str:
    """Deterministic category from product_id."""
    h = int(hashlib.md5(product_id.encode()).hexdigest(), 16)
    return config.CATEGORIES[h % len(config.CATEGORIES)]


def fetch_competitor_price(product_id: str) -> dict:
    """
    Simulate fetching a competitor price for a given product.
    Returns realistic fluctuations around a base price.
    """
    base = _base_price(product_id)

    # Time-based fluctuation (changes throughout the day)
    now = datetime.now()
    hour_seed = now.hour + now.minute / 60.0
    time_factor = 0.03 * (hash(f"{product_id}-{now.strftime('%Y%m%d%H')}") % 100 - 50) / 50

    # Random daily variation
    daily_noise = random.gauss(0, 0.02)

    # Seasonal adjustment
    month = now.month
    if month in [11, 12, 1]:   # Holiday season — prices up
        seasonal = 0.05
    elif month in [6, 7]:       # Summer sales — prices down
        seasonal = -0.04
    else:
        seasonal = 0.0

    competitor_price = base * (1 + time_factor + daily_noise + seasonal)
    competitor_price = round(max(50, competitor_price), 2)

    return {
        "product_id": product_id,
        "competitor_price": competitor_price,
        "source": "MockCompetitorAPI",
        "category": _get_category(product_id),
        "timestamp": now.isoformat(),
    }


def fetch_all_competitor_prices() -> list:
    """Fetch competitor prices for all products."""
    results = []
    for pid in config.PRODUCT_IDS:
        results.append(fetch_competitor_price(pid))
    return results


if __name__ == "__main__":
    for item in fetch_all_competitor_prices()[:5]:
        print(f"  {item['product_id']}: ${item['competitor_price']:.2f}  ({item['source']})")
