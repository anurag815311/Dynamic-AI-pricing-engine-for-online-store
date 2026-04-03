"""
Price Optimization Engine — find the price that maximises revenue.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.predict import predict_demand


def optimize_price(
    current_price: float,
    competitor_price: float,
    category: str = "Electronics",
    season: str = "Summer",
    day_of_week: str = "Monday",
    discount: float = 0.0,
    stock_available: int = 100,
    marketing_spend: float = 500.0,
    customer_rating: float = 4.0,
    price_range_pct: float = config.PRICE_RANGE_PERCENT,
    num_steps: int = config.PRICE_STEPS,
) -> dict:
    """
    Evaluate candidate prices in [current_price ± range%] and return the
    price that maximises revenue = price × predicted_demand.
    """
    low = max(config.MIN_PRICE, current_price * (1 - price_range_pct))
    high = min(config.MAX_PRICE, current_price * (1 + price_range_pct))
    candidate_prices = np.linspace(low, high, num_steps)

    results = []
    for p in candidate_prices:
        demand = predict_demand(
            price=p,
            competitor_price=competitor_price,
            discount=discount,
            category=category,
            season=season,
            day_of_week=day_of_week,
            stock_available=stock_available,
            marketing_spend=marketing_spend,
            customer_rating=customer_rating,
        )
        revenue = p * demand
        results.append({
            "price": round(p, 2),
            "predicted_demand": round(demand, 2),
            "expected_revenue": round(revenue, 2),
        })

    results_df = pd.DataFrame(results)
    best_idx = results_df["expected_revenue"].idxmax()
    best = results_df.iloc[best_idx]

    # ── Price elasticity (at optimal price) ───────────────────────────────
    # Elasticity = (% change in demand) / (% change in price)
    if best_idx > 0 and best_idx < len(results_df) - 1:
        prev = results_df.iloc[best_idx - 1]
        nxt = results_df.iloc[best_idx + 1]
        pct_demand_change = (nxt["predicted_demand"] - prev["predicted_demand"]) / max(prev["predicted_demand"], 1)
        pct_price_change = (nxt["price"] - prev["price"]) / max(prev["price"], 1)
        elasticity = pct_demand_change / pct_price_change if pct_price_change != 0 else 0
    else:
        elasticity = 0.0

    return {
        "recommended_price": float(best["price"]),
        "expected_demand": float(best["predicted_demand"]),
        "expected_revenue": float(best["expected_revenue"]),
        "price_elasticity": round(float(elasticity), 4),
        "current_price": current_price,
        "competitor_price": competitor_price,
        "price_curve": results_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    result = optimize_price(
        current_price=500,
        competitor_price=520,
        category="Electronics",
        season="Summer",
    )
    print(f"✅ Optimal Price: ₹{result['recommended_price']}")
    print(f"   Expected Demand:  {result['expected_demand']}")
    print(f"   Expected Revenue: ₹{result['expected_revenue']}")
    print(f"   Price Elasticity: {result['price_elasticity']}")
