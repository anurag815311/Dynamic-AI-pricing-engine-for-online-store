"""
SQLite store for competitor price history.
"""
import os
import sys
import sqlite3
import logging
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("price_store")


def _get_connection() -> sqlite3.Connection:
    """Get SQLite connection, creating tables if needed."""
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS competitor_prices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id  TEXT    NOT NULL,
            price       REAL    NOT NULL,
            source      TEXT    DEFAULT 'mock',
            category    TEXT,
            timestamp   TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id        TEXT NOT NULL,
            current_price     REAL,
            competitor_price  REAL,
            recommended_price REAL,
            expected_demand   REAL,
            expected_revenue  REAL,
            price_elasticity  REAL,
            timestamp         TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_cp_product_time
        ON competitor_prices (product_id, timestamp DESC)
    """)
    conn.commit()
    return conn


def store_price(product_id: str, price: float, source: str = "mock", category: str = None):
    """Store a competitor price observation."""
    conn = _get_connection()
    conn.execute(
        "INSERT INTO competitor_prices (product_id, price, source, category, timestamp) VALUES (?, ?, ?, ?, ?)",
        (product_id, price, source, category, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    logger.debug(f"Stored price for {product_id}: ${price:.2f}")


def store_prices_batch(prices: list):
    """Store multiple price observations at once."""
    conn = _get_connection()
    now = datetime.now().isoformat()
    rows = [
        (p["product_id"], p["competitor_price"], p.get("source", "mock"), p.get("category"), now)
        for p in prices
    ]
    conn.executemany(
        "INSERT INTO competitor_prices (product_id, price, source, category, timestamp) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    logger.info(f"Stored {len(rows)} competitor prices")


def get_latest_price(product_id: str) -> Optional[dict]:
    """Get the most recent competitor price for a product."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM competitor_prices WHERE product_id = ? ORDER BY timestamp DESC LIMIT 1",
        (product_id,),
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_price_history(product_id: str, limit: int = 100) -> list:
    """Get historical competitor prices for a product."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM competitor_prices WHERE product_id = ? ORDER BY timestamp DESC LIMIT ?",
        (product_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_latest_prices() -> list:
    """Get the latest competitor price for each product."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT cp.* FROM competitor_prices cp
        INNER JOIN (
            SELECT product_id, MAX(timestamp) as max_ts
            FROM competitor_prices
            GROUP BY product_id
        ) latest ON cp.product_id = latest.product_id AND cp.timestamp = latest.max_ts
        ORDER BY cp.product_id
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def store_prediction(product_id: str, current_price: float, competitor_price: float,
                     recommended_price: float, expected_demand: float,
                     expected_revenue: float, price_elasticity: float):
    """Log a pricing prediction."""
    conn = _get_connection()
    conn.execute(
        """INSERT INTO predictions_log
           (product_id, current_price, competitor_price, recommended_price,
            expected_demand, expected_revenue, price_elasticity, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (product_id, current_price, competitor_price, recommended_price,
         expected_demand, expected_revenue, price_elasticity, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_prediction_history(product_id: str = None, limit: int = 50) -> list:
    """Get prediction history, optionally filtered by product."""
    conn = _get_connection()
    if product_id:
        rows = conn.execute(
            "SELECT * FROM predictions_log WHERE product_id = ? ORDER BY timestamp DESC LIMIT ?",
            (product_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM predictions_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_price_count() -> int:
    """Get total number of competitor price records."""
    conn = _get_connection()
    count = conn.execute("SELECT COUNT(*) FROM competitor_prices").fetchone()[0]
    conn.close()
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Quick test
    store_price("P001", 499.99, "test", "Electronics")
    store_price("P001", 509.50, "test", "Electronics")
    latest = get_latest_price("P001")
    print(f"Latest price for P001: {latest}")
    history = get_price_history("P001")
    print(f"Price history ({len(history)} records): {history}")
