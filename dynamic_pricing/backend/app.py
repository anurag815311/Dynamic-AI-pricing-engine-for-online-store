"""
FastAPI backend for the Dynamic AI Pricing Engine.
"""
import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("api")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on app startup, stop on shutdown."""
    from scheduler.scheduler import start_scheduler, stop_scheduler
    start_scheduler()
    logger.info("🚀 API started with background scheduler")
    yield
    stop_scheduler()
    logger.info("🛑 API shutdown")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Dynamic AI Pricing Engine",
    description="Real-time demand prediction and price optimization with competitor tracking",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────
class PricePredictionRequest(BaseModel):
    product_id: str = Field(..., example="P001")
    current_price: float = Field(..., example=500.0)
    category: str = Field(default="Electronics", example="Electronics")
    season: str = Field(default="Summer", example="Summer")
    day_of_week: str = Field(default="Monday", example="Monday")
    discount: float = Field(default=0.0, example=5.0)
    stock_available: int = Field(default=100, example=100)
    marketing_spend: float = Field(default=500.0, example=500.0)
    customer_rating: float = Field(default=4.0, example=4.0)


class PricePredictionResponse(BaseModel):
    product_id: str
    recommended_price: float
    expected_demand: float
    expected_revenue: float
    price_elasticity: float
    current_price: float
    competitor_price: float
    price_curve: list


class CompetitorPriceResponse(BaseModel):
    product_id: str
    competitor_price: float
    source: str
    timestamp: str


class RetrainResponse(BaseModel):
    status: str
    train_rmse: float
    test_rmse: float
    train_r2: float
    test_r2: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from scheduler.scheduler import get_scheduler_status
    from scraping.price_store import get_price_count
    return {
        "status": "healthy",
        "model_exists": os.path.exists(config.MODEL_PATH),
        "training_data_exists": os.path.exists(config.TRAINING_DATA_PATH),
        "competitor_prices_count": get_price_count(),
        "scheduler": get_scheduler_status(),
    }


@app.post("/predict-price", response_model=PricePredictionResponse)
async def predict_price(request: PricePredictionRequest):
    """
    Predict optimal price for a product.
    Fetches latest competitor price, runs the optimizer, returns recommendation.
    """
    try:
        from scraping.scraper import fetch_competitor_price
        from scraping.price_store import get_latest_price, store_prediction
        from models.optimizer import optimize_price

        # Get competitor price — try DB first, then fetch live
        db_price = get_latest_price(request.product_id)
        if db_price:
            competitor_price = db_price["price"]
        else:
            live = fetch_competitor_price(request.product_id)
            competitor_price = live["competitor_price"]

        # Run optimizer
        result = optimize_price(
            current_price=request.current_price,
            competitor_price=competitor_price,
            category=request.category,
            season=request.season,
            day_of_week=request.day_of_week,
            discount=request.discount,
            stock_available=request.stock_available,
            marketing_spend=request.marketing_spend,
            customer_rating=request.customer_rating,
        )

        # Log prediction
        store_prediction(
            product_id=request.product_id,
            current_price=request.current_price,
            competitor_price=competitor_price,
            recommended_price=result["recommended_price"],
            expected_demand=result["expected_demand"],
            expected_revenue=result["expected_revenue"],
            price_elasticity=result["price_elasticity"],
        )

        return PricePredictionResponse(
            product_id=request.product_id,
            **result,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/competitor-price/{product_id}", response_model=CompetitorPriceResponse)
async def get_competitor_price(product_id: str):
    """Get latest competitor price for a product."""
    from scraping.price_store import get_latest_price
    from scraping.scraper import fetch_competitor_price

    # Try database first
    db_price = get_latest_price(product_id)
    if db_price:
        return CompetitorPriceResponse(
            product_id=product_id,
            competitor_price=db_price["price"],
            source=db_price.get("source", "database"),
            timestamp=db_price["timestamp"],
        )

    # Fetch live
    live = fetch_competitor_price(product_id)
    return CompetitorPriceResponse(
        product_id=product_id,
        competitor_price=live["competitor_price"],
        source=live["source"],
        timestamp=live["timestamp"],
    )


@app.get("/competitor-prices")
async def get_all_competitor_prices():
    """Get latest competitor prices for all products."""
    from scraping.price_store import get_all_latest_prices
    prices = get_all_latest_prices()
    if not prices:
        from scraping.scraper import fetch_all_competitor_prices
        prices = fetch_all_competitor_prices()
    return {"prices": prices}


@app.get("/price-history/{product_id}")
async def get_price_history(product_id: str, limit: int = 100):
    """Get historical competitor prices for a product."""
    from scraping.price_store import get_price_history
    history = get_price_history(product_id, limit=limit)
    return {"product_id": product_id, "history": history}


@app.get("/prediction-history")
async def get_prediction_history(product_id: str = None, limit: int = 50):
    """Get prediction history."""
    from scraping.price_store import get_prediction_history
    history = get_prediction_history(product_id=product_id, limit=limit)
    return {"history": history}


@app.post("/retrain", response_model=RetrainResponse)
async def retrain_model():
    """Trigger model retraining."""
    try:
        from models.retrain import retrain
        metrics = retrain()
        return RetrainResponse(status="success", **metrics)
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products")
async def get_products():
    """Get list of available product IDs."""
    return {"products": config.PRODUCT_IDS, "categories": config.CATEGORIES}


@app.get("/scheduler-status")
async def scheduler_status():
    """Get scheduler status."""
    from scheduler.scheduler import get_scheduler_status
    return get_scheduler_status()


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="info")
