# 💰 Dynamic AI Pricing Engine

> Real-time demand prediction, competitor tracking, and revenue-optimal pricing powered by XGBoost, FastAPI, and Streamlit.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Competitor   │────▶│   SQLite DB  │◀────│  APScheduler    │
│  Scraper/Mock │     │  Price Store │     │  (Background)   │
└──────────────┘     └──────┬───────┘     └────────┬────────┘
                            │                       │
                     ┌──────▼───────┐       ┌───────▼────────┐
                     │   XGBoost    │       │   Retraining   │
                     │  Demand Model│◀──────│   Pipeline     │
                     └──────┬───────┘       └────────────────┘
                            │
                     ┌──────▼───────┐
                     │   Price      │
                     │  Optimizer   │
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐     ┌─────────────────┐
                     │   FastAPI    │◀────│   Streamlit     │
                     │   Backend    │────▶│   Dashboard     │
                     └──────────────┘     └─────────────────┘
```

## 📁 Project Structure

```
dynamic_pricing/
├── config.py               # Central configuration
├── main.py                 # Orchestrator (CLI entry point)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   ├── preprocessing.py    # Feature engineering & encoding
│   ├── eda.py              # Exploratory data analysis plots
│   └── plots/              # Generated EDA visualizations
├── models/
│   ├── train_model.py      # XGBoost training & evaluation
│   ├── predict.py          # Cached inference module
│   ├── optimizer.py        # Revenue-maximizing price optimizer
│   └── retrain.py          # Automated retraining pipeline
├── scraping/
│   ├── mock_scraper.py     # Simulated competitor API
│   ├── scraper.py          # BeautifulSoup web scraper
│   └── price_store.py      # SQLite storage layer
├── backend/
│   └── app.py              # FastAPI application
├── frontend/
│   └── app.py              # Streamlit dashboard
└── scheduler/
    └── scheduler.py        # APScheduler automation
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd dynamic_pricing
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Recommended)
```bash
python main.py
```
This will:

1. Run EDA and save plots
2. Train XGBoost model
3. Seed competitor prices
4. Start FastAPI server (with background scheduler)

### 3. Launch Streamlit Dashboard (in a new terminal)
```bash
cd dynamic_pricing
streamlit run frontend/app.py --server.port 8501
```

### 4. Open the Dashboard
Navigate to **http://localhost:8501** in your browser.

---

## 🔧 Individual Steps

```bash

python main.py --eda           # Run EDA only
python main.py --train         # Train model only
python main.py --seed          # Seed competitor prices only
python main.py --serve         # Start API server only
python main.py --retrain       # Retrain model with latest data
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check & system status |
| `POST` | `/predict-price` | Get optimal price recommendation |
| `GET`  | `/competitor-price/{product_id}` | Latest competitor price |
| `GET`  | `/competitor-prices` | All latest competitor prices |
| `GET`  | `/price-history/{product_id}` | Historical competitor prices |
| `GET`  | `/prediction-history` | Past predictions log |
| `POST` | `/retrain` | Trigger model retraining |
| `GET`  | `/products` | List of product IDs & categories |
| `GET`  | `/scheduler-status` | Background job status |

### Example: Get Optimal Price
```bash
curl -X POST http://localhost:8000/predict-price \
  -H "Content-Type: application/json" \
  -d '{"product_id": "P001", "current_price": 500}'
```

Response:
```json
{
  "product_id": "P001",
  "recommended_price": 520.41,
  "expected_demand": 45,
  "expected_revenue": 23418.45,
  "price_elasticity": -0.3421,
  "competitor_price": 510.25,
  "price_curve": [...]
}
```

## 📊 Features

- **Demand Prediction**: XGBoost regressor with 17 engineered features
- **Price Optimization**: Revenue-maximizing search across ±20% price range
- **Competitor Tracking**: Mock API + BeautifulSoup scraper with SQLite history
- **Automation**: APScheduler for periodic price fetches and model retraining
- **Price Elasticity**: Computed at the optimal price point
- **Interactive Dashboard**: Plotly charts for revenue/demand curves
- **Model Retraining**: Automatic and on-demand retraining pipeline

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | XGBoost |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Scheduler | APScheduler |
| Scraping | BeautifulSoup + Requests |
| Database | SQLite |
| Data | Pandas + NumPy |

---


