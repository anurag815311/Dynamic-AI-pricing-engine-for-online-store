"""
Streamlit Dashboard for the Dynamic AI Pricing Engine.
Modern, interactive UI with real-time competitor tracking and price optimization.
"""
import os
import sys
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic AI Pricing Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = f"http://localhost:{config.API_PORT}"


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }

    .main-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }

    .metric-card h3 {
        color: #a0aec0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .metric-card .value {
        color: #e2e8f0;
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-card .value.green { color: #48bb78; }
    .metric-card .value.blue  { color: #63b3ed; }
    .metric-card .value.purple { color: #b794f4; }
    .metric-card .value.orange { color: #f6ad55; }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-online {
        background: rgba(72, 187, 120, 0.2);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.4);
    }

    .status-offline {
        background: rgba(245, 101, 101, 0.2);
        color: #f56565;
        border: 1px solid rgba(245, 101, 101, 0.4);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    div[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────
def api_get(endpoint: str, params: dict = None):
    """Make GET request to API."""
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint: str, data: dict):
    """Make POST request to API."""
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def check_api_health():
    """Check if API is online."""
    return api_get("/health") is not None


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💰 Dynamic AI Pricing Engine</h1>
    <p>Real-time demand prediction • Competitor tracking • Revenue optimization</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_online = check_api_health()
    if api_online:
        st.markdown('<span class="status-badge status-online">● API Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-offline">● API Offline</span>', unsafe_allow_html=True)
        st.warning("Start the API server first:\n```\npython backend/app.py\n```")

    st.markdown("---")

    # Product selection
    products = api_get("/products")
    product_list = products["products"] if products else config.PRODUCT_IDS
    categories = products["categories"] if products else config.CATEGORIES

    selected_product = st.selectbox("🏷️ Product ID", product_list, index=0)
    current_price = st.number_input("💵 Current Price ($)", min_value=50.0, max_value=5000.0, value=500.0, step=10.0)
    selected_category = st.selectbox("📦 Category", categories, index=0)
    selected_season = st.selectbox("🌤️ Season", config.SEASONS, index=2)
    selected_day = st.selectbox("📅 Day of Week", config.DAYS_OF_WEEK, index=0)

    st.markdown("---")
    st.markdown("### 📊 Advanced Settings")
    discount = st.slider("Discount %", 0.0, 40.0, 5.0, 0.5)
    stock = st.slider("Stock Available", 5, 500, 100)
    marketing = st.slider("Marketing Spend ($)", 0.0, 5000.0, 500.0, 50.0)
    rating = st.slider("Customer Rating", 1.0, 5.0, 4.0, 0.1)

    st.markdown("---")
    optimize_btn = st.button("🚀 Optimize Price", use_container_width=True, type="primary")
    retrain_btn = st.button("🔄 Retrain Model", use_container_width=True)


# ── Main Content ──────────────────────────────────────────────────────────────
if not api_online:
    st.info("👆 Please start the FastAPI backend to use the dashboard. See sidebar for instructions.")
    st.stop()

# ── Competitor Price Display ──────────────────────────────────────────────────
col_comp1, col_comp2 = st.columns([2, 1])
with col_comp1:
    st.markdown("### 📡 Real-Time Competitor Price")
    comp_data = api_get(f"/competitor-price/{selected_product}")
    if comp_data:
        comp_price = comp_data["competitor_price"]
        price_diff = current_price - comp_price
        diff_pct = (price_diff / comp_price * 100) if comp_price > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Competitor Price", f"${comp_price:.2f}", f"{diff_pct:+.1f}% vs yours")
        c2.metric("Your Price", f"${current_price:.2f}")
        c3.metric("Price Gap", f"${abs(price_diff):.2f}", "Higher" if price_diff > 0 else "Lower")

with col_comp2:
    st.markdown("### 🕐 Last Updated")
    if comp_data:
        st.markdown(f"**Source:** {comp_data.get('source', 'N/A')}")
        st.markdown(f"**Time:** {comp_data.get('timestamp', 'N/A')[:19]}")


st.markdown("---")

# ── Price Optimization ────────────────────────────────────────────────────────
if optimize_btn:
    with st.spinner("🔄 Running price optimization..."):
        result = api_post("/predict-price", {
            "product_id": selected_product,
            "current_price": current_price,
            "category": selected_category,
            "season": selected_season,
            "day_of_week": selected_day,
            "discount": discount,
            "stock_available": stock,
            "marketing_spend": marketing,
            "customer_rating": rating,
        })

    if result:
        st.session_state["last_result"] = result
        st.session_state["last_product"] = selected_product

# Display results if available
if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.markdown("### 🎯 Optimization Results")

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Recommended Price</h3>
            <div class="value green">${result['recommended_price']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Expected Demand</h3>
            <div class="value blue">{result['expected_demand']:.0f} units</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Expected Revenue</h3>
            <div class="value purple">${result['expected_revenue']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Price Elasticity</h3>
            <div class="value orange">{result['price_elasticity']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────
    if "price_curve" in result and result["price_curve"]:
        curve_df = pd.DataFrame(result["price_curve"])

        chart1, chart2 = st.columns(2)

        with chart1:
            fig_revenue = go.Figure()
            fig_revenue.add_trace(go.Scatter(
                x=curve_df["price"], y=curve_df["expected_revenue"],
                mode="lines+markers",
                line=dict(color="#667eea", width=3),
                marker=dict(size=4),
                name="Revenue",
                fill="tozeroy",
                fillcolor="rgba(102, 126, 234, 0.1)",
            ))
            # Mark optimal price
            fig_revenue.add_trace(go.Scatter(
                x=[result["recommended_price"]],
                y=[result["expected_revenue"]],
                mode="markers+text",
                marker=dict(size=14, color="#48bb78", symbol="star"),
                text=[f"Optimal: ${result['recommended_price']:.0f}"],
                textposition="top center",
                name="Optimal",
            ))
            fig_revenue.update_layout(
                title="💰 Price vs Revenue Curve",
                xaxis_title="Price ($)",
                yaxis_title="Expected Revenue ($)",
                template="plotly_dark",
                height=420,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,26,46,0.8)",
            )
            st.plotly_chart(fig_revenue, use_container_width=True)

        with chart2:
            fig_demand = go.Figure()
            fig_demand.add_trace(go.Scatter(
                x=curve_df["price"], y=curve_df["predicted_demand"],
                mode="lines+markers",
                line=dict(color="#f6ad55", width=3),
                marker=dict(size=4),
                name="Demand",
                fill="tozeroy",
                fillcolor="rgba(246, 173, 85, 0.1)",
            ))
            fig_demand.add_trace(go.Scatter(
                x=[result["recommended_price"]],
                y=[result["expected_demand"]],
                mode="markers+text",
                marker=dict(size=14, color="#48bb78", symbol="star"),
                text=[f"Demand: {result['expected_demand']:.0f}"],
                textposition="top center",
                name="At Optimal",
            ))
            fig_demand.update_layout(
                title="📈 Price vs Demand Curve",
                xaxis_title="Price ($)",
                yaxis_title="Predicted Demand (units)",
                template="plotly_dark",
                height=420,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,26,46,0.8)",
            )
            st.plotly_chart(fig_demand, use_container_width=True)

    # ── Price Comparison Bar ──────────────────────────────────────────────
    st.markdown("### 📊 Price Comparison")
    comp_fig = go.Figure()
    prices_to_compare = {
        "Your Current": result["current_price"],
        "Competitor": result["competitor_price"],
        "Recommended": result["recommended_price"],
    }
    colors = ["#63b3ed", "#f56565", "#48bb78"]
    comp_fig.add_trace(go.Bar(
        x=list(prices_to_compare.keys()),
        y=list(prices_to_compare.values()),
        marker_color=colors,
        text=[f"${v:.2f}" for v in prices_to_compare.values()],
        textposition="outside",
        textfont=dict(size=14, color="white"),
    ))
    comp_fig.update_layout(
        template="plotly_dark",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,46,0.8)",
        yaxis_title="Price ($)",
        showlegend=False,
    )
    st.plotly_chart(comp_fig, use_container_width=True)

st.markdown("---")

# ── Competitor Price History ──────────────────────────────────────────────────
st.markdown("### 📈 Competitor Price History")
history = api_get(f"/price-history/{selected_product}", params={"limit": 200})
if history and history.get("history"):
    hist_df = pd.DataFrame(history["history"])
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
    hist_df = hist_df.sort_values("timestamp")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_df["timestamp"], y=hist_df["price"],
        mode="lines+markers",
        line=dict(color="#b794f4", width=2),
        marker=dict(size=5),
        name="Competitor Price",
        fill="tozeroy",
        fillcolor="rgba(183, 148, 244, 0.1)",
    ))
    fig_hist.update_layout(
        title=f"Competitor Price History — {selected_product}",
        xaxis_title="Timestamp",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,46,0.8)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("No competitor price history yet. The scheduler will populate this automatically.")

# ── Retrain Model ─────────────────────────────────────────────────────────────
if retrain_btn:
    with st.spinner("🔄 Retraining model... this may take a minute."):
        retrain_result = api_post("/retrain", {})
    if retrain_result:
        st.success(f"✅ Model retrained! Test R²: {retrain_result['test_r2']:.4f} | RMSE: {retrain_result['test_rmse']:.3f}")
    else:
        st.error("Retraining failed. Check API logs.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096; padding: 1rem;'>"
    "Dynamic AI Pricing Engine v1.0 • Powered by XGBoost, FastAPI & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
