"""
Exploratory Data Analysis — generates and saves key visualizations.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def run_eda(df: pd.DataFrame = None, save_dir: str = None):
    """Generate EDA plots and save to disk."""
    if df is None:
        from data.preprocessing import load_data
        df = load_data()

    save_dir = save_dir or config.PLOTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(style="darkgrid", palette="viridis")

    # ── 1. Price vs Demand scatter ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["price"], df["units_sold"],
        c=df["discount"], cmap="coolwarm", alpha=0.4, s=10,
    )
    plt.colorbar(scatter, label="Discount %")
    ax.set_xlabel("Price ($)", fontsize=12)
    ax.set_ylabel("Units Sold", fontsize=12)
    ax.set_title("Price vs Demand (colored by Discount)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "price_vs_demand.png"), dpi=150)
    plt.close(fig)

    # ── 2. Competitor price impact ────────────────────────────────────────
    df["price_diff"] = df["price"] - df["competitor_price"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = pd.cut(df["price_diff"], bins=20)
    grouped = df.groupby(bins, observed=True)["units_sold"].mean()
    grouped.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(grouped)))
    ax.set_xlabel("Price − Competitor Price (binned)", fontsize=12)
    ax.set_ylabel("Average Units Sold", fontsize=12)
    ax.set_title("Competitor Price Impact on Demand", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "competitor_price_impact.png"), dpi=150)
    plt.close(fig)

    # ── 3. Seasonal trends ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_demand = df.groupby("season")["units_sold"].mean().reindex(season_order)
    bars = ax.bar(season_demand.index, season_demand.values,
                  color=["#4FC3F7", "#81C784", "#FFD54F", "#FF8A65"])
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Average Units Sold", fontsize=12)
    ax.set_title("Seasonal Demand Trends", fontsize=14)
    for bar, val in zip(bars, season_demand.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "seasonal_trends.png"), dpi=150)
    plt.close(fig)

    # ── 4. Category-wise demand ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="category", y="units_sold", palette="Set2", ax=ax)
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Units Sold", fontsize=12)
    ax.set_title("Demand Distribution by Category", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "category_demand.png"), dpi=150)
    plt.close(fig)

    # ── 5. Correlation heatmap ────────────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # ── 6. Day-of-week demand ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_demand = df.groupby("day_of_week")["units_sold"].mean().reindex(day_order)
    ax.plot(day_demand.index, day_demand.values, "o-", color="#7E57C2", linewidth=2, markersize=8)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("Average Units Sold", fontsize=12)
    ax.set_title("Demand by Day of Week", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "day_of_week_demand.png"), dpi=150)
    plt.close(fig)

    print(f"✅ EDA complete — {6} plots saved to {save_dir}")


if __name__ == "__main__":
    run_eda()
