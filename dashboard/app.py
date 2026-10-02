"""
BasketIQ Dashboard
-------------------
An interactive dashboard that presents the key findings from the BasketIQ
analysis notebooks (EDA, RFM segmentation, clustering, market basket
analysis, and cohort retention) in one place.

This file does NOT recompute anything heavy itself — it only reads the
processed CSV files produced by the notebooks in `notebooks/`. Run the
notebooks first (in order) so these files exist:
    data/processed/online_retail_clean.csv
    data/processed/rfm_with_clusters.csv
    data/processed/cohort_retention.csv
    data/processed/association_rules_france.csv

Run with:
    streamlit run dashboard/app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BasketIQ Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# A consistent color palette used across every chart
PALETTE = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# Data loading (cached so the app stays fast on reruns/interactions)
# ---------------------------------------------------------------------------
@st.cache_data
def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "online_retail_clean.csv", parse_dates=["InvoiceDate"])
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    return df


@st.cache_data
def load_rfm() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "rfm_with_clusters.csv")


@st.cache_data
def load_cohort() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "cohort_retention.csv", index_col=0)


@st.cache_data
def load_basket_rules() -> pd.DataFrame:
    path = DATA_DIR / "association_rules_france.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def missing_file_message(filename: str):
    st.warning(
        f"`{filename}` was not found in `data/processed/`. "
        f"Run the corresponding notebook first to generate it."
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛒 BasketIQ")
st.sidebar.caption("Retail analytics built on the UCI Online Retail dataset")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Sales Trends", "Customer Segments", "Market Basket", "Retention"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data source: processed CSVs from `notebooks/01–06`")

# ---------------------------------------------------------------------------
# Load data once (each loader is individually cached and fails gracefully)
# ---------------------------------------------------------------------------
try:
    df = load_transactions()
except FileNotFoundError:
    df = None

try:
    rfm = load_rfm()
except FileNotFoundError:
    rfm = None

try:
    cohort = load_cohort()
except FileNotFoundError:
    cohort = None

rules = load_basket_rules()


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Overview")
    st.caption("Headline numbers across the full cleaned dataset.")

    if df is None:
        missing_file_message("online_retail_clean.csv")
    else:
        total_revenue = df["TotalPrice"].sum()
        total_orders = df["InvoiceNo"].nunique()
        total_customers = df["CustomerID"].nunique()
        avg_order_value = total_revenue / total_orders

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"£{total_revenue:,.0f}")
        col2.metric("Total Orders", f"{total_orders:,}")
        col3.metric("Unique Customers", f"{total_customers:,}")
        col4.metric("Avg. Order Value", f"£{avg_order_value:,.2f}")

        st.markdown("---")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            monthly_revenue = df.groupby("InvoiceMonth")["TotalPrice"].sum().reset_index()
            fig = px.line(
                monthly_revenue,
                x="InvoiceMonth",
                y="TotalPrice",
                markers=True,
                title="Monthly Revenue",
                color_discrete_sequence=[PALETTE[0]],
            )
            fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (£)")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            top_countries = (
                df[df["Country"] != "United Kingdom"]
                .groupby("Country")["TotalPrice"]
                .sum()
                .sort_values(ascending=False)
                .head(8)
                .reset_index()
            )
            fig = px.bar(
                top_countries,
                x="TotalPrice",
                y="Country",
                orientation="h",
                title="Top Markets (excl. UK)",
                color_discrete_sequence=[PALETTE[1]],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Revenue (£)")
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Sales Trends
# ---------------------------------------------------------------------------
elif page == "Sales Trends":
    st.title("Sales Trends")
    st.caption("Revenue patterns over time and by product.")

    if df is None:
        missing_file_message("online_retail_clean.csv")
    else:
        top_by_revenue = (
            df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10).reset_index()
        )
        top_by_quantity = (
            df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10).reset_index()
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                top_by_revenue.sort_values("TotalPrice"),
                x="TotalPrice",
                y="Description",
                orientation="h",
                title="Top 10 Products by Revenue",
                color_discrete_sequence=[PALETTE[2]],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                top_by_quantity.sort_values("Quantity"),
                x="Quantity",
                y="Description",
                orientation="h",
                title="Top 10 Products by Quantity Sold",
                color_discrete_sequence=[PALETTE[3]],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        order_value = df.groupby("InvoiceNo")["TotalPrice"].sum().clip(upper=1000)
        fig = px.histogram(
            order_value,
            nbins=40,
            title="Distribution of Order Value (capped at £1,000)",
            color_discrete_sequence=[PALETTE[4]],
        )
        fig.update_layout(xaxis_title="Order Value (£)", yaxis_title="Number of Orders", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Customer Segments
# ---------------------------------------------------------------------------
elif page == "Customer Segments":
    st.title("Customer Segments")
    st.caption("RFM scoring and K-Means clustering results, from `03_rfm_analysis.ipynb` and `04_clustering.ipynb`.")

    if rfm is None:
        missing_file_message("rfm_with_clusters.csv")
    else:
        col1, col2 = st.columns(2)

        with col1:
            segment_counts = rfm["Segment"].value_counts().reset_index()
            segment_counts.columns = ["Segment", "Customers"]
            fig = px.bar(
                segment_counts.sort_values("Customers"),
                x="Customers",
                y="Segment",
                orientation="h",
                title="Customers per RFM Segment",
                color_discrete_sequence=[PALETTE[0]],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            segment_revenue = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False).reset_index()
            fig = px.pie(
                segment_revenue,
                names="Segment",
                values="Monetary",
                title="Revenue Share by Segment",
                color_discrete_sequence=PALETTE,
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("K-Means Clusters")

        if "Cluster" in rfm.columns:
            cluster_profile = (
                rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
                .mean()
                .round(1)
            )
            cluster_profile["Customers"] = rfm["Cluster"].value_counts()
            st.dataframe(cluster_profile, use_container_width=True)

            fig = px.scatter(
                rfm,
                x="Frequency",
                y="Monetary",
                color=rfm["Cluster"].astype(str),
                title="Clusters: Frequency vs Monetary",
                color_discrete_sequence=PALETTE,
                opacity=0.6,
                log_y=True,
            )
            fig.update_layout(legend_title="Cluster")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cluster column not found — re-run `04_clustering.ipynb` to generate it.")


# ---------------------------------------------------------------------------
# Page: Market Basket
# ---------------------------------------------------------------------------
elif page == "Market Basket":
    st.title("Market Basket Analysis")
    st.caption("Association rules mined from France transactions, from `05_market_basket_analysis.ipynb`.")

    if rules.empty:
        missing_file_message("association_rules_france.csv")
    else:
        col1, col2 = st.columns(2)
        min_confidence = col1.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05)
        min_lift = col2.slider("Minimum lift", 1.0, float(max(rules["lift"].max(), 3)), 3.0, 0.5)

        filtered = rules[(rules["confidence"] >= min_confidence) & (rules["lift"] >= min_lift)]
        filtered = filtered.sort_values("lift", ascending=False)

        st.write(f"**{len(filtered)}** rules match the current thresholds.")
        st.dataframe(
            filtered[["antecedents", "consequents", "support", "confidence", "lift"]].head(30),
            use_container_width=True,
        )

        if not filtered.empty:
            top20 = filtered.head(20).copy()
            top20["rule"] = top20["antecedents"] + " → " + top20["consequents"]
            fig = px.bar(
                top20.sort_values("lift"),
                x="lift",
                y="rule",
                orientation="h",
                title="Top Association Rules by Lift",
                color_discrete_sequence=[PALETTE[5]],
            )
            fig.update_layout(yaxis_title="", height=600)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Retention
# ---------------------------------------------------------------------------
elif page == "Retention":
    st.title("Customer Retention")
    st.caption("Cohort retention heatmap, from `06_cohort_analysis.ipynb`.")

    if cohort is None:
        missing_file_message("cohort_retention.csv")
    else:
        fig = go.Figure(
            data=go.Heatmap(
                z=cohort.values,
                x=[str(c) for c in cohort.columns],
                y=cohort.index.astype(str),
                colorscale="YlGnBu",
                colorbar=dict(title="Retention %"),
                text=cohort.round(1).values,
                texttemplate="%{text}",
                hovertemplate="Cohort: %{y}<br>Month: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title="Retention by Cohort (%)",
            xaxis_title="Months Since First Purchase",
            yaxis_title="Cohort (first purchase month)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        average_retention = cohort.mean(axis=0).reset_index()
        average_retention.columns = ["CohortIndex", "AverageRetention"]
        fig = px.line(
            average_retention,
            x="CohortIndex",
            y="AverageRetention",
            markers=True,
            title="Average Retention Curve (all cohorts)",
            color_discrete_sequence=[PALETTE[6 % len(PALETTE)]],
        )
        fig.update_layout(xaxis_title="Months Since First Purchase", yaxis_title="Average Retention %")
        st.plotly_chart(fig, use_container_width=True)
