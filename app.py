import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit.components.v1 as components

# 1. Page Config
st.set_page_config(page_title="Favorita Forecast Command Center", page_icon="🛒", layout="wide")
st.title("🛒 Favorita Demand Forecasting Command Center")
st.markdown("An end-to-end dashboard for inventory planning, ROI simulation, and model monitoring.")

# 2. Load Data (Cached)
@st.cache_data
def load_data():
    df = pd.read_parquet("data/daily_sales.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

df = load_data()

# Get unique stores and families for the dropdown
stores = sorted(df["store_nbr"].unique())
families = sorted(df["family"].unique())

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📈 Forecast Explorer", "💰 Business Impact", "🩺 Data Drift"])

# ==========================================
# TAB 1: FORECAST EXPLORER
# ==========================================
with tab1:
    st.header("Historical Sales vs. Forecast")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_store = st.selectbox("Select Store", stores, index=0)
    with col2:
        selected_family = st.selectbox("Select Product Family", families, index=0)

    # Filter Data
    store_df = df[(df["store_nbr"] == selected_store) & (df["family"] == selected_family)].set_index("date")
    
    # Train on 2017 data, forecast Aug 2017
    train = store_df[: "2017-08-01"]["total_sales"]
    test = store_df["2017-08-01" : "2017-08-15"]["total_sales"]

    if len(train) > 30:
        # Train ETS
        model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(len(test))

        # Plotly Chart
        fig = go.Figure()
        
        # Plot Historical (last 30 days of train)
        fig.add_trace(go.Scatter(x=train.index[-30:], y=train.values[-30:], mode='lines', name='Historical Sales', line=dict(color='blue')))
        
        # Plot Actual Test
        fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines', name='Actual Demand', line=dict(color='green')))
        
        # Plot Forecast
        fig.add_trace(go.Scatter(x=test.index, y=forecast.values, mode='lines', name='ETS Forecast', line=dict(color='red', dash='dash')))

        fig.update_layout(title=f"Store {selected_store} - {selected_family}", xaxis_title="Date", yaxis_title="Units Sold")
        st.plotly_chart(fig, use_container_width=True)

        # Show metrics
        def wape(y_true, y_pred):
            return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
        
        score = wape(test.values, forecast.values)
        st.metric(label="Forecast Accuracy (WAPE)", value=f"{score:.2%}")
    else:
        st.warning("Not enough data for this store/family combo.")

# ==========================================
# TAB 2: BUSINESS IMPACT
# ==========================================
with tab2:
    st.header("Financial Impact of Forecasting Errors")
    st.markdown("Adjust the business costs to see how much money is lost due to imperfect forecasting.")
    
    col1, col2 = st.columns(2)
    with col1:
        cost_stockout = st.number_input("Cost of Stockout (Lost Sale) per unit", value=10.0, step=1.0)
    with col2:
        cost_overstock = st.number_input("Cost of Overstock (Waste) per unit", value=5.0, step=1.0)
        
    if len(train) > 30:
        ORDERED = forecast.values
        ACTUAL = test.values

        # Calculate Lost Sales and Wasted Inventory
        lost_sales = np.where(ACTUAL > ORDERED, ACTUAL - ORDERED, 0).sum()
        wasted_inventory = np.where(ORDERED > ACTUAL, ORDERED - ACTUAL, 0).sum()

        # Calculate Financial Impact
        financial_loss_stockout = lost_sales * cost_stockout
        financial_loss_overstock = wasted_inventory * cost_overstock
        total_loss = financial_loss_stockout + financial_loss_overstock
        
        # Display Metrics
        st.subheader(f"Results for Store {selected_store} - {selected_family}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Lost Sales (Units)", f"{lost_sales:,.0f}", delta=f"-R$ {financial_loss_stockout:,.2f}", delta_color="inverse")
        m2.metric("Wasted Inventory (Units)", f"{wasted_inventory:,.0f}", delta=f"-R$ {financial_loss_overstock:,.2f}", delta_color="inverse")
        m3.metric("Total Financial Loss", f"R$ {total_loss:,.2f}")

        # Create a bar chart
        loss_df = pd.DataFrame({
            "Category": ["Lost Sales (Stockout)", "Wasted Inventory (Overstock)"],
            "Cost (R$)": [financial_loss_stockout, financial_loss_overstock]
        })
        fig_bar = px.bar(loss_df, x="Category", y="Cost (R$)", color="Category", title="Breakdown of Financial Losses")
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 3: DATA DRIFT
# ==========================================
with tab3:
    st.header("Data Drift & Model Health")
    st.markdown("This report compares the data the model was trained on (Early 2017) vs. the data it is currently seeing (Late 2017).")
    
    drift_file = "docs/favorita_drift_report.html"
    if os.path.exists(drift_file):
        with open(drift_file, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, height=800, scrolling=True)
    else:
        st.warning("Drift report not found. Please run `python src/check_drift.py` first.")