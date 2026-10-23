# Favorita Demand Forecasting & Inventory Optimization

## The Problem
We need to forecast daily sales for thousands of store-product combinations to optimize inventory. Under-ordering causes stockouts (lost revenue), and over-ordering causes spoilage (wasted capital).

## The Solution
An end-to-end time-series forecasting pipeline. We tested a Gradient Boosted model (LightGBM) and a classical Exponential Smoothing (ETS) model against a Naive Baseline.

## Key Results (Store 1, Grocery I)
*   **Naive Baseline:** 30.88% WAPE
*   **LightGBM (ML):** 55.56% WAPE (Failed)
*   **ETS (Classical):** **28.66% WAPE** (Won)
*   **Business Impact:** Even with the winning model, forecasting errors resulted in R$ 96,943 in combined lost sales and wasted inventory over a 15-day simulation.

## The AI PM Insights
1. **The "Fake Lag" Trap:** LightGBM failed because we couldn't provide it with real future lags in the test set. The simpler ETS model handled seasonality natively and beat the complex ML algorithm. *Simple > Complex when assumptions are wrong.*
2. **The Cost of Imperfection:** A 28% error rate translates to massive financial loss. In a production system, we would need to adjust our ordering strategy (safety stock) to minimize total cost, not just minimize WAPE.

## Repository Structure
- `data/` (Contains the compressed Parquet file)
- `docs/` (Problem brief, decisions log, Evidently drift report)
- `src/` (Exploration, Baselines, LightGBM, ETS, Simulation, Drift check)
- `api/` (FastAPI endpoint for real-time forecasting)
- `app.py` (Streamlit dashboard for Ops and Business teams)

## How to Run
1. Install requirements: `pip install pandas numpy scikit-learn lightgbm statsmodels fastapi uvicorn streamlit plotly evidently`
2. Run the Streamlit app: `streamlit run app.py`
3. Run the FastAPI server: `uvicorn api.main:app --reload`