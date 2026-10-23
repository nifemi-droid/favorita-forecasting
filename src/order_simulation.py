import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("Simulating Order Decisions...")
df = pd.read_parquet("data/daily_sales.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Focus on Store 1, Grocery I
store = 1
family = "GROCERY I"
store_df = df[(df["store_nbr"] == store) & (df["family"] == family)].set_index("date")

train = store_df[: "2017-08-01"]["total_sales"]
test = store_df["2017-08-01" : "2017-08-15"]["total_sales"]

# Train ETS
model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7)
fitted_model = model.fit()
forecast = fitted_model.forecast(len(test))

# --- THE BUSINESS ASSUMPTIONS ---
# Let's say we sell Grocery I for R$ 10.
# If we under-stock and run out, we lose the sale (R$ 10 lost).
# If we over-stock, we throw it away (R$ 10 lost).
# Let's assume we order exactly what the model predicts.
ORDERED = forecast.values
ACTUAL = test.values

# Calculate lost sales (stockouts) and wasted inventory (overstock)
lost_sales = np.where(ACTUAL > ORDERED, ACTUAL - ORDERED, 0).sum()
wasted_inventory = np.where(ORDERED > ACTUAL, ORDERED - ACTUAL, 0).sum()

print(f"\n--- Simulation Results (Store {store}, {family}) ---")
print(f"Total Actual Demand: {ACTUAL.sum():,.0f} units")
print(f"Total Ordered:       {ORDERED.sum():,.0f} units")
print(f"\nLost Sales (Stockouts): {lost_sales:,.0f} units")
print(f"Wasted Inventory:       {wasted_inventory:,.0f} units")

# If a unit costs R$ 10, we can calculate the financial impact
COST_PER_UNIT = 10
print(f"\nEstimated Financial Impact: R$ {((lost_sales + wasted_inventory) * COST_PER_UNIT):,.2f} lost due to imperfect forecasting.")