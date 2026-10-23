import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("Loading data...")
df = pd.read_parquet("data/daily_sales.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# We will test on ONE store and ONE family (Store 1, Grocery I) for the demo
# ETS is slow, so doing this for all 3 million rows would take hours
store = 1
family = "GROCERY I"

store_df = df[(df["store_nbr"] == store) & (df["family"] == family)].set_index("date")

# Split: Train on 2016-2017, Test on Aug 2017
train = store_df[: "2017-08-01"]["total_sales"]
test = store_df["2017-08-01" : "2017-08-15"]["total_sales"]

print(f"Training on {len(train)} days of data...")

# Train an Exponential Smoothing Model (captures trend and weekly seasonality)
model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7)
fitted_model = model.fit()

# Forecast the next 15 days
predictions = fitted_model.forecast(len(test))

# Evaluate
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

score = wape(test.values, predictions.values)
print(f"\n--- ETS Model (Store {store}, {family}) ---")
print(f"WAPE: {score:.2%}")

# Show the first 5 predictions vs actuals
results = pd.DataFrame({"Actual": test.values, "Predicted": predictions.values})
print("\nFirst 5 days of the forecast:")
print(results.head())