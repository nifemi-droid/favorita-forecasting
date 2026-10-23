import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

print("Loading data...")
df = pd.read_parquet("data/daily_sales.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# 1. Time-Based Split
# Train on everything before August 1, 2017
# Test on the last 2 weeks (August 1 - August 15, 2017)
train_df = df[df["date"] < "2017-08-01"].copy()
test_df = df[(df["date"] >= "2017-08-01") & (df["date"] <= "2017-08-15")].copy()

print(f"Train rows: {len(train_df):,}")
print(f"Test rows:  {len(test_df):,}")

# 2. Baseline: "Last week's sales"
# The simplest guess: whatever we sold on this day LAST week, we will sell this week.
# We shift the data by 7 days to get last week's sales.
train_df["lag_7"] = train_df.groupby(["store_nbr", "family"])["total_sales"].shift(7)

# For the test set, we just use the average of the last 7 days of train as our guess
# (Because we don't have the target for the test set)
last_week_avg = train_df.groupby(["store_nbr", "family"])["total_sales"].mean().reset_index()
last_week_avg.rename(columns={"total_sales": "baseline_prediction"}, inplace=True)

# Merge the baseline predictions into the test set
test_df = test_df.merge(last_week_avg, on=["store_nbr", "family"], how="left")

# 3. Evaluate the Baseline (WAPE)
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

# Fill any NaNs with 0 (if a store/family combo wasn't in the train set)
test_df["baseline_prediction"] = test_df["baseline_prediction"].fillna(0)

score = wape(test_df["total_sales"], test_df["baseline_prediction"])
print(f"\n--- Baseline Model (Last Week Average) ---")
print(f"WAPE: {score:.2%}")
print(f"(This means our baseline is off by {score:.2%} on average)")