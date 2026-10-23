import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("Loading data...")
df = pd.read_parquet("data/daily_sales.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# We only need the features for drift, not the target
# Let's use Store 1, Grocery I as a sample to see if the behavior changed
store_df = df[(df["store_nbr"] == 1) & (df["family"] == "GROCERY I")].copy()

# 1. Reference Data (Early 2017)
reference_data = store_df[store_df["date"] < "2017-05-01"][["total_sales", "promo_count"]]

# 2. Current Data (August 2017)
current_data = store_df[store_df["date"] >= "2017-08-01"][["total_sales", "promo_count"]]

print(f"Reference rows: {len(reference_data)}")
print(f"Current rows: {len(current_data)}")

print("Generating Evidently Drift Report...")
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Save as HTML
report.save_html("docs/favorita_drift_report.html")
print("Success! Open docs/favorita_drift_report.html in your browser to see the drift.")