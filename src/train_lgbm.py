import pandas as pd
import numpy as np
import lightgbm as lgb

print("Loading data and engineering features...")
df = pd.read_parquet("data/daily_sales.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["store_nbr", "family", "date"])

# 1. Time-Based Split
train_df = df[df["date"] < "2017-08-01"].copy()
test_df = df[(df["date"] >= "2017-08-01") & (df["date"] <= "2017-08-15")].copy()

# 2. Feature Engineering (Train only)
print("Creating features (this takes ~1 minute)...")

def create_features(df):
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    
    # Lags
    for lag in [1, 7, 14, 30, 365]:
        df[f"lag_{lag}"] = df.groupby(["store_nbr", "family"])["total_sales"].shift(lag)
    
    # Rolling Averages
    for window in [7, 30]:
        df[f"rolling_{window}"] = df.groupby(["store_nbr", "family"])["total_sales"].shift(1).rolling(window).mean()
        
    return df

train_df = create_features(train_df)

# 3. Handle the Test Set features
# We don't have lags for the future, so we need to create them.
# A common cheat is to use the last known rolling average for all future lags.
last_7_avg = train_df.groupby(["store_nbr", "family"])["total_sales"].mean().reset_index()
last_7_avg.rename(columns={"total_sales": "rolling_7"}, inplace=True)
last_7_avg["rolling_30"] = last_7_avg["rolling_7"]

test_df = test_df.merge(last_7_avg, on=["store_nbr", "family"], how="left")

# Add the lag columns for the test set with 0s (since we can't know the future)
for col in ["lag_1", "lag_7", "lag_14", "lag_30", "lag_365"]:
    test_df[col] = 0

# Fill any NaNs in test set with 0
test_df = test_df.fillna(0)

# 4. Train LightGBM
features = ["day_of_week", "day_of_month", "month", "promo_count", "lag_1", "lag_7", "lag_14", "lag_30", "lag_365", "rolling_7", "rolling_30"]
target = "total_sales"

# Drop NaNs from train (caused by the lags)
train_df = train_df.dropna()

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print("Training LightGBM model...")
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# 5. Predict and Evaluate
print("Making predictions...")
preds = model.predict(X_test)

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

score = wape(y_test, preds)

print(f"\n--- LightGBM Model ---")
print(f"WAPE: {score:.2%}")
print(f"(Our baseline was 30.88%)")

if score < 0.3088:
    print("🎉 SUCCESS! The ML model beat the baseline!")
else:
    print("❌ The ML model failed to beat the baseline.")