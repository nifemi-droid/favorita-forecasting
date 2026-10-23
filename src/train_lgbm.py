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

# 2. Feature Engineering (ONLY on train to prevent leakage)
def create_features(df, is_train=True):
    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    
    if is_train:
        # Lags (Sales from N days ago)
        for lag in [1, 7, 14, 30, 365]:
            df[f"lag_{lag}"] = df.groupby(["store_nbr", "family"])["total_sales"].shift(lag)
        
        # Rolling Averages (The average over the last N days)
        for window in [7, 30]:
            df[f"rolling_{window}"] = df.groupby(["store_nbr", "family"])["total_sales"].shift(1).rolling(window).mean()
            
    return df

print("Creating features (this takes ~1 minute)...")
train_df = create_features(train_df, is_train=True)

# For the test set, we create a fixed "last 7 days" average from the train set to use as a feature
last_7_days = train_df.groupby(["store_nbr", "family"])["total_sales"].mean().reset_index()
last_7_days.rename(columns={"total_sales": "rolling_7"}, inplace=True)

test_df = create_features(test_df, is_train=False)
test_df = test_df.merge(last_7_days, on=["store_nbr", "family"], how="left")
test_df["rolling_30"] = test_df["rolling_7"] # Approximate for the test set

# Drop missing values (since lags create NaNs at the start of the series)
train_df = train_df.dropna()

# 3. Train LightGBM
features = ["day_of_week", "day_of_month", "month", "promo_count", "lag_1", "lag_7", "lag_14", "lag_30", "lag_365", "rolling_7", "rolling_30"]
target = "total_sales"

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print("Training LightGBM model...")
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# 4. Predict and Evaluate
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
    