import pandas as pd

# 1. Load the Parquet file
print("Loading data...")
df = pd.read_parquet("data/daily_sales.parquet")

print(f"Total rows: {len(df):,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# 2. Check for Zero-Sales Days (The PM Trap)
zero_sales = (df['total_sales'] == 0).mean()
print(f"Percentage of zero-sales days: {zero_sales:.2%}")

# 3. Look at the top 5 product families by total sales
print("\nTop 5 Product Families:")
top_families = df.groupby('family')['total_sales'].sum().sort_values(ascending=False).head(5)
print(top_families)