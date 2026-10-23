# Decisions Log: Favorita Forecasting

## Decision 1: Data Aggregation & Compression
**Context:** The raw Kaggle train.csv is 116MB and has millions of rows, which is too heavy for local modeling. 
**Decision:** We aggregated the data to `date`, `store_nbr`, and `family` levels using DuckDB in Colab, and exported it as a compressed ZSTD Parquet file.
**Results:** The data is now 3,000,888 rows and loads in under 1 second. 

## Decision 2: Handling the "Zero-Sales Trap"
**Context:** 31.3% of our daily data has zero sales. 
**Decision:** We will NOT drop these zeros (that would cause massive data leakage and bias). We will treat them as real signal (e.g., store closed), but we will evaluate our models using WAPE, which handles zeros gracefully.