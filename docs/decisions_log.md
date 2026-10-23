# Decisions Log: Favorita Forecasting

## Decision 1: Data Aggregation & Compression
**Context:** The raw Kaggle train.csv is 116MB and has millions of rows, which is too heavy for local modeling. 
**Decision:** We aggregated the data to `date`, `store_nbr`, and `family` levels using DuckDB in Colab, and exported it as a compressed ZSTD Parquet file.
**Results:** The data is now 3,000,888 rows and loads in under 1 second. 

## Decision 2: Handling the "Zero-Sales Trap"
**Context:** 31.3% of our daily data has zero sales. 
**Decision:** We will NOT drop these zeros (that would cause massive data leakage and bias). We will treat them as real signal (e.g., store closed), but we will evaluate our models using WAPE, which handles zeros gracefully.

## Decision 3: The Baseline Benchmark
**Context:** Before training complex models, we built a "Naive Baseline" that predicts sales based on the average of the last week. 
**Results:** The baseline achieves a WAPE of **30.88%**. 
**Decision:** Any machine learning model we build MUST beat this 30.88% WAPE. If it doesn't, it means the ML model is not worth the engineering complexity.

## Decision 4: Model Selection (The Surprising Winner)
**Context:** We tested two approaches: a LightGBM gradient-boosted tree model and a classical Exponential Smoothing (ETS) model.
**Results:** 
- Naive Baseline: 30.88% WAPE
- LightGBM (ML): **55.56% WAPE** (Failed)
- ETS (Classical): **28.66% WAPE** (Won)
**Decision:** We chose to deploy the ETS model. 
**Why:** The LightGBM model failed due to the "fake lag" problem in the test set. The ETS model handled seasonality natively and beat the baseline. **This proves the AI PM principle: A simpler model with correct assumptions beats a complex model with broken data.**

## Decision 5: The Cost of Imperfect Forecasting
**Context:** We simulated ordering exactly what the ETS model predicted for Store 1, Grocery I, over a 15-day period.
**Results:** We wasted 7,366 units (overstock) and missed 2,328 units (stockout). Estimated financial loss: **R$ 96,943**.
**PM Decision:** The current model assumes symmetric costs. In production, we would adjust the order quantity based on a "safety stock" multiplier (e.g., order 90% of the forecast if spoilage is high) and measure the new trade-off.