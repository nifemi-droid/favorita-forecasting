# Problem Brief: Favorita Demand Forecasting

## 1. The Problem
We need to predict how many units of each product family (e.g., "Beverages", "Produce") each store will sell over the next 2 to 4 weeks. 

## 2. The Action
The inventory team uses these forecasts to place stock orders. 

## 3. The Costs
* **Stockout (Under-ordering):** We run out of stock. The customer leaves empty-handed, and we lose the sale plus brand trust.
* **Overstock (Over-ordering):** We buy too much, it spoils (especially for perishables), and we waste capital on storage.

## 4. The Success Metric
We will use **WAPE (Weighted Absolute Percentage Error)**. It is better than RMSE for retail because it handles zero-sales days gracefully. 
We will also measure **Bias** (are we consistently under-ordering or over-ordering?).

## 5. The Data
Kaggle Store Sales Time Series Dataset (Corporación Favorita). We have aggregated 3 million rows into a daily `daily_sales.parquet` file containing store, family, date, sales, and promo count.