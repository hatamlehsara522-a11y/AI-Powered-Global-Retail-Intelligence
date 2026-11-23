# AI-Powered Global Retail Intelligence Dashboard

An end-to-end retail intelligence project combining Machine Learning, Python, SQL Server, and Power BI to transform retail data into actionable business insights.

## Project Overview

This project develops an intelligent retail analytics solution using historical sales data. It integrates data preprocessing, machine learning models, SQL Server data management, and interactive Power BI dashboards.

The solution focuses on three main analytical capabilities:

- Sales Forecasting
- Customer Segmentation
- Sales Anomaly Detection

## Business Objectives

- Analyze historical retail sales and profitability.
- Forecast future sales using Machine Learning.
- Identify meaningful customer segments based on purchasing behavior.
- Detect unusual sales patterns that may require further investigation.
- Present analytical results through interactive dashboards.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Data processing and Machine Learning |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | Machine Learning models |
| SQL Server | Data storage and management |
| Power BI | Interactive dashboards |
| Joblib | Model serialization |

## Project Architecture

The project follows an integrated analytical workflow:

1. Retail dataset ingestion.
2. Data cleaning and preprocessing using Python.
3. Storage and management using SQL Server.
4. Machine Learning model development.
5. Exporting model outputs into SQL tables.
6. Connecting analytical results to Power BI.
7. Visualizing business insights through interactive dashboards.

## Machine Learning Implementation

### 1. Sales Forecasting

- Model: Random Forest Regressor.
- Uses historical monthly sales and lag-based features.
- Evaluates forecasting performance using MAE, RMSE, and WAPE.
- Generates forecasts for the next six months.

**Model Performance:**

| Metric | Result |
|---|---:|
| MAE | 88,681.52 |
| RMSE | 102,202.85 |
| WAPE | 21.07% |

### 2. Customer Segmentation

- Model: K-Means Clustering.
- Identifies four customer segments.
- Uses purchasing and profitability-related features.
- Evaluates clustering quality using the Silhouette
