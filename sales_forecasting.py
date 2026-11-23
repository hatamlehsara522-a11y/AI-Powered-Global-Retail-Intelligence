import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================
# 1. Load Data
# =====================================

print("Loading cleaned data...")

df = pd.read_csv("cleaned_orders.csv")

df["order_date"] = pd.to_datetime(df["order_date"])

print("Data loaded:", df.shape)

# =====================================
# 2. Prepare Monthly Sales
# =====================================

df["month"] = df["order_date"].dt.to_period("M")

monthly_sales = (
    df.groupby("month")["sales"]
    .sum()
    .reset_index()
)

monthly_sales["month"] = monthly_sales["month"].dt.to_timestamp()

monthly_sales = monthly_sales.sort_values("month").reset_index(drop=True)

print("\nMonthly sales data:")
print(monthly_sales.tail())

# =====================================
# 3. Create Features
# =====================================

monthly_sales["month_number"] = monthly_sales["month"].dt.month
monthly_sales["year"] = monthly_sales["month"].dt.year

for lag in [1, 2, 3, 6, 12]:
    monthly_sales[f"sales_lag_{lag}"] = monthly_sales["sales"].shift(lag)

model_data = monthly_sales.dropna().copy()

features = [
    "month_number",
    "year",
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_3",
    "sales_lag_6",
    "sales_lag_12",
]

X = model_data[features]
y = model_data["sales"]

# =====================================
# 4. Split Data Chronologically
# =====================================

split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# =====================================
# 5. Train Model
# =====================================

print("\nTraining Machine Learning model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# 6. Evaluate Model
# =====================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

actual_total = y_test.sum()

if actual_total != 0:
    wape = np.abs(y_test - predictions).sum() / np.abs(actual_total) * 100
else:
    wape = np.nan

print("\n========== MODEL RESULTS ==========")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"WAPE: {wape:.2f}%")

# =====================================
# 7. Save Test Predictions
# =====================================

results = model_data.iloc[split_index:][["month", "sales"]].copy()

results["predicted_sales"] = predictions
results = results.rename(columns={"sales": "actual_sales"})

results.to_csv("sales_forecast_evaluation.csv", index=False)

# =====================================
# 8. Forecast Next 6 Months
# =====================================

history = monthly_sales["sales"].tolist()
last_month = monthly_sales["month"].max()

forecast_rows = []

for i in range(1, 7):
    future_month = last_month + pd.DateOffset(months=i)

    row = {
        "month_number": future_month.month,
        "year": future_month.year,
        "sales_lag_1": history[-1],
        "sales_lag_2": history[-2],
        "sales_lag_3": history[-3],
        "sales_lag_6": history[-6],
        "sales_lag_12": history[-12],
    }

    future_sales = max(0, float(model.predict(pd.DataFrame([row]))[0]))

    forecast_rows.append({
        "month": future_month,
        "forecast_sales": round(future_sales, 2)
    })

    history.append(future_sales)

forecast_df = pd.DataFrame(forecast_rows)

forecast_df.to_csv("future_sales_forecast.csv", index=False)

# =====================================
# 9. Save Trained Model
# =====================================

import joblib

joblib.dump(model, "sales_forecasting_model.pkl")

print("\n========== FORECAST ==========")
print(forecast_df.to_string(index=False))

print("\nFiles saved successfully:")
print("- sales_forecast_evaluation.csv")
print("- future_sales_forecast.csv")
print("- sales_forecasting_model.pkl")