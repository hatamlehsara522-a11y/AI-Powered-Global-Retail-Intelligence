import pyodbc
import pandas as pd
import numpy as np
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# SQL Server connection
connection = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=(localdb)\\MSSQLLocalDB;"
    "DATABASE=superstore2016;"
    "Trusted_Connection=yes;"
)

cursor = connection.cursor()
cursor.fast_executemany = True


def insert_dataframe(csv_name, table_name, columns, create_sql, date_columns=None):
    """Create a result table if needed, then replace its contents with CSV data."""

    file_path = BASE_DIR / csv_name

    if not file_path.exists():
        print(f"SKIPPED: {csv_name} was not found.")
        return

    df = pd.read_csv(file_path)

    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"SKIPPED: {csv_name} is missing columns: {missing}")
        return

    df = df[columns].copy()

    # Convert date columns
    for col in date_columns or []:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert pandas/numpy values to Python values for SQL Server
    rows = []
    for row in df.itertuples(index=False, name=None):
        clean_row = []

        for col, value in zip(columns, row):
            if pd.isna(value):
                clean_row.append(None)
            elif col in (date_columns or []):
                clean_row.append(value.to_pydatetime())
            elif isinstance(value, np.generic):
                clean_row.append(value.item())
            else:
                clean_row.append(value)

        rows.append(tuple(clean_row))

    cursor.execute(create_sql)

    # Refresh only this AI result table
    cursor.execute(f"DELETE FROM dbo.[{table_name}]")

    placeholders = ", ".join("?" for _ in columns)
    column_names = ", ".join(f"[{col}]" for col in columns)

    sql = (
        f"INSERT INTO dbo.[{table_name}] ({column_names}) "
        f"VALUES ({placeholders})"
    )

    if rows:
        cursor.executemany(sql, rows)

    connection.commit()
    print(f"SUCCESS: dbo.{table_name} — {len(rows)} rows imported.")


try:
    # 1. Future sales forecasts
    insert_dataframe(
        "future_sales_forecast.csv",
        "sales_forecasts",
        ["month", "forecast_sales"],
        """
        IF OBJECT_ID('dbo.sales_forecasts', 'U') IS NULL
        CREATE TABLE dbo.sales_forecasts (
            month DATE,
            forecast_sales FLOAT
        );
        """,
        date_columns=["month"]
    )

    # 2. Forecast model evaluation
    insert_dataframe(
        "sales_forecast_evaluation.csv",
        "sales_forecast_evaluation",
        ["month", "actual_sales", "predicted_sales"],
        """
        IF OBJECT_ID('dbo.sales_forecast_evaluation', 'U') IS NULL
        CREATE TABLE dbo.sales_forecast_evaluation (
            month DATE,
            actual_sales FLOAT,
            predicted_sales FLOAT
        );
        """,
        date_columns=["month"]
    )

    # 3. Customer segmentation
    insert_dataframe(
        "customer_segments.csv",
        "customer_segments",
        [
            "customer_id",
            "total_sales",
            "total_profit",
            "total_orders",
            "total_quantity",
            "avg_discount",
            "last_purchase",
            "days_since_last_purchase",
            "customer_segment"
        ],
        """
        IF OBJECT_ID('dbo.customer_segments', 'U') IS NULL
        CREATE TABLE dbo.customer_segments (
            customer_id NVARCHAR(100),
            total_sales FLOAT,
            total_profit FLOAT,
            total_orders INT,
            total_quantity INT,
            avg_discount FLOAT,
            last_purchase DATE,
            days_since_last_purchase INT,
            customer_segment INT
        );
        """,
        date_columns=["last_purchase"]
    )

    # 4. Unusual sales transactions
    insert_dataframe(
        "detected_anomalies.csv",
        "sales_anomalies",
        [
            "row_id",
            "order_id",
            "customer_id",
            "order_date",
            "sales",
            "profit",
            "quantity",
            "discount",
            "shipping_cost",
            "anomaly_label",
            "anomaly_status"
        ],
        """
        IF OBJECT_ID('dbo.sales_anomalies', 'U') IS NULL
        CREATE TABLE dbo.sales_anomalies (
            row_id INT,
            order_id NVARCHAR(100),
            customer_id NVARCHAR(100),
            order_date DATE,
            sales FLOAT,
            profit FLOAT,
            quantity INT,
            discount FLOAT,
            shipping_cost FLOAT,
            anomaly_label INT,
            anomaly_status NVARCHAR(30)
        );
        """,
        date_columns=["order_date"]
    )

    print("\nAI results export finished.")

except Exception as error:
    connection.rollback()
    print("\nERROR:", error)
    print("The current transaction was rolled back.")

finally:
    cursor.close()
    connection.close()