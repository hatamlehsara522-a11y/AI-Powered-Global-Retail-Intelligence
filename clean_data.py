import pyodbc
import pandas as pd

connection = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=(localdb)\\MSSQLLocalDB;"
    "DATABASE=superstore2016;"
    "Trusted_Connection=yes;"
)

print("Connected to SQL Server!")

try:
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM dbo.orders")

    columns = [column[0] for column in cursor.description]
    chunks = []
    total_rows = 0

    while True:
        batch = cursor.fetchmany(1000)

        if not batch:
            break

        chunks.extend(batch)
        total_rows += len(batch)

        if total_rows % 5000 == 0:
            print(f"Loaded {total_rows} rows")

    df = pd.DataFrame.from_records(chunks, columns=columns)

    print("\nOriginal shape:", df.shape)

    # Convert dates
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["ship_date"] = pd.to_datetime(df["ship_date"])

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    print("Duplicate rows:", duplicates)
    df = df.drop_duplicates()

    # Calculate delivery duration
    df["delivery_days"] = (
        df["ship_date"] - df["order_date"]
    ).dt.days

    # Calculate profit margin
    df["profit_margin"] = (
        df["profit"] / df["sales"].replace(0, float("nan"))
    )

    # Save cleaned data
    df.to_csv("cleaned_orders.csv", index=False)

    print("\nCleaned shape:", df.shape)
    print("Cleaned data saved successfully!")

finally:
    connection.close()