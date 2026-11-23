import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Loading data...")

# 1. Load cleaned data
df = pd.read_csv("cleaned_orders.csv")

# 2. Prepare customer-level features
df["order_date"] = pd.to_datetime(df["order_date"])

customers = df.groupby("customer_id").agg(
    total_sales=("sales", "sum"),
    total_profit=("profit", "sum"),
    total_orders=("order_id", "nunique"),
    total_quantity=("quantity", "sum"),
    avg_discount=("discount", "mean"),
    last_purchase=("order_date", "max")
).reset_index()

# 3. Calculate recency
reference_date = df["order_date"].max() + pd.Timedelta(days=1)

customers["days_since_last_purchase"] = (
    reference_date - customers["last_purchase"]
).dt.days

# 4. Select features
features = [
    "total_sales",
    "total_profit",
    "total_orders",
    "total_quantity",
    "avg_discount",
    "days_since_last_purchase"
]

X = customers[features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train K-Means
print("Training K-Means model...")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

customers["customer_segment"] = kmeans.fit_predict(X_scaled)

# 7. Evaluate clustering
score = silhouette_score(
    X_scaled,
    customers["customer_segment"]
)

print(f"\nSilhouette Score: {score:.3f}")

# 8. Create segment summary
segment_summary = customers.groupby("customer_segment").agg(
    customer_count=("customer_id", "count"),
    avg_sales=("total_sales", "mean"),
    avg_profit=("total_profit", "mean"),
    avg_orders=("total_orders", "mean"),
    avg_discount=("avg_discount", "mean"),
    avg_recency=("days_since_last_purchase", "mean")
).round(2)

print("\nCustomer Segment Summary:")
print(segment_summary)

# 9. Save results
customers.to_csv("customer_segments.csv", index=False)
segment_summary.to_csv("customer_segment_summary.csv")

print("\nFiles saved successfully:")
print("- customer_segments.csv")
print("- customer_segment_summary.csv")