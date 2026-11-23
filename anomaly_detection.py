import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest

print("Loading cleaned data...")

# 1. Load data
df = pd.read_csv("cleaned_orders.csv")

print("Dataset shape:", df.shape)

# 2. Select features for anomaly detection
features = [
    "sales",
    "profit",
    "quantity",
    "discount",
    "shipping_cost"
]

X = df[features].copy()

# Handle missing and infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# 3. Train Isolation Forest
print("\nTraining anomaly detection model...")

model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42
)

df["anomaly_label"] = model.fit_predict(X)

# -1 means unusual
#  1 means normal
df["anomaly_status"] = df["anomaly_label"].map({
    -1: "Anomaly",
    1: "Normal"
})

# 4. Calculate anomaly statistics
anomaly_count = (df["anomaly_label"] == -1).sum()
normal_count = (df["anomaly_label"] == 1).sum()

print("\n========== ANOMALY RESULTS ==========")
print("Total records:", len(df))
print("Normal records:", normal_count)
print("Anomalous records:", anomaly_count)
print(f"Anomaly percentage: {anomaly_count / len(df) * 100:.2f}%")

# 5. Show unusual records
anomalies = df[df["anomaly_label"] == -1].copy()

anomalies = anomalies.sort_values(
    by="profit",
    ascending=True
)

print("\nSample unusual transactions:")
print(
    anomalies[
        ["order_id", "sales", "profit", "quantity",
         "discount", "shipping_cost", "anomaly_status"]
    ].head(10).to_string(index=False)
)

# 6. Save results
df.to_csv("sales_anomaly_results.csv", index=False)
anomalies.to_csv("detected_anomalies.csv", index=False)

print("\nFiles saved successfully:")
print("- sales_anomaly_results.csv")
print("- detected_anomalies.csv")