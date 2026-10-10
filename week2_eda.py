from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

if not RAW.exists():
    raise FileNotFoundError("Run python week1_setup.py first.")

df = pd.read_csv(RAW)
print("Raw shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Blank TotalCharges:", df["TotalCharges"].astype(str).str.strip().eq("").sum())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
for col in service_cols:
    if col in df.columns:
        df[col] = df[col].replace("No internet service", "No")

df["CLV"] = df["MonthlyCharges"] * df["tenure"]
out_csv = PROCESSED / "telco_customer_churn_clean.csv"
df.to_csv(out_csv, index=False)
print("Clean data saved to:", out_csv)

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=140, bbox_inches="tight")
    plt.close()

# 1. Churn distribution
df["Churn"].value_counts().reindex(["No", "Yes"]).plot(kind="bar", title="Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Number of customers")
save_plot("01_churn_distribution.png")

# 2. Contract type
pd.crosstab(df["Contract"], df["Churn"], normalize="index").mul(100).plot(
    kind="bar", title="Contract Type vs Churn", ylabel="Percent", xlabel="Contract"
)
save_plot("02_contract_vs_churn.png")

# 3. Tenure buckets
df["tenure_group"] = pd.cut(
    df["tenure"], bins=[-1, 12, 24, 36, 48, 60, 72],
    labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
)
pd.crosstab(df["tenure_group"], df["Churn"], normalize="index").mul(100).plot(
    kind="bar", title="Tenure Group vs Churn", ylabel="Percent", xlabel="Tenure (months)"
)
save_plot("03_tenure_vs_churn.png")

# 4. Payment method
pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index").mul(100).plot(
    kind="bar", title="Payment Method vs Churn", ylabel="Percent", xlabel="Payment method"
)
save_plot("04_payment_method_vs_churn.png")

# 5. Monthly charges
df.boxplot(column="MonthlyCharges", by="Churn")
plt.title("Monthly Charges by Churn")
plt.suptitle("")
plt.xlabel("Churn")
save_plot("05_monthly_charges_vs_churn.png")

# 6. Total charges
df.boxplot(column="TotalCharges", by="Churn")
plt.title("Total Charges by Churn")
plt.suptitle("")
plt.xlabel("Churn")
save_plot("06_total_charges_vs_churn.png")

# 7. Internet service
pd.crosstab(df["InternetService"], df["Churn"], normalize="index").mul(100).plot(
    kind="bar", title="Internet Service vs Churn", ylabel="Percent", xlabel="Internet service"
)
save_plot("07_internet_service_vs_churn.png")

# 8. Tech support
pd.crosstab(df["TechSupport"], df["Churn"], normalize="index").mul(100).plot(
    kind="bar", title="Tech Support vs Churn", ylabel="Percent", xlabel="Tech support"
)
save_plot("08_techsupport_vs_churn.png")

# 9. Online security
pd.crosstab(df["OnlineSecurity"], df["Churn"], normalize="index").mul(100).plot(
    kind="bar", title="Online Security vs Churn", ylabel="Percent", xlabel="Online security"
)
save_plot("09_onlinesecurity_vs_churn.png")

print("Graphs saved to:", FIGURES)
