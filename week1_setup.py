from pathlib import Path
import urllib.request
import pandas as pd

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
csv_path = RAW_DIR / "telco_customer_churn.csv"
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

if not csv_path.exists():
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, csv_path)

df = pd.read_csv(csv_path)
print("Dataset shape:", df.shape)
print(df.head())

if "Churn" not in df.columns:
    raise ValueError("The dataset does not contain a Churn column.")

print("Raw data saved to:", csv_path)
