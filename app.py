import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
RAW_FILE = ROOT / "data" / "raw" / "telco_customer_churn.csv"
CLEAN_FILE = ROOT / "data" / "processed" / "telco_customer_churn_clean.csv"
FIGURES = ROOT / "outputs" / "figures"

st.set_page_config(page_title="Telco Churn", page_icon="📊", layout="wide")

@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if path.exists() else None

raw = load_csv(RAW_FILE)
clean = load_csv(CLEAN_FILE)

st.sidebar.title("Telco Churn")
page = st.sidebar.radio("Menu", [
    "Overview", "Raw Data", "Clean Data", "Data Quality", "Graphs", "Column Analysis"
])

if raw is None:
    st.title("Telco Customer Churn")
    st.warning("Raw dataset not found. First run: python week1_setup.py")
    st.stop()

if page == "Overview":
    st.title("Telco Customer Churn")
    st.write("A simple dashboard for exploring customer churn.")
    a, b, c = st.columns(3)
    a.metric("Customers", f"{len(raw):,}")
    b.metric("Columns", raw.shape[1])
    c.metric("Churn rate", f'{raw["Churn"].eq("Yes").mean() * 100:.1f}%')
    st.subheader("Project steps")
    st.markdown("1. Download the data\n2. Check and clean the data\n3. Explore churn patterns\n4. Use the findings to prepare for ML")
    st.info("Goal: understand which customer characteristics are associated with leaving the company.")

elif page == "Raw Data":
    st.title("Raw Data")
    st.caption("Original dataset before cleaning.")
    st.write(f"{len(raw):,} rows · {raw.shape[1]} columns")
    st.dataframe(raw, use_container_width=True, height=500)
    st.download_button("Download raw CSV", raw.to_csv(index=False).encode("utf-8"),
                       "telco_customer_churn_raw.csv", "text/csv")

elif page == "Clean Data":
    st.title("Clean Data")
    if clean is None:
        st.warning("Clean data not found. Run: python week2_eda.py")
    else:
        st.caption("Dataset after the Week 2 cleaning steps.")
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(clean):,}")
        b.metric("Columns", clean.shape[1])
        c.metric("Missing cells", f"{int(clean.isna().sum().sum()):,}")
        st.dataframe(clean, use_container_width=True, height=500)
        st.download_button("Download clean CSV", clean.to_csv(index=False).encode("utf-8"),
                           "telco_customer_churn_clean.csv", "text/csv")

elif page == "Data Quality":
    st.title("Data Quality")
    blank_count = raw["TotalCharges"].astype(str).str.strip().eq("").sum()
    duplicate_rows = int(raw.duplicated().sum())
    duplicate_ids = int(raw["customerID"].duplicated().sum()) if "customerID" in raw else 0
    a, b, c = st.columns(3)
    a.metric("Blank TotalCharges", int(blank_count))
    b.metric("Duplicate rows", duplicate_rows)
    c.metric("Duplicate customer IDs", duplicate_ids)
    st.subheader("How we cleaned the data")
    st.markdown("""
    - Converted `TotalCharges` to numeric values.
    - Handled invalid or blank values using the project's documented fallback.
    - Standardized selected service columns by changing `No internet service` to `No`.
    """)
    st.code('df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")')

elif page == "Graphs":
    st.title("Graphs")
    graph_names = [
        ("01_churn_distribution.png", "1. Churn distribution"),
        ("02_contract_vs_churn.png", "2. Contract type and churn"),
        ("03_tenure_vs_churn.png", "3. Tenure and churn"),
        ("04_payment_method_vs_churn.png", "4. Payment method and churn"),
        ("05_monthly_charges_vs_churn.png", "5. Monthly charges and churn"),
        ("06_total_charges_vs_churn.png", "6. Total charges and churn"),
        ("07_internet_service_vs_churn.png", "7. Internet service and churn"),
        ("08_techsupport_vs_churn.png", "8. Tech support and churn"),
        ("09_onlinesecurity_vs_churn.png", "9. Online security and churn"),
    ]
    shown = 0
    for filename, title in graph_names:
        image_path = FIGURES / filename
        if image_path.exists():
            st.subheader(title)
            st.image(str(image_path), use_container_width=True)
            shown += 1
    if shown == 0:
        st.warning("No graphs found yet. Run: python week2_eda.py")


elif page == "Column Analysis":
    st.title("Analysis Between Two Columns")
    st.write(
        "Choose any two columns to explore how they relate. "
        "The dashboard automatically selects a suitable analysis based on their data types."
    )

    df = clean if clean is not None else raw
    if clean is None:
        st.info("Showing raw data because cleaned data was not found. Run `python week2_eda.py` for cleaned data.")

    available_columns = [c for c in df.columns if c != "customerID"]
    left, right = st.columns(2)
    col_x = left.selectbox("Select first column (X)", available_columns, index=0)
    default_y_index = available_columns.index("Churn") if "Churn" in available_columns else min(1, len(available_columns) - 1)
    col_y = right.selectbox("Select second column (Y)", available_columns, index=default_y_index)

    if col_x == col_y:
        st.warning("Choose two different columns to compare.")
    else:
        pair = df[[col_x, col_y]].copy()
        pair = pair.dropna()
        st.caption(f"Comparing **{col_x}** with **{col_y}** · {len(pair):,} complete rows")

        numeric_x = pd.api.types.is_numeric_dtype(pair[col_x])
        numeric_y = pd.api.types.is_numeric_dtype(pair[col_y])

        if numeric_x and numeric_y:
            st.subheader("Numeric vs numeric")
            corr = pair[col_x].corr(pair[col_y])
            a, b, c = st.columns(3)
            a.metric("Correlation (Pearson)", "Not available" if pd.isna(corr) else f"{corr:.3f}")
            b.metric(f"{col_x} average", f"{pair[col_x].mean():,.2f}")
            c.metric(f"{col_y} average", f"{pair[col_y].mean():,.2f}")
            st.scatter_chart(pair, x=col_x, y=col_y)
            st.caption("Correlation ranges from -1 to +1. It measures a linear relationship, not causation.")

        elif numeric_x or numeric_y:
            st.subheader("Numeric vs category")
            numeric_col = col_x if numeric_x else col_y
            category_col = col_y if numeric_x else col_x
            summary = pair.groupby(category_col, dropna=False)[numeric_col].agg(
                Count="count", Mean="mean", Median="median", Minimum="min", Maximum="max"
            ).round(2).reset_index()
            st.dataframe(summary, use_container_width=True)
            chart_data = summary.set_index(category_col)[["Mean"]]
            st.bar_chart(chart_data)
            st.caption(f"The chart compares the average **{numeric_col}** for each **{category_col}** group.")

        else:
            st.subheader("Category vs category")
            table = pd.crosstab(pair[col_x], pair[col_y], margins=False)
            st.write("Count table")
            st.dataframe(table, use_container_width=True)
            st.write("Row percentages")
            row_percent = (table.div(table.sum(axis=1).replace(0, 1), axis=0) * 100).round(1)
            st.dataframe(row_percent, use_container_width=True)
            st.bar_chart(table)

        st.subheader("Sample rows")
        st.dataframe(pair.head(100), use_container_width=True)
        st.download_button(
            "Download comparison data",
            pair.to_csv(index=False).encode("utf-8"),
            "two_column_analysis.csv",
            "text/csv"
        )
