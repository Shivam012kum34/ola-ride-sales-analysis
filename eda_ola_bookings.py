# ==============================
# Ola Ride Data - EDA Script
# ==============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# USER SETTINGS
# ------------------------------
DATA_PATH = "Bookings-100000-Rows.xlsx"   # or .csv
DATE_COL = "booking_date"                 # change if needed
SAVE_DIR = "eda_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------
# LOAD DATA
# ------------------------------
def load_data(path):
    print("Loading data...")
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# ------------------------------
# BASIC CLEANING
# ------------------------------
def clean_data(df):
    print("Cleaning data...")

    # Trim spaces in object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col].replace({"": np.nan, "nan": np.nan}, inplace=True)

    # Convert date column
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    print("Missing values (%):")
    print((df.isna().mean() * 100).sort_values(ascending=False).head(10))

    return df

# ------------------------------
# BASIC STATS
# ------------------------------
def save_basic_stats(df):
    num = df.select_dtypes(include=np.number).describe().T
    cat = df.select_dtypes(include="object").describe().T

    num.to_csv(os.path.join(SAVE_DIR, "numeric_summary.csv"))
    cat.to_csv(os.path.join(SAVE_DIR, "categorical_summary.csv"))

    print("Saved numeric and categorical summaries")

# ------------------------------
# DAILY BOOKINGS
# ------------------------------
def aggregate_daily_bookings(df):
    if DATE_COL not in df.columns:
        print("Date column not found. Skipping daily aggregation.")
        return None

    daily = (
        df.dropna(subset=[DATE_COL])
          .groupby(df[DATE_COL].dt.date)
          .size()
          .reset_index(name="booking_count")
    )

    daily.columns = ["date", "booking_count"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    daily.to_csv(os.path.join(SAVE_DIR, "daily_bookings.csv"), index=False)
    print("Saved daily bookings")

    return daily

# ------------------------------
# VISUALIZATION
# ------------------------------
def plot_daily_bookings(daily):
    if daily is None:
        return

    plt.figure(figsize=(12, 4))
    plt.plot(daily["date"], daily["booking_count"])
    plt.title("Daily Bookings Trend")
    plt.xlabel("Date")
    plt.ylabel("Bookings")
    plt.grid(alpha=0.3)

    out = os.path.join(SAVE_DIR, "daily_bookings_trend.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    print("Saved daily bookings plot")

# ------------------------------
# CORRELATION HEATMAP
# ------------------------------
def correlation_heatmap(df):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        print("Not enough numeric columns for correlation")
        return

    corr = num_df.corr()
    corr.to_csv(os.path.join(SAVE_DIR, "correlation_matrix.csv"))

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix")
    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    print("Saved correlation heatmap")

# ------------------------------
# MAIN FLOW
# ------------------------------
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Save small raw sample
    df.head(30).to_csv(os.path.join(SAVE_DIR, "raw_sample.csv"), index=False)

    save_basic_stats(df)

    daily = aggregate_daily_bookings(df)
    plot_daily_bookings(daily)

    correlation_heatmap(df)

    print("EDA completed. Check the 'eda_outputs' folder.")
