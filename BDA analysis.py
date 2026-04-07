import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FILE = "modified_air_quality.csv"
MODEL_FILE = "aqi_model.pkl"
TARGET = "AQI"
EPS = 1e-8

CITY_SELECTED = "All Cities"
YEAR_SELECTED = "All Years"
BACKTEST_DAYS = 30
# ==========================================


def load_model(filepath):
    if not os.path.exists(filepath):
        print("Warning: Model file not found.")
        return None

    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_data(filepath):

    print("Loading AQI dataset (large file)...")

    df = pd.read_csv(filepath)

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df = df.sort_values("Datetime")

    return df


def backtest_accuracy(model, df, days):

    test_df = df.iloc[-days:].copy()

    X_test = test_df[["PM2.5","PM10","NO2","CO","SO2","O3"]]
    y_test = test_df[TARGET]

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    smape = np.mean(
        2 * np.abs(preds - y_test) /
        (np.abs(y_test) + np.abs(preds) + EPS)
    ) * 100

    accuracy = (1 - mae / np.mean(y_test)) * 100

    result_df = test_df[["Datetime", TARGET]].copy()
    result_df["Predicted_AQI"] = preds.round(2)

    return result_df, mae, r2, smape, accuracy


def plot_aqi_dashboard(df, target_col):

    df_plot = df.copy()

    df_plot["Year"] = df_plot["Datetime"].dt.year
    df_plot["Month"] = df_plot["Datetime"].dt.month

    monthly_trend = df_plot.groupby(["Year","Month"])[target_col].mean().unstack(fill_value=0)
    yearly_avg = df_plot.groupby("Year")[target_col].mean()

    years = sorted(df_plot["Year"].unique())

    fig, axs = plt.subplots(2,2,figsize=(14,10))

    fig.suptitle("Air Quality Index Big Data Analysis", fontsize=16, fontweight='bold')

    # 1 Trend
    ax1 = axs[0,0]

    for month in monthly_trend.columns:
        ax1.plot(monthly_trend.index, monthly_trend[month], marker='o', label=f"Month {month}")

    ax1.set_title("Monthly AQI Trend")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Average AQI")
    ax1.legend()
    ax1.grid(True)

    # 2 Year comparison
    ax2 = axs[0,1]

    x = np.arange(len(years))
    width = 0.2

    for i, month in enumerate(monthly_trend.columns[:4]):
        ax2.bar(x + width*i, monthly_trend[month], width, label=f"M{month}")

    ax2.set_xticks(x + width, years)
    ax2.set_title("Year-wise AQI Comparison")
    ax2.legend()

    # 3 Yearly AQI
    ax3 = axs[1,0]

    bars = ax3.bar(yearly_avg.index.astype(str), yearly_avg.values)

    ax3.set_title("Average AQI Per Year")
    ax3.set_ylabel("AQI")

    for bar in bars:
        ax3.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height()*1.01,
            f"{bar.get_height():.1f}",
            ha='center'
        )

    # 4 Pollutant importance
    ax4 = axs[1,1]

    pollutants = ["PM2.5","PM10","NO2","CO","SO2","O3"]

    corr = df[pollutants + [target_col]].corr()[target_col].drop(target_col)

    ax4.barh(corr.index, corr.values)

    ax4.set_title("Pollutant Correlation with AQI")
    ax4.set_xlabel("Correlation")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    return fig


if __name__ == "__main__":

    print("\n"+"="*50)
    print(" AIR QUALITY BIG DATA ANALYTICS ENGINE ")
    print("="*50)

    if not os.path.exists(DATA_FILE):
        print("Dataset not found.")
        exit()

    df = load_data(DATA_FILE)

    if CITY_SELECTED != "All Cities":
        df = df[df["City"] == CITY_SELECTED]

    if YEAR_SELECTED != "All Years":
        df = df[df["Datetime"].dt.year == YEAR_SELECTED]

    print(f"Dataset loaded. Records: {len(df):,}")

    model = load_model(MODEL_FILE)

    if model is None:
        print("No model found.")
        exit()

    X = df[["PM2.5","PM10","NO2","CO","SO2","O3"]]
    y = df[TARGET]

    split_idx = int(len(df)*0.8)

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    preds = model.predict(X_test)

    print("\n--- Model Performance ---")

    print("MAE:", mean_absolute_error(y_test,preds))
    print("R2:", r2_score(y_test,preds))

    bt_df, bt_mae, bt_r2, bt_smape, bt_acc = backtest_accuracy(model, df, BACKTEST_DAYS)

    print("\nRecent Accuracy:", bt_acc)

    print("\nGenerating AQI dashboards...")

    fig1 = plot_aqi_dashboard(df, TARGET)

    fig2, ax = plt.subplots(figsize=(8,5))

    ax.plot(bt_df["Datetime"], bt_df[TARGET], label="Actual AQI", marker="o")

    ax.plot(bt_df["Datetime"], bt_df["Predicted_AQI"], label="Predicted AQI", linestyle="--")

    ax.set_title("AQI Prediction Accuracy")

    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")

    ax.legend()

    plt.tight_layout()

    plt.show()

    output_csv = "aqi_predicted_vs_actual.csv"

    bt_df.to_csv(output_csv, index=False)

    print("\nSaved predictions to:", output_csv)