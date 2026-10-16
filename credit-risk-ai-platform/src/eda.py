"""Exploratory data analysis with Pandas, Matplotlib and Seaborn.

Nothing fancy here on purpose -- this is the same kind of EDA you'd do
in any intro ML project: look at the data with .describe()/.value_counts(),
check the class balance, and plot a few relationships that explain *why*
later feature engineering and class-weighting decisions were made.

Plots are saved to reports/figures/ instead of just shown, so they can be
looked at later without re-running the script.

Run with: python -m src.eda
"""

import os

import matplotlib
matplotlib.use("Agg")  # save plots to file instead of opening a GUI window
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_prep import load_clean
from src.features import add_engineered_features

FIGURES_DIR = "reports/figures"


def basic_summary(df):
    print("Shape:", df.shape)
    print("\nMissing values per column:\n", df.isnull().sum().sum(), "total missing values")
    print("\n.describe() on numeric columns:")
    print(df.describe().T)

    print("\nDefault class balance:")
    print(df["default"].value_counts(normalize=True).rename("proportion"))


def plot_class_balance(df):
    plt.figure(figsize=(5, 4))
    sns.countplot(x="default", data=df)
    plt.title("Default vs. Non-Default (0 = no default, 1 = default)")
    plt.xlabel("default")
    plt.ylabel("number of customers")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/class_balance.png")
    plt.close()


def plot_default_rate_by_education(df):
    rate_by_edu = df.groupby("EDUCATION")["default"].mean()
    plt.figure(figsize=(5, 4))
    sns.barplot(x=rate_by_edu.index, y=rate_by_edu.values)
    plt.title("Default Rate by Education Level")
    plt.xlabel("EDUCATION (1=grad school, 2=university, 3=high school, 4=other)")
    plt.ylabel("default rate")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/default_rate_by_education.png")
    plt.close()


def plot_utilization_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="CREDIT_UTIL_RATIO", hue="default", bins=40, element="step", stat="density", common_norm=False)
    plt.xlim(-0.5, 2)
    plt.title("Credit Utilization Ratio by Default Status")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/utilization_by_default.png")
    plt.close()


def plot_delayed_payments(df):
    plt.figure(figsize=(6, 4))
    sns.barplot(x="NUM_DELAYED_PAYMENTS", y="default", data=df, errorbar=None)
    plt.title("Default Rate by Number of Delayed Payments (last 6 months)")
    plt.ylabel("default rate")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/default_rate_by_delayed_payments.png")
    plt.close()


def plot_correlation_heatmap(df):
    numeric_cols = ["LIMIT_BAL", "AGE", "CREDIT_UTIL_RATIO", "NUM_DELAYED_PAYMENTS", "default"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png")
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = add_engineered_features(load_clean())
    basic_summary(df)

    plot_class_balance(df)
    plot_default_rate_by_education(df)
    plot_utilization_distribution(df)
    plot_delayed_payments(df)
    plot_correlation_heatmap(df)

    print(f"\nSaved 5 plots to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
