"""Stage 1: Exploratory Data Analysis.

Produces:
  * printed summary statistics + variable-type inventory for both raw tables
  * outputs/plots/eda_distributions.png   (distributions of numeric vars)
  * outputs/plots/eda_outliers.png        (boxplots -> outlier inspection)
  * outputs/plots/eda_correlations.png    (correlation heatmaps)
  * outputs/plots/eda_target_balance.png  (class balance of the target)

The goal is to understand each source *before* merging, so the merge and the
feature engineering that follow are driven by evidence rather than guesswork.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import RAW, PLOTS

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 140, "display.max_columns", 30)


def load_dataset():
    social = pd.read_csv(RAW / "customer_social_profiles.csv")
    trans = pd.read_csv(RAW / "customer_transactions.csv")
    return social, trans


def describe_table(name, df):
    print(f"\n{'='*70}\n{name}  ->  shape={df.shape}\n{'='*70}")
    print("\n-- variable types --")
    types = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(1),
    })
    print(types.to_string())
    print("\n-- numeric summary --")
    print(df.describe(include=[np.number]).round(2).to_string())
    cat = df.select_dtypes(include=["object", "string"])
    if not cat.empty:
        print("\n-- categorical summary --")
        for c in cat.columns:
            vc = df[c].value_counts(dropna=False).head(6)
            print(f"  {c}: {dict(vc)}")
    # duplicate audit
    print(f"\n-- fully duplicated rows: {df.duplicated().sum()}")


def main():
    social, trans = load_dataset()
    describe_table("customer_social_profiles", social)
    describe_table("customer_transactions", trans)

    # numeric coercion for plotting
    for df in (social, trans):
        for c in df.columns:
            if c not in ("customer_id_new", "customer_id_legacy", "transaction_id",
                         "purchase_date", "social_media_platform", "review_sentiment",
                         "product_category"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Plot 1: distributions
    num_social = ["engagement_score", "purchase_interest_score"]
    num_trans = ["purchase_amount", "customer_rating"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), num_social + num_trans):
        src = social if col in num_social else trans
        sns.histplot(src[col].dropna(), kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribution: {col}")
    fig.suptitle("EDA - Distributions of numeric variables", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "product_model" / "eda_distributions.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: outliers (boxplots)
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    for ax, col in zip(axes, num_social + num_trans):
        src = social if col in num_social else trans
        sns.boxplot(y=src[col].dropna(), ax=ax, color="#DD8452")
        ax.set_title(col)
    fig.suptitle("EDA - Outlier inspection (boxplots)", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOTS / "product_model" /"eda_outliers.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: correlations 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(social[num_social].corr(), annot=True, cmap="coolwarm", center=0,
                ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title("Social profile numeric corr")
    sns.heatmap(trans[num_trans].corr(), annot=True, cmap="coolwarm", center=0,
                ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title("Transactions numeric corr")
    fig.suptitle("EDA - Correlation heatmaps", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(PLOTS / "product_model" / "eda_correlations.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: target balance 
    fig, ax = plt.subplots(figsize=(8, 5))
    order = trans["product_category"].value_counts().index
    sns.countplot(data=trans, x="product_category", order=order, ax=ax,
                  hue="product_category", palette="viridis", legend=False)
    ax.set_title("EDA - Target class balance (product_category)")
    ax.set_xlabel("")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOTS / "product_model" /"eda_target_balance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n[Done] EDA plots written to outputs/plots/product_model/")


if __name__ == "__main__":
    main()
