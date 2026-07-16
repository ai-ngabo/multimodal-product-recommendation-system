"""Stage 2: Cleaning + validated merge.

further justifications are provided in the report

Outputs:
  data/cleaned/customer_features.csv   (clean, merged, ready for modeling)
"""
import pandas as pd
import numpy as np
from config import RAW, PROC, SEED


def load_dataset():
    social = pd.read_csv(RAW / "customer_social_profiles.csv")
    trans = pd.read_csv(RAW / "customer_transactions.csv")
    return social, trans


def clean_social(social):
    print("\n[socials] raw shape:", social.shape)
    # 1. drop exact duplicated rows shown in EDA
    dups = social.duplicated().sum()
    social = social.drop_duplicates().copy()
    print(f"[socials] dropped {dups} exact duplicate rows -> {social.shape}")
    # 2. types
    social["engagement_score"] = pd.to_numeric(social["engagement_score"], errors="coerce")
    social["purchase_interest_score"] = pd.to_numeric(social["purchase_interest_score"], errors="coerce")
    # 3. reconcile key: 'A178' -> 178 (to match the Ids)
    social["customer_id"] = social["customer_id_new"].str.lstrip("A").astype(int)
    return social


def build_social_fingerprint(social):
    """Aggregate the multi-row social table to one row per customer."""
    sent_map = {"Negative": -1, "Neutral": 0, "Positive": 1}
    social["sentiment_num"] = social["review_sentiment"].map(sent_map)

    g = social.groupby("customer_id")
    fp = pd.DataFrame({
        "engagement_mean": g["engagement_score"].mean(),
        "engagement_max": g["engagement_score"].max(),
        "interest_mean": g["purchase_interest_score"].mean(),
        "interest_max": g["purchase_interest_score"].max(),
        "sentiment_mean": g["sentiment_num"].mean(),
        "num_platforms": g["social_media_platform"].nunique(),
        "num_social_rows": g.size(),
    })
    # dominant platform (most frequent) as a categorical signal
    dom = (social.groupby(["customer_id", "social_media_platform"]).size()
           .reset_index(name="n")
           .sort_values(["customer_id", "n"], ascending=[True, False])
           .drop_duplicates("customer_id")
           .set_index("customer_id")["social_media_platform"]
           .rename("dominant_platform"))
    fp = fp.join(dom)
    fp = fp.reset_index()
    print(f"[socials] fingerprint: {fp.shape[0]} unique customers, {fp.shape[1]-1} features")
    return fp


def clean_transactions(trans):
    print("\n[transactions] raw shape:", trans.shape)
    print("[transactions] duplicate rows:", trans.duplicated().sum())
    trans = trans.copy()
    trans["customer_id"] = trans["customer_id_legacy"].astype(int)
    trans["purchase_amount"] = pd.to_numeric(trans["purchase_amount"], errors="coerce")
    trans["customer_rating"] = pd.to_numeric(trans["customer_rating"], errors="coerce")
    trans["purchase_date"] = pd.to_datetime(trans["purchase_date"], errors="coerce")
    # missing-rating handling: FLAG then impute with median
    trans["rating_was_missing"] = trans["customer_rating"].isna().astype(int)
    med = trans["customer_rating"].median()
    trans["customer_rating"] = trans["customer_rating"].fillna(med)
    print(f"[transactions] imputed {int(trans['rating_was_missing'].sum())} missing ratings with median={med}")
    # light date features
    trans["purchase_month"] = trans["purchase_date"].dt.month
    trans["purchase_dow"] = trans["purchase_date"].dt.dayofweek
    return trans


def validate_join(trans, fp):
    trans_ids = set(trans["customer_id"])
    soc_ids = set(fp["customer_id"])
    print("\n[join] key range trans:", min(trans_ids), "-", max(trans_ids))
    print("[join] key range social:", min(soc_ids), "-", max(soc_ids))
    overlap = trans_ids & soc_ids
    print(f"[join] customers in BOTH: {len(overlap)}")
    print(f"[join] transactions' customers WITHOUT a social profile: {len(trans_ids - soc_ids)}")


def main():
    social, trans = load_dataset()
    social = clean_social(social)
    fp = build_social_fingerprint(social)
    trans = clean_transactions(trans)
    validate_join(trans, fp)

    # Merging
    merged = trans.merge(fp, on="customer_id", how="left")
    print("\n[merge] result shape:", merged.shape)

    # Checks for post_merge
    assert len(merged) == len(trans), "Row count changed -> fan-out bug!"
    print("[check] row count preserved (no fan-out):", len(merged) == len(trans))
    unmatched = merged["engagement_mean"].isna().sum()
    print(f"[check] transactions with no social match: {unmatched} "
          f"({unmatched/len(merged)*100:.1f}%)")

    # impute the unmatched social features + add a match flag
    merged["has_social_profile"] = merged["engagement_mean"].notna().astype(int)
    social_cols = ["engagement_mean", "engagement_max", "interest_mean", "interest_max",
                   "sentiment_mean", "num_platforms", "num_social_rows"]
    for c in social_cols:
        merged[c] = merged[c].fillna(merged[c].median())
    merged["dominant_platform"] = merged["dominant_platform"].fillna("None")

    print("[check] remaining nulls after impute:",
          int(merged[social_cols].isna().sum().sum()))
    print("[check] final dtypes ok; target classes:",
          sorted(merged["product_category"].unique()))

    out = PROC / "customer_features.csv"
    merged.to_csv(out, index=False)
    print(f"\n[Done] wrote {out}  shape={merged.shape}")


if __name__ == "__main__":
    main()
