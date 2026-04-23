"""GradientBoostingClassifier AML model trained on synthetic data."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

_CAT_FEATURE_COLS = ["channel", "currency", "merchant_category"]
_TARGET_COL = "is_suspicious"


def _ensure_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "hour_of_day" not in df.columns or "day_of_week" not in df.columns:
        df = df.copy()
        df["hour_of_day"] = df["timestamp"].dt.hour.astype(int)
        df["day_of_week"] = df["timestamp"].dt.day_of_week.astype(int)
    return df


def _sender_txns_per_day(df: pd.DataFrame) -> pd.Series:
    """For each row, count transactions by the same sender within a ±12h window."""
    result = pd.Series(0.0, index=df.index)
    half_ns = pd.Timedelta("12h").value  # nanoseconds

    for _, group in df.groupby("sender_customer_id", sort=False):
        ts_ns = group["timestamp"].astype(np.int64).values
        sorted_ts = np.sort(ts_ns)
        counts = np.empty(len(ts_ns), dtype=np.int64)
        for i, t in enumerate(ts_ns):
            lo = np.searchsorted(sorted_ts, t - half_ns, side="left")
            hi = np.searchsorted(sorted_ts, t + half_ns, side="right")
            counts[i] = hi - lo
        result.loc[group.index] = counts.astype(float)

    return result


def build_features(
    df: pd.DataFrame,
    encoder: OneHotEncoder | None = None,
) -> pd.DataFrame:
    """Return engineered feature DataFrame with no target column and no PII.

    Args:
        df: Transaction DataFrame. Must include timestamp if hour_of_day/day_of_week
            absent.
        encoder: Pre-fit OneHotEncoder. If None, fits a new encoder on this DataFrame.

    Returns:
        Feature DataFrame ready for sklearn estimators.
    """
    df = _ensure_time_features(df)

    base = pd.DataFrame(index=df.index)
    base["amount_eur"] = df["amount_eur"].astype(float)
    base["hour_of_day"] = df["hour_of_day"].astype(float)
    base["day_of_week"] = df["day_of_week"].astype(float)
    base["is_round_amount"] = df["amount_eur"].isin(
        {10_000.0, 20_000.0, 50_000.0, 100_000.0}
    ).astype(int)
    base["amount_near_10k"] = (
        (df["amount_eur"] >= 9_000.0) & (df["amount_eur"] <= 9_999.0)
    ).astype(int)
    base["is_high_risk_country"] = (
        df["country_code"].isin({"KY", "PA", "VG"}).astype(int)
    )
    base["is_cross_border"] = (df["country_code"] != "DE").astype(int)
    base["sender_txns_per_day"] = _sender_txns_per_day(df).values

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(df[_CAT_FEATURE_COLS])

    ohe_values = encoder.transform(df[_CAT_FEATURE_COLS])
    ohe_cols = encoder.get_feature_names_out(_CAT_FEATURE_COLS)
    ohe_df = pd.DataFrame(ohe_values, columns=ohe_cols, index=df.index)

    return pd.concat([base, ohe_df], axis=1)


def _eval_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    auc = float(roc_auc_score(y_true, proba))
    top_k = max(1, int(0.01 * len(y_true)))
    top_idx = np.argsort(proba)[::-1][:top_k]
    n_suspicious = int(y_true.sum())
    precision_at_1 = float(y_true[top_idx].sum() / top_k)
    recall_at_1 = float(y_true[top_idx].sum() / max(1, n_suspicious))
    preds = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, preds)
    return {
        "auc": auc,
        "precision_at_1pct": precision_at_1,
        "recall_at_1pct": recall_at_1,
        "confusion": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def train(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    random_state: int = 42,
) -> dict:
    """Train on full synthetic, evaluate on held-out real split.

    Args:
        synth_df: Synthetic transactions from CTGAN.
        real_df: Real transactions (ground truth, never used for training).
        random_state: Seed for reproducibility.

    Returns:
        Dict with synthetic_trained metrics, real_trained_baseline, privacy_cost_auc.
    """
    real_train, real_test = train_test_split(
        real_df,
        test_size=0.2,
        random_state=random_state,
        stratify=real_df[_TARGET_COL],
    )
    print(
        f"Discarded real_train ({len(real_train)} rows) to simulate"
        " no-real-data-exposure training."
    )

    # Fit OHE on combined vocab (synth + real) to prevent column mismatch at inference
    combined_cat = pd.concat(
        [synth_df[_CAT_FEATURE_COLS], real_df[_CAT_FEATURE_COLS]], ignore_index=True
    )
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(combined_cat)

    X_synth = build_features(synth_df, encoder)
    y_synth = synth_df[_TARGET_COL].astype(int).values
    feature_cols = list(X_synth.columns)

    X_real_test = build_features(real_test, encoder)[feature_cols]
    y_real_test = real_test[_TARGET_COL].astype(int).values

    # --- Synthetic-trained model (the privacy-preserving path) ---
    sw_synth = compute_sample_weight("balanced", y_synth)
    model_synth = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=random_state
    )
    model_synth.fit(X_synth[feature_cols], y_synth, sample_weight=sw_synth)
    proba_synth = model_synth.predict_proba(X_real_test)[:, 1]
    synth_metrics = _eval_metrics(y_real_test, proba_synth)

    # --- Real-trained baseline (the "cheating ceiling") ---
    X_real_train = build_features(real_train, encoder)[feature_cols]
    y_real_train = real_train[_TARGET_COL].astype(int).values
    sw_real = compute_sample_weight("balanced", y_real_train)
    model_real = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=random_state
    )
    model_real.fit(X_real_train, y_real_train, sample_weight=sw_real)
    proba_real = model_real.predict_proba(X_real_test)[:, 1]
    real_full = _eval_metrics(y_real_test, proba_real)
    real_baseline = {
        "auc": real_full["auc"],
        "precision_at_1pct": real_full["precision_at_1pct"],
        "recall_at_1pct": real_full["recall_at_1pct"],
    }

    # Persist artifacts for explain.py
    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {"model": model_synth, "encoder": encoder, "feature_cols": feature_cols},
        Path("models/aml_model.joblib"),
    )

    return {
        "synthetic_trained": synth_metrics,
        "real_trained_baseline": real_baseline,
        "privacy_cost_auc": round(real_baseline["auc"] - synth_metrics["auc"], 4),
        "n_train_synthetic": len(synth_df),
        "n_eval_real": len(real_test),
    }


if __name__ == "__main__":
    real_df = pd.read_csv(
        Path("data/raw/transactions.csv"), parse_dates=["timestamp"]
    )
    synth_df = pd.read_csv(
        Path("data/synth/transactions_synth.csv"), parse_dates=["timestamp"]
    )

    results = train(synth_df, real_df)

    s = results["synthetic_trained"]
    r = results["real_trained_baseline"]
    print("\n=== AML Detection Results ===")
    hdr = f"{'Metric':<25} {'Synth-trained':>14} {'Real-trained (ceil)':>20}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'AUC':<25} {s['auc']:>14.4f} {r['auc']:>20.4f}")
    p1s, p1r = s["precision_at_1pct"], r["precision_at_1pct"]
    r1s, r1r = s["recall_at_1pct"], r["recall_at_1pct"]
    print(f"{'Precision@1%':<25} {p1s:>14.4f} {p1r:>20.4f}")
    print(f"{'Recall@1%':<25} {r1s:>14.4f} {r1r:>20.4f}")
    print(f"\nPrivacy cost (AUC delta):  {results['privacy_cost_auc']:+.4f}")
    print(f"N train (synthetic):       {results['n_train_synthetic']}")
    print(f"N eval  (real held-out):   {results['n_eval_real']}")
    cm = s["confusion"]
    print("\nConfusion matrix (synth-trained, threshold=0.5):")
    print(f"  TN={cm['tn']}  FP={cm['fp']}")
    print(f"  FN={cm['fn']}  TP={cm['tp']}")
    print("\nModel saved to models/aml_model.joblib")
