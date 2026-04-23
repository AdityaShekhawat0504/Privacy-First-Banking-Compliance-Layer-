"""Statistical fidelity validation: KS (numeric), chi-squared (categorical), TSTR/TRTS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

_NUMERIC_COLS = ["amount_eur", "hour_of_day", "day_of_week"]
_CAT_COLS = [
    "currency", "channel", "country_code", "merchant_category", "is_suspicious"
]
_CAT_FEATURE_COLS = ["currency", "channel", "country_code", "merchant_category"]
_TARGET_COL = "is_suspicious"


def validate(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    random_state: int = 42,
) -> dict:
    """Run KS, chi-squared, and TSTR/TRTS validation between real and synthetic data.

    One-hot encoding is fit on the combined vocabulary of real+synth to prevent
    column mismatch when categories appear in one set but not the other.

    Args:
        real: Real transaction DataFrame (must include hour_of_day, day_of_week).
        synth: Synthetic transaction DataFrame from generate.train_and_generate().
        random_state: Seed for train/test split and classifiers.

    Returns:
        Dict with ks_tests, chi2_tests, tstr_auc, trts_auc, tstr_trts_ratio,
        overall_pass.
    """
    # --- KS tests (numeric columns) ---
    ks_tests: dict = {}
    for col in _NUMERIC_COLS:
        d_stat, p_val = ks_2samp(real[col].dropna(), synth[col].dropna())
        ks_tests[col] = {
            "D": float(d_stat), "p": float(p_val), "pass": bool(d_stat < 0.1)
        }

    # --- Chi-squared tests (categorical columns) ---
    chi2_tests: dict = {}
    for col in _CAT_COLS:
        real_cats = sorted(real[col].astype(str).unique())
        real_counts = (
            real[col].astype(str).value_counts().reindex(real_cats, fill_value=0)
        )
        synth_counts = (
            synth[col].astype(str).value_counts().reindex(real_cats, fill_value=0)
        )
        # Scale real proportions to synth total — keeps chi2 sum constraint satisfied
        expected = real_counts / real_counts.sum() * synth_counts.sum()
        # Replace zero expected values with a small epsilon to avoid divide-by-zero
        expected = expected.replace(0, 1e-9)
        stat, p_val = chisquare(synth_counts.values, expected.values)
        chi2_tests[col] = {
            "statistic": float(stat),
            "p": float(p_val),
            "pass": bool(p_val > 0.05),
        }

    # --- Feature engineering (shared for TSTR + TRTS) ---
    real_train, real_test = train_test_split(
        real,
        test_size=0.2,
        random_state=random_state,
        stratify=real[_TARGET_COL],
    )

    # Fit OHE on combined vocab to avoid column mismatch
    combined_cat = pd.concat(
        [real[_CAT_FEATURE_COLS], synth[_CAT_FEATURE_COLS]], ignore_index=True
    )
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(combined_cat)

    def _features(df: pd.DataFrame) -> np.ndarray:
        num = df[_NUMERIC_COLS].values.astype(float)
        cat = ohe.transform(df[_CAT_FEATURE_COLS])
        return np.hstack([num, cat])

    X_real_train = _features(real_train)
    y_real_train = real_train[_TARGET_COL].astype(int).values
    X_real_test = _features(real_test)
    y_real_test = real_test[_TARGET_COL].astype(int).values
    X_synth = _features(synth)
    y_synth = synth[_TARGET_COL].astype(int).values

    # --- TSTR: train on synthetic, evaluate on real test set ---
    sw_synth = compute_sample_weight("balanced", y_synth)
    tstr_clf = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    tstr_clf.fit(X_synth, y_synth, sample_weight=sw_synth)
    try:
        tstr_proba = tstr_clf.predict_proba(X_real_test)[:, 1]
        tstr_auc = float(roc_auc_score(y_real_test, tstr_proba))
    except ValueError:
        tstr_auc = 0.5  # fallback if only one class in test set

    # --- TRTS: train on real, evaluate on synthetic ---
    sw_real_train = compute_sample_weight("balanced", y_real_train)
    trts_clf = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    trts_clf.fit(X_real_train, y_real_train, sample_weight=sw_real_train)
    try:
        trts_proba = trts_clf.predict_proba(X_synth)[:, 1]
        trts_auc = float(roc_auc_score(y_synth, trts_proba))
    except ValueError:
        trts_auc = 0.5

    ratio = tstr_auc / trts_auc if trts_auc > 0 else 0.0

    ks_all_pass = all(v["pass"] for v in ks_tests.values())
    chi2_all_pass = all(v["pass"] for v in chi2_tests.values())
    overall_pass = ks_all_pass and chi2_all_pass and ratio >= 0.85

    return {
        "ks_tests": ks_tests,
        "chi2_tests": chi2_tests,
        "tstr_auc": tstr_auc,
        "trts_auc": trts_auc,
        "tstr_trts_ratio": ratio,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    from privacy_aml.reports.html import generate_report

    real_df = pd.read_csv(
        Path("data/raw/transactions.csv"), parse_dates=["timestamp"]
    )
    real_df["hour_of_day"] = real_df["timestamp"].dt.hour.astype(int)
    real_df["day_of_week"] = real_df["timestamp"].dt.day_of_week.astype(int)

    synth_df = pd.read_csv(
        Path("data/synth/transactions_synth.csv"), parse_dates=["timestamp"]
    )

    results = validate(real_df, synth_df)

    print(f"\nTSTR AUC:        {results['tstr_auc']:.4f}")
    print(f"TRTS AUC:        {results['trts_auc']:.4f}")
    ratio = results["tstr_trts_ratio"]
    verdict = "PASS" if ratio >= 0.85 else "FAIL (below 0.85 — see report for analysis)"
    print(f"TSTR/TRTS Ratio: {ratio:.4f}  [{verdict}]")
    print(f"Overall:         {'PASS' if results['overall_pass'] else 'FAIL'}")

    report_path = Path("reports/validation_report.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(results, real_df, synth_df, report_path)
