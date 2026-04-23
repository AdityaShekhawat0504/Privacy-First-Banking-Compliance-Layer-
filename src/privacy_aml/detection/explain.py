"""SHAP top-3 per-alert explanations for AML model predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from privacy_aml.detection.train import _TARGET_COL, build_features


def load_artifacts(path: Path = Path("models/aml_model.joblib")) -> dict:
    """Load model artifacts saved by train().

    Args:
        path: Path to the joblib file produced by train().

    Returns:
        Dict with keys: model, encoder, feature_cols.
    """
    return joblib.load(path)


def explain_predictions(
    df: pd.DataFrame,
    artifacts: dict,
    top_k: int = 3,
) -> pd.DataFrame:
    """Return df with probability and top SHAP reasons per row.

    Args:
        df: Transaction DataFrame (same schema as training data).
        artifacts: Dict from load_artifacts() containing model, encoder, feature_cols.
        top_k: Number of top features by absolute SHAP value to return per row.

    Returns:
        DataFrame indexed like df with columns: probability, top_reasons.
        top_reasons is a list of (feature_name, row_value, shap_value) tuples.
    """
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    feature_cols = artifacts["feature_cols"]

    X = build_features(df, encoder)[feature_cols]
    proba = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # sklearn GBM binary returns a 2D array; list handles multi-output edge case
    shap_arr: np.ndarray = (
        shap_values[1] if isinstance(shap_values, list) else shap_values
    )

    rows = []
    for i in range(len(df)):
        abs_sv = np.abs(shap_arr[i])
        top_idx = np.argsort(abs_sv)[::-1][:top_k]
        reasons = [
            (feature_cols[j], float(X.iloc[i, j]), float(shap_arr[i, j]))
            for j in top_idx
        ]
        rows.append({"probability": float(proba[i]), "top_reasons": reasons})

    return pd.DataFrame(rows, index=df.index)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    real_df = pd.read_csv(
        Path("data/raw/transactions.csv"), parse_dates=["timestamp"]
    )

    # Re-derive real_test using identical split parameters as train()
    _, real_test = train_test_split(
        real_df,
        test_size=0.2,
        random_state=42,
        stratify=real_df[_TARGET_COL],
    )
    real_test = real_test.reset_index(drop=True)

    artifacts = load_artifacts()
    explained = explain_predictions(real_test, artifacts, top_k=3)
    explained["txn_id"] = real_test["txn_id"].values

    top20 = explained.sort_values("probability", ascending=False).head(20)

    def _fmt(reason: tuple) -> str:
        feat, val, sv = reason
        return f"{feat}={val:.2f} (shap:{sv:+.3f})"

    output_rows = []
    for _, row in top20.iterrows():
        reasons = row["top_reasons"]
        output_rows.append(
            {
                "txn_id": row["txn_id"],
                "probability": f"{row['probability']:.4f}",
                "reason_1": _fmt(reasons[0]) if len(reasons) > 0 else "",
                "reason_2": _fmt(reasons[1]) if len(reasons) > 1 else "",
                "reason_3": _fmt(reasons[2]) if len(reasons) > 2 else "",
            }
        )

    out_df = pd.DataFrame(output_rows)
    print("\n=== Top 20 AML Alerts with SHAP Explanations ===")
    print(out_df.to_string(index=False))

    Path("reports").mkdir(exist_ok=True)
    out_df.to_csv(Path("reports/top_alerts.csv"), index=False)
    print("\nSaved to reports/top_alerts.csv")
