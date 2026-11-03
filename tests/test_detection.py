"""AML detection tests: feature engineering, metrics, training, SHAP explanations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_transactions

from privacy_aml.detection.train import (
    _eval_metrics,
    _sender_txns_per_day,
    build_features,
    train,
)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _rows(amounts, countries, senders=None, timestamps=None) -> pd.DataFrame:
    n = len(amounts)
    return pd.DataFrame(
        {
            "timestamp": timestamps
            if timestamps is not None
            else pd.date_range("2024-01-01", periods=n, freq="D"),
            "sender_customer_id": senders or [f"S{i}" for i in range(n)],
            "amount_eur": amounts,
            "country_code": countries,
            "channel": ["online"] * n,
            "currency": ["EUR"] * n,
            "merchant_category": ["retail"] * n,
        }
    )


def test_build_features_rule_flags() -> None:
    df = _rows(
        amounts=[20_000.0, 9_500.0, 123.45, 9_999.5],
        countries=["KY", "DE", "FR", "DE"],
    )
    X = build_features(df)

    assert X["is_round_amount"].tolist() == [1, 0, 0, 0]
    assert X["amount_near_10k"].tolist() == [0, 1, 0, 0]
    assert X["is_high_risk_country"].tolist() == [1, 0, 0, 0]
    assert X["is_cross_border"].tolist() == [1, 0, 1, 0]


def test_build_features_has_no_pii_or_target(transactions: pd.DataFrame) -> None:
    X = build_features(transactions)
    forbidden = {
        "txn_id",
        "sender_customer_id",
        "sender_iban",
        "receiver_counterparty_id",
        "receiver_iban",
        "country_code",
        "is_suspicious",
    }
    assert forbidden.isdisjoint(X.columns)
    assert len(X) == len(transactions)
    assert not X.isna().any().any()


def test_build_features_derives_time_features() -> None:
    df = _rows(
        amounts=[1.0, 2.0],
        countries=["DE", "DE"],
        timestamps=pd.to_datetime(["2024-01-01 08:30", "2024-01-06 23:10"]),
    )
    X = build_features(df)
    assert X["hour_of_day"].tolist() == [8.0, 23.0]
    assert X["day_of_week"].tolist() == [0.0, 5.0]  # Monday, Saturday


def test_sender_velocity_window() -> None:
    ts = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 06:00",
            "2024-01-01 11:00",
            "2024-01-02 12:00",  # > 12h after the others
            "2024-01-01 00:00",  # different sender
        ]
    )
    df = pd.DataFrame(
        {"sender_customer_id": ["A", "A", "A", "A", "B"], "timestamp": ts}
    )
    counts = _sender_txns_per_day(df).tolist()
    assert counts == [3.0, 3.0, 3.0, 1.0, 1.0]


@pytest.mark.parametrize("unit", ["s", "us", "ns"])
def test_sender_velocity_independent_of_timestamp_resolution(unit: str) -> None:
    ts = pd.Series(
        pd.to_datetime(["2024-01-01 00:00", "2024-01-01 06:00", "2024-01-05 00:00"])
    ).astype(f"datetime64[{unit}]")
    df = pd.DataFrame({"sender_customer_id": ["A", "A", "A"], "timestamp": ts})
    assert _sender_txns_per_day(df).tolist() == [2.0, 2.0, 1.0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_eval_metrics_perfect_ranking() -> None:
    y = np.array([0] * 198 + [1, 1])
    proba = np.linspace(0, 1, 200)  # positives get the two highest scores
    m = _eval_metrics(y, proba)

    assert m["auc"] == pytest.approx(1.0)
    # top 1 % of 200 rows = 2 rows, both positive
    assert m["precision_at_1pct"] == pytest.approx(1.0)
    assert m["recall_at_1pct"] == pytest.approx(1.0)
    cm = m["confusion"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 200
    assert cm["tp"] == 2


# ---------------------------------------------------------------------------
# Training (train on synthetic, evaluate on real hold-out)
# ---------------------------------------------------------------------------


@pytest.fixture()
def trained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)  # train() writes models/ relative to cwd
    real = make_transactions(n=600, seed=1)
    synth = make_transactions(n=600, seed=2)
    results = train(synth, real)
    return results, real, tmp_path / "models" / "aml_model.joblib"


def test_train_returns_metrics_and_saves_model(trained) -> None:
    results, real, model_path = trained

    assert model_path.exists()
    assert results["n_train_synthetic"] == 600
    assert results["n_eval_real"] == 120  # 20 % stratified hold-out
    for key in ("synthetic_trained", "real_trained_baseline"):
        assert 0.0 <= results[key]["auc"] <= 1.0
    # Toy data is trivially separable, so both models should rank well.
    assert results["synthetic_trained"]["auc"] > 0.9
    assert results["privacy_cost_auc"] == pytest.approx(
        results["real_trained_baseline"]["auc"] - results["synthetic_trained"]["auc"],
        abs=1e-4,
    )


def test_explain_predictions_top_k(trained) -> None:
    pytest.importorskip("shap")
    from privacy_aml.detection.explain import explain_predictions, load_artifacts

    _, real, model_path = trained
    artifacts = load_artifacts(model_path)
    sample = real.head(8)

    out = explain_predictions(sample, artifacts, top_k=3)

    assert list(out.index) == list(sample.index)
    assert out["probability"].between(0, 1).all()
    for reasons in out["top_reasons"]:
        assert len(reasons) == 3
        names = [r[0] for r in reasons]
        assert set(names) <= set(artifacts["feature_cols"])
        mags = [abs(r[2]) for r in reasons]
        assert mags == sorted(mags, reverse=True)
