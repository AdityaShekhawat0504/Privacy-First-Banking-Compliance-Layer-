"""Synthetic-data fidelity validation and HTML report tests."""

from pathlib import Path

import pandas as pd
import pytest
from conftest import make_transactions

from privacy_aml.reports.html import generate_report
from privacy_aml.synthesis.validate import validate


def _with_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour.astype(int)
    df["day_of_week"] = df["timestamp"].dt.day_of_week.astype(int)
    return df


@pytest.fixture(scope="module")
def identical_results():
    real = _with_time_features(make_transactions(n=600, seed=3))
    return validate(real, real.copy()), real


def test_validate_result_shape(identical_results) -> None:
    results, _ = identical_results
    assert set(results) == {
        "ks_tests",
        "chi2_tests",
        "tstr_auc",
        "trts_auc",
        "tstr_trts_ratio",
        "overall_pass",
    }
    assert set(results["ks_tests"]) == {"amount_eur", "hour_of_day", "day_of_week"}
    assert set(results["chi2_tests"]) == {
        "currency",
        "channel",
        "country_code",
        "merchant_category",
        "is_suspicious",
    }


def test_validate_identical_data_passes(identical_results) -> None:
    results, _ = identical_results
    for v in results["ks_tests"].values():
        assert v["D"] == pytest.approx(0.0)
        assert v["pass"]
    for v in results["chi2_tests"].values():
        assert v["pass"]
    assert results["tstr_trts_ratio"] > 0.85
    assert results["overall_pass"]


def test_validate_detects_shifted_amounts() -> None:
    real = _with_time_features(make_transactions(n=600, seed=4))
    synth = real.copy()
    synth["amount_eur"] = synth["amount_eur"] * 10
    results = validate(real, synth)
    assert not results["ks_tests"]["amount_eur"]["pass"]
    assert not results["overall_pass"]


def test_validate_detects_category_drift() -> None:
    real = _with_time_features(make_transactions(n=600, seed=5))
    synth = real.copy()
    synth["channel"] = "online"
    results = validate(real, synth)
    assert not results["chi2_tests"]["channel"]["pass"]


def test_generate_report_writes_html(identical_results, tmp_path: Path) -> None:
    results, real = identical_results
    out = tmp_path / "report.html"
    generate_report(results, real, real.copy(), out)

    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "TSTR/TRTS Ratio" in html
    assert "data:image/png;base64," in html
    assert ">PASS<" in html
