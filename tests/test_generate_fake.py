"""Raw data generator tests: schema, base rate, injected AML patterns."""

from pathlib import Path

import pytest

pytest.importorskip("faker")

from privacy_aml.data.generate_fake import (  # noqa: E402
    generate_transactions,
    verify_patterns,
)

_EXPECTED_COLUMNS = [
    "txn_id",
    "timestamp",
    "sender_customer_id",
    "sender_iban",
    "receiver_counterparty_id",
    "receiver_iban",
    "amount_eur",
    "currency",
    "channel",
    "country_code",
    "merchant_category",
    "is_suspicious",
]


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory):
    workdir = tmp_path_factory.mktemp("gen")
    mp = pytest.MonkeyPatch()
    mp.chdir(workdir)  # generator writes data/raw/ relative to cwd
    try:
        # 20k rows: enough senders with >= 10 txns for the velocity pattern.
        df = generate_transactions(n=20_000, random_state=7)
    finally:
        mp.undo()
    return df, workdir


def test_schema_and_output_file(generated) -> None:
    df, workdir = generated
    assert list(df.columns) == _EXPECTED_COLUMNS
    assert len(df) == 20_000
    assert (workdir / Path("data/raw/transactions.csv")).exists()
    assert df["txn_id"].is_unique


def test_suspicious_base_rate(generated) -> None:
    df, _ = generated
    assert df["is_suspicious"].mean() == pytest.approx(0.015, abs=0.003)


def test_injected_patterns_present(generated) -> None:
    df, _ = generated
    verify_patterns(df)  # raises ValueError if any pattern is missing


def test_country_code_matches_receiver_iban(generated) -> None:
    df, _ = generated
    assert (df["receiver_iban"].str[:2] == df["country_code"]).all()
