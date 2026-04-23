"""Privacy layer tests: HMAC determinism, vault round-trip, no cross-domain collisions."""

import sqlite3

import numpy as np
import pandas as pd
import pytest
from cryptography.fernet import Fernet

from privacy_aml.privacy.pipeline import apply_privacy
from privacy_aml.privacy.pseudonymise import pseudonymise, pseudonymise_series
from privacy_aml.privacy.tokenise import DomainMismatchError, Vault

# 32-byte hex keys (deterministic, never used outside tests)
_CUSTOMER_KEY = "aa" * 32
_COUNTERPARTY_KEY = "bb" * 32
_ACCOUNT_KEY = "cc" * 32
_TRANSACTION_KEY = "dd" * 32
_INDEX_KEY = "ee" * 32

_TEST_IBAN = "DE89370400440532013000"


@pytest.fixture()
def hmac_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate HMAC domain keys in env."""
    monkeypatch.setenv("HMAC_KEY_CUSTOMER", _CUSTOMER_KEY)
    monkeypatch.setenv("HMAC_KEY_COUNTERPARTY", _COUNTERPARTY_KEY)
    monkeypatch.setenv("HMAC_KEY_ACCOUNT", _ACCOUNT_KEY)
    monkeypatch.setenv("HMAC_KEY_TRANSACTION", _TRANSACTION_KEY)


@pytest.fixture()
def all_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Populate all privacy env keys; return tmp_path for vault DB placement."""
    monkeypatch.setenv("HMAC_KEY_CUSTOMER", _CUSTOMER_KEY)
    monkeypatch.setenv("HMAC_KEY_COUNTERPARTY", _COUNTERPARTY_KEY)
    monkeypatch.setenv("HMAC_KEY_ACCOUNT", _ACCOUNT_KEY)
    monkeypatch.setenv("HMAC_KEY_TRANSACTION", _TRANSACTION_KEY)
    monkeypatch.setenv("VAULT_INDEX_KEY", _INDEX_KEY)
    monkeypatch.setenv("VAULT_KEY", Fernet.generate_key().decode())
    return tmp_path


# ---------------------------------------------------------------------------
# Pseudonymisation tests
# ---------------------------------------------------------------------------


def test_pseudonymise_deterministic(hmac_env: None) -> None:
    results = {pseudonymise("ID-1", "customer") for _ in range(1000)}
    assert len(results) == 1


def test_pseudonymise_domain_separation(hmac_env: None) -> None:
    assert pseudonymise("ID-1", "customer") != pseudonymise("ID-1", "counterparty")


def test_pseudonymise_avalanche(hmac_env: None) -> None:
    h1 = pseudonymise("ID-1", "customer")
    h2 = pseudonymise("ID-2", "customer")
    assert h1 is not None and h2 is not None
    differing = sum(c1 != c2 for c1, c2 in zip(h1, h2))
    assert differing / len(h1) >= 0.40


def test_pseudonymise_none_passthrough(hmac_env: None) -> None:
    assert pseudonymise(None, "customer") is None
    s = pseudonymise_series(pd.Series([None, np.nan]), "customer")
    assert s.isna().all()


# ---------------------------------------------------------------------------
# Vault / tokenisation tests
# ---------------------------------------------------------------------------


def test_vault_round_trip(all_env) -> None:
    with Vault(db_path=all_env / "tokens.db") as vault:
        token = vault.tokenise(_TEST_IBAN, domain="iban")
        recovered = vault.reverse(token, domain="iban")
    assert recovered == _TEST_IBAN


def test_vault_dedup(all_env) -> None:
    db_path = all_env / "tokens.db"
    with Vault(db_path=db_path) as vault:
        token1 = vault.tokenise(_TEST_IBAN, domain="iban")
        token2 = vault.tokenise(_TEST_IBAN, domain="iban")

    assert token1 == token2

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM vault").fetchone()[0]
    conn.close()
    assert count == 1


def test_vault_no_plaintext_stored(all_env) -> None:
    db_path = all_env / "tokens.db"
    with Vault(db_path=db_path) as vault:
        vault.tokenise(_TEST_IBAN, domain="iban")

    raw = db_path.read_bytes()
    assert _TEST_IBAN.encode() not in raw


def test_vault_domain_mismatch(all_env) -> None:
    with Vault(db_path=all_env / "tokens.db") as vault:
        token = vault.tokenise(_TEST_IBAN, domain="iban")
        with pytest.raises(DomainMismatchError):
            vault.reverse(token, domain="account")


def test_vault_audit_log(all_env) -> None:
    db_path = all_env / "tokens.db"
    with Vault(db_path=db_path) as vault:
        token = vault.tokenise(_TEST_IBAN, domain="iban")
        vault.reverse(token, domain="iban")

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT operation FROM audit ORDER BY id").fetchall()
    conn.close()

    assert len(rows) == 2
    assert rows[0][0] == "insert"
    assert rows[1][0] == "reverse"


# ---------------------------------------------------------------------------
# Pipeline end-to-end test
# ---------------------------------------------------------------------------


def _sample_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "txn_id": [f"TXN{i:04d}" for i in range(n)],
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "sender_customer_id": [f"CUST{i:04d}" for i in range(n)],
            "sender_iban": [f"DE89{i:016d}" for i in range(n)],
            "receiver_counterparty_id": [f"CP{i:04d}" for i in range(n)],
            "receiver_iban": [f"GB29{i:016d}" for i in range(n)],
            "amount_eur": [float(100 * i) for i in range(n)],
            "currency": ["EUR"] * n,
            "channel": ["ONLINE"] * n,
            "country_code": ["DE"] * n,
            "merchant_category": ["retail"] * n,
            "is_suspicious": [0] * (n - 1) + [1],
        }
    )


def test_pipeline_end_to_end(all_env) -> None:
    df = _sample_df()

    with Vault(db_path=all_env / "tokens.db") as vault:
        result = apply_privacy(df, vault)

    # (a) shape preserved
    assert result.shape == df.shape

    # (b) pseudonymised columns contain 64-char hex strings
    for col in ("txn_id", "sender_customer_id", "receiver_counterparty_id"):
        assert result[col].apply(lambda v: isinstance(v, str) and len(v) == 64).all(), col

    # (c) tokenised columns contain 32-char UUID hex strings
    for col in ("sender_iban", "receiver_iban"):
        assert result[col].apply(lambda v: isinstance(v, str) and len(v) == 32).all(), col

    # (d) passthrough columns are bit-identical to input
    passthrough = [
        "timestamp", "amount_eur", "currency", "channel",
        "country_code", "merchant_category", "is_suspicious",
    ]
    for col in passthrough:
        pd.testing.assert_series_equal(result[col], df[col])
