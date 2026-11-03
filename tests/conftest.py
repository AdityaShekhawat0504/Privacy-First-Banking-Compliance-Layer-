"""Shared fixtures: small, fast, fully offline transaction tables."""

import numpy as np
import pandas as pd
import pytest


def make_transactions(n: int = 600, suspicious_rate: float = 0.1, seed: int = 0):
    """Build a toy raw-schema transaction table with an easy-to-learn signal.

    Suspicious rows get round amounts and high-risk countries so models can
    separate them; normal rows get small log-normal amounts in EU countries.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n * suspicious_rate)
    is_susp = np.array([1] * n_pos + [0] * (n - n_pos))
    rng.shuffle(is_susp)

    amounts = np.where(
        is_susp == 1,
        rng.choice([10_000.0, 20_000.0, 50_000.0], n),
        np.clip(rng.lognormal(np.log(200), 1.0, n), 1.0, 8_000.0),
    )
    countries = np.where(
        is_susp == 1,
        rng.choice(["KY", "PA", "VG"], n),
        rng.choice(["DE", "FR", "NL"], n),
    )
    start = pd.Timestamp("2024-01-01").value
    end = pd.Timestamp("2024-03-31").value
    timestamps = pd.to_datetime(rng.integers(start, end, n)).astype("datetime64[ns]")

    return pd.DataFrame(
        {
            "txn_id": [f"TXN{i:06d}" for i in range(n)],
            "timestamp": timestamps,
            "sender_customer_id": [f"CUST-{i:04d}" for i in rng.integers(0, 50, n)],
            "sender_iban": [f"DE89{i:018d}" for i in range(n)],
            "receiver_counterparty_id": [f"CPTY-{i:04d}" for i in range(n)],
            "receiver_iban": [f"FR76{i:023d}" for i in range(n)],
            "amount_eur": amounts,
            "currency": rng.choice(["EUR", "USD", "GBP"], n),
            "channel": rng.choice(["online", "mobile", "branch", "atm"], n),
            "country_code": countries,
            "merchant_category": rng.choice(["retail", "travel", "gambling"], n),
            "is_suspicious": is_susp,
        }
    )


@pytest.fixture()
def transactions() -> pd.DataFrame:
    return make_transactions()
