"""CTGAN wrapper via SDV for synthetic transaction generation."""

from __future__ import annotations

import uuid
from pathlib import Path

import faker
import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import CTGANSynthesizer

_PII_COLS = [
    "txn_id",
    "sender_customer_id",
    "receiver_counterparty_id",
    "sender_iban",
    "receiver_iban",
]
_HIGH_RISK_COUNTRIES = {"KY", "PA", "VG"}


def _make_high_risk_iban(country: str, rng: np.random.Generator) -> str:
    check = rng.integers(10, 99)
    digits = "".join(str(d) for d in rng.integers(0, 10, 16))
    return f"{country}{check}{digits}"


def _iban_pool(
    country_codes: pd.Series, rng: np.random.Generator, random_state: int
) -> list[str]:
    """Generate one synthetic IBAN per row, keyed to CTGAN-generated country_code."""
    faker_map: dict[str, faker.Faker] = {}
    for c in ["DE", "FR", "NL", "IT"]:
        f = faker.Faker({"DE": "de_DE", "FR": "fr_FR", "NL": "nl_NL", "IT": "it_IT"}[c])
        f.seed_instance(random_state)
        faker_map[c] = f

    result = []
    for country in country_codes:
        if country in faker_map:
            result.append(faker_map[country].iban())
        else:
            result.append(_make_high_risk_iban(str(country), rng))
    return result


def train_and_generate(
    real_df: pd.DataFrame,
    n_synthetic: int,
    epochs: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit CTGANSynthesizer on analytical columns and return a synthetic DataFrame.

    PII columns (IDs, IBANs) are excluded from CTGAN and reconstructed post-hoc.
    Temporal features (hour_of_day, day_of_week) are extracted before fitting and
    overwritten after synthesis by re-deriving from freshly generated timestamps.

    Note:
        epochs=100 on 50k rows takes ~15 min on CPU. Use epochs=5 for a smoke test.

    Args:
        real_df: Raw transaction DataFrame from generate_fake.generate_transactions().
        n_synthetic: Number of synthetic rows to generate.
        epochs: CTGAN training epochs.
        random_state: Seed for reproducibility.

    Returns:
        Synthetic DataFrame with same schema as real_df plus hour_of_day, day_of_week.
    """
    rng = np.random.default_rng(random_state)

    real_suspicious_rate = real_df["is_suspicious"].mean()
    n_pos = round(n_synthetic * real_suspicious_rate)
    n_neg = n_synthetic - n_pos

    # --- Prepare training data: drop PII, extract temporal features ---
    train_df = real_df.drop(columns=_PII_COLS).copy()
    train_df["hour_of_day"] = real_df["timestamp"].dt.hour.astype(int)
    train_df["day_of_week"] = real_df["timestamp"].dt.day_of_week.astype(int)
    train_df = train_df.drop(columns=["timestamp"])

    # --- Build SDV metadata ---
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)

    # Override auto-detected types that CTGAN handles better as categorical
    for col in [
        "country_code", "is_suspicious", "currency", "channel", "merchant_category"
    ]:
        metadata.update_column(col, sdtype="categorical")

    # SDV sometimes auto-detects a column as primary key — remove it
    try:
        metadata.remove_primary_key()
    except Exception:
        pass

    # --- Fit CTGAN ---
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        verbose=True,
        cuda=False,  # force CPU for reproducibility on Mac M-series
    )
    synthesizer.fit(train_df)

    # --- Generate with conditional sampling to preserve real suspicious rate ---
    synth_core = synthesizer.sample_from_conditions(conditions=[
        Condition(num_rows=n_neg, column_values={"is_suspicious": 0}),
        Condition(num_rows=n_pos, column_values={"is_suspicious": 1}),
    ])
    synth_core["is_suspicious"] = synth_core["is_suspicious"].astype(int)

    # --- Post-hoc PII + temporal reconstruction ---
    ts_start = real_df["timestamp"].min()
    ts_end = real_df["timestamp"].max()

    ts_floats = rng.uniform(ts_start.timestamp(), ts_end.timestamp(), n_synthetic)
    synth_core["timestamp"] = pd.to_datetime(ts_floats, unit="s")

    # Re-derive temporal features from new timestamps (discards CTGAN-generated values)
    synth_core["hour_of_day"] = synth_core["timestamp"].dt.hour.astype(int)
    synth_core["day_of_week"] = synth_core["timestamp"].dt.day_of_week.astype(int)

    synth_core["txn_id"] = [uuid.uuid4().hex for _ in range(n_synthetic)]
    synth_core["sender_customer_id"] = [
        f"SCUST-{i:06d}" for i in rng.integers(0, 5_000, n_synthetic)
    ]
    synth_core["receiver_counterparty_id"] = [
        f"SCPTY-{i:06d}" for i in rng.integers(0, 8_000, n_synthetic)
    ]

    fake_de = faker.Faker("de_DE")
    fake_de.seed_instance(random_state)
    sender_iban_pool = [fake_de.iban() for _ in range(1_000)]
    synth_core["sender_iban"] = [
        sender_iban_pool[i % 1_000] for i in rng.integers(0, 1_000, n_synthetic)
    ]
    synth_core["receiver_iban"] = _iban_pool(
        synth_core["country_code"], rng, random_state
    )

    # --- Reorder to match raw schema ---
    out_cols = [
        "txn_id", "timestamp", "sender_customer_id", "sender_iban",
        "receiver_counterparty_id", "receiver_iban", "amount_eur",
        "currency", "channel", "country_code", "merchant_category",
        "is_suspicious", "hour_of_day", "day_of_week",
    ]
    synth_core = synth_core[out_cols]

    output = Path("data/synth/transactions_synth.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    synth_core.to_csv(output, index=False)
    print(f"Saved {len(synth_core):,} synthetic rows → {output}")
    return synth_core


if __name__ == "__main__":
    real_df = pd.read_csv(
        Path("data/raw/transactions.csv"),
        parse_dates=["timestamp"],
    )
    train_and_generate(real_df, n_synthetic=50_000, epochs=100)
