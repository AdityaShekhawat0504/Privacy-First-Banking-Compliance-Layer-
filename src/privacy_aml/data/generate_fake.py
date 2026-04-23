"""Faker-based generator: 50k raw transactions, ~1.5% injected suspicious patterns."""

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from pathlib import Path

import faker
import numpy as np
import pandas as pd

_N_SENDERS = 5_000
_N_RECEIVERS = 8_000
_MCC_CODES = [
    "retail", "food_beverage", "travel", "entertainment", "utilities",
    "healthcare", "education", "real_estate", "financial_services", "telecom",
    "auto", "insurance", "government", "charity", "fuel", "clothing",
    "electronics", "gambling", "crypto_exchange", "money_transfer",
]
_ROUND_AMOUNTS = [10_000, 20_000, 50_000, 100_000]
_HIGH_RISK_COUNTRIES = {"KY", "PA", "VG"}
_RECEIVER_COUNTRIES = ["DE", "FR", "NL", "IT", "PA", "VG", "KY"]
_RECEIVER_COUNTRY_PROBS = [0.55, 0.20, 0.10, 0.05, 0.04, 0.03, 0.03]


def _make_high_risk_iban(country: str, rng: np.random.Generator) -> str:
    """Generate a plausible-looking but non-validating IBAN for high-risk countries."""
    check = rng.integers(10, 99)
    digits = "".join(str(d) for d in rng.integers(0, 10, 16))
    return f"{country}{check}{digits}"


def generate_transactions(n: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic raw transaction table with injected suspicious patterns.

    Args:
        n: Total number of transactions to generate.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with schema matching CLAUDE.md spec.
    """
    rng = np.random.default_rng(random_state)
    py_rng = random.Random(random_state)

    fake_de = faker.Faker("de_DE")
    fake_de.seed_instance(random_state)
    fake_fr = faker.Faker("fr_FR")
    fake_fr.seed_instance(random_state)
    fake_nl = faker.Faker("nl_NL")
    fake_nl.seed_instance(random_state)
    fake_it = faker.Faker("it_IT")
    fake_it.seed_instance(random_state)
    faker_map = {"DE": fake_de, "FR": fake_fr, "NL": fake_nl, "IT": fake_it}

    # --- Pre-generate sender / receiver pools ---
    sender_ids = [f"CUST-{i:06d}" for i in range(_N_SENDERS)]
    sender_ibans = [fake_de.iban() for _ in range(_N_SENDERS)]

    receiver_ids = [f"CPTY-{i:06d}" for i in range(_N_RECEIVERS)]
    receiver_countries_pool = rng.choice(
        _RECEIVER_COUNTRIES, size=_N_RECEIVERS, p=_RECEIVER_COUNTRY_PROBS
    )
    receiver_ibans_pool = [
        faker_map[c].iban() if c in faker_map else _make_high_risk_iban(c, rng)
        for c in receiver_countries_pool
    ]

    sender_iban_map = dict(zip(sender_ids, sender_ibans))
    receiver_iban_map = dict(zip(receiver_ids, receiver_ibans_pool))
    receiver_country_map = {
        rid: iban[:2] for rid, iban in zip(receiver_ids, receiver_ibans_pool)
    }

    # --- Base rows (all normal) ---
    now = pd.Timestamp.now().floor("s")
    start = now - pd.Timedelta(days=90)

    sender_idx = rng.integers(0, _N_SENDERS, n)
    receiver_idx = rng.integers(0, _N_RECEIVERS, n)

    sender_cust_ids = [sender_ids[i] for i in sender_idx]
    receiver_cpty_ids = [receiver_ids[i] for i in receiver_idx]

    ts_floats = rng.uniform(start.timestamp(), now.timestamp(), n)
    timestamps = pd.to_datetime(ts_floats, unit="s")

    amounts = np.clip(rng.lognormal(np.log(200), 1.8, n), 1.0, 500_000.0)

    mcc_weights = np.array([1.0 / (i + 1) for i in range(len(_MCC_CODES))])
    mcc_weights /= mcc_weights.sum()

    df = pd.DataFrame(
        {
            "txn_id": [uuid.uuid4().hex for _ in range(n)],
            "timestamp": timestamps,
            "sender_customer_id": sender_cust_ids,
            "sender_iban": [sender_iban_map[s] for s in sender_cust_ids],
            "receiver_counterparty_id": receiver_cpty_ids,
            "receiver_iban": [receiver_iban_map[r] for r in receiver_cpty_ids],
            "amount_eur": amounts,
            "currency": rng.choice(["EUR", "USD", "GBP"], n, p=[0.95, 0.03, 0.02]),
            "channel": rng.choice(
                ["online", "mobile", "branch", "atm"], n, p=[0.60, 0.25, 0.10, 0.05]
            ),
            "country_code": [receiver_country_map[r] for r in receiver_cpty_ids],
            "merchant_category": rng.choice(_MCC_CODES, n, p=mcc_weights),
            "is_suspicious": np.zeros(n, dtype=int),
        }
    )

    # --- Build sender → row-index mapping ---
    sender_to_indices: dict[str, list[int]] = defaultdict(list)
    for row_idx, sid in enumerate(sender_cust_ids):
        sender_to_indices[sid].append(row_idx)

    used_indices: set[int] = set()

    # Allocation: ~1.5% of n
    n_target = int(n * 0.015)
    n_round = int(n_target * 0.25)
    n_crossborder = int(n_target * 0.20)
    n_velocity = int(n_target * 0.25)
    n_struct = n_target - n_round - n_crossborder - n_velocity

    all_idx = np.arange(n)

    # 1. Round numbers
    avail = all_idx[~np.isin(all_idx, list(used_indices))]
    chosen = rng.choice(avail, size=n_round, replace=False)
    for idx in chosen:
        df.at[idx, "amount_eur"] = float(py_rng.choice(_ROUND_AMOUNTS))
        df.at[idx, "is_suspicious"] = 1
    used_indices.update(chosen.tolist())

    # 2. High-risk cross-border
    avail = all_idx[~np.isin(all_idx, list(used_indices))]
    chosen = rng.choice(avail, size=n_crossborder, replace=False)
    hr_list = list(_HIGH_RISK_COUNTRIES)
    for idx in chosen:
        country = py_rng.choice(hr_list)
        df.at[idx, "receiver_iban"] = _make_high_risk_iban(country, rng)
        df.at[idx, "country_code"] = country
        df.at[idx, "is_suspicious"] = 1
    used_indices.update(chosen.tolist())

    # 3. Velocity spikes (≥10 txns from same sender in one day)
    eligible_vel = [
        (sid, idxs)
        for sid, idxs in sender_to_indices.items()
        if len([i for i in idxs if i not in used_indices]) >= 10
    ]
    py_rng.shuffle(eligible_vel)
    vel_injected = 0
    for sid, idxs in eligible_vel:
        if vel_injected >= n_velocity:
            break
        unused = [i for i in idxs if i not in used_indices]
        if len(unused) < 10:
            continue
        cluster = unused[:10]
        spike_day = now - pd.Timedelta(seconds=float(rng.uniform(3600, 88 * 86400)))
        offsets = rng.uniform(0, 86400, size=10)
        for k, row_idx in enumerate(cluster):
            offset_td = pd.Timedelta(seconds=float(offsets[k]))
            df.at[row_idx, "timestamp"] = spike_day + offset_td
            df.at[row_idx, "is_suspicious"] = 1
        used_indices.update(cluster)
        vel_injected += len(cluster)

    # 4. Structuring (sub-10k clusters within 48h from same sender)
    # Build unused-index list per sender AFTER earlier injections
    struct_eligible = []
    for sid, idxs in sender_to_indices.items():
        unused = [i for i in idxs if i not in used_indices]
        if len(unused) >= 3:
            struct_eligible.append((sid, unused))

    py_rng.shuffle(struct_eligible)
    struct_injected = 0
    # Clamp anchor so cluster stays within window
    anchor_min = (start + pd.Timedelta(hours=1)).timestamp()
    anchor_max = (now - pd.Timedelta(hours=49)).timestamp()

    for sid, unused_idxs in struct_eligible:
        if struct_injected >= n_struct:
            break
        cluster_size = min(py_rng.randint(3, 5), len(unused_idxs))
        cluster = py_rng.sample(unused_idxs, cluster_size)
        anchor = float(rng.uniform(anchor_min, anchor_max))
        offsets = rng.uniform(0, 48 * 3600, size=cluster_size)
        for k, row_idx in enumerate(cluster):
            df.at[row_idx, "amount_eur"] = float(rng.uniform(9_000.0, 9_999.99))
            ts = pd.Timestamp(anchor + float(offsets[k]), unit="s")
            df.at[row_idx, "timestamp"] = ts
            df.at[row_idx, "is_suspicious"] = 1
        used_indices.update(cluster)
        struct_injected += cluster_size

    output = Path("data/raw/transactions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    n_susp = df["is_suspicious"].sum()
    pct = df["is_suspicious"].mean() * 100
    print(f"Wrote {len(df):,} rows → {output}")
    print(f"Suspicious: {n_susp} ({pct:.2f}%)")
    return df


def verify_patterns(df: pd.DataFrame) -> None:
    """Print group-by summaries; raise ValueError if expected patterns are absent.

    Args:
        df: Output of generate_transactions().

    Raises:
        ValueError: If any expected pattern is absent.
    """
    susp = df[df["is_suspicious"] == 1]
    print(f"\n--- Pattern verification (n_suspicious={len(susp)}) ---")

    # Round numbers
    round_count = susp["amount_eur"].isin(_ROUND_AMOUNTS).sum()
    print(f"Round-number rows:         {round_count}")
    if round_count == 0:
        raise ValueError("No round-number suspicious rows found — injection failed")

    # High-risk cross-border
    hr_count = susp["country_code"].isin(_HIGH_RISK_COUNTRIES).sum()
    print(f"High-risk country rows:    {hr_count}")
    if hr_count == 0:
        raise ValueError("No high-risk cross-border suspicious rows — injection failed")

    # Structuring (amounts in [9000, 9999.99])
    struct_count = susp["amount_eur"].between(9_000, 9_999.99).sum()
    print(f"Structuring-amount rows:   {struct_count}")
    if struct_count == 0:
        raise ValueError("No structuring suspicious rows — injection failed")

    # Velocity (at least one sender with ≥10 suspicious rows)
    sender_counts = susp.groupby("sender_customer_id").size()
    max_vel = int(sender_counts.max()) if len(sender_counts) > 0 else 0
    print(f"Max suspicious/sender:     {max_vel}")
    if max_vel < 10:
        raise ValueError(f"Velocity spike missing: max is {max_vel}, need ≥10")

    # Summary aggregates
    print("\nMean amount by is_suspicious:")
    print(df.groupby("is_suspicious")["amount_eur"].mean().to_string())
    print("\nCountry distribution by is_suspicious:")
    print(
        df.groupby(["is_suspicious", "country_code"])
        .size()
        .unstack(fill_value=0)
        .loc[:, list(_HIGH_RISK_COUNTRIES)]
        .to_string()
    )
    print("\nAll pattern checks passed.")


if __name__ == "__main__":
    df = generate_transactions()
    verify_patterns(df)
