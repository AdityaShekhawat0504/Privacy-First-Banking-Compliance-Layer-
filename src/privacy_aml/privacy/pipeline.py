"""Privacy transform: pseudonymise (HMAC-SHA256) + tokenise (vault) all columns.

Two architecturally distinct mechanisms are applied — they must NOT be merged:

  Pseudonymisation (HMAC-SHA256, one-way) — for analytics join keys:
    txn_id                 → domain='transaction'
    sender_customer_id     → domain='customer'
    receiver_counterparty_id → domain='counterparty'

    These fields must survive cross-row aggregation (e.g., "all transactions
    from sender X"), so they need a deterministic, consistent digest.
    One-way means even full vault access cannot reconstruct the original ID.

  Tokenisation (Fernet + UUID, reversible via vault) — for PII that may need
  unmasking under legitimate authority:
    sender_iban            → domain='iban'
    receiver_iban          → domain='iban'

    Regulators (BaFin), SAR filing, and correspondent-bank AML queries may
    require the actual IBAN.  The analytics layer sees only a UUID; the vault
    (inside the bank's trust boundary) holds the mapping.

  Passthrough (non-identifying analytics attributes):
    timestamp, amount_eur, currency, channel, country_code,
    merchant_category, is_suspicious

    These carry statistical signal for AML models without identifying
    individuals.  They are not pseudonymised or tokenised.

This split is the "pseudonymised records with tokenised PII" claim in the
architecture, directly supporting GDPR Art. 4(5) and Art. 25 defences.
"""

import pandas as pd

from privacy_aml.privacy.pseudonymise import pseudonymise_series
from privacy_aml.privacy.tokenise import Vault

_PSEUDONYMISED = {
    "txn_id": "transaction",
    "sender_customer_id": "customer",
    "receiver_counterparty_id": "counterparty",
}

_TOKENISED = {
    "sender_iban": "iban",
    "receiver_iban": "iban",
}


def apply_privacy(df: pd.DataFrame, vault: Vault) -> pd.DataFrame:
    """Apply the full privacy transform to a raw transaction DataFrame.

    Args:
        df: Raw transaction DataFrame.  Expected columns:
            txn_id, timestamp, sender_customer_id, sender_iban,
            receiver_counterparty_id, receiver_iban, amount_eur,
            currency, channel, country_code, merchant_category, is_suspicious.
        vault: An open Vault instance used for IBAN tokenisation.

    Returns:
        New DataFrame with privacy transforms applied.  Input is not mutated.
        Shape is identical to input; only identifier columns change values.
    """
    out = df.copy()

    for col, domain in _PSEUDONYMISED.items():
        out[col] = pseudonymise_series(df[col], domain=domain)

    for col, domain in _TOKENISED.items():
        out[col] = vault.tokenise_series(df[col], domain=domain)

    return out
