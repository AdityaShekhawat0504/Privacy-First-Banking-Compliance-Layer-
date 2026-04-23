"""HMAC-SHA256 pseudonymisation with per-domain keys.

Keys are loaded from environment variables (never hardcoded).
Separate keys per domain prevent cross-domain linkage (GDPR Art. 4(5)):
the same raw identifier hashed under 'customer' and 'counterparty' produces
unrelated digests, so a compromise of one domain's analytics data cannot be
joined to another domain's records.
"""

import hashlib
import hmac
import os
from typing import Literal

import pandas as pd

DomainLiteral = Literal["customer", "counterparty", "account", "transaction"]

_KEY_ENV: dict[str, str] = {
    "customer": "HMAC_KEY_CUSTOMER",
    "counterparty": "HMAC_KEY_COUNTERPARTY",
    "account": "HMAC_KEY_ACCOUNT",
    "transaction": "HMAC_KEY_TRANSACTION",
}


def load_domain_key(domain: DomainLiteral) -> bytes:
    """Load and hex-decode the HMAC key for the given domain from env.

    Args:
        domain: One of 'customer', 'counterparty', 'account', 'transaction'.

    Returns:
        Raw key bytes (32 bytes for a 64-char hex env var).

    Raises:
        ValueError: If domain is not recognised.
        KeyError: If the env var is absent (with pointer to .env.example).
    """
    env_name = _KEY_ENV.get(domain)
    if env_name is None:
        raise ValueError(
            f"Unknown domain {domain!r}. Must be one of: {sorted(_KEY_ENV)}"
        )
    raw = os.environ.get(env_name)
    if raw is None:
        raise KeyError(
            f"Missing env var {env_name!r} for domain {domain!r}. "
            "Copy .env.example to .env and populate real hex keys."
        )
    return bytes.fromhex(raw)


def pseudonymise(value: str | None, domain: str) -> str | None:
    """One-way HMAC-SHA256 pseudonymisation of a single identifier.

    Args:
        value: Raw identifier string, or None / NaN.
        domain: Domain key selector ('customer', 'counterparty', 'account', 'transaction').

    Returns:
        64-char hex digest, or None if value is None/NaN.
    """
    if value is None or (not isinstance(value, str) and _is_null(value)):
        return None
    key = load_domain_key(domain)
    return hmac.new(key, str(value).encode(), hashlib.sha256).hexdigest()


def pseudonymise_series(s: pd.Series, domain: str) -> pd.Series:
    """Vectorised pseudonymisation — domain key loaded once for the whole series.

    Args:
        s: Series of identifier strings (may contain None / NaN).
        domain: Domain key selector.

    Returns:
        Series of 64-char hex digests (None where input was None/NaN).
    """
    key = load_domain_key(domain)

    def _hash(value: str | None) -> str | None:
        if value is None or (not isinstance(value, str) and _is_null(value)):
            return None
        return hmac.new(key, str(value).encode(), hashlib.sha256).hexdigest()

    return s.map(_hash)


def _is_null(value: object) -> bool:
    try:
        return pd.isna(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
