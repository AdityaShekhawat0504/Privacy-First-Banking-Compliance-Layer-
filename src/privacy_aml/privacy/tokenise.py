"""SQLite-backed tokenisation vault for IBANs and account numbers.

Tokenisation is architecturally distinct from pseudonymisation:

  Pseudonymisation (pseudonymise.py) — one-way HMAC.  Used for analytics
  join keys (customer IDs, counterparty IDs) that must survive aggregation
  across many rows without ever being reversible.

  Tokenisation (this module) — reversible via the vault.  Used for PII
  (IBANs) that may need unmasking for legitimate operations: SAR filing,
  BaFin regulator requests, or correspondent-bank queries.  The analytics
  layer sees only opaque UUID tokens; the vault is the sole mapping holder
  and stays inside the bank's trust boundary.

Schema
------
vault(plaintext_hash PK, token, ciphertext, domain, created_at)
  plaintext_hash: HMAC(plaintext, VAULT_INDEX_KEY) — enables dedup without
                  storing plaintext; the index key is separate from the
                  encryption key so a compromise of one does not expose both.
  ciphertext:     Fernet(plaintext) — symmetric authenticated encryption.

audit(id, timestamp, operation, domain, token)
  operation:      'insert' | 'lookup' | 'reverse'
"""

import hashlib
import hmac
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet


class DomainMismatchError(Exception):
    """Raised when a token is reversed under the wrong domain label."""


class Vault:
    """SQLite-backed tokenisation vault with Fernet encryption at rest."""

    def __init__(self, db_path: Path = Path("data/tokens.db")) -> None:
        """Open the vault database and create tables on first use.

        Args:
            db_path: Path to the SQLite file (created if absent).
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._fernet = Fernet(self._load_vault_key())
        self._index_key = self._load_index_key()
        self._create_tables()

    # ------------------------------------------------------------------
    # Key loading
    # ------------------------------------------------------------------

    def _load_vault_key(self) -> bytes:
        key = os.environ.get("VAULT_KEY")
        if key is None:
            raise KeyError(
                "Missing env var 'VAULT_KEY'. Copy .env.example to .env and set a "
                "Fernet key (generate: python -c "
                '"from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())")'
            )
        return key.encode()

    def _load_index_key(self) -> bytes:
        raw = os.environ.get("VAULT_INDEX_KEY")
        if raw is None:
            raise KeyError(
                "Missing env var 'VAULT_INDEX_KEY'. Copy .env.example to .env "
                "and set a 32-byte hex string (generate: openssl rand -hex 32)."
            )
        return bytes.fromhex(raw)

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vault (
                plaintext_hash TEXT PRIMARY KEY,
                token          TEXT UNIQUE NOT NULL,
                ciphertext     BLOB NOT NULL,
                domain         TEXT NOT NULL,
                created_at     TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_token ON vault(token);
            CREATE TABLE IF NOT EXISTS audit (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TIMESTAMP NOT NULL,
                operation  TEXT NOT NULL,
                domain     TEXT NOT NULL,
                token      TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_plaintext(self, plaintext: str) -> str:
        return hmac.new(
            self._index_key, plaintext.encode(), hashlib.sha256
        ).hexdigest()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _audit(self, operation: str, domain: str, token: str) -> None:
        self._conn.execute(
            "INSERT INTO audit(timestamp, operation, domain, token) VALUES(?,?,?,?)",
            (self._now(), operation, domain, token),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenise(self, plaintext: str | None, domain: str) -> str | None:
        """Return a UUID token for plaintext, inserting into the vault if new.

        Deduplicates via HMAC index: tokenising the same plaintext twice
        returns the same token with a single vault row.

        Args:
            plaintext: Raw PII string (e.g., IBAN), or None / NaN.
            domain: Domain label (e.g., 'iban').

        Returns:
            32-char UUID hex token, or None if plaintext is None/NaN.
        """
        if plaintext is None or (
            not isinstance(plaintext, str) and _is_null(plaintext)
        ):
            return None

        ph = self._hash_plaintext(plaintext)
        row = self._conn.execute(
            "SELECT token FROM vault WHERE plaintext_hash = ?", (ph,)
        ).fetchone()

        if row is not None:
            token: str = row[0]
            self._audit("lookup", domain, token)
            self._conn.commit()
            return token

        token = uuid.uuid4().hex
        ciphertext = self._fernet.encrypt(plaintext.encode())
        self._conn.execute(
            "INSERT INTO vault(plaintext_hash, token, ciphertext, domain, created_at)"
            " VALUES(?,?,?,?,?)",
            (ph, token, ciphertext, domain, self._now()),
        )
        self._audit("insert", domain, token)
        self._conn.commit()
        return token

    def reverse(self, token: str, domain: str) -> str:
        """Decrypt and return the plaintext for a token.

        Args:
            token: 32-char UUID hex token.
            domain: Expected domain label; raises DomainMismatchError if wrong.

        Returns:
            Original plaintext string.

        Raises:
            KeyError: Token not found.
            DomainMismatchError: Stored domain differs from requested domain.
        """
        row = self._conn.execute(
            "SELECT ciphertext, domain FROM vault WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Token {token!r} not found in vault.")
        ciphertext, stored_domain = row
        if stored_domain != domain:
            raise DomainMismatchError(
                f"Domain mismatch: token was stored under {stored_domain!r}, "
                f"but reversal requested under {domain!r}."
            )
        self._audit("reverse", domain, token)
        self._conn.commit()
        return self._fernet.decrypt(ciphertext).decode()

    def tokenise_series(self, s: pd.Series, domain: str) -> pd.Series:
        """Vectorised tokenisation.

        Args:
            s: Series of plaintext strings (may contain None / NaN).
            domain: Domain label.

        Returns:
            Series of 32-char UUID hex tokens.
        """
        return s.map(lambda v: self.tokenise(v, domain))

    def close(self) -> None:
        """Commit and close the SQLite connection."""
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "Vault":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _is_null(value: object) -> bool:
    try:
        return pd.isna(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
