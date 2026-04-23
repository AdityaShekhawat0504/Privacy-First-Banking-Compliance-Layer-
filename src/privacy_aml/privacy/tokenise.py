"""SQLite-backed tokenisation vault for IBANs and account numbers.

Tables: iban_vault(token, ciphertext, domain) and audit(timestamp, operation, domain).
Tokens are UUIDs; only the vault knows the plaintext mapping.
"""

# TODO: Phase 1
