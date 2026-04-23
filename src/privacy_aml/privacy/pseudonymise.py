"""HMAC-SHA256 pseudonymisation with per-domain keys.

Keys are loaded from environment variables (never hardcoded).
Separate keys per domain prevent cross-domain linkage (GDPR Art. 4(5)).
"""

# TODO: Phase 1
