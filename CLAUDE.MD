# Privacy-First Banking Compliance Layer

Context file for Claude Code. Read this first at the start of every session.

## Project purpose

Proof-of-concept that backs this resume claim:

> Designed a compliance-analytics architecture for banks that stores transactions as pseudonymised records with tokenised PII; original-data keys remain inside the bank, so a breach at the analytics layer exposes no customer identity. Generated synthetic transaction datasets statistically aligned with real flows to train AML and fraud-detection models without exposing production PII. Aligned with GDPR (Art. 4(5), Art. 25) and BaFin-regulated workflows. Stack: Python, tokenisation & key-management logic, synthetic data generation, scikit-learn for downstream AML models.

This is a **pitched concept / demo**, not production code. Every design decision optimises for (1) interview defensibility and (2) README quality. Clarity beats completeness.

## Scope — IN

- HMAC-SHA256 pseudonymisation with per-domain keys (customer, counterparty, account)
- SQLite-backed tokenisation vault for IBANs and account numbers
- CTGAN synthetic transaction generation via SDV
- Statistical fidelity validation: KS (numeric), chi-squared (categorical), TSTR/TRTS (utility)
- scikit-learn `GradientBoostingClassifier` + SHAP for AML detection, trained on synthetic data
- Jupyter notebook showing end-to-end story
- README with Mermaid architecture diagram and regulatory alignment section

## Scope — OUT (document as "production architecture" in README, don't build)

- HSM / FIPS 140-3 key management → env vars in demo
- Kafka ingestion → batch CSV
- Format-preserving encryption (AES-FF3-1)
- Differential privacy at query layer
- Graph Neural Networks, federated learning
- FastAPI serving (only if Phase 4 finishes early)
- goAML / XBRL regulatory reporting
- MMD, Wasserstein, membership-inference attack tests

## Data

The "real" transaction table is **Faker-generated**, not a real bank dataset. This is deliberate: self-contained demo, no licensing issues, no Kaggle auth. In the README, frame this as "in production, the real-side feed would be ISO 20022 MX messages from the bank's core."

Schema for the generated transaction table:
`txn_id, timestamp, sender_customer_id, sender_iban, receiver_counterparty_id, receiver_iban, amount_eur, currency, channel, country_code, merchant_category, is_suspicious`

Generate ~50k rows. Inject ~1.5% suspicious transactions with patterns the AML model should catch (structuring under 10k EUR thresholds, round-number amounts, velocity spikes from one sender, cross-border to high-risk jurisdictions). This gives both the CTGAN and the AML model something real to learn.

## Stack

- Python 3.11, `uv` for env and dependencies
- `pandas`, `numpy`, `faker`
- `cryptography` (HMAC), stdlib `sqlite3` (vault)
- `sdv`, `ctgan`
- `scikit-learn`, `shap`
- `jupyter`, `matplotlib` for the demo notebook
- `pytest` for privacy-layer tests
- `ruff` for linting

## Layout

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── Makefile
├── .env.example          # HMAC_KEY_CUSTOMER, HMAC_KEY_COUNTERPARTY, HMAC_KEY_ACCOUNT
├── .gitignore            # .env, data/, *.db, mlruns/
├── data/
│   ├── raw/              # Faker-generated "real" transactions
│   ├── synth/            # CTGAN output
│   └── tokens.db         # SQLite vault (gitignored)
├── src/privacy_aml/
│   ├── __init__.py
│   ├── privacy/
│   │   ├── pseudonymise.py   # HMAC per-domain
│   │   ├── tokenise.py       # SQLite vault: insert/lookup/reverse
│   │   └── pipeline.py       # Applies full privacy transform to a DataFrame
│   ├── synthesis/
│   │   ├── generate.py       # CTGAN wrapper via SDV
│   │   └── validate.py       # KS, chi2, TSTR/TRTS
│   ├── detection/
│   │   ├── train.py          # GradientBoosting on synthetic
│   │   └── explain.py        # SHAP top-3 reasons per prediction
│   ├── data/
│   │   └── generate_fake.py  # Faker-based "real" data
│   └── reports/
│       └── html.py           # Simple HTML validation report
├── notebooks/
│   └── demo.ipynb            # End-to-end narrative for README screenshots
└── tests/
    └── test_privacy.py       # Determinism, round-trip, no-collision
```

## Conventions

- Type hints on all public functions; Google-style docstrings
- Each module under ~200 lines; split if larger
- Privacy keys loaded from env vars. **Never hardcode keys.** `.env.example` shows names with dummy values.
- Separate HMAC keys per domain (customer / counterparty / account) — prevents cross-domain linkage. This is the GDPR Art. 4(5) defensibility point, make sure it's in the README.
- Tokenisation vault is SQLite with two tables: `iban_vault(token, ciphertext, domain)` and `audit(timestamp, operation, domain)`. Tokens are UUIDs. Only the vault knows the mapping.
- Deterministic seeds everywhere in the demo notebook (random_state=42) so the README screenshots are reproducible
- `data/raw/` and `data/synth/` gitignored; regeneration via `make data` and `make synth`
- Tests cover the privacy layer only (HMAC determinism, vault round-trip, no collisions across 50k customer IDs). Skip tests for synthesis/detection — not a good use of Phase budget.

## Regulatory claims to support in the README

These map 1:1 to the resume bullet. The README needs a short "Regulatory alignment" section hitting each:

- **GDPR Art. 4(5)** (pseudonymisation): mapping keys never leave the bank vault; analytics layer sees only HMAC digests and opaque tokens
- **GDPR Art. 25** (data protection by design): architecture makes re-identification structurally impossible at the analytics layer, not just policy-enforced
- **EBA Guidelines on ML in AML/CFT (2023)**: synthetic data validation (KS + chi2 + TSTR/TRTS) addresses the "training data representativeness" requirement
- **BaFin Circular 03/2017** on AI/ML: SHAP per-alert explanations satisfy the explainability requirement
- **AMLD6 / GwG**: note as future typology extension (22 predicate offense categories)

## Phase budget (1 day, ~8.5h productive)

| Phase | Deliverable | Time |
|-------|-------------|------|
| 0 | uv scaffold, pyproject, Makefile, .env.example, .gitignore, empty module tree | 30 min |
| 1 | privacy/pseudonymise.py + privacy/tokenise.py + privacy/pipeline.py + tests | 2h |
| 2 | data/generate_fake.py + synthesis/generate.py + synthesis/validate.py + HTML report | 3h |
| 3 | detection/train.py + detection/explain.py | 1.5h |
| 4 | notebooks/demo.ipynb + README.md with Mermaid diagram | 1.5h |

**Overrun rules:** if Phase 2 exceeds 3h, cut the HTML report and put validation numbers as a markdown table in the README. If Phase 3 exceeds 1.5h, drop SHAP (not in the resume bullet). If Phase 4 exceeds 1.5h, ship the notebook without extra polish — the README is more important than the notebook.

## Working style with Claude Code

- **One phase per session.** Run `/clear` between phases so context stays tight.
- **Commit after each phase.** If a phase breaks, `git reset --hard` and retry with a tighter prompt rather than letting Claude Code dig deeper.
- **Use plan mode (Shift+Tab twice) before Phase 2 and Phase 3.** These are the phases most likely to over-engineer. Review the plan, cut anything not in scope, then approve.
- When Claude Code proposes anything from the "Scope — OUT" list above, reject it and point back to this file.
- Every phase ends with: tests pass (where applicable), `ruff check` is clean, and the Makefile target for that phase runs end-to-end.