.PHONY: install data synth train demo test lint

install:
	uv sync --all-groups

data:
	uv run python -m privacy_aml.data.generate_fake

synth:
	uv run python -m privacy_aml.synthesis.generate
	uv run python -m privacy_aml.synthesis.validate

train:
	@echo "not implemented yet"

demo:
	@echo "not implemented yet"

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
