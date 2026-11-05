.PHONY: install data synth train explain demo test lint

install:
	uv sync --all-groups

data:
	uv run python -m privacy_aml.data.generate_fake

synth:
	uv run python -m privacy_aml.synthesis.generate
	uv run python -m privacy_aml.synthesis.validate

train:
	uv run python -m privacy_aml.detection.train

explain:
	uv run python -m privacy_aml.detection.explain

demo:
	uv run jupyter lab notebooks/demo.ipynb

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
