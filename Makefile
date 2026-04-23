.PHONY: install data synth train demo test lint

install:
	uv sync --all-groups

data:
	@echo "not implemented yet"

synth:
	@echo "not implemented yet"

train:
	@echo "not implemented yet"

demo:
	@echo "not implemented yet"

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
