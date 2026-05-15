.PHONY: lint format format-check fix typecheck test all

lint:
	uv run ruff check ondes/

format:
	uv run ruff format ondes/

format-check:
	uv run ruff format --check ondes/

fix:
	uv run ruff check --fix ondes/

typecheck:
	uv run pyright ondes/

test:
	uv run pytest tests/ -v || [ $$? -eq 5 ]

all: format-check lint typecheck test
