.PHONY: install test check broker-check
install:
	uv sync --locked --extra mt5
test:
	uv run --locked pytest
check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked mypy
broker-check:
	uv run --locked --extra mt5 sniper broker-check --capital 10 --symbol EURUSD
