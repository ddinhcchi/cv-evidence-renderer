.PHONY: install dev test lint fmt clean benchmark

install:
	pip install -e .

dev:
	pip install -e ".[dev,supervision]"

test:
	pytest -v

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

benchmark:
	python scripts/benchmark.py
