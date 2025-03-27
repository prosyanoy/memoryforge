PY ?= .venv/bin/python

.PHONY: install test run bench eval clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	$(PY) -m pytest

run:            ## API on :8000 with 20 seeded synthetic customers
	$(PY) -m memoryforge.cli serve --seed 20

bench:
	$(PY) scripts/benchmark.py --whale 5000

eval:           ## extraction F1 + crossed-memory (leakage) check on synthetic data
	$(PY) scripts/evaluate.py

clean:
	rm -rf .pytest_cache data build *.egg-info src/*.egg-info
