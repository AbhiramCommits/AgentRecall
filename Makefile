PYTHON ?= python

.PHONY: up down test lint eval

up:
	docker compose up -d --wait

down:
	docker compose down

test:
	$(PYTHON) -m pytest -q --cov=src/agentrecall --cov-report=term-missing --cov-fail-under=85

lint:
	$(PYTHON) -m ruff check .

eval:
	PYTHON_BIN=$(PYTHON) bash scripts/run_eval.sh
