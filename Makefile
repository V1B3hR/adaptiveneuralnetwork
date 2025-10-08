.PHONY: help install install-dev lint format type-check test test-all test-cov clean

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev,nlp]"

lint:  ## Run linting checks with ruff
	ruff check .

lint-fix:  ## Run linting and auto-fix issues
	ruff check --fix .

format:  ## Format code with ruff and black
	ruff format .
	black .

format-check:  ## Check code formatting without changes
	ruff format --check .
	black --check .

type-check:  ## Run type checking with mypy
	mypy adaptiveneuralnetwork --ignore-missing-imports --no-strict-optional

test:  ## Run fast unit tests
	pytest tests/ -v -m "not slow" --ignore=tests/integration/ --ignore=tests/phase_specific/

test-all:  ## Run all tests including slow ones
	pytest tests/ -v

test-cov:  ## Run tests with coverage report
	pytest tests/ --cov=adaptiveneuralnetwork --cov-report=html --cov-report=term -m "not slow" --ignore=tests/integration/ --ignore=tests/phase_specific/
	@echo "Coverage report generated in htmlcov/index.html"

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

pre-commit-install:  ## Install pre-commit hooks
	pre-commit install

pre-commit-run:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean:  ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation (placeholder for future)
	@echo "Documentation build not yet configured"

all-checks:  ## Run all checks (lint, format-check, type-check, test)
	@echo "Running all checks..."
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) type-check
	$(MAKE) test
	@echo "All checks passed!"
