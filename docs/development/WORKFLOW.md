# Development Workflow Guide

This guide describes the standardized development workflow for the Adaptive Neural Network project, implementing **Phase 3: Standardize Tooling** of the consolidation plan.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/V1B3hR/adaptiveneuralnetwork.git
cd adaptiveneuralnetwork

# Install development dependencies
make install-dev

# Install pre-commit hooks
make pre-commit-install
```

## Development Tools

### Code Quality Tools

The project uses the following tools for maintaining code quality:

1. **ruff** - Fast Python linter and formatter (replaces flake8, isort, and many others)
2. **black** - Uncompromising Python code formatter
3. **mypy** - Static type checker for Python
4. **pytest** - Testing framework with coverage support

### Configuration

All tools are configured in `pyproject.toml`:

- **ruff**: Line length 100, Python 3.12 target, auto-fix enabled
- **black**: Line length 100, Python 3.12 target
- **mypy**: Strict type checking (with some practical overrides)
- **pytest**: Test discovery, markers for test categories

## Common Development Tasks

### Using Make Commands

The project includes a Makefile with common development tasks:

```bash
make help              # Show all available commands
make install           # Install package in development mode
make install-dev       # Install with development dependencies
make lint              # Run linting checks
make lint-fix          # Run linting and auto-fix issues
make format            # Format code with ruff and black
make format-check      # Check formatting without changes
make type-check        # Run type checking
make test              # Run fast unit tests
make test-all          # Run all tests including slow ones
make test-cov          # Run tests with coverage report
make all-checks        # Run all checks (lint, format, type, test)
make clean             # Clean up build artifacts
```

### Manual Commands

If you prefer to run commands directly:

```bash
# Linting
ruff check .                    # Check for issues
ruff check --fix .              # Fix issues automatically

# Formatting
ruff format .                   # Format with ruff
black .                         # Format with black
ruff format --check .           # Check formatting (no changes)

# Type checking
mypy adaptiveneuralnetwork --ignore-missing-imports --no-strict-optional

# Testing
pytest tests/ -v                                    # Run all tests
pytest tests/ -v -m "not slow"                     # Run fast tests only
pytest tests/ --cov=adaptiveneuralnetwork          # Run with coverage
pytest tests/test_specific.py -v                   # Run specific test file
pytest tests/ -k test_function_name                # Run tests matching name
```

## Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit:

```bash
# Install hooks (run once after cloning)
make pre-commit-install
# or
pre-commit install

# Run hooks manually on all files
make pre-commit-run
# or
pre-commit run --all-files

# Skip hooks for a specific commit (not recommended)
git commit --no-verify -m "message"
```

The following hooks are configured:
- Trailing whitespace removal
- End of file fixing
- YAML/JSON/TOML validation
- Large file detection
- ruff linting and formatting
- black formatting
- mypy type checking (on core modules)

## Continuous Integration

The project uses GitHub Actions for CI/CD with three main jobs:

### 1. Lint and Format Check
- Runs on: Every push and PR
- Checks: ruff lint, ruff format, black format
- Purpose: Ensure code style consistency

### 2. Type Check
- Runs on: Every push and PR
- Checks: mypy static type analysis
- Purpose: Catch type-related issues early

### 3. Test Matrix
- Runs on: Every push and PR
- Python versions: 3.10, 3.11, 3.12
- Purpose: Ensure compatibility across Python versions
- Coverage: Generated for Python 3.12 and uploaded to Codecov

## Development Workflow

### 1. Before Starting Work

```bash
# Update your branch
git pull origin main

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. During Development

```bash
# Make changes to code

# Run linting and fix issues
make lint-fix

# Format code
make format

# Run tests
make test

# Check types (optional but recommended)
make type-check
```

### 3. Before Committing

```bash
# Run all checks
make all-checks

# Or rely on pre-commit hooks (runs automatically)
git add .
git commit -m "Your commit message"
```

### 4. Before Pushing

```bash
# Make sure all tests pass
make test-all

# Push your changes
git push origin feature/your-feature-name
```

### 5. Creating a Pull Request

1. Push your branch to GitHub
2. Create a PR from your branch to `main` or `develop`
3. Wait for CI checks to pass
4. Request review from maintainers
5. Address any feedback

## Testing Guidelines

### Test Organization

Tests are organized by category using pytest markers:

```python
@pytest.mark.unit          # Unit tests (fast)
@pytest.mark.integration   # Integration tests (slower)
@pytest.mark.slow          # Slow tests (skip by default)
```

### Running Specific Test Categories

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"

# Run slow tests only
pytest tests/ -m slow
```

### Writing Tests

Follow these guidelines when writing tests:

1. **Use descriptive names**: `test_trainer_saves_checkpoint_correctly`
2. **Mark appropriately**: Use `@pytest.mark` decorators
3. **Keep tests isolated**: Each test should be independent
4. **Use fixtures**: Define common setup in fixtures
5. **Assert clearly**: Use specific assertions with helpful messages

Example:

```python
import pytest
from adaptiveneuralnetwork.training import Trainer

@pytest.mark.unit
def test_trainer_initializes_with_defaults():
    """Test that Trainer initializes with default parameters."""
    trainer = Trainer(model=dummy_model)
    assert trainer.seed == 42
    assert trainer.use_amp is False
```

## Code Style Guidelines

### General Principles

1. **Follow PEP 8**: With line length of 100 characters
2. **Use type hints**: For function signatures and important variables
3. **Write docstrings**: For modules, classes, and public functions
4. **Keep it simple**: Prefer readability over cleverness
5. **DRY principle**: Don't Repeat Yourself

### Type Hints

```python
from typing import Optional, List, Dict, Any

def process_data(
    data: List[Dict[str, Any]],
    threshold: float = 0.5,
    verbose: bool = False
) -> Optional[Dict[str, float]]:
    """Process data and return metrics.
    
    Args:
        data: List of data dictionaries
        threshold: Confidence threshold
        verbose: Enable verbose logging
        
    Returns:
        Dictionary of metrics or None if processing fails
    """
    pass
```

### Docstring Format

Use Google-style docstrings:

```python
def train_model(model, data_loader, epochs=10):
    """Train a model on the given data.
    
    This function trains the model for the specified number of epochs
    and returns training metrics.
    
    Args:
        model: PyTorch model to train
        data_loader: DataLoader for training data
        epochs: Number of training epochs (default: 10)
        
    Returns:
        Dictionary containing training metrics:
            - loss: Final training loss
            - accuracy: Final training accuracy
            
    Raises:
        ValueError: If epochs is negative
        RuntimeError: If training fails
        
    Example:
        >>> model = MyModel()
        >>> loader = get_data_loader()
        >>> metrics = train_model(model, loader, epochs=5)
    """
    pass
```

## Troubleshooting

### Common Issues

#### Import Errors in Tests

If you see import errors when running tests:

```bash
# Reinstall in development mode
pip install -e ".[dev,nlp]"
```

#### Pre-commit Hook Failures

If pre-commit hooks fail:

```bash
# Fix formatting issues
make format

# Fix linting issues
make lint-fix

# Try commit again
git commit
```

#### mypy Type Check Errors

If mypy reports errors you can't fix:

1. Add `# type: ignore` comment on the specific line
2. Or configure in `pyproject.toml` under `[[tool.mypy.overrides]]`

```python
# For specific line
result = some_untyped_library_call()  # type: ignore

# For whole file (add to pyproject.toml)
[[tool.mypy.overrides]]
module = "problematic_module.*"
ignore_missing_imports = true
```

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search [GitHub Issues](https://github.com/V1B3hR/adaptiveneuralnetwork/issues)
- **Contributing**: See [CONTRIBUTING.md](../../CONTRIBUTING.md)
- **Examples**: Check `examples/` directory

## Summary

The standardized tooling ensures:
- ✅ Consistent code style across the project
- ✅ Early detection of bugs and type issues
- ✅ Automated quality checks in CI/CD
- ✅ Fast feedback for developers
- ✅ Reduced manual review burden

By following this workflow, you contribute to a high-quality, maintainable codebase!
