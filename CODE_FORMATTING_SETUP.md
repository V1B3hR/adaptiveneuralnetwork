# Code Formatting Setup Guide

This repository is now configured with automated code formatting and linting tools.

## Installed Tools

- **pre-commit** (v4.3.0) - Git hook framework
- **ruff** (v0.13.1) - Fast Python linter and formatter
- **black** (v25.9.0) - Python code formatter
- **isort** (v6.0.1) - Import statement organizer

## How to Use

### Manual Formatting Commands

```bash
# Run ruff linting with automatic fixes
ruff check --fix .

# Run ruff formatting
ruff format .

# Run black formatting
black adaptiveneuralnetwork/

# Run isort for import organization
isort adaptiveneuralnetwork/ --profile=black
```

### Pre-commit Hooks

Pre-commit hooks are installed and will automatically run on every commit:

```bash
# Run pre-commit on all files manually
pre-commit run --all-files

# Run pre-commit on specific files
pre-commit run --files path/to/file.py

# Skip pre-commit hooks (not recommended)
git commit --no-verify -m "commit message"
```

## Configuration

### pyproject.toml
- Ruff configuration with line length 100, Python 3.9+ target
- Black configuration with line length 100
- MyPy configuration for type checking

### .pre-commit-config.yaml
- Local hooks using system-installed tools
- Runs ruff check, ruff format, black, and isort

## Results

The setup has been applied to the entire codebase:
- **170 files** reformatted by ruff
- **17 additional files** reformatted by black
- **10,852 linting issues** automatically fixed
- **2 files** had import sorting fixed by isort

## Integration with Development Workflow

1. Install development dependencies: `pip install -r dev-requirements.txt`
2. Install pre-commit hooks: `pre-commit install`
3. Make your changes
4. Commit - hooks will run automatically
5. If hooks fail, fix issues and commit again

## Contributing

All code contributions should follow the established formatting standards. The pre-commit hooks ensure consistency across the codebase.