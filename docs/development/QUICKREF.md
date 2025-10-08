# Developer Quick Reference Card

**Phase 3 Standardized Tooling - Quick Commands**

## Setup (One Time)

```bash
make install-dev        # Install development dependencies
make pre-commit-install # Install pre-commit hooks
```

## Daily Workflow

### Before Starting Work
```bash
git pull origin main
git checkout -b feature/my-feature
```

### While Coding
```bash
make lint              # Check code style (no changes)
make lint-fix          # Auto-fix linting issues
make format            # Format code with ruff + black
make test              # Run fast unit tests
```

### Before Committing
```bash
make all-checks        # Run all quality checks
git add .
git commit -m "message"  # Pre-commit hooks run automatically
```

### Before Pushing
```bash
make test-all          # Run all tests (including slow)
git push origin feature/my-feature
```

## Common Commands

### Linting & Formatting
| Command | Description |
|---------|-------------|
| `make lint` | Check for linting issues |
| `make lint-fix` | Auto-fix linting issues |
| `make format` | Format code |
| `make format-check` | Check formatting (no changes) |

### Testing
| Command | Description |
|---------|-------------|
| `make test` | Run fast unit tests |
| `make test-all` | Run all tests |
| `make test-cov` | Run tests with coverage |
| `make test-integration` | Run integration tests |

### Type Checking
| Command | Description |
|---------|-------------|
| `make type-check` | Run mypy type analysis |

### Pre-commit
| Command | Description |
|---------|-------------|
| `make pre-commit-install` | Install hooks |
| `make pre-commit-run` | Run hooks on all files |

### Utilities
| Command | Description |
|---------|-------------|
| `make clean` | Clean build artifacts |
| `make all-checks` | Run everything |
| `make help` | Show all commands |

## Manual Commands (if needed)

```bash
# Linting
ruff check .                    # Check issues
ruff check --fix .              # Fix issues

# Formatting  
ruff format .                   # Format with ruff
black .                         # Format with black

# Type checking
mypy adaptiveneuralnetwork --ignore-missing-imports

# Testing
pytest tests/ -v                # All tests
pytest tests/ -m "not slow"     # Fast tests only
pytest tests/ --cov=adaptiveneuralnetwork  # With coverage
```

## CI/CD Pipeline

### Jobs Run on Every Push/PR

1. **Lint and Format Check**
   - ruff check
   - ruff format --check
   - black --check

2. **Type Check**
   - mypy (informational)

3. **Test Matrix**
   - Python 3.10, 3.11, 3.12
   - Coverage report (3.12 only)

## Pre-commit Hooks

Automatically run before each commit:
- Trailing whitespace removal
- End of file fixing
- YAML/JSON/TOML validation
- Merge conflict detection
- Debug statement detection
- ruff linting and formatting
- black formatting
- mypy type checking

**Skip hooks** (not recommended):
```bash
git commit --no-verify -m "message"
```

## Troubleshooting

### Pre-commit fails
```bash
make format
make lint-fix
git add .
git commit
```

### Import errors in tests
```bash
pip install -e ".[dev,nlp]"
```

### Too many mypy errors
Add `# type: ignore` to specific lines or configure in `pyproject.toml`

## Configuration Files

- **pyproject.toml** - All tool configurations
- **.pre-commit-config.yaml** - Pre-commit hooks
- **.github/workflows/ci.yml** - CI/CD pipeline
- **Makefile** - Development commands
- **.editorconfig** - Editor settings

## Documentation

- **Workflow Guide**: `docs/development/WORKFLOW.md`
- **Phase 3 Summary**: `docs/development/PHASE3_SUMMARY.md`
- **Consolidation Plan**: `CONSOLIDATION_PLAN.md`
- **Main README**: `README.md`

## Setup Verification

```bash
python scripts/setup_phase3.py
```

This checks:
- ✅ All tools installed
- ✅ Config files present
- ✅ Current code state

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: [GitHub Issues](https://github.com/V1B3hR/adaptiveneuralnetwork/issues)
- **Contributing**: See `CONTRIBUTING.md`

---

**Quick Start**: `make install-dev && make pre-commit-install && make test`

**Daily Use**: `make lint-fix && make format && make test`

**Before PR**: `make all-checks`
