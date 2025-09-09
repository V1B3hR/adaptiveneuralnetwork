# Contributing to Adaptive Neural Network

Thank you for your interest in contributing to the Adaptive Neural Network project! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Ethics Compliance](#ethics-compliance)
- [Submitting Changes](#submitting-changes)
- [Documentation](#documentation)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/adaptiveneuralnetwork.git
   cd adaptiveneuralnetwork
   ```
3. Set up your development environment (see below)
4. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Environment

### Requirements

- Python 3.7 or higher
- pip or conda for package management

### Installation

1. Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install development dependencies (optional but recommended):
   ```bash
   pip install -r dev-requirements.txt
   ```

3. Verify installation:
   ```bash
   python -c "import numpy, scipy, matplotlib; print('Dependencies OK')"
   ```

## Running Tests

### All Tests
```bash
# Run all tests
python -m unittest discover tests/ -p "test_*.py" -v

# Or using pytest (if installed)
pytest tests/ -v
```

### Specific Test Categories
```bash
# Basic problem solving
python -m unittest tests.test_basic_problem_solving -v

# Adaptive learning
python -m unittest tests.test_adaptive_learning -v

# Cognitive functioning
python -m unittest tests.test_cognitive_functioning -v

# Pattern recognition
python -m unittest tests.test_pattern_recognition -v

# Ethics compliance (always required)
python -m unittest tests.test_basic_problem_solving.TestBasicProblemSolving.test_ethics_compliance -v
```

### Ethics Framework Testing
```bash
# Test ethics module directly
python -c "
from core.ai_ethics import audit_decision
result = audit_decision({
    'action': 'test_action',
    'preserve_life': True,
    'absolute_honesty': True,
    'privacy': True
})
print('Ethics audit:', result['compliant'])
"
```

## Code Style

### Formatting
- Use **black** for code formatting: `black .`
- Use **isort** for import sorting: `isort .`
- Maximum line length: 88 characters
- Follow PEP 8 guidelines

### Type Hints (Optional)
- While not required, type hints are encouraged for new code
- Use **mypy** for type checking: `mypy core/`

### Linting
- Use **flake8** for linting: `flake8 core/ tests/`

### Pre-commit Hooks (Recommended)
If you have pre-commit installed:
```bash
pre-commit install
pre-commit run --all-files
```

## Ethics Compliance

**CRITICAL**: All contributions must maintain ethics compliance.

### Requirements
1. **Every test class** must include a `test_ethics_compliance()` method
2. **All major actions** in AliveLoopNode must call `audit_decision()`
3. **Ethics violations** are not permitted and will block PR acceptance

### Example Ethics Test
```python
def test_ethics_compliance(self):
    """Ensure major action is checked against the ethics audit."""
    from core.ai_ethics import audit_decision
    
    # Test decision parameters
    decision_log = {
        "action": "test_action",
        "preserve_life": True,
        "absolute_honesty": True,
        "privacy": True
    }
    
    # Audit decision
    audit_result = audit_decision(decision_log)
    
    # Verify compliance
    self.assertTrue(audit_result["compliant"], 
                   f"Ethics audit failed: {audit_result['violations']}")
```

## Submitting Changes

### Pull Request Process

1. **Update tests**: Add or modify tests for your changes
2. **Run all tests**: Ensure all tests pass, including ethics compliance
3. **Update documentation**: Add or update relevant documentation
4. **Commit with clear messages**: Use descriptive commit messages
5. **Push to your fork**: `git push origin feature/your-feature-name`
6. **Create pull request**: Submit PR with detailed description

### Pull Request Requirements

- [ ] All tests pass
- [ ] Ethics compliance tests included and passing
- [ ] Code follows style guidelines (black, isort)
- [ ] Documentation updated (if applicable)
- [ ] No linting errors
- [ ] Clear description of changes and reasoning

### Commit Message Format
```
type: brief description

Longer description if needed, explaining what and why.

- Add specific details
- Include any breaking changes
- Reference issues: Fixes #123
```

## Documentation

### Adding New Documentation
- Follow the existing markdown style
- Add entries to [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
- Include "Back to Documentation Index" links
- Update README.md if adding major sections

### Documentation Standards
- Use clear, concise language
- Include code examples where helpful
- Cross-reference related documents
- Keep formatting consistent

## Questions and Support

- **Documentation**: Check [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions

## License

By contributing, you agree that your contributions will be licensed under the GNU GPL v3.0 license.

---

**[⬅ Back to Documentation Index](DOCUMENTATION_INDEX.md)**