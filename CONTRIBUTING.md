# Contributing to Adaptive Neural Network

Thank you for your interest in contributing to the Adaptive Neural Network project! This guide will help you get started with development and ensure your contributions align with the project's goals and standards.

## 🚀 Quick Start for Contributors

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/V1B3hR/adaptiveneuralnetwork.git
   cd adaptiveneuralnetwork
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode with all dependencies
   pip install -e .[dev]
   ```

3. **Verify setup**
   ```bash
   # Run tests
   pytest adaptiveneuralnetwork/tests/ -v
   
   # Run quick benchmark
   python scripts/run_benchmark.py --quick-test --epochs 1
   ```

## 📋 Development Workflow

### Branch Strategy
- `main` — Stable releases and hotfixes
- `develop` — Integration branch for new features
- `feature/feature-name` — New features and enhancements
- `fix/issue-description` — Bug fixes
- `docs/topic` — Documentation improvements

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run full test suite
   pytest adaptiveneuralnetwork/tests/ -v
   
   # Run linting
   ruff check adaptiveneuralnetwork/
   black --check adaptiveneuralnetwork/
   
   # Run type checking
   mypy adaptiveneuralnetwork/core/ adaptiveneuralnetwork/api/
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add description of your feature"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use the PR template
   - Link to relevant issues
   - Ensure CI passes

## 🎨 Code Style Guidelines

### Python Code Standards

- **Line length**: 100 characters (enforced by black)
- **Import organization**: Use ruff/isort formatting
- **Type hints**: Required for all public APIs
- **Docstrings**: Google-style docstrings for all public functions/classes

```python
def example_function(param: int, optional_param: Optional[str] = None) -> bool:
    """
    Brief description of the function.
    
    Args:
        param: Description of the parameter
        optional_param: Description of optional parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this is raised
    """
    pass
```

### Code Quality Tools

```bash
# Format code
black adaptiveneuralnetwork/

# Lint code
ruff check adaptiveneuralnetwork/ --fix

# Type check
mypy adaptiveneuralnetwork/core/ adaptiveneuralnetwork/api/

# Run all checks
make lint  # If using Makefile
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Benchmark tests**: Verify end-to-end functionality

### Writing Tests

```python
import pytest
import torch
from adaptiveneuralnetwork.core.nodes import NodeState, NodeConfig

class TestNodeState:
    """Test NodeState functionality."""
    
    def test_initialization(self):
        """Test basic initialization."""
        config = NodeConfig(num_nodes=10, hidden_dim=8)
        state = NodeState(config)
        
        assert state.hidden_state.shape == (1, 10, 8)
        
    @pytest.mark.slow
    def test_training_integration(self):
        """Test training integration (marked as slow)."""
        # Longer running test
        pass
```

### Test Markers

- `@pytest.mark.slow` — Tests that take >5 seconds
- `@pytest.mark.integration` — Integration tests
- `@pytest.mark.unit` — Unit tests (default)

### Running Tests

```bash
# All tests
pytest adaptiveneuralnetwork/tests/ -v

# Specific test file
pytest adaptiveneuralnetwork/tests/test_nodes.py -v

# Skip slow tests (for quick development)
pytest -m "not slow"

# Only integration tests
pytest -m integration
```

## 📝 Documentation Standards

### Code Documentation

- All public APIs must have docstrings
- Use Google-style docstring format
- Include examples for complex functions
- Document parameter types and return values

### README and Guides

- Use clear, concise language
- Include code examples
- Update relevant sections when adding features
- Follow markdown best practices

## 🐛 Issue Guidelines

### Reporting Bugs

Use the bug report template and include:

- **Environment**: OS, Python version, package versions
- **Reproduction steps**: Clear, minimal example
- **Expected vs actual behavior**
- **Error messages/stack traces**
- **Additional context**: Related issues, potential causes

### Feature Requests

Use the feature request template and include:

- **Problem description**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches
- **Implementation notes**: Technical considerations

### Issue Labels

- `bug` — Something isn't working
- `enhancement` — New feature or improvement
- `documentation` — Documentation improvements
- `good first issue` — Good for newcomers
- `help wanted` — Looking for contributors
- `priority:high` — Critical issues

## 🔧 Pull Request Guidelines

### PR Checklist

- [ ] Branch is up to date with develop
- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] New tests added for new functionality
- [ ] Documentation updated (if applicable)
- [ ] Changelog updated (if applicable)
- [ ] PR description is clear and complete

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review of code performed
- [ ] Tests pass locally
- [ ] Documentation updated
```

## 🎯 Contribution Priorities

### High Priority
- Bug fixes and stability improvements
- Performance optimizations
- Test coverage improvements
- Documentation enhancements

### Medium Priority
- New benchmark implementations
- Additional dataset support
- Configuration and API improvements
- Developer tooling enhancements

### Future Considerations
- Advanced features (continual learning, domain robustness)
- Alternative backends (JAX, custom CUDA)
- Visualization and analysis tools
- Integration with other frameworks

## 💬 Community Guidelines

### Communication Channels
- **GitHub Issues**: Bug reports, feature requests, technical discussions
- **GitHub Discussions**: General questions, ideas, community chat
- **Pull Requests**: Code review and technical feedback

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Maintain professional communication

## 🏆 Recognition

Contributors are recognized through:
- Contributor mentions in release notes
- GitHub contributor graphs and statistics
- Special recognition for significant contributions

## 📞 Getting Help

- **Technical questions**: Open a GitHub discussion or issue
- **Contribution guidance**: Comment on relevant issues or PRs
- **General support**: Check documentation and existing issues first

Thank you for contributing to making adaptive neural networks more accessible and powerful!