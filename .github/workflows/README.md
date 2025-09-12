# CI/CD Workflows

## CI - Train, Test, Coverage & Artifacts

This workflow implements the complete CI pipeline as requested: **train → test → coverage badge → artifact upload**

### Workflow Stages

1. **Training Phase** 🧠
   - Runs the adaptive neural network simulation (`runsimulation.py`)
   - Generates training artifacts (database, JSON outputs, visualizations)

2. **Testing Phase** ✅
   - Executes complete test suite with pytest
   - Generates coverage reports (HTML, XML, terminal)
   - Current coverage: **71%**

3. **Coverage Badge Generation** 📊
   - Creates coverage badge JSON for shields.io
   - Generates markdown coverage summary
   - Auto-updates coverage badges in README

4. **Artifact Upload** 📦
   - Test reports and coverage data
   - Training outputs (JSON, DB files, images)  
   - Badge data for external services
   - 30-day retention policy

### Triggers

- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

### Matrix Testing

- Python 3.11 and 3.12
- Ubuntu latest environment
- Pip dependency caching for performance

### Artifacts Generated

- `test-results-pythonX.X/`: Test reports and coverage
- `coverage-badge/`: Coverage badge JSON
- `training-artifacts-pythonX.X/`: Neural network outputs

### PR Integration

On pull requests, the workflow automatically comments the coverage report for easy review.

### Badges

The README now includes:
- CI status badge (workflow status) 
- Coverage percentage badge (auto-updated)