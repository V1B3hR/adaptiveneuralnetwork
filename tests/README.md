# Continual Learning Test Suite

This directory contains a layered testing infrastructure for the adaptive neural network's continual learning capabilities. The tests are organized into several categories to systematically validate different aspects of the system.

## Test Structure

### Core Test Categories

#### `sanity/` - Sanity Tests
Basic checks to detect common failure modes and ensure test validity:

- **`test_random_labels.py`** - Verifies that models achieve near-chance performance on random labels, detecting data leakage
- **`test_permutation.py`** - Confirms that permuting input features degrades performance, ensuring models learn meaningful patterns

#### `memory/` - Memory & Forgetting Tests  
Tests related to catastrophic forgetting and memory systems:

- **`test_forgetting_matrix.py`** - Validates forgetting matrix construction and continual learning performance tracking

#### `fewshot/` - Few-Shot Learning Tests
Tests for learning with limited data:

- **`test_fewshot_basic.py`** - Basic few-shot learning functionality, forward pass validation, and probability consistency

#### `drift/` - Distribution Drift Tests
Tests for concept drift and distribution shift handling:

- **`test_gaussian_drift.py`** - Validates drift injection utilities and measures their impact on model performance

### Shared Infrastructure

- **`conftest.py`** - Shared pytest fixtures for creating synthetic DataLoaders and common test components

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run by Category
```bash
# Sanity checks only
pytest -m sanity -v

# Memory/forgetting tests only  
pytest -m memory -v

# Few-shot learning tests
pytest -m fewshot -v

# Distribution drift tests
pytest -m drift -v
```

### Run Specific Test Files
```bash
# Random label detection
pytest tests/sanity/test_random_labels.py -v

# Forgetting matrix functionality
pytest tests/memory/test_forgetting_matrix.py -v
```

### Quiet Mode (Minimal Output)
```bash
pytest tests/ -q
```

## Test Design Principles

### 1. Synthetic Data
All tests use synthetic data to ensure:
- Fast execution (under 10 seconds total)
- Reproducible results via manual seeding
- No dependency on external datasets
- Controlled test scenarios

### 2. Robust Assertions
Tests use bounds-based assertions rather than exact values to avoid flakiness:
- Performance comparisons use meaningful margins (e.g., ">0.1 improvement")
- Random processes include appropriate tolerances
- Chance-level performance is bounded rather than exact

### 3. Relational Checks
Tests focus on relative comparisons rather than absolute values:
- "Permutation should reduce accuracy" rather than "accuracy should be 0.85"
- "Strong drift should hurt more than mild drift"
- "Structured data should outperform random labels"

### 4. Controlled Randomness
All random processes use manual seeding for reproducibility:
- Fixed seeds for dataset generation
- Separate seeds for train/validation splits
- Deterministic shuffling in DataLoaders

## Test Markers

Tests are tagged with markers for selective execution:

- `@pytest.mark.sanity` - Basic sanity checks
- `@pytest.mark.memory` - Memory and forgetting tests
- `@pytest.mark.fewshot` - Few-shot learning tests
- `@pytest.mark.drift` - Distribution drift tests

## Expected Outcomes

### Sanity Tests
- Random label accuracy: ~20% ± 30% (near chance for 5-class)
- Permutation should reduce accuracy by >10%
- Structured data should outperform random labels by >15%

### Memory Tests
- Forgetting matrix construction should handle various task scenarios
- Continual learning should show measurable but bounded forgetting
- Metrics should correctly compute average/maximum forgetting

### Few-Shot Tests  
- Models should handle small batch sizes (1-5 examples per class)
- Few-shot should outperform chance by >10% but underperform regular learning
- Output shapes and probability distributions should be valid

### Drift Tests
- Gaussian drift should inject measurable noise (σ ± 0.1)
- Constant shift should be exactly applied
- Drift should cause >10% performance drop
- Stronger drift should cause larger drops than mild drift

## Integration with Continual Learning System

These tests validate the utilities and infrastructure used by the main `ContinualLearningSystem`:

- `utils/drift.py` functions are tested in `drift/` tests
- `utils/forgetting.py` functions are tested in `memory/` tests  
- The validation-aware training loop uses patterns validated in `sanity/` tests
- Few-shot capabilities support the broader continual learning framework

## Maintenance Notes

- Tests should complete in <10 seconds on a typical CPU
- Add new test categories as subdirectories with appropriate markers
- Keep synthetic data patterns simple but learnable
- Update this README when adding new test categories or changing test structure