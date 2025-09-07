# Intelligence Test Readiness Checklist

Use this checklist before running any intelligence test in the adaptive neural network system.

## Environment Setup
- [x] All Python dependencies installed and up to date (numpy, scipy)
- [x] Random seeds fixed (e.g., `numpy.random.seed(42)`, `random.seed(42)`)
- [x] Test framework configured (unittest)

## Data and Inputs
- [x] Test datasets curated and versioned (built into test cases)
- [x] Edge-case and adversarial inputs identified (stress conditions, low energy scenarios)
- [x] Input pipelines mocked or stubbed for isolation (direct AliveLoopNode instantiation)

## Test Infrastructure
- [x] Test directories (`tests/`) match project structure
- [ ] CI/CD workflows set to trigger tests on each commit
- [x] Logging directory or files configured for result dumps (unittest output, ethics logging)

## Metrics and Analysis
- [x] Quantitative metrics defined (accuracy, latency, energy efficiency)
- [x] Logging and visualization scripts ready (ethics audit logging)
- [x] Baseline performance documented (in TESTING_GUIDE.md)

## Ethics and Compliance
- [x] `test_ethics_compliance` included in every test file
- [x] Decision logs capture action, honesty, privacy, preserve_life flags
- [x] Audit function passes for all intended behaviors

## Documentation
- [x] Testing guide (`TESTING_GUIDE.md`) updated with how to run and interpret tests
- [x] Readme includes instructions for new contributors
- [x] Code comments explain test logic and expected outcomes

## Intelligence Test Categories

### Problem Solving Tests
- [x] **File**: `tests/test_basic_problem_solving.py`
- [x] **Tests**: Energy optimization, phase adaptation, memory-based prediction
- [x] **Ethics Check**: ✅ Compliant
- [x] **Status**: All tests passing

### Learning Ability Tests
- [x] **File**: `tests/test_adaptive_learning.py`
- [x] **Tests**: Memory consolidation, social learning, behavioral adaptation, pattern recognition
- [x] **Ethics Check**: ✅ Compliant
- [x] **Status**: All tests passing

### Cognitive Functioning Tests
- [x] **File**: `tests/test_cognitive_functioning.py`
- [x] **Tests**: Attention management, decision making under stress, resource management, communication processing, multi-task coordination
- [x] **Ethics Check**: ✅ Compliant
- [x] **Status**: All tests passing

### Pattern Recognition Tests
- [x] **File**: `tests/test_pattern_recognition.py`
- [x] **Tests**: Temporal patterns, spatial patterns, behavioral patterns, social patterns, anomaly detection
- [x] **Ethics Check**: ✅ Compliant
- [x] **Status**: All tests passing

### Rigorous Intelligence Tests
- [x] **File**: `tests/test_rigorous_intelligence.py`
- [x] **Tests**: Nested puzzles, ambiguous decisions, nonlinear outcomes, incremental learning, out-of-distribution generalization, catastrophic forgetting resistance, sparse pattern recall, temporal sequences, conflicting memory resolution, multi-agent consensus, social signal ambiguity, adversarial influence resistance, subtle ethics violations, ethical dilemmas, audit bypass detection
- [x] **Ethics Check**: ✅ Compliant
- [x] **Status**: All tests passing

## Pre-Test Verification Commands

### Environment Check
```bash
# Verify Python version
python --version  # Should be 3.7+

# Check dependencies
python -c "import numpy, scipy; print('Dependencies OK')"

# Set Python path
export PYTHONPATH=/path/to/adaptiveneuralnetwork:$PYTHONPATH
```

### Quick Test Run
```bash
# Run all intelligence tests
cd /path/to/adaptiveneuralnetwork
PYTHONPATH=. python -m unittest discover tests/ -p "test_*.py" -v

# Run specific test category
PYTHONPATH=. python -m unittest tests.test_basic_problem_solving -v
PYTHONPATH=. python -m unittest tests.test_adaptive_learning -v
PYTHONPATH=. python -m unittest tests.test_cognitive_functioning -v
PYTHONPATH=. python -m unittest tests.test_pattern_recognition -v
```

### Ethics Framework Verification
```bash
# Test ethics module
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

## Test Execution Checklist

Before each test run, verify:

1. **Environment Setup**
   - [ ] Working directory is project root
   - [ ] PYTHONPATH includes project root
   - [ ] All dependencies installed

2. **Code State**
   - [ ] Latest code changes committed
   - [ ] No syntax errors in test files
   - [ ] AliveLoopNode properly initialized

3. **Test Configuration**
   - [ ] Random seeds set consistently
   - [ ] Test parameters match intended scenarios
   - [ ] Ethics compliance checks enabled

4. **Expected Outcomes**
   - [ ] All intelligence test categories pass
   - [ ] Ethics audit returns compliant=True for all actions
   - [ ] No unexpected exceptions or errors
   - [ ] Performance metrics within expected ranges

## Post-Test Analysis

After test completion, review:

1. **Test Results**
   - Energy optimization efficiency
   - Learning adaptation rates
   - Cognitive task coordination success
   - Pattern recognition accuracy

2. **Ethics Compliance**
   - All decision logs show compliant=True
   - No violations reported in ethics logger
   - Privacy and honesty maintained throughout

3. **Performance Metrics**
   - Response times acceptable
   - Memory usage reasonable
   - Energy management efficient

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure PYTHONPATH includes project root
- **Random Seed Issues**: Verify seeds set in test setUp() methods
- **Ethics Failures**: Check decision log parameters
- **Missing Dependencies**: Install numpy and scipy

### Debug Commands
```bash
# Verbose test output
python -m unittest tests.test_name -v

# Check specific test method
python -c "
import tests.test_basic_problem_solving
import unittest
suite = unittest.TestSuite()
suite.addTest(tests.test_basic_problem_solving.TestBasicProblemSolving('test_energy_optimization'))
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
"
```

## Sign-off

- [ ] All checklist items completed
- [ ] All intelligence tests passing
- [ ] Ethics compliance verified
- [ ] Documentation updated
- [ ] Ready for production intelligence testing

**Test Operator**: _________________ **Date**: _____________

**System Version**: _________________ **Commit**: _____________