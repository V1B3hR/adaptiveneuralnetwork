# Testing Guide for Adaptive Neural Network

This guide explains how to run and interpret intelligence tests for the adaptive neural network system.

## Overview

The adaptive neural network includes comprehensive intelligence testing capabilities that evaluate various cognitive aspects of the AliveLoopNode agents. All tests follow ethical guidelines and include mandatory ethics compliance checks.

## Intelligence Test Categories

### 1. Problem Solving Tests
**Location**: `tests/test_basic_problem_solving.py`
**Purpose**: Evaluate the node's ability to solve optimization problems and adapt to environmental constraints.

**Key Tests**:
- Energy optimization under resource constraints
- Phase adaptation based on environmental conditions
- Memory-based prediction capabilities

**Run with**:
```bash
python -m unittest tests.test_basic_problem_solving
```

### 2. Learning Ability Tests
**Location**: `tests/test_adaptive_learning.py`
**Purpose**: Assess the node's capacity for learning from experience and integrating new knowledge.

**Key Tests**:
- Memory consolidation based on importance
- Social learning from other nodes
- Behavioral adaptation through experience
- Pattern recognition in environmental data

**Run with**:
```bash
python -m unittest tests.test_adaptive_learning
```

### 3. Cognitive Functioning Tests
**Location**: `tests/test_cognitive_functioning.py`
**Purpose**: Test cognitive abilities including attention management, decision making, and multi-tasking.

**Key Tests**:
- Attention management and focus redirection
- Decision making under stress conditions
- Resource management and optimization
- Communication processing capabilities
- Multi-task coordination

**Run with**:
```bash
python -m unittest tests.test_cognitive_functioning
```

### 4. Pattern Recognition Tests
**Location**: `tests/test_pattern_recognition.py`
**Purpose**: Evaluate the node's ability to identify, learn, and respond to various types of patterns.

**Key Tests**:
- Temporal pattern recognition in sequences
- Spatial pattern identification
- Behavioral pattern detection
- Social interaction pattern analysis
- Anomaly detection capabilities

**Run with**:
```bash
python -m unittest tests.test_pattern_recognition
```

### 5. Rigorous Intelligence Tests
**Location**: `tests/test_rigorous_intelligence.py`
**Purpose**: Comprehensive intelligence validation under challenging conditions, testing advanced problem solving, learning, memory, social intelligence, and ethics.

**Key Test Categories**:
- **Problem Solving & Reasoning**: Nested puzzles, ambiguous decisions, nonlinear outcomes
- **Learning & Adaptation**: Incremental difficulty, out-of-distribution generalization, catastrophic forgetting resistance
- **Memory & Pattern Recognition**: Sparse pattern recall, temporal sequences, conflicting memory resolution
- **Social/Collaborative Intelligence**: Multi-agent consensus, social signal ambiguity, adversarial influence resistance
- **Ethics & Safety**: Subtle violation detection, ethical dilemma resolution, audit bypass detection

**Run with**:
```bash
python -m unittest tests.test_rigorous_intelligence
```

## Running All Intelligence Tests

To run all intelligence tests at once:

```bash
# Run all intelligence tests
python -m unittest discover tests/ -p "test_*.py" -v

# Run with coverage (if coverage tool is installed)
coverage run -m unittest discover tests/ -p "test_*.py"
coverage report
```

## Test Environment Setup

### Prerequisites
1. **Python Dependencies**: Ensure all required packages are installed:
   ```bash
   pip install numpy scipy
   ```

2. **Random Seeds**: All tests use fixed random seeds for reproducibility:
   - `random.seed(42)`
   - `np.random.seed(42)`

3. **Test Framework**: Tests use Python's built-in `unittest` framework.

### Environment Variables
Set the Python path to include the project root:
```bash
export PYTHONPATH=/path/to/adaptiveneuralnetwork:$PYTHONPATH
```

Or run tests with:
```bash
PYTHONPATH=. python -m unittest tests.test_name
```

## Test Result Interpretation

### Success Criteria
- **All assertions pass**: Core functionality works as expected
- **Ethics compliance**: All tests include mandatory ethics checks
- **Performance metrics**: Tests validate quantitative performance criteria
- **Reproducibility**: Results are consistent across runs due to fixed seeds

### Common Test Patterns
Each test follows this structure:
1. **Setup**: Initialize node with specific parameters
2. **Action**: Execute the behavior being tested
3. **Assertion**: Verify expected outcomes
4. **Ethics Check**: Validate ethical compliance

### Expected Behaviors

#### Energy Management
- Nodes should conserve energy when resources are low
- Energy predictions should improve with experience
- Resource interactions should be mutually beneficial

#### Learning Capabilities
- Important memories should be retained longer
- Social learning should integrate trusted information
- Behavioral adaptations should improve performance

#### Cognitive Functions
- Attention should focus on relevant stimuli
- Decision making should adapt to stress levels
- Multi-tasking should handle concurrent demands

#### Pattern Recognition
- Temporal patterns should enable predictions
- Spatial patterns should optimize navigation
- Social patterns should inform trust relationships

## Ethics Framework Integration

### Mandatory Ethics Checks
Every test includes a `test_ethics_compliance()` method that validates:
- **Preserve Life**: Actions don't cause harm
- **Absolute Honesty**: Truthfulness in all operations
- **Privacy**: Respect for confidential information

### Ethics Module
The `core.ai_ethics` module provides:
- `audit_decision()`: Validate decision against 25-law framework
- `log_ethics_event()`: Record ethics compliance events
- `enforce_ethics_compliance()`: Automatically enforce compliance

### Example Ethics Check
```python
def test_ethics_compliance(self):
    decision_log = {
        "action": "test_action_name",
        "preserve_life": True,
        "absolute_honesty": True,
        "privacy": True
    }
    from core.ai_ethics import audit_decision
    audit = audit_decision(decision_log)
    self.assertTrue(audit["compliant"])
```

## Creating New Intelligence Tests

### Test Template
Use the provided templates in each test file as starting points:

1. **Copy template**: Start with appropriate category template
2. **Customize setup**: Modify node initialization for your test case
3. **Implement test logic**: Add specific behavior validation
4. **Include ethics check**: Always add mandatory ethics compliance test
5. **Document expected outcomes**: Clearly describe what the test validates

### Test Naming Convention
- File: `test_[category]_[specific_area].py`
- Class: `Test[CategoryName]`
- Methods: `test_[specific_behavior]()`
- Ethics: `test_ethics_compliance()`

### Key Guidelines
- **Reproducibility**: Always set random seeds
- **Isolation**: Each test should be independent
- **Clarity**: Document what aspect of intelligence is being tested
- **Ethics**: Include mandatory compliance checks
- **Assertions**: Use meaningful assertion messages

## Performance Monitoring

### Metrics to Track
- **Energy Efficiency**: How well nodes optimize energy usage
- **Learning Rate**: Speed of adaptation to new patterns
- **Decision Quality**: Appropriateness of responses to stimuli
- **Social Integration**: Effectiveness of collaborative behaviors

### Logging
Test results and ethics events are automatically logged. Check logs for:
- Compliance violations
- Performance degradation
- Unexpected behaviors

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure PYTHONPATH includes project root
2. **Missing Dependencies**: Install numpy and scipy
3. **Random Seed Issues**: Verify seeds are set in setUp()
4. **Ethics Failures**: Check that all ethical parameters are properly set

### Debugging
- Use verbose unittest output: `python -m unittest -v`
- Check individual test files independently
- Verify node initialization parameters
- Review ethics decision logs

## Contributing New Tests

When adding new intelligence tests:
1. Follow the established template structure
2. Include comprehensive documentation
3. Add ethics compliance checks
4. Update this guide with new test information
5. Ensure reproducibility with fixed random seeds

## Contact

For questions about the testing framework or to suggest improvements, please open an issue or pull request in the repository.