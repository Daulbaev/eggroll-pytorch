# EGGROLL PyTorch Test Suite

This directory contains comprehensive tests for the EGGROLL PyTorch implementation.

## Test Files

### `test_basic.py`
Basic functionality tests:
- Trainer initialization
- Training step execution
- Model forward interface
- Low-level `eggroll_step` function

### `test_jax_comparison.py`
Tests comparing PyTorch implementation with JAX version:
- Deterministic seed generation
- Low-rank perturbation generation
- Fitness normalization
- Update formula structure
- Parameter update direction
- Loss convergence

### `test_jax_differences.py`
Documents and tests known differences between PyTorch and JAX:
- Thread ID alternation (sigma sign)
- Lora generation method
- Update sign handling
- Noise reuse feature
- Group size normalization
- Optimizer differences

### `test_comprehensive.py`
Comprehensive tests for various scenarios:
- Different model sizes
- Different rank values
- Different numbers of workers
- Different sigma values
- Gradient clipping variations
- Fitness normalization toggle
- Multiple training steps
- Checkpoint saving
- Invalid loss handling
- CUDA support

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_basic.py -v
```

### Run specific test
```bash
pytest tests/test_basic.py::test_trainer_initialization -v
```

### Run with coverage
```bash
pytest tests/ --cov=eggroll --cov-report=html
```

### Use test runner script
```bash
./tests/run_all_tests.sh
```

## Test Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov
```

## Expected Results

All tests should pass. The JAX comparison tests verify that:
- Core algorithm components match JAX behavior
- Deterministic behavior is preserved
- Known differences are documented and tested

## Known Differences

See `COMPATIBILITY.md` for detailed analysis of differences between PyTorch and JAX implementations.




