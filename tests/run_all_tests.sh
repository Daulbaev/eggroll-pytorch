#!/bin/bash
# Run all tests for eggroll-pytorch

echo "=========================================="
echo "Running EGGROLL PyTorch Test Suite"
echo "=========================================="
echo ""

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing..."
    pip install pytest
fi

echo "1. Running basic tests..."
pytest tests/test_basic.py -v
echo ""

echo "2. Running JAX comparison tests..."
pytest tests/test_jax_comparison.py -v
echo ""

echo "3. Running JAX differences tests..."
pytest tests/test_jax_differences.py -v
echo ""

echo "4. Running comprehensive tests..."
pytest tests/test_comprehensive.py -v
echo ""

echo "=========================================="
echo "All tests completed!"
echo "=========================================="




