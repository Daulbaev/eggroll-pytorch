# PyTorch vs JAX EGGROLL - Compatibility Analysis

This document describes the compatibility between the PyTorch and JAX implementations of EGGROLL.

## Core Algorithm Compatibility

### ✅ Compatible Components

1. **Low-rank perturbation structure**: Both use A (m×r) and B (n×r) matrices
2. **Fitness normalization**: Both use z-score normalization: `(x - mean) / sqrt(var + eps)`
3. **Update formula**: Both use `einsum('nir,njr->ij', A_weighted, B) / num_envs`
4. **Scaling factor**: Both multiply by `sqrt(num_samples)` before applying learning rate
5. **Deterministic generation**: Both use deterministic seed generation (though methods differ)

## Known Differences

### 1. Thread ID Alternation (Sigma Sign)

**JAX Implementation:**
```python
true_thread_idx = thread_id // 2
sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)
```

**PyTorch Implementation:**
- ✅ **NOW SUPPORTED!** Fully implemented, matching JAX behavior
- Uses `sample_idx // 2` for `true_thread_idx` (matching JAX)
- Alternates sigma sign: positive for even `sample_idx`, negative for odd `sample_idx`

**What is Thread ID Alternation?**
Thread ID alternation means that perturbations alternate in sign based on the worker thread ID:
- Even thread IDs (0, 2, 4, ...) use positive sigma: `+sigma`
- Odd thread IDs (1, 3, 5, ...) use negative sigma: `-sigma`
- This creates pairs of perturbations with opposite signs, which can help with exploration

**Impact:**
- ✅ Fully compatible with JAX implementation

### 2. Lora Parameter Generation

**JAX Implementation:**
```python
lora_params = jax.random.normal(..., (a+b, rank), ...)
B = lora_params[:b]  # b x r
A = lora_params[b:]   # a x r
```

**PyTorch Implementation:**
```python
A = torch.randn(m, r, ...)
B = torch.randn(n, r, ...)
```

**Impact:**
- Different random sequences but equivalent statistical properties
- Results will differ but behavior is equivalent
- This is a **minor difference** (implementation detail)

### 3. Update Sign

**JAX Implementation:**
```python
return -(new_grad * jnp.sqrt(fitnesses.size))
```

**PyTorch Implementation:**
```python
param.data.add_(learning_rate * update)
# where update already has correct sign via fitness = -loss
```

**Impact:**
- Equivalent behavior (sign handled via fitness function)
- This is **compatible** (different but equivalent)

### 4. Noise Reuse Feature

**JAX Implementation:**
```python
true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse
```

**PyTorch Implementation:**
- ✅ **NOW SUPPORTED!** Fully implemented, matching JAX behavior

**What is Noise Reuse?**
Noise reuse allows reusing the same random perturbations across multiple epochs. When `noise_reuse = N > 0`, epochs are grouped into blocks of size N, and all epochs in the same block use identical perturbations. For example:
- `noise_reuse = 2`: epochs 0-1 share perturbations, epochs 2-3 share perturbations, etc.
- `noise_reuse = 0`: each epoch uses different perturbations (default)

**Impact:**
- ✅ Fully compatible with JAX implementation

### 5. Group Size Normalization

**JAX Implementation:**
```python
if group_size == 0:
    true_scores = (raw_scores - mean) / sqrt(var + eps)
else:
    group_scores = raw_scores.reshape((-1, group_size))
    true_scores = (group_scores - mean(group_scores, axis=-1)) / sqrt(var(raw_scores) + eps)
    true_scores = true_scores.ravel()
```

**PyTorch Implementation:**
- ✅ **NOW SUPPORTED!** Fully implemented, matching JAX behavior

**What is Group Normalization?**
Group normalization divides fitness values into groups and normalizes within each group. Each group's fitness values have their group mean subtracted, but all groups use the same global standard deviation. This is useful for comparing fitness within groups while maintaining global scale.

**Impact:**
- ✅ Fully compatible with JAX implementation

### 6. Optimizer/Solver

**JAX Implementation:**
- Uses optax solvers (SGD, AdamW, etc.)
- Supports optimizer state and momentum

**PyTorch Implementation:**
- ✅ **NOW SUPPORTED!** Supports `torch.optim` optimizers (SGD, Adam, AdamW)
- Supports optimizer state and momentum via PyTorch optimizers
- Can also use simple SGD-like update (default, when optimizer=None)

**Impact:**
- ✅ Fully compatible with JAX implementation
- PyTorch version supports all major optimizers via torch.optim

## Summary

| Component | Status | Impact |
|-----------|--------|--------|
| Low-rank structure | ✅ Compatible | None |
| Fitness normalization | ✅ Compatible | None |
| Update formula | ✅ Compatible | None |
| Thread ID alternation | ✅ Supported | None - Fully implemented |
| Lora generation | ⚠️ Different method | Low - Equivalent behavior |
| Update sign | ✅ Compatible | None |
| Noise reuse | ✅ Supported | None - Fully implemented |
| Group normalization | ✅ Supported | None - Fully implemented |
| Optimizer | ✅ Supported | None - Now supports torch.optim (SGD, Adam, AdamW) |

## Recommendations

1. **For research reproducibility**: All major features are now implemented, including thread ID alternation. Results should match JAX implementation closely.

2. **For practical use**: The PyTorch implementation should work well for most use cases, as the core algorithm is preserved.

3. **For feature parity**: To match JAX exactly, would need to implement:
   - ✅ Thread ID alternation (NOW SUPPORTED)
   - ✅ Noise reuse (NOW SUPPORTED)
   - ✅ Group normalization (NOW SUPPORTED)
   - ✅ Advanced optimizers (NOW SUPPORTED via torch.optim)

## Testing

Run compatibility tests:
```bash
pytest tests/test_jax_comparison.py -v
pytest tests/test_jax_differences.py -v
```

These tests verify:
- Deterministic behavior
- Correct shapes and structures
- Fitness normalization
- Update formula structure
- Known differences are documented

