# EGGROLL PyTorch

Unofficial PyTorch implementation of **EGGROLL** (Evolution Guided General Optimization via Low-rank Learning), an algorithm for training neural networks without backpropagation.

**This is a PyTorch port of the JAX implementation from [HyperscaleES](https://github.com/ESHyperscale/HyperscaleES).**

> **Note:** A significant portion of this PyTorch port (including code and documentation) was developed with the assistance of large language models (LLMs) and reviewed and adapted by a human author.

## Overview

EGGROLL is an evolutionary strategy algorithm that uses low-rank perturbations to efficiently optimize neural networks. Instead of computing gradients via backpropagation, EGGROLL:

1. Generates low-rank perturbations of the model parameters
2. Evaluates fitness for each perturbed model
3. Updates parameters based on weighted perturbations

The key innovation is using low-rank matrices (A, B) to represent perturbations, making the algorithm memory-efficient for large models.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install torch numpy
```

## Quick Start

```python
import torch
import torch.nn as nn
from eggroll import EGGROLLTrainer

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, batch, labels=None, is_training=True):
        output = self.net(batch)
        loss = torch.mean((output - 1.0) ** 2)
        return {'loss': loss}

# Create model and trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)

trainer = EGGROLLTrainer(
    model=model,
    device=device,
    rank=8,              # Rank of low-rank perturbation
    sigma=0.2,           # Perturbation scale
    learning_rate=0.03,   # Learning rate
    n_workers=64,        # Number of perturbation samples per iteration
    grad_clip=1.0,        # Gradient clipping threshold
    normalize_fitness=True,
    base_seed=0,
    optimizer='adamw',    # Optional: 'sgd', 'adam', 'adamw', or None for simple SGD
    optimizer_kwargs={'betas': (0.9, 0.999)},  # Optional optimizer arguments
)

# Training loop
for epoch in range(100):
    batch = torch.randn(64, 10, device=device)
    metrics = trainer.train_step(batch)
    print(f"Epoch {epoch}: Loss = {metrics['loss']:.6f}")
```

See `examples/simple_example.py` for a complete working example.

## API Reference

### `EGGROLLTrainer`

Main trainer class for EGGROLL optimization.

#### Parameters

- **model** (`nn.Module`): PyTorch model to train. Must have a `forward` method that accepts `(batch, labels=None, is_training=True)` and returns a dict with `'loss'` key.
- **device** (`torch.device`): Device to train on (CPU or CUDA).
- **rank** (`int`, default=30): Rank of low-rank perturbation. Lower rank = less memory but potentially less expressive.
- **sigma** (`float`, default=0.01): Perturbation scale (σ in the paper).
- **learning_rate** (`float`, default=0.001): Learning rate for weight updates (α_t in the paper).
- **n_workers** (`int`, default=30): Number of perturbation samples per iteration (N_workers in the paper).
- **grad_clip** (`float`, optional, default=1.0): Gradient clipping threshold. Set to `None` to disable.
- **use_weighted_loss** (`bool`, default=False): Whether to use weighted loss (not used in standalone version).
- **log_steps** (`int`, default=100): Frequency of logging.
- **save_steps** (`int`, optional): Frequency of checkpoint saving. Set to `None` to disable.
- **results_dir** (`str`, optional): Directory to save checkpoints and logs.
- **normalize_fitness** (`bool`, default=True): Whether to normalize fitness values using z-score (recommended).
- **base_seed** (`int`, default=0): Base random seed for deterministic perturbation generation.
- **optimizer** (`str` or `torch.optim.Optimizer`, optional): Optimizer to use:
    - `None`: Simple SGD-like update (default, matches original implementation)
    - `'sgd'`: PyTorch SGD optimizer
    - `'adam'`: PyTorch Adam optimizer
    - `'adamw'`: PyTorch AdamW optimizer (recommended, matches JAX default)
    - `torch.optim.Optimizer`: Pre-initialized optimizer instance
- **optimizer_kwargs** (`dict`, optional): Additional keyword arguments for optimizer (e.g., `{'betas': (0.9, 0.999)}` for AdamW).
- **noise_reuse** (`int`, default=0): Number of epochs to reuse the same noise. If > 0, epochs are grouped into blocks and share perturbations. For example, `noise_reuse=2` means epochs 0-1 share perturbations, epochs 2-3 share perturbations, etc.
- **group_size** (`int`, default=0): Size of groups for group-wise fitness normalization. If 0, uses global normalization (default). If > 0, fitness values are grouped and normalized within each group (subtract group mean, divide by global std). Must divide `n_workers` evenly. For example, `group_size=4` with `n_workers=8` creates 2 groups of 4 samples each.

#### Methods

- **`train_step(batch)`**: Perform one training step. Returns a dictionary of metrics.
- **`save_checkpoint(checkpoint_name=None)`**: Save model checkpoint.

### `eggroll_step`

Low-level function to perform one EGGROLL optimization step.

```python
from eggroll import eggroll_step

loss, metrics = eggroll_step(
    model=model,
    batch=batch,
    device=device,
    n_workers=64,
    rank=8,
    sigma=0.2,
    learning_rate=0.03,
    grad_clip=1.0,
    normalize_fitness=True,
    base_seed=0,
    epoch=0,
)
```

## Algorithm Details

EGGROLL implements the following update rule (from the paper, formula 8):

```
µ_{t+1} = µ_t + (α_t / N_workers) * Σ_{i=1}^{N_workers} E_{i,t} * f(M = µ_t + σE_{i,t})
```

where:
- `µ_t` are the current model parameters
- `E_{i,t} = (1/√r) * A_{i,t} * B_{i,t}^T` is the low-rank perturbation
- `f(M)` is the fitness function (typically `-loss`)
- `α_t` is the learning rate
- `N_workers` is the number of perturbation samples
- `σ` is the perturbation scale
- `r` is the rank of the low-rank perturbation

### Key Features

1. **Low-rank perturbations**: Instead of full-rank perturbations (which would be memory-intensive), EGGROLL uses low-rank matrices A (m×r) and B (n×r) where r << min(m, n).

2. **On-the-fly application**: For 2D parameters (Linear layers), perturbations are applied on-the-fly using forward hooks, avoiding the need to form full perturbation matrices.

3. **Deterministic generation**: Perturbations are generated deterministically based on seeds, ensuring reproducibility.

4. **Fitness normalization**: By default, fitness values are normalized using z-score normalization, matching the official JAX implementation.

## Model Requirements

Your PyTorch model must:

1. Have a `forward` method with signature: `forward(batch, labels=None, is_training=True)`
2. Return a dictionary with at least a `'loss'` key
3. The loss should be a scalar tensor

Example:

```python
class MyModel(nn.Module):
    def forward(self, batch, labels=None, is_training=True):
        output = self.network(batch)
        loss = self.loss_fn(output, labels) if labels is not None else self.loss_fn(output)
        return {'loss': loss}
```

## Differences from JAX Implementation

This PyTorch implementation:

- Removes dependencies on domain-specific code (e.g., affinity fitness computation)
- Uses PyTorch's forward hooks for efficient on-the-fly perturbation application
- Maintains compatibility with the core EGGROLL algorithm as described in the paper
- Uses deterministic seed generation compatible with PyTorch's random number generator

## Examples

See `examples/simple_example.py` for a complete example training an MLP to minimize squared error.

### Using Different Optimizers

```python
# Simple SGD (default, no optimizer)
trainer = EGGROLLTrainer(model=model, device=device, learning_rate=0.01, ...)

# AdamW optimizer (matching JAX default)
trainer = EGGROLLTrainer(
    model=model,
    device=device,
    learning_rate=0.03,
    optimizer='adamw',
    optimizer_kwargs={'betas': (0.9, 0.999)},
    ...
)

# Adam optimizer
trainer = EGGROLLTrainer(
    model=model,
    device=device,
    learning_rate=0.01,
    optimizer='adam',
    ...
)

# Custom optimizer instance
custom_optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
trainer = EGGROLLTrainer(
    model=model,
    device=device,
    optimizer=custom_optimizer,
    ...
)
```

## Citation

If you use this implementation, please cite the original EGGROLL paper:

```bibtex
@article{eggroll2024,
  title={EGGROLL: Evolution Guided General Optimization via Low-rank Learning},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This implementation is licensed under GPL-3.0, matching the original JAX implementation.

## Acknowledgments

This is an unofficial PyTorch port of the JAX implementation in [HyperscaleES](https://github.com/ESHyperscale/HyperscaleES). The original JAX implementation is the work of the HyperscaleES team.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Testing

Run the test suite to verify functionality and compare with JAX implementation:

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_basic.py -v              # Basic functionality
pytest tests/test_jax_comparison.py -v      # JAX compatibility tests
pytest tests/test_jax_differences.py -v     # Known differences
pytest tests/test_comprehensive.py -v       # Comprehensive scenarios

# Or use the test runner script
./tests/run_all_tests.sh
```

## Compatibility with JAX Implementation

This PyTorch implementation is **functionally compatible** with the JAX version for core algorithm behavior, but there are some **known differences**:

### ✅ Compatible
- Low-rank perturbation structure (A, B matrices)
- Fitness normalization (z-score)
- Update formula computation
- Deterministic seed generation
- Core training loop

### ✅ Fully Compatible Features
1. **Thread ID alternation**: ✅ **NOW SUPPORTED!** PyTorch now alternates sigma sign based on `sample_idx % 2`, matching JAX behavior
   - Even sample indices (0, 2, 4, ...) use positive sigma
   - Odd sample indices (1, 3, 5, ...) use negative sigma

See `tests/COMPATIBILITY.md` for detailed analysis.

**Impact**: The PyTorch implementation now matches JAX behavior for all major features. Results should be very similar to the JAX version.

#### Noise Reuse Feature

**Noise reuse** is now supported! It allows reusing the same random perturbations across multiple epochs.

When `noise_reuse > 0`, the effective epoch used for generating perturbations is `epoch // noise_reuse` instead of the actual `epoch`. This means:
- If `noise_reuse = 2`: epochs 0-1 use the same perturbations, epochs 2-3 use the same perturbations, etc.
- If `noise_reuse = 0`: each epoch uses different perturbations (default behavior)

**Usage:**
```python
trainer = EGGROLLTrainer(
    model=model,
    device=device,
    noise_reuse=2,  # Reuse perturbations every 2 epochs
    ...
)
```

#### Thread ID Alternation Feature

**Thread ID alternation** is now supported! It alternates the sign of sigma based on the worker thread ID, matching JAX behavior.

When enabled (default, always active):
- Even worker indices (0, 2, 4, ...) use positive sigma: `+sigma`
- Odd worker indices (1, 3, 5, ...) use negative sigma: `-sigma`
- This creates pairs of perturbations with opposite signs, which can help with exploration

This feature is automatically enabled and matches the JAX implementation exactly.

#### Group Normalization Feature

**Group normalization** is now supported! It allows normalizing fitness values within groups instead of globally.

When `group_size > 0`, fitness values are divided into groups, and each group is normalized separately:
- Each group's fitness values have their group mean subtracted
- All groups use the same global standard deviation (from all raw fitness values)
- This is useful when you want to compare fitness within groups while maintaining global scale

**Usage:**
```python
trainer = EGGROLLTrainer(
    model=model,
    device=device,
    n_workers=8,
    group_size=4,  # 2 groups of 4 samples each
    ...
)
```

## Disclaimer

This is an unofficial implementation. For the official implementation, please refer to the [HyperscaleES repository](https://github.com/ESHyperscale/HyperscaleES).

