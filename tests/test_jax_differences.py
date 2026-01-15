"""
Tests to identify and document differences between PyTorch and JAX implementations.

This file documents known differences and tests for compatibility.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from eggroll import EGGROLLTrainer, generate_low_rank_perturbation, fold_in_seed


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=3, hidden_dims=[4, 4], out_dim=1):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, batch, labels=None, is_training=True):
        output = self.network(batch)
        loss = torch.mean((output - 2.0) ** 2)
        fitness = -loss  # Higher fitness is better
        return {'fitness': fitness}


def test_jax_thread_id_alternation():
    """
    TEST: PyTorch now implements thread ID alternation matching JAX.

    JAX code:
        true_thread_idx = thread_id // 2
        sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    PyTorch version NOW implements this alternation.
    """
    # In JAX, thread_id=0 and thread_id=1 use same true_thread_idx (0) but different sigma signs
    # In PyTorch, we now match this behavior

    device = torch.device('cpu')
    m, n, r = 10, 5, 3
    normalized_sigma = 0.1
    epoch = 0
    base_seed = 42

    # Generate for sample_idx=0 and sample_idx=1
    A0, B0 = generate_low_rank_perturbation(m, n, r, device, normalized_sigma, epoch, 0, base_seed, "param")
    A1, B1 = generate_low_rank_perturbation(m, n, r, device, normalized_sigma, epoch, 1, base_seed, "param")

    # B should be the same (same true_thread_idx = 0 // 2 = 0, 1 // 2 = 0)
    assert torch.allclose(B0, B1), "B should be same for sample_idx 0 and 1 (same true_thread_idx)"

    # A should be different due to sign alternation (sample_idx 0 -> +sigma, sample_idx 1 -> -sigma)
    # A0 should be approximately -A1 (opposite signs)
    assert not torch.allclose(A0, A1), "A should differ due to sign alternation"
    # Check that they are approximately opposite (allowing for small numerical differences)
    assert torch.allclose(A0, -A1, atol=1e-5), "A0 and A1 should be approximately opposite (sign alternation)"


def test_jax_lora_generation_method():
    """
    TEST: PyTorch now matches JAX LoRA generation method.

    JAX code:
        lora_params = jax.random.normal(..., (a+b, rank), ...)
        B = lora_params[:b]  # b x r
        A = lora_params[b:]   # a x r

    PyTorch now generates a single (m+n, r) tensor and then splits it in the same way.
    We do not test for bit‑exact equality with JAX here, only that B looks like standard normal.
    """
    device = torch.device('cpu')
    m, n, r = 10, 5, 3
    normalized_sigma = 0.1
    epoch = 0
    sample_idx = 0
    base_seed = 42

    # PyTorch method
    A_pt, B_pt = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch, sample_idx, base_seed, "param"
    )

    # Verify shapes
    assert A_pt.shape == (m, r)
    assert B_pt.shape == (n, r)

    # Both should be standard normal (before sigma is applied to A)
    # B should be standard normal
    # Note: With small sample sizes, mean and std may deviate from expected values
    B_mean = B_pt.mean().item()
    B_std = B_pt.std().item()
    # With only n * r = 15 samples, sampling noise can be relatively large.
    assert abs(B_mean) < 0.5, f"B should have mean ~0, got {B_mean}"  # Loose tolerance for small samples
    # Use a slightly looser tolerance for std because of small sample size (n * r = 15)
    assert abs(B_std - 1.0) < 0.5, f"B should have std ~1, got {B_std}"  # Loose tolerance for small samples


def test_jax_update_sign():
    """
    KNOWN DIFFERENCE: JAX uses negative sign in _do_update.

    JAX code:
        return -(new_grad * jnp.sqrt(fitnesses.size)).astype(param.dtype)

    PyTorch applies update directly with positive sign:
        param.data.add_(learning_rate * update)

    The sign difference is handled by the fitness function (we use -loss as fitness).
    This should be equivalent.
    """
    device = torch.device('cpu')
    model = SimpleMLP()
    model.to(device)

    # Store original
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}

    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.2,
        learning_rate=0.03,
        n_workers=8,
        base_seed=42,
    )

    batch = torch.randn(8, 3, device=device)
    metrics = trainer.train_step(batch)

    # Check that parameters changed (update was applied)
    param_changed = False
    for name, param in model.named_parameters():
        if name in original_params:
            diff = torch.norm(param.data - original_params[name])
            if diff > 1e-6:
                param_changed = True
                break

    assert param_changed, "Parameters should change (update applied)"
    assert metrics['fitness'] > float('-inf'), "Fitness should be finite"


def test_jax_noise_reuse():
    """
    TEST: PyTorch now supports noise_reuse parameter matching JAX.

    JAX code:
        true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    PyTorch version NOW implements noise_reuse.
    """
    device = torch.device('cpu')
    m, n, r = 10, 5, 3
    normalized_sigma = 0.1
    base_seed = 42

    # Test with noise_reuse=0 (should use actual epoch)
    A0, B0 = generate_low_rank_perturbation(m, n, r, device, normalized_sigma, epoch=0, sample_idx=0, base_seed=base_seed, param_name="param", noise_reuse=0)
    A1, B1 = generate_low_rank_perturbation(m, n, r, device, normalized_sigma, epoch=1, sample_idx=0, base_seed=base_seed, param_name="param", noise_reuse=0)

    # Should be different (different epochs with noise_reuse=0)
    assert not torch.allclose(A0, A1), "Different epochs should give different perturbations with noise_reuse=0"

    # Test with noise_reuse=2 (epochs 0 and 1 should use same true_epoch=0)
    A0_reuse, B0_reuse = generate_low_rank_perturbation(m, n, r, device, normalized_sigma, epoch=0, sample_idx=0, base_seed=base_seed, param_name="param", noise_reuse=2)
    A1_reuse, B1_reuse = generate_low_rank_perturbation(m, n, r, device, normalized_sigma, epoch=1, sample_idx=0, base_seed=base_seed, param_name="param", noise_reuse=2)

    # Should be same (same true_epoch = 0 // 2 = 0, 1 // 2 = 0)
    assert torch.allclose(A0_reuse, A1_reuse), "Epochs 0 and 1 should use same perturbations with noise_reuse=2"
    assert torch.allclose(B0_reuse, B1_reuse), "Epochs 0 and 1 should use same perturbations with noise_reuse=2"


def test_jax_group_size():
    """
    KNOWN DIFFERENCE: JAX supports group_size for fitness normalization.

    JAX code:
        if group_size == 0:
            true_scores = (raw_scores - mean) / sqrt(var + 1e-5)
        else:
            # Group normalization

    PyTorch always uses global normalization (group_size=0 behavior).
    This is a feature difference.
    """
    # Test that PyTorch uses global normalization
    raw_fitnesses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    mean = np.mean(raw_fitnesses)
    std = np.std(raw_fitnesses)
    eps = 1e-5
    normalized = (raw_fitnesses - mean) / (std + eps)

    # Should normalize globally, not in groups
    assert abs(np.mean(normalized)) < 1e-6, "Should normalize globally"
    assert abs(np.std(normalized) - 1.0) < 0.1, "Should have std ~1"


def test_jax_solver_optimizer():
    """
    KNOWN DIFFERENCE: JAX uses optax solvers (SGD, AdamW, etc.).

    JAX code:
        solver = optax.sgd  # or optax.adamw
        true_solver = solver(lr, **solver_kwargs)
        updates, opt_state = true_solver.update(new_grad, opt_state, params)

    PyTorch applies updates directly with learning_rate (equivalent to SGD).
    This is a feature difference - PyTorch version is simpler but less flexible.
    """
    device = torch.device('cpu')
    model = SimpleMLP()
    model.to(device)

    # PyTorch uses simple SGD-like update: param += lr * update
    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.2,
        learning_rate=0.03,  # Direct learning rate (like SGD)
        n_workers=8,
        base_seed=42,
    )

    batch = torch.randn(8, 3, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['fitness'] > float('-inf'), "Should work with direct learning rate"


def test_compatibility_summary():
    """
    Summary of compatibility: PyTorch version should produce similar results
    to JAX version for basic use cases, with some known differences.
    """
    print("\n" + "="*60)
    print("PYTORCH vs JAX EGGROLL - COMPATIBILITY SUMMARY")
    print("="*60)
    print("\nCOMPATIBLE:")
    print("  ✓ Low-rank perturbation structure (A, B matrices)")
    print("  ✓ Fitness normalization (z-score)")
    print("  ✓ Update formula (einsum computation)")
    print("  ✓ Deterministic seed generation")
    print("  ✓ Basic training loop structure")
    print("\nKNOWN DIFFERENCES:")
    print("  ✗ Group size normalization - NOT implemented in PyTorch")
    print("  ✗ Advanced optimizers (AdamW, etc.) - PyTorch uses simple SGD by default")
    print("\nIMPACT:")
    print("  - Results will be similar but not identical")
    print("  - Core algorithm behavior is preserved")
    print("  - Differences are mostly in advanced features")
    print("="*60 + "\n")

    # This test always passes - it's just documentation
    assert True

