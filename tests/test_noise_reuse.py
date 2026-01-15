"""
Tests for noise_reuse feature in EGGROLL PyTorch implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from eggroll import EGGROLLTrainer, generate_low_rank_perturbation, fold_in_seed


class SimpleModel(nn.Module):
    def __init__(self, in_dim=5, out_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, batch, labels=None, is_training=True):
        output = self.linear(batch)
        loss = torch.mean((output - 1.0) ** 2)
        fitness = -loss  # Higher fitness is better
        return {'fitness': fitness}


def test_noise_reuse_zero():
    """Test that noise_reuse=0 uses different perturbations for each epoch."""
    device = torch.device('cpu')
    m, n, r = 10, 5, 3
    normalized_sigma = 0.1
    base_seed = 42
    sample_idx = 0

    # Generate for different epochs with noise_reuse=0
    A0, B0 = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=0, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=0
    )
    A1, B1 = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=1, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=0
    )

    # Should be different (different epochs, noise_reuse=0 means use actual epoch)
    # Note: B might be same if sample_idx is same and true_thread_idx is same
    # But A should differ due to different true_epoch (0 vs 1)
    assert not torch.allclose(A0, A1), "Different epochs should give different A with noise_reuse=0"
    # B might be same if using same sample_idx, so we check A only


def test_noise_reuse_two():
    """Test that noise_reuse=2 reuses perturbations every 2 epochs."""
    device = torch.device('cpu')
    m, n, r = 10, 5, 3
    normalized_sigma = 0.1
    base_seed = 42
    sample_idx = 0

    # Generate for epochs 0 and 1 with noise_reuse=2
    A0, B0 = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=0, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=2
    )
    A1, B1 = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=1, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=2
    )

    # Should be identical (same true_epoch = 0 // 2 = 0, 1 // 2 = 0)
    assert torch.allclose(A0, A1), "Epochs 0 and 1 should use same perturbations with noise_reuse=2"
    assert torch.allclose(B0, B1), "Epochs 0 and 1 should use same perturbations with noise_reuse=2"

    # Epoch 2 should be different (true_epoch = 2 // 2 = 1)
    A2, B2 = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=2, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=2
    )

    assert not torch.allclose(A0, A2), "Epoch 2 should use different perturbations from epochs 0-1"
    assert not torch.allclose(B0, B2), "Epoch 2 should use different perturbations from epochs 0-1"

    # Epoch 3 should be same as epoch 2 (true_epoch = 3 // 2 = 1)
    A3, B3 = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=3, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=2
    )

    assert torch.allclose(A2, A3), "Epochs 2 and 3 should use same perturbations with noise_reuse=2"
    assert torch.allclose(B2, B3), "Epochs 2 and 3 should use same perturbations with noise_reuse=2"


def test_noise_reuse_five():
    """Test that noise_reuse=5 reuses perturbations every 5 epochs."""
    device = torch.device('cpu')
    m, n, r = 10, 5, 3
    normalized_sigma = 0.1
    base_seed = 42
    sample_idx = 0

    # Epochs 0-4 should share perturbations (true_epoch = 0)
    A0, _ = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=0, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=5
    )
    A1, _ = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=1, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=5
    )
    A4, _ = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=4, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=5
    )

    assert torch.allclose(A0, A1), "Epochs 0 and 1 should share perturbations"
    assert torch.allclose(A0, A4), "Epochs 0 and 4 should share perturbations"

    # Epoch 5 should be different (true_epoch = 5 // 5 = 1)
    A5, _ = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=5, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=5
    )

    assert not torch.allclose(A0, A5), "Epoch 5 should use different perturbations from epochs 0-4"


def test_noise_reuse_trainer():
    """Test noise_reuse in EGGROLLTrainer."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.1,
        learning_rate=0.01,
        n_workers=8,
        noise_reuse=2,  # Reuse every 2 epochs
        base_seed=42,
    )

    assert trainer.noise_reuse == 2

    batch = torch.randn(8, 5, device=device)

    # Train for a few steps
    for step in range(5):
        metrics = trainer.train_step(batch)
        assert metrics['fitness'] > float('-inf')
        assert metrics['valid_samples'] > 0


def test_noise_reuse_determinism():
    """Test that noise_reuse produces deterministic results."""
    device = torch.device('cpu')
    m, n, r = 10, 5, 3
    normalized_sigma = 0.1
    base_seed = 42
    sample_idx = 0

    # Generate twice with same parameters - should be identical
    A1, B1 = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=0, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=2
    )
    A2, B2 = generate_low_rank_perturbation(
        m, n, r, device, normalized_sigma, epoch=0, sample_idx=sample_idx,
        base_seed=base_seed, param_name="param", noise_reuse=2
    )

    assert torch.allclose(A1, A2), "Should be deterministic"
    assert torch.allclose(B1, B2), "Should be deterministic"


def test_noise_reuse_vs_no_reuse():
    """Test that noise_reuse changes behavior compared to no reuse."""
    device = torch.device('cpu')

    def train_with_noise_reuse(noise_reuse_val):
        model = SimpleModel(in_dim=5)
        model.to(device)

        trainer = EGGROLLTrainer(
            model=model,
            device=device,
            rank=4,
            sigma=0.1,
            learning_rate=0.01,
            n_workers=8,
            noise_reuse=noise_reuse_val,
            base_seed=42,
        )

        torch.manual_seed(42)
        np.random.seed(42)
        batch = torch.randn(8, 5, device=device)

        # Train for 3 steps
        for _ in range(3):
            metrics = trainer.train_step(batch)

        params = {name: param.data.clone() for name, param in model.named_parameters()}
        return params

    # Train with and without noise reuse
    params_no_reuse = train_with_noise_reuse(noise_reuse_val=0)
    params_with_reuse = train_with_noise_reuse(noise_reuse_val=2)

    # Results should be different
    param_different = False
    for name in params_no_reuse:
        if not torch.allclose(params_no_reuse[name], params_with_reuse[name], atol=1e-4):
            param_different = True
            break

    assert param_different, "Noise reuse should produce different results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

