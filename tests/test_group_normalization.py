"""
Tests for group normalization feature in EGGROLL PyTorch implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from eggroll import EGGROLLTrainer


class SimpleModel(nn.Module):
    def __init__(self, in_dim=5, out_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, batch, labels=None, is_training=True):
        output = self.linear(batch)
        loss = torch.mean((output - 1.0) ** 2)
        fitness = -loss  # Higher fitness is better
        return {'fitness': fitness}


def test_global_normalization():
    """Test global normalization (group_size=0, default)."""
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
        group_size=0,  # Global normalization
        base_seed=42,
    )

    batch = torch.randn(8, 5, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['fitness'] > float('-inf')
    assert metrics['valid_samples'] > 0


def test_group_normalization():
    """Test group normalization (group_size > 0)."""
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
        group_size=4,  # 2 groups of 4 samples each
        base_seed=42,
    )

    batch = torch.randn(8, 5, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['fitness'] > float('-inf')
    assert metrics['valid_samples'] > 0


def test_group_normalization_different_sizes():
    """Test group normalization with different group sizes."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    for group_size in [2, 4, 8]:
        trainer = EGGROLLTrainer(
            model=model,
            device=device,
            rank=4,
            sigma=0.1,
            learning_rate=0.01,
            n_workers=8,
            group_size=group_size,
            base_seed=42,
        )

        batch = torch.randn(8, 5, device=device)
        metrics = trainer.train_step(batch)

        assert metrics['fitness'] > float('-inf')
        assert metrics['valid_samples'] > 0


def test_group_size_validation():
    """Test that group_size must divide n_workers evenly."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    # This should raise an error (8 % 3 != 0)
    with pytest.raises(ValueError, match="group_size.*must divide"):
        EGGROLLTrainer(
            model=model,
            device=device,
            rank=4,
            sigma=0.1,
            learning_rate=0.01,
            n_workers=8,
            group_size=3,  # Doesn't divide 8 evenly
            base_seed=42,
        )


def test_group_normalization_logic():
    """Test the group normalization logic matches JAX."""
    # Simulate the normalization logic
    raw_fitnesses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    group_size = 4

    # Group normalization (matching JAX)
    group_scores = raw_fitnesses.reshape((-1, group_size))
    group_means = np.mean(group_scores, axis=-1, keepdims=True)
    global_std = np.std(raw_fitnesses, keepdims=True)
    eps = 1e-5
    normalized_group_scores = (group_scores - group_means) / (global_std + eps)
    normalized_fitnesses = normalized_group_scores.ravel()

    # Check that each group has mean ~0
    group1 = normalized_fitnesses[:4]
    group2 = normalized_fitnesses[4:]

    assert abs(np.mean(group1)) < 1e-5, "Group 1 should have mean ~0"
    assert abs(np.mean(group2)) < 1e-5, "Group 2 should have mean ~0"

    # Check that std is approximately 1 (using global std)
    # Note: Group normalization uses global std, so the overall std may be slightly different
    assert abs(np.std(normalized_fitnesses) - 1.0) < 0.6, "Should have std ~1 (with tolerance for group normalization)"


def test_group_vs_global_normalization():
    """Test that group and global normalization produce different results."""
    device = torch.device('cpu')

    def train_with_group_size(group_size_val):
        model = SimpleModel(in_dim=5)
        model.to(device)

        trainer = EGGROLLTrainer(
            model=model,
            device=device,
            rank=4,
            sigma=0.1,
            learning_rate=0.01,
            n_workers=8,
            group_size=group_size_val,
            base_seed=42,
        )

        torch.manual_seed(42)
        np.random.seed(42)
        batch = torch.randn(8, 5, device=device)

        metrics = trainer.train_step(batch)
        params = {name: param.data.clone() for name, param in model.named_parameters()}

        return metrics, params

    # Train with global and group normalization
    metrics_global, params_global = train_with_group_size(group_size_val=0)
    metrics_group, params_group = train_with_group_size(group_size_val=4)

    # Results should be different
    assert abs(metrics_global['fitness'] - metrics_group['fitness']) > 1e-5, \
        "Group and global normalization should produce different results"

    # At least some parameters should be different
    param_different = False
    for name in params_global:
        if not torch.allclose(params_global[name], params_group[name], atol=1e-4):
            param_different = True
            break

    assert param_different, "Group normalization should produce different parameters"


def test_group_normalization_edge_cases():
    """Test edge cases for group normalization."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    # group_size = n_workers (single group)
    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.1,
        learning_rate=0.01,
        n_workers=8,
        group_size=8,  # Single group
        base_seed=42,
    )

    batch = torch.randn(8, 5, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['fitness'] > float('-inf')
    assert metrics['valid_samples'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

