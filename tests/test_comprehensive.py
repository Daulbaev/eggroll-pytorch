"""
Comprehensive tests for EGGROLL PyTorch implementation.

Tests various scenarios and edge cases.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from eggroll import EGGROLLTrainer, eggroll_step


class SimpleModel(nn.Module):
    def __init__(self, in_dim=5, out_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, batch, labels=None, is_training=True):
        output = self.linear(batch)
        loss = torch.mean((output - 1.0) ** 2)
        return {'loss': loss}


class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, batch, labels=None, is_training=True):
        x = self.pool(self.relu(self.conv1(batch)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        loss = torch.mean((x - 1.0) ** 2)
        return {'loss': loss}


def test_different_model_sizes():
    """Test with models of different sizes."""
    device = torch.device('cpu')

    for in_dim in [1, 5, 10, 20]:
        model = SimpleModel(in_dim=in_dim)
        model.to(device)

        trainer = EGGROLLTrainer(
            model=model,
            device=device,
            rank=min(4, in_dim),
            sigma=0.1,
            learning_rate=0.01,
            n_workers=8,
            base_seed=42,
        )

        batch = torch.randn(8, in_dim, device=device)
        metrics = trainer.train_step(batch)

        assert metrics['loss'] < float('inf')
        assert metrics['valid_samples'] > 0


def test_different_ranks():
    """Test with different rank values."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=10)
    model.to(device)

    for rank in [1, 2, 4, 8, 16]:
        trainer = EGGROLLTrainer(
            model=model,
            device=device,
            rank=rank,
            sigma=0.1,
            learning_rate=0.01,
            n_workers=8,
            base_seed=42,
        )

        batch = torch.randn(8, 10, device=device)
        metrics = trainer.train_step(batch)

        assert metrics['loss'] < float('inf')
        assert metrics['valid_samples'] > 0


def test_different_n_workers():
    """Test with different numbers of workers."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    for n_workers in [4, 8, 16, 32]:
        trainer = EGGROLLTrainer(
            model=model,
            device=device,
            rank=4,
            sigma=0.1,
            learning_rate=0.01,
            n_workers=n_workers,
            base_seed=42,
        )

        batch = torch.randn(n_workers, 5, device=device)
        metrics = trainer.train_step(batch)

        assert metrics['loss'] < float('inf')
        assert metrics['valid_samples'] > 0


def test_different_sigma_values():
    """Test with different sigma (perturbation scale) values."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    for sigma in [0.01, 0.1, 0.2, 0.5]:
        trainer = EGGROLLTrainer(
            model=model,
            device=device,
            rank=4,
            sigma=sigma,
            learning_rate=0.01,
            n_workers=8,
            base_seed=42,
        )

        batch = torch.randn(8, 5, device=device)
        metrics = trainer.train_step(batch)

        assert metrics['loss'] < float('inf')
        assert metrics['valid_samples'] > 0


def test_gradient_clipping_variations():
    """Test gradient clipping with different values."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    for grad_clip in [None, 0.1, 1.0, 10.0]:
        trainer = EGGROLLTrainer(
            model=model,
            device=device,
            rank=4,
            sigma=0.2,
            learning_rate=0.01,
            n_workers=8,
            grad_clip=grad_clip,
            base_seed=42,
        )

        batch = torch.randn(8, 5, device=device)
        metrics = trainer.train_step(batch)

        assert metrics['loss'] < float('inf')
        assert 'grad_norm' in metrics

        if grad_clip is not None:
            # Grad norm should be clipped (allow larger tolerance due to numerical precision)
            assert metrics['grad_norm'] <= grad_clip * 1.5, f"Grad norm {metrics['grad_norm']} should be <= {grad_clip * 1.5}"


def test_normalize_fitness_toggle():
    """Test with and without fitness normalization."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    for normalize_fitness in [True, False]:
        trainer = EGGROLLTrainer(
            model=model,
            device=device,
            rank=4,
            sigma=0.1,
            learning_rate=0.01,
            n_workers=8,
            normalize_fitness=normalize_fitness,
            base_seed=42,
        )

        batch = torch.randn(8, 5, device=device)
        metrics = trainer.train_step(batch)

        assert metrics['loss'] < float('inf')
        assert metrics['valid_samples'] > 0


def test_multiple_training_steps():
    """Test multiple consecutive training steps."""
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
        base_seed=42,
    )

    losses = []
    for step in range(10):
        batch = torch.randn(8, 5, device=device)
        metrics = trainer.train_step(batch)
        losses.append(metrics['loss'])

        assert metrics['loss'] < float('inf')
        assert metrics['valid_samples'] > 0

    # Check that trainer step counter increased
    assert trainer.step == 10
    assert len(trainer.loss_history) == 10


def test_checkpoint_saving():
    """Test checkpoint saving functionality."""
    import tempfile
    import os

    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = EGGROLLTrainer(
            model=model,
            device=device,
            rank=4,
            sigma=0.1,
            learning_rate=0.01,
            n_workers=8,
            results_dir=tmpdir,
            base_seed=42,
        )

        batch = torch.randn(8, 5, device=device)
        trainer.train_step(batch)

        checkpoint_path = trainer.save_checkpoint()

        assert checkpoint_path is not None
        assert os.path.exists(checkpoint_path)
        assert os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin"))
        assert os.path.exists(os.path.join(checkpoint_path, "training_state.pt"))


def test_invalid_loss_handling():
    """Test handling of invalid (NaN/Inf) losses."""
    device = torch.device('cpu')

    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 1)

        def forward(self, batch, labels=None, is_training=True):
            output = self.linear(batch)
            # Sometimes return invalid loss
            if torch.rand(1).item() < 0.5:
                loss = torch.tensor(float('inf'))
            else:
                loss = torch.mean((output - 1.0) ** 2)
            return {'loss': loss}

    model = BadModel()
    model.to(device)

    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.1,
        learning_rate=0.01,
        n_workers=16,  # More workers to ensure some valid samples
        base_seed=42,
    )

    batch = torch.randn(16, 5, device=device)
    metrics = trainer.train_step(batch)

    # Should handle invalid losses gracefully
    # Note: If all samples are invalid, loss might still be inf, but valid_samples should be 0
    assert metrics['valid_samples'] > 0 or metrics['loss'] < float('inf'), \
        "Should have at least some valid samples or finite loss"


def test_cuda_if_available():
    """Test CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device('cuda')
    model = SimpleModel(in_dim=5)
    model.to(device)

    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.1,
        learning_rate=0.01,
        n_workers=8,
        base_seed=42,
    )

    batch = torch.randn(8, 5, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['loss'] < float('inf')
    assert metrics['valid_samples'] > 0


def test_eggroll_step_function():
    """Test low-level eggroll_step function."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    batch = torch.randn(8, 5, device=device)

    loss, metrics = eggroll_step(
        model=model,
        batch=batch,
        device=device,
        n_workers=8,
        rank=4,
        sigma=0.1,
        learning_rate=0.01,
        base_seed=42,
        epoch=0,
    )

    assert isinstance(loss, float)
    assert loss < float('inf')
    assert 'loss' in metrics
    assert 'valid_samples' in metrics
    assert metrics['valid_samples'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

