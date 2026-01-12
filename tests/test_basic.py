"""
Basic tests for EGGROLL PyTorch implementation.
"""

import torch
import torch.nn as nn
import pytest
from eggroll import EGGROLLTrainer, eggroll_step


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, batch, labels=None, is_training=True):
        output = self.linear(batch)
        loss = torch.mean((output - 1.0) ** 2)
        return {'loss': loss}


def test_trainer_initialization():
    """Test that EGGROLLTrainer can be initialized."""
    device = torch.device('cpu')
    model = SimpleModel()

    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.1,
        learning_rate=0.01,
        n_workers=8,
    )

    assert trainer.model == model
    assert trainer.device == device
    assert trainer.rank == 4
    assert trainer.sigma == 0.1
    assert trainer.learning_rate == 0.01
    assert trainer.n_workers == 8


def test_train_step():
    """Test that train_step works."""
    device = torch.device('cpu')
    model = SimpleModel()

    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.1,
        learning_rate=0.01,
        n_workers=8,
    )

    batch = torch.randn(8, 5)
    metrics = trainer.train_step(batch)

    assert 'loss' in metrics
    assert 'valid_samples' in metrics
    assert isinstance(metrics['loss'], float)
    assert metrics['valid_samples'] > 0


def test_eggroll_step():
    """Test that eggroll_step function works."""
    device = torch.device('cpu')
    model = SimpleModel()

    batch = torch.randn(8, 5)

    loss, metrics = eggroll_step(
        model=model,
        batch=batch,
        device=device,
        n_workers=8,
        rank=4,
        sigma=0.1,
        learning_rate=0.01,
        base_seed=0,
        epoch=0,
    )

    assert isinstance(loss, float)
    assert 'loss' in metrics
    assert 'valid_samples' in metrics
    assert metrics['valid_samples'] > 0


def test_model_forward_interface():
    """Test that model forward interface is correct."""
    model = SimpleModel()
    batch = torch.randn(4, 5)

    output = model(batch, labels=None, is_training=True)

    assert isinstance(output, dict)
    assert 'loss' in output
    assert isinstance(output['loss'], torch.Tensor)
    assert output['loss'].dim() == 0  # Scalar


if __name__ == "__main__":
    pytest.main([__file__])




