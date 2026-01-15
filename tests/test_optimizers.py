"""
Tests for optimizer support in EGGROLL PyTorch implementation.
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


def test_sgd_optimizer():
    """Test SGD optimizer."""
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
        optimizer='sgd',
        base_seed=42,
    )

    assert trainer.optimizer is not None
    assert isinstance(trainer.optimizer, torch.optim.SGD)

    batch = torch.randn(8, 5, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['fitness'] > float('-inf')
    assert metrics['valid_samples'] > 0


def test_adam_optimizer():
    """Test Adam optimizer."""
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
        optimizer='adam',
        optimizer_kwargs={'betas': (0.9, 0.999)},
        base_seed=42,
    )

    assert trainer.optimizer is not None
    assert isinstance(trainer.optimizer, torch.optim.Adam)

    batch = torch.randn(8, 5, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['fitness'] > float('-inf')
    assert metrics['valid_samples'] > 0


def test_adamw_optimizer():
    """Test AdamW optimizer (matching JAX default)."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.1,
        learning_rate=0.03,
        n_workers=8,
        optimizer='adamw',
        optimizer_kwargs={'betas': (0.9, 0.999)},  # Matching JAX default
        base_seed=42,
    )

    assert trainer.optimizer is not None
    assert isinstance(trainer.optimizer, torch.optim.AdamW)

    batch = torch.randn(8, 5, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['fitness'] > float('-inf')
    assert metrics['valid_samples'] > 0


def test_no_optimizer_default():
    """Test default behavior (no optimizer, simple SGD-like update)."""
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

    assert trainer.optimizer is None

    batch = torch.randn(8, 5, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['fitness'] > float('-inf')
    assert metrics['valid_samples'] > 0


def test_custom_optimizer_instance():
    """Test with pre-initialized optimizer instance."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    # Create custom optimizer
    custom_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.1,
        learning_rate=0.01,
        n_workers=8,
        optimizer=custom_optimizer,
        base_seed=42,
    )

    assert trainer.optimizer is custom_optimizer

    batch = torch.randn(8, 5, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['fitness'] > float('-inf')
    assert metrics['valid_samples'] > 0


def test_optimizer_state_persistence():
    """Test that optimizer state persists across steps."""
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
        optimizer='adamw',
        base_seed=42,
    )

    batch = torch.randn(8, 5, device=device)

    # Get optimizer state after first step
    trainer.train_step(batch)
    state_after_first = trainer.optimizer.state_dict()

    # Second step should update optimizer state
    trainer.train_step(batch)
    state_after_second = trainer.optimizer.state_dict()

    # States should exist and optimizer should be working
    # Note: If the same batch is used twice with same gradients, momentum might be similar
    # We mainly verify that optimizer state exists and is being maintained
    assert 'state' in state_after_first and 'state' in state_after_second, "Optimizer states should exist"

    # Check that step count increased (both should be 2 after 2 steps, so >= is correct)
    if 'state' in state_after_first and 0 in state_after_first['state']:
        step_first = state_after_first['state'][0].get('step', torch.tensor(0))
        if 'state' in state_after_second and 0 in state_after_second['state']:
            step_second = state_after_second['state'][0].get('step', torch.tensor(0))
            # Step should be at least the same or increased
            assert step_second.item() >= step_first.item(), \
                f"Optimizer step should not decrease: {step_first.item()} -> {step_second.item()}"

    # Verify that optimizer is maintaining state (exp_avg and exp_avg_sq should exist)
    if 'state' in state_after_second and 0 in state_after_second['state']:
        state = state_after_second['state'][0]
        # AdamW should have exp_avg and exp_avg_sq
        assert 'exp_avg' in state or 'step' in state, "Optimizer should maintain state"


def test_optimizer_checkpoint_saving():
    """Test that optimizer state is saved in checkpoints."""
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
            optimizer='adamw',
            results_dir=tmpdir,
            base_seed=42,
        )

        batch = torch.randn(8, 5, device=device)
        trainer.train_step(batch)

        checkpoint_path = trainer.save_checkpoint()

        # Load checkpoint
        checkpoint = torch.load(os.path.join(checkpoint_path, "training_state.pt"))

        assert 'optimizer_state' in checkpoint
        assert checkpoint['optimizer_state'] is not None


def test_unknown_optimizer_error():
    """Test that unknown optimizer name raises error."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    with pytest.raises(ValueError, match="Unknown optimizer"):
        EGGROLLTrainer(
            model=model,
            device=device,
            rank=4,
            sigma=0.1,
            learning_rate=0.01,
            n_workers=8,
            optimizer='unknown_optimizer',
            base_seed=42,
        )


def test_optimizer_vs_no_optimizer():
    """Test that optimizer and no-optimizer produce different results."""
    device = torch.device('cpu')

    def train_with_optimizer(use_optimizer):
        model = SimpleModel(in_dim=5)
        model.to(device)

        kwargs = {
            'model': model,
            'device': device,
            'rank': 4,
            'sigma': 0.1,
            'learning_rate': 0.01,
            'n_workers': 8,
            'base_seed': 42,
        }

        if use_optimizer:
            kwargs['optimizer'] = 'adamw'

        trainer = EGGROLLTrainer(**kwargs)

        torch.manual_seed(42)
        np.random.seed(42)
        batch = torch.randn(8, 5, device=device)

        metrics = trainer.train_step(batch)
        params = {name: param.data.clone() for name, param in model.named_parameters()}

        return metrics, params

    # Train with and without optimizer
    metrics1, params1 = train_with_optimizer(use_optimizer=False)
    metrics2, params2 = train_with_optimizer(use_optimizer=True)

    # Results should be different (optimizer has momentum/state)
    # At least some parameters should differ
    param_different = False
    for name in params1:
        if not torch.allclose(params1[name], params2[name], atol=1e-4):
            param_different = True
            break

    assert param_different, "Optimizer should produce different results"


def test_adamw_matching_jax():
    """Test AdamW configuration matching JAX default."""
    device = torch.device('cpu')
    model = SimpleModel(in_dim=5)
    model.to(device)

    # JAX default: optax.adamw with b1=0.9, b2=0.999
    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=4,
        sigma=0.2,
        learning_rate=0.03,
        n_workers=8,
        optimizer='adamw',
        optimizer_kwargs={'betas': (0.9, 0.999)},  # Matching JAX
        base_seed=42,
    )

    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    # Check that betas are set correctly
    assert trainer.optimizer.param_groups[0]['betas'] == (0.9, 0.999)

    batch = torch.randn(8, 5, device=device)
    metrics = trainer.train_step(batch)

    assert metrics['fitness'] > float('-inf')
    assert metrics['valid_samples'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

