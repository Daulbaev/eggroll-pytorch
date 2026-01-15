"""
Simple example for PyTorch EGGROLL implementation.

This example trains a simple MLP to minimize (x - target)^2 using EGGROLL optimization.
Similar to the JAX end_to_end_test.py example.
"""

import torch
import torch.nn as nn
import numpy as np
from eggroll import EGGROLLTrainer


class SimpleMLP(nn.Module):
    """
    Simple MLP model for testing EGGROLL.
    Similar to the MLP used in JAX end_to_end_test.
    """

    def __init__(self, in_dim=3, hidden_dims=[16, 16], out_dim=1, use_bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Build layers
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=use_bias))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim, bias=use_bias))
        self.network = nn.Sequential(*layers)

    def forward(self, batch, labels=None, is_training=True):
        """
        Forward pass.

        Args:
            batch: Input tensor of shape (batch_size, in_dim)
            labels: Not used, kept for compatibility
            is_training: Not used, kept for compatibility

        Returns:
            Dictionary with 'fitness' key
        """
        # Simple fitness: maximize -(output - target)^2 (higher is better)
        # Target is 2.0 for all inputs (matching JAX test)
        target = 2.0
        output = self.network(batch)
        loss = torch.mean((output - target) ** 2)
        # Fitness is negative loss (higher fitness is better)
        fitness = -loss

        return {'fitness': fitness}


def calculate_fitness(outputs, target=2.0):
    """
    Calculate fitness: -((output - target)^2)
    Higher fitness is better (we maximize fitness, minimize loss).

    Args:
        outputs: Model outputs of shape (batch_size, 1)
        target: Target value

    Returns:
        Fitness values (negative squared error)
    """
    return -((outputs - target) ** 2).squeeze(-1)


def main():
    print("=" * 60)
    print("PyTorch EGGROLL Simple Example")
    print("=" * 60)

    # Hyperparameters (matching JAX test)
    sigma = 0.2
    learning_rate = 0.03
    num_epochs = 50
    n_workers = 64  # n_workers (N_workers in EGGROLL paper, num_envs in JAX)
    rank = 8
    in_dim = 3
    hidden_dims = [16, 16]
    out_dim = 1
    base_seed = 0

    print(f"Hyperparameters:")
    print(f"  sigma: {sigma}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  n_workers: {n_workers}")
    print(f"  rank: {rank}")
    print(f"  in_dim: {in_dim}, hidden_dims: {hidden_dims}, out_dim: {out_dim}")
    print()

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = SimpleMLP(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=out_dim, use_bias=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()

    # Create EGGROLL trainer
    # You can use optimizer='adamw' to match JAX default, or None for simple SGD
    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=rank,
        sigma=sigma,
        learning_rate=learning_rate,
        n_workers=n_workers,
        grad_clip=1.0,
        use_weighted_loss=False,
        log_steps=1,
        save_steps=None,
        results_dir=None,
        normalize_fitness=True,
        base_seed=base_seed,
        # optimizer='adamw',  # Uncomment to use AdamW (matching JAX default)
        # optimizer_kwargs={'betas': (0.9, 0.999)},
        # noise_reuse=0,  # Set to > 0 to reuse perturbations across epochs (e.g., noise_reuse=2)
        # group_size=0,  # Set to > 0 for group-wise fitness normalization (must divide n_workers)
    )

    # Training loop
    print("Starting training...")
    print("-" * 60)

    # Set random seed for data generation
    torch.manual_seed(42)
    np.random.seed(42)

    for epoch in range(num_epochs):
        # Generate random input batch
        input_batch = torch.randn(n_workers, in_dim, device=device)

        # Create batch dict (matching our interface)
        batch = input_batch

        # Evaluate model before training step (validation)
        model.eval()
        with torch.no_grad():
            validation_output = model(batch, labels=None, is_training=False)
            validation_fitness = validation_output['fitness'].item()
            validation_outputs = model.network(batch)
            validation_fitness_calc = calculate_fitness(validation_outputs).cpu().numpy()
            avg_validation_fitness = np.mean(validation_fitness_calc)

        # Training step
        model.train()
        metrics = trainer.train_step(batch)

        # Calculate fitness from outputs for comparison
        model.eval()
        with torch.no_grad():
            train_outputs = model.network(batch)
            train_fitness = calculate_fitness(train_outputs).cpu().numpy()
            avg_fitness = np.mean(train_fitness)
            min_fitness = np.min(train_fitness)
            max_fitness = np.max(train_fitness)

        # Print every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Validation fitness: {validation_fitness:.6f}")
            print(f"  Avg validation fitness (calc): {avg_validation_fitness:.6f}")
            print(f"  Training metrics:")
            print(f"    Fitness: {metrics.get('fitness', 'N/A'):.6f}")
            print(f"    Avg fitness: {avg_fitness:.6f}")
            print(f"    Min fitness: {min_fitness:.6f}")
            print(f"    Max fitness: {max_fitness:.6f}")
            print(f"    Valid samples: {metrics.get('valid_samples', 'N/A')}")
            print(f"    Grad norm: {metrics.get('grad_norm', 'N/A'):.6f}")

            # Check parameter updates
            if epoch > 0:
                with torch.no_grad():
                    param_norms = []
                    for param in model.parameters():
                        if param.requires_grad:
                            param_norms.append(torch.norm(param).item())
                    avg_param_norm = np.mean(param_norms)
                    print(f"    Avg param norm: {avg_param_norm:.6f}")
            print()

    print("-" * 60)
    print("Training completed!")
    print()

    # Final evaluation
    print("Final evaluation:")
    model.eval()
    with torch.no_grad():
        # Test on multiple random batches
        test_batches = [torch.randn(n_workers, in_dim, device=device) for _ in range(5)]
        all_fitnesses = []
        all_fitnesses_direct = []

        for test_batch in test_batches:
            output = model(test_batch, labels=None, is_training=False)
            fitness = output['fitness'].item()
            outputs = model.network(test_batch)
            fitness_calc = calculate_fitness(outputs).cpu().numpy()

            all_fitnesses.extend(fitness_calc.tolist())
            all_fitnesses_direct.append(fitness)

        avg_final_fitness = np.mean(all_fitnesses)
        avg_final_fitness_direct = np.mean(all_fitnesses_direct)
        std_final_fitness = np.std(all_fitnesses)

        print(f"  Average final fitness (calc): {avg_final_fitness:.6f} ± {std_final_fitness:.6f}")
        print(f"  Average final fitness (direct): {avg_final_fitness_direct:.6f}")
        print(f"  Target fitness (optimal): 0.0")
        print()

        # Check if model learned something (fitness > -1.0 means loss < 1.0)
        if avg_final_fitness_direct > -1.0:
            print("✓ Model successfully learned! (fitness > -1.0)")
        else:
            print("⚠ Model may need more training (fitness <= -1.0)")

    print("=" * 60)


if __name__ == "__main__":
    main()

