"""
MNIST example for PyTorch EGGROLL implementation.

This example trains a simple neural network on MNIST digit classification
using EGGROLL optimization (no backpropagation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from eggroll import EGGROLLTrainer


class SimpleMNIST(nn.Module):
    """
    Simple MLP model for MNIST classification.
    Compatible with EGGROLL trainer.
    """

    def __init__(self, input_size=784, hidden_dims=[128, 64], num_classes=10, use_bias=True):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # Build layers
        layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=use_bias))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes, bias=use_bias))
        self.network = nn.Sequential(*layers)

    def forward(self, batch, labels=None, is_training=True):
        """
        Forward pass.

        Args:
            batch: Input tensor of shape (batch_size, input_size) or (batch_size, 1, 28, 28),
                   or dict with 'batch'/'images' key and optionally 'labels' key
            labels: Target labels tensor of shape (batch_size,) or None
            is_training: Not used, kept for compatibility

        Returns:
            Dictionary with 'fitness' key (and optionally 'output' key)
        """
        # Extract images and labels from batch if it's a dict
        if isinstance(batch, dict):
            # Support both 'batch' and 'images' keys for compatibility
            images = batch.get('images', batch.get('batch', batch))
            # Extract labels from batch dict if available (EGGROLL doesn't pass labels parameter)
            if labels is None and 'labels' in batch:
                labels = batch['labels']
        else:
            images = batch

        # Flatten input if needed (handle both flattened and image formats)
        if images.dim() > 2:
            images = images.view(images.size(0), -1)

        # Forward pass
        output = self.network(images)

        # Compute fitness if labels are provided (fitness = -loss, higher is better)
        if labels is not None:
            loss = F.cross_entropy(output, labels)
            fitness = -loss  # Higher fitness is better
        else:
            # If no labels, return dummy fitness (shouldn't happen during training)
            fitness = torch.tensor(0.0, device=output.device, requires_grad=False)

        return {'fitness': fitness, 'output': output}


class ConvMNIST(nn.Module):
    """
    Lightweight CNN model for MNIST classification.
    Compatible with EGGROLL trainer.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, batch, labels=None, is_training=True):
        """
        Forward pass.

        Args:
            batch: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784),
                   or dict with 'batch'/'images' key and optionally 'labels' key
            labels: Target labels tensor of shape (batch_size,) or None
            is_training: Not used, kept for compatibility

        Returns:
            Dictionary with 'fitness' key (and optionally 'output' key)
        """
        # Extract images and labels from batch if it's a dict
        if isinstance(batch, dict):
            # Support both 'batch' and 'images' keys for compatibility
            images = batch.get('images', batch.get('batch', batch))
            # Extract labels from batch dict if available (EGGROLL doesn't pass labels parameter)
            if labels is None and 'labels' in batch:
                labels = batch['labels']
        else:
            images = batch

        # Reshape if flattened (batch_size, 784) -> (batch_size, 1, 28, 28)
        if images.dim() == 2:
            images = images.view(images.size(0), 1, 28, 28)

        # Forward pass
        x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        output = self.fc2(x)

        # Compute fitness if labels are provided (fitness = -loss, higher is better)
        if labels is not None:
            loss = F.cross_entropy(output, labels)
            fitness = -loss  # Higher fitness is better
        else:
            # If no labels, return dummy fitness (shouldn't happen during training)
            fitness = torch.tensor(0.0, device=output.device, requires_grad=False)

        return {'fitness': fitness, 'output': output}


def load_mnist_data(batch_size=64, data_dir='./data'):
    """
    Load MNIST dataset.

    Args:
        batch_size: Batch size for data loader
        data_dir: Directory to store/load MNIST data

    Returns:
        train_loader, test_loader: Data loaders for training and testing
    """
    # Transform: convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def evaluate_accuracy(model, data_loader, device):
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to compute on

    Returns:
        accuracy: Classification accuracy (0-1)
        avg_loss: Average loss
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Flatten images for MLP (if using SimpleMNIST)
            if isinstance(model, SimpleMNIST):
                images = images.view(images.size(0), -1)

            # Forward pass
            output_dict = model(images, labels=labels, is_training=False)
            output = output_dict['output']
            fitness = output_dict['fitness']
            loss = -fitness.item()  # Convert fitness back to loss for reporting

            # Get predictions
            _, predicted = torch.max(output.data, 1)

            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss
            num_batches += 1

    accuracy = correct / total
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return accuracy, avg_loss


def main():
    print("=" * 60)
    print("PyTorch EGGROLL MNIST Example")
    print("=" * 60)

    # Hyperparameters
    model_type = 'mlp'  # 'mlp' or 'cnn'
    sigma = 0.05
    learning_rate = 0.01
    num_epochs = 20
    n_workers = 64  # Number of perturbation samples per iteration
    rank = 16
    batch_size = 64
    base_seed = 42

    print(f"Hyperparameters:")
    print(f"  Model type: {model_type}")
    print(f"  sigma: {sigma}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  n_workers: {n_workers}")
    print(f"  rank: {rank}")
    print(f"  batch_size: {batch_size}")
    print()

    # Create device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()

    # Load MNIST data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size=batch_size)
    print(f"  Training samples: {len(train_loader.dataset):,}")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    print()

    # Create model
    if model_type == 'mlp':
        model = SimpleMNIST(input_size=784, hidden_dims=[128, 64], num_classes=10)
    elif model_type == 'cnn':
        model = ConvMNIST(num_classes=10)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_type.upper()}")
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()

    # Create EGGROLL trainer
    trainer = EGGROLLTrainer(
        model=model,
        device=device,
        rank=rank,
        sigma=sigma,
        learning_rate=learning_rate,
        n_workers=n_workers,
        grad_clip=1.0,
        use_weighted_loss=False,
        log_steps=100,
        save_steps=None,
        results_dir=None,
        normalize_fitness=True,
        base_seed=base_seed,
        optimizer='adamw',  # Use AdamW for better convergence
        optimizer_kwargs={'betas': (0.9, 0.999)},
    )

    # Training loop
    print("Starting training...")
    print("-" * 60)

    # Set random seeds
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    # Evaluate initial accuracy
    print("Initial evaluation:")
    initial_acc, initial_loss = evaluate_accuracy(model, test_loader, device)
    print(f"  Test accuracy: {initial_acc * 100:.2f}%")
    print(f"  Test loss: {initial_loss:.4f}")
    print()

    for epoch in range(num_epochs):
        epoch_losses = []
        num_batches = 0

        # Training over all batches
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Flatten images for MLP (if using SimpleMNIST)
            if isinstance(model, SimpleMNIST):
                images = images.view(images.size(0), -1)

            # Prepare batch for EGGROLL
            # EGGROLL trainer checks if batch is dict with 'batch' key and extracts it
            # So we pass dict with 'images' and 'labels' keys (not 'batch' key)
            # The trainer will pass the entire dict to model, and model will extract labels
            batch_for_eggroll = {'images': images, 'labels': labels}

            # Training step
            metrics = trainer.train_step(batch_for_eggroll)
            # Convert fitness to loss for reporting (fitness = -loss)
            train_fitness = metrics.get('fitness', float('-inf'))
            train_loss = -train_fitness if train_fitness != float('-inf') else float('inf')
            epoch_losses.append(train_loss)
            num_batches += 1

        # Average loss for epoch
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')

        # Evaluate on test set every epoch
        test_acc, test_loss = evaluate_accuracy(model, test_loader, device)

        # Print progress
        if (epoch + 1) % 1 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train loss: {avg_epoch_loss:.4f}")
            print(f"  Train fitness: {metrics.get('fitness', 'N/A'):.6f}")
            print(f"  Test accuracy: {test_acc * 100:.2f}%")
            print(f"  Test loss: {test_loss:.4f}")
            print(f"  Valid samples: {metrics.get('valid_samples', 'N/A')}")
            print(f"  Grad norm: {metrics.get('grad_norm', 'N/A'):.4f}")
            print()

    print("-" * 60)
    print("Training completed!")
    print()

    # Final evaluation
    print("Final evaluation:")
    final_acc, final_loss = evaluate_accuracy(model, test_loader, device)
    print(f"  Final test accuracy: {final_acc * 100:.2f}%")
    print(f"  Final test loss: {final_loss:.4f}")
    print()

    # Compare with random baseline
    random_baseline = 1.0 / 10  # 10 classes
    print(f"  Random baseline: {random_baseline * 100:.2f}%")
    if final_acc > random_baseline:
        improvement = (final_acc - random_baseline) / random_baseline * 100
        print(f"  Improvement over random: {improvement:.1f}%")
        print("✓ Model successfully learned!")
    else:
        print("⚠ Model did not learn (accuracy <= random baseline)")

    print("=" * 60)


if __name__ == "__main__":
    main()
