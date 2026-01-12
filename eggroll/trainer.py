"""
EGGROLL Trainer for PyTorch Models

Implements Evolution Guided General Optimization via Low-rank Learning (EGGROLL)
algorithm for training neural networks without backpropagation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import tempfile
import os
import shutil


def fold_in_seed(base_seed: int, epoch: int, thread_id: int) -> int:
    """
    Deterministic seed generation similar to JAX fold_in.

    Combines base_seed with epoch and thread_id deterministically to create
    a new seed. This ensures that the same (base_seed, epoch, thread_id)
    combination always produces the same seed.

    Args:
        base_seed: Base random seed
        epoch: Epoch number (or step number)
        thread_id: Thread/sample index

    Returns:
        Combined seed value

    Note: Determinism can be verified by generating perturbations twice with
    the same (base_seed, epoch, thread_id) and checking they are identical.
    """
    # Use hash to combine seeds deterministically (similar to JAX fold_in)
    # This ensures good mixing while being deterministic
    combined = hash((base_seed, epoch, thread_id))
    # Ensure positive value (hash can be negative)
    return abs(combined) % (2**31)


def generate_low_rank_perturbation(
    m: int,
    n: int,
    r: int,
    device: torch.device,
    normalized_sigma: float,
    epoch: int,
    sample_idx: int,
    base_seed: int,
    param_name: str = "",
    noise_reuse: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate low-rank perturbation matrices A and B such that E = (1/√r) * A B^T
    according to EGGROLL paper (formula 8, page 7).

    This matches the JAX implementation where normalized_sigma (sigma/sqrt(rank))
    is applied during generation, and generation is deterministic based on seed.

    Args:
        m: First dimension
        n: Second dimension
        r: Rank (r << min(m, n))
        device: Device to create tensors on
        normalized_sigma: Normalized sigma value (sigma / sqrt(rank))
        epoch: Epoch number for deterministic generation
        sample_idx: Sample index (thread_id) for deterministic generation
        base_seed: Base random seed
        param_name: Parameter name (for additional seed mixing, optional)
        noise_reuse: Number of epochs to reuse the same noise (0 = no reuse, default)

    Returns:
        A: (m, r) tensor with normalized_sigma * sign already applied
        B: (n, r) tensor (standard normal)

    Note: Matches JAX implementation exactly:
    - lora_params generated as single (m+n, r) array, then split into B[:n] and A[n:]
    - sigma is applied to A: return A * sigma, B
    - normalized_sigma already includes (sigma / sqrt(rank))
    """
    # Generate deterministic seed using fold_in (similar to JAX)
    # Use param_name hash for additional mixing if provided
    param_hash = hash(param_name) if param_name else 0
    # Compute true_epoch based on noise_reuse
    # When noise_reuse == 0, use actual epoch (each epoch gets different perturbations)
    # When noise_reuse > 0, group epochs: true_epoch = epoch // noise_reuse
    # This matches JAX behavior where noise_reuse=0 means no reuse (each epoch different)
    true_epoch = epoch if noise_reuse == 0 else epoch // noise_reuse
    true_thread_idx = sample_idx // 2  # Match JAX behavior

    # Thread ID alternation: alternate sigma sign based on sample_idx % 2 (matching JAX)
    # JAX: sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)
    # If sample_idx is even -> positive sigma, if odd -> negative sigma
    sigma_sign = 1.0 if (sample_idx % 2 == 0) else -1.0
    # Apply sign to normalized_sigma
    signed_normalized_sigma = normalized_sigma * sigma_sign

    # Combine seeds deterministically
    seed_for_generation = fold_in_seed(base_seed, true_epoch, true_thread_idx)
    if param_hash != 0:
        seed_for_generation = fold_in_seed(seed_for_generation, param_hash, 0)

    # Create generator with deterministic seed
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_for_generation)

    # Generate A and B as standard normal (matching JAX random.normal)
    # In JAX version, lora_params is generated as (a+b, rank), then split:
    # lora_params = jax.random.normal(..., (a+b, rank))
    # B = lora_params[:b]  # b x r (first b rows)
    # A = lora_params[b:]  # a x r (remaining a rows)
    # We must match this exactly to get the same random values
    lora_params = torch.randn(m + n, r, device=device, generator=generator)
    B = lora_params[:n]  # (n, r) - first n rows
    A = lora_params[n:]  # (m, r) - remaining m rows

    # Apply signed normalized_sigma to A (matching JAX: return A * sigma, B)
    # Note: normalized_sigma already includes sigma / sqrt(rank), and we apply sign alternation
    A = A * signed_normalized_sigma

    return A, B


class PerturbationHookManager:
    """
    Context manager for applying low-rank perturbations on-the-fly using forward hooks.
    This avoids forming full perturbation matrices E = A @ B.T, saving memory.

    Matches JAX's do_mm: base_ans + x @ B @ A.T
    where A already contains normalized_sigma * sign (sigma/sqrt(rank) * sign).
    """
    def __init__(self, model: nn.Module, low_rank_perturbations: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                 sigma: float, rank: int):
        """
        Args:
            model: PyTorch model
            low_rank_perturbations: Dict mapping parameter names to (A, B) tuples
                A: (m, r) tensor with normalized_sigma already applied
                B: (n, r) tensor
            sigma: Perturbation scale
            rank: Rank of low-rank perturbation
        """
        self.model = model
        self.low_rank_perturbations = low_rank_perturbations
        self.sigma = sigma
        self.rank = rank
        self.hooks = []
        self.param_to_module = {}  # Map parameter names to their modules

        # Build mapping from parameter names to modules
        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}" if name else param_name
                if full_name in low_rank_perturbations:
                    self.param_to_module[full_name] = (module, param_name)

    def _create_hook(self, A: torch.Tensor, B: torch.Tensor):
        """
        Create a forward hook for applying low-rank perturbation on-the-fly.

        Matches JAX implementation: base_ans + x @ B @ A.T
        where A already contains normalized_sigma * sign (from generate_low_rank_perturbation).

        For nn.Linear: output = input @ weight.T + bias
        With perturbation: output = input @ weight.T + (input @ B @ A.T) + bias
        """
        def hook(module, input, output):
            x = input[0]  # Input tensor: (batch_size, in_features)

            # Apply perturbation: x @ B @ A.T
            # A already contains: A = A_original * (sigma/sqrt(rank)) * sign
            # This matches JAX do_mm: base_ans + x @ B @ A.T
            perturbation = x @ B @ A.T

            # Add perturbation to output
            output = output + perturbation
            return output
        return hook

    def __enter__(self):
        """Register forward hooks for nn.Linear layers."""
        for param_name, (A, B) in self.low_rank_perturbations.items():
            if param_name in self.param_to_module:
                module, local_param_name = self.param_to_module[param_name]
                # Only apply hooks to nn.Linear weight parameters
                if isinstance(module, nn.Linear) and local_param_name == 'weight':
                    hook = module.register_forward_hook(self._create_hook(A, B))
                    self.hooks.append((hook, param_name))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove forward hooks."""
        for hook, _ in self.hooks:
            hook.remove()
        self.hooks.clear()


def compute_model_loss(
    model: nn.Module,
    batch,
    device: torch.device,
    use_weighted_loss: bool = False
) -> torch.Tensor:
    """
    Compute loss for a model with a batch of data.

    Args:
        model: PyTorch model
        batch: Batch of data (can be dict with 'batch' key or direct batch)
        device: Device to compute on
        use_weighted_loss: Whether to use affinity-based weights (not used in standalone version)

    Returns:
        Scalar loss tensor
    """
    # Set model to training mode but disable gradient computation
    model.train()
    with torch.no_grad():
        # Handle batch format - can be dict with 'batch' key or direct batch
        if isinstance(batch, dict) and 'batch' in batch:
            actual_batch = batch['batch']
        else:
            actual_batch = batch

        # Forward pass with is_training=True but no gradients
        # Need to temporarily disable report_metrics if it doesn't exist
        has_report_metrics = hasattr(model, 'report_metrics')
        if not has_report_metrics:
            # Add dummy report_metrics method if it doesn't exist
            def dummy_report_metrics(**kwargs):
                pass
            model.report_metrics = dummy_report_metrics

        try:
            output = model(actual_batch, labels=None, is_training=True)
        finally:
            # Remove dummy method if we added it
            if not has_report_metrics and hasattr(model, 'report_metrics'):
                delattr(model, 'report_metrics')

        if output is None or 'loss' not in output:
            return torch.tensor(float('inf'), device=device)

        loss = output['loss']

        # Check for invalid loss
        if not torch.isfinite(loss):
            return torch.tensor(float('inf'), device=device)

        return loss


def apply_perturbations_to_model(
    model: nn.Module,
    perturbations: Dict[str, torch.Tensor],
    sigma: float,
    mode: str = 'add'
) -> None:
    """
    Apply perturbations to model parameters.

    Args:
        model: PyTorch model
        perturbations: Dictionary mapping parameter names to perturbation tensors
        sigma: Perturbation scale
        mode: 'add' to add perturbation, 'restore' to restore original parameters
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name not in perturbations:
            continue

        pert = perturbations[name]
        # Ensure perturbation has same shape as parameter
        if pert.shape != param.shape:
            # Try to reshape
            try:
                pert = pert.reshape(param.shape)
            except:
                continue

        if mode == 'add':
            param.data.add_(sigma * pert)
        elif mode == 'restore':
            param.data.sub_(sigma * pert)  # Restore from +sigma to original
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'add' or 'restore'.")


def eggroll_step(
    model: nn.Module,
    batch,
    device: torch.device,
    n_workers: int,
    rank: int,
    sigma: float,
    learning_rate: float,
    grad_clip: Optional[float] = None,
    use_weighted_loss: bool = False,
    normalize_fitness: bool = True,
    base_seed: int = 0,
    epoch: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    noise_reuse: int = 0,
    group_size: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """
    Perform one step of EGGROLL optimization.

    Implements the EGGROLL update formula from the paper (formula 8, page 7):
    µ_{t+1} = µ_t + (α_t / N_workers) * Σ_{i=1}^{N_workers} E_{i,t} * f(M = µ_t + σE_{i,t})

    where:
    - E_{i,t} = (1/√r) * A_{i,t} * B_{i,t}^T is the low-rank perturbation
    - f(M) is the fitness function (in our case: -loss)
    - α_t is the learning rate
    - N_workers is the number of perturbation samples per iteration

    Args:
        model: PyTorch model to optimize
        batch: Batch of training data
        device: Device to compute on
        n_workers: Number of perturbation samples per iteration (N_workers in EGGROLL paper)
        rank: Rank of low-rank perturbation (r)
        sigma: Perturbation scale (σ)
        learning_rate: Learning rate for weight updates (α_t)
        grad_clip: Gradient clipping threshold (None to disable)
        use_weighted_loss: Whether to use affinity-based weights (not used in standalone version)
        normalize_fitness: Whether to normalize fitness values using z-score (default: True, as in official EGGROLL)
        base_seed: Base random seed for deterministic perturbation generation
        epoch: Epoch/step number for deterministic perturbation generation
        optimizer: Optional PyTorch optimizer (if None, uses simple SGD-like update)
        noise_reuse: Number of epochs to reuse the same noise (0 = no reuse, default).
                     If > 0, epochs are grouped and share perturbations.
        group_size: Size of groups for group-wise fitness normalization (0 = global normalization, default).
                    If > 0, fitness values are grouped and normalized within each group using group mean
                    but global std. Must divide n_workers evenly.

    Returns:
        Current loss value and metrics dictionary
    """
    # Get all trainable parameters
    trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}

    # Store original parameters
    original_params = {name: param.data.clone() for name, param in trainable_params.items()}

    # Initialize update accumulator (will accumulate E_i * f_i for each sample)
    # For efficiency, we store low-rank components (A, B) instead of full matrices E
    update_accumulator = {}
    low_rank_components = {}  # Store (A_list, B_list) for each parameter
    for name, param in trainable_params.items():
        if param.dim() >= 2:
            # For 2D+ parameters, store low-rank components
            low_rank_components[name] = ([], [])  # (A_list, B_list)
        else:
            # For 1D parameters, use full-rank accumulator
            update_accumulator[name] = torch.zeros_like(param)

    sample_losses = []
    valid_samples = 0

    # Store raw fitness values and perturbations for normalization
    raw_fitnesses = []
    stored_perturbations = []  # Still store full perturbations for 1D params and for applying to model

    # Generate n_workers perturbations and evaluate
    for sample_idx in range(n_workers):
        # Generate low-rank perturbations for each parameter
        # Store both full perturbations (for applying to model) and low-rank components (for efficient update)
        perturbations = {}
        current_low_rank = {}  # Temporary storage for current sample's low-rank components (cleared each iteration)

        # Compute normalized_sigma = sigma / sqrt(rank) as in JAX version
        # This is used for deterministic generation, but sigma is applied separately to E
        normalized_sigma = sigma / np.sqrt(rank)

        # Compute true_epoch based on noise_reuse (matching JAX behavior)
        # When noise_reuse == 0: use actual epoch (each epoch gets different perturbations)
        # When noise_reuse > 0: group epochs by dividing (epoch // noise_reuse)
        true_epoch = epoch if noise_reuse == 0 else epoch // noise_reuse

        for name, param in trainable_params.items():
            if param.dim() < 2:
                # For 1D parameters (bias, etc.), use full-rank perturbation
                # Generate deterministically using true_epoch
                true_thread_idx = sample_idx // 2  # Match JAX behavior
                param_seed = fold_in_seed(base_seed, true_epoch, true_thread_idx)
                param_seed = fold_in_seed(param_seed, hash(name), 0)
                generator = torch.Generator(device=device)
                generator.manual_seed(param_seed)
                # torch.randn_like doesn't support generator, so use torch.randn with explicit shape
                # Thread ID alternation: alternate sign based on sample_idx % 2 (matching JAX)
                sigma_sign = 1.0 if (sample_idx % 2 == 0) else -1.0
                perturbations[name] = (torch.randn(param.shape, device=device, generator=generator) / np.sqrt(param.numel())) * sigma_sign
            elif param.dim() == 2:
                # For 2D parameters (Linear layers), use low-rank perturbation WITHOUT forming full E matrix
                # We'll apply perturbation on-the-fly using forward hooks
                m, n = param.shape
                r = min(rank, m, n)  # Ensure rank doesn't exceed dimensions
                # Generate A, B with deterministic seed
                # In JAX version, normalized_sigma (sigma/sqrt(rank)) is applied to A during generation
                A_with_sigma, B = generate_low_rank_perturbation(
                    m, n, r, device, normalized_sigma, epoch, sample_idx, base_seed, name, noise_reuse
                )
                # Store A with sigma for update computation (matching JAX behavior)
                current_low_rank[name] = (A_with_sigma, B)
                # DO NOT form E = A @ B.T - we'll apply perturbation on-the-fly via hooks
            else:
                # For multi-dimensional parameters, flatten first two dimensions
                orig_shape = param.shape
                first_dim = orig_shape[0]
                rest_dims = orig_shape[1:]
                rest_size = int(np.prod(rest_dims))

                # Use low-rank perturbation
                r = min(rank, first_dim, rest_size)
                A_with_sigma, B = generate_low_rank_perturbation(
                    first_dim, rest_size, r, device, normalized_sigma, epoch, sample_idx, base_seed, name, noise_reuse
                )
                # Store A with sigma for update computation (matching JAX behavior)
                current_low_rank[name] = (A_with_sigma, B)
                # For E, we need A without sigma since sigma is applied separately
                A_original = A_with_sigma / normalized_sigma if normalized_sigma != 0 else A_with_sigma
                # Apply normalization factor (1/√r) when forming the perturbation for model application
                E = (1.0 / np.sqrt(r)) * (A_original @ B.T)
                perturbations[name] = E.reshape(orig_shape)

        # Evaluate fitness: f(M = µ + σE_i)
        # Apply perturbation: µ + σE_i
        # For 2D parameters, use forward hooks to apply perturbation on-the-fly (no full E matrix)
        # For 1D and multi-dimensional parameters, apply directly to parameters

        # Separate low-rank perturbations (for hooks) from full perturbations (for direct application)
        low_rank_perturbations_for_hooks = {name: (A, B) for name, (A, B) in current_low_rank.items()}
        full_perturbations_for_direct = {name: pert for name, pert in perturbations.items()
                                         if name not in current_low_rank}

        # Apply low-rank perturbations via hooks (for 2D parameters)
        with PerturbationHookManager(model, low_rank_perturbations_for_hooks, sigma, rank):
            # Apply full perturbations directly (for 1D and multi-dimensional parameters)
            apply_perturbations_to_model(model, full_perturbations_for_direct, sigma, mode='add')

            # Use loss-based fitness
            loss = compute_model_loss(model, batch, device, use_weighted_loss)

            # Restore original parameters (hooks are already removed by context manager)
            apply_perturbations_to_model(model, full_perturbations_for_direct, sigma, mode='restore')

            # Skip if loss is invalid
            if not torch.isfinite(loss):
                continue

            # Compute fitness: f_i = -loss (we maximize fitness, minimize loss)
            fitness = -loss.item()
            sample_losses.append(loss.item())

        # Final check: ensure fitness is valid before using it
        if not np.isfinite(fitness) or abs(fitness) > 1000.0:
            print(f"Warning: Invalid fitness {fitness} after all checks, skipping sample")
            continue

        valid_samples += 1

        # Store raw fitness and perturbations for later normalization
        raw_fitnesses.append(fitness)
        stored_perturbations.append(perturbations)

        # Store low-rank components for valid samples only (for efficient update computation)
        for name, (A, B) in current_low_rank.items():
            if name not in low_rank_components:
                low_rank_components[name] = ([], [])
            low_rank_components[name][0].append(A)
            low_rank_components[name][1].append(B)

    # Normalize fitness values using z-score normalization (as in official EGGROLL implementation)
    if valid_samples > 0 and len(raw_fitnesses) > 0:
        if normalize_fitness:
            raw_fitnesses_array = np.array(raw_fitnesses)
            # Z-score normalization: (x - mean) / std
            # This matches the official EGGROLL implementation (convert_fitnesses method)
            # If group_size > 0, normalize within groups but use global std
            if group_size > 0 and len(raw_fitnesses_array) % group_size == 0:
                # Group normalization: reshape into groups, normalize within each group
                # JAX: group_scores = raw_scores.reshape((-1, group_size))
                #      true_scores = (group_scores - mean(group_scores, axis=-1)) / sqrt(var(raw_scores) + eps)
                group_scores = raw_fitnesses_array.reshape((-1, group_size))
                # Mean within each group (axis=-1 means last dimension)
                group_means = np.mean(group_scores, axis=-1, keepdims=True)
                # Global std (from all raw_scores, matching JAX: jnp.var(raw_scores, keepdims=True))
                global_std = np.std(raw_fitnesses_array, keepdims=True)
                eps = 1e-5
                # Normalize: subtract group mean, divide by global std
                normalized_group_scores = (group_scores - group_means) / (global_std + eps)
                # Flatten back to original shape
                normalized_fitnesses = normalized_group_scores.ravel()
            else:
                # Global normalization (group_size = 0 or not divisible)
                mean_fitness = np.mean(raw_fitnesses_array)
                std_fitness = np.std(raw_fitnesses_array)
                eps = 1e-5  # Small epsilon to prevent division by zero
                normalized_fitnesses = (raw_fitnesses_array - mean_fitness) / (std_fitness + eps)
        else:
            # Use raw fitness values without normalization
            normalized_fitnesses = np.array(raw_fitnesses)

        # Accumulate updates with (normalized) fitness values
        # According to EGGROLL formula: µ += (α/N) * Σ E_i * f_i
        # For efficiency, use low-rank components and einsum (as in official implementation)
        # Instead of forming full matrices E_i, we compute: Σ_i (f_i * A_i) @ B_i^T efficiently

        # Process low-rank parameters efficiently using einsum
        # low_rank_components already contains only valid samples (we only append on valid samples)
        for name, (A_list, B_list) in low_rank_components.items():
            if len(A_list) != len(normalized_fitnesses) or len(B_list) != len(normalized_fitnesses):
                continue

            param = trainable_params[name]

            # Stack all A and B matrices: (n_workers, m, r) and (n_workers, n, r)
            A_stack = torch.stack(A_list, dim=0)  # (n_workers, m, r)
            B_stack = torch.stack(B_list, dim=0)  # (n_workers, n, r)
            fitness_tensor = torch.tensor(normalized_fitnesses, device=device, dtype=A_stack.dtype)

            # Reshape fitness for broadcasting: (n_workers, 1, 1)
            fitness_broadcast = fitness_tensor.view(-1, 1, 1)

            # Weight A by fitness: A_weighted = fitness * A  -> (n_workers, m, r)
            # Note: A_stack already contains normalized_sigma (sigma/sqrt(rank)) as in JAX version
            A_weighted = fitness_broadcast * A_stack

            # Compute update efficiently using einsum: Σ_i (f_i * A_i) @ B_i^T
            # einsum('nir,njr->ij', A_weighted, B_stack) computes Σ_n (A_weighted[n,i,r] * B_stack[n,j,r])
            # This is equivalent to Σ_n (A_weighted[n] @ B_stack[n].T) but more efficient
            # Matching JAX version: einsum('nir,njr->ij', A, B) / num_envs
            # Note: A already contains sigma/sqrt(rank), so we don't need additional (1/sqrt(r)) factor
            num_envs = len(normalized_fitnesses)
            if param.dim() == 2:
                # For 2D: einsum('nir,njr->ij', A_weighted, B_stack) -> (m, n)
                # JAX: einsum(...) / num_envs, then multiplied by sqrt(num_envs) in _do_update
                update = torch.einsum('nir,njr->ij', A_weighted, B_stack) / num_envs
            else:
                # For multi-dimensional: flatten, compute, then reshape
                orig_shape = param.shape
                first_dim = orig_shape[0]
                rest_dims = orig_shape[1:]
                rest_size = int(np.prod(rest_dims))

                # Reshape A and B to match flattened dimensions
                A_flat = A_weighted.view(num_envs, first_dim, -1)
                B_flat = B_stack.view(num_envs, rest_size, -1)

                # Matching JAX version: A already contains sigma/sqrt(rank), so no additional normalization
                update_flat = torch.einsum('nir,njr->ij', A_flat, B_flat) / num_envs
                update = update_flat.reshape(orig_shape)

            update_accumulator[name] = update

        # Process 1D parameters (full-rank) - accumulate normally
        for i, fitness in enumerate(normalized_fitnesses):
            if i < len(stored_perturbations):
                for name, pert in stored_perturbations[i].items():
                    if name in update_accumulator and name not in low_rank_components:
                        update_accumulator[name] += pert * fitness

    if valid_samples == 0:
        # All samples were invalid, return current loss
        current_loss = compute_model_loss(model, batch, device, use_weighted_loss)
        return current_loss.item(), {'valid_samples': 0}

    # Apply EGGROLL update: µ_{t+1} = µ_t + (α_t / N_workers) * Σ E_i * f_i
    # Matching JAX version: multiply by sqrt(fitnesses.size) before applying learning_rate
    # In JAX: update = einsum(...) / num_envs, then * sqrt(num_envs) in _do_update
    # So final update = einsum(...) * sqrt(num_envs) / num_envs
    # JAX uses: new_grad = -(update * sqrt(fitnesses.size)), then optimizer applies it

    # Prepare gradients for optimizer (if using optimizer) or apply directly
    if optimizer is not None:
        # Use optimizer: set gradients and let optimizer handle the update
        optimizer.zero_grad()

        for name, param in trainable_params.items():
            if name in update_accumulator:
                # update_accumulator already contains einsum(...) / num_envs
                # Scale by sqrt(valid_samples) as in JAX version (line 125: * jnp.sqrt(fitnesses.size))
                # This gives: einsum(...) * sqrt(valid_samples) / valid_samples
                update = update_accumulator[name] * np.sqrt(valid_samples)

                # Clip update if needed
                if grad_clip is not None:
                    update_norm = torch.norm(update)
                    if update_norm > grad_clip:
                        update = update * (grad_clip / update_norm)

                # Check for NaN/Inf in update
                if torch.isfinite(update).all():
                    # JAX uses negative sign: return -(new_grad * sqrt(fitnesses.size))
                    # So we set grad = -update (optimizer will apply it with learning rate)
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad.data.copy_(-update)
                else:
                    print(f"Warning: Non-finite update detected for parameter {name}, skipping")
                    if param.grad is not None:
                        param.grad.zero_()

        # Apply optimizer step (handles learning rate and optimizer-specific logic)
        optimizer.step()
    else:
        # Direct update (simple SGD-like behavior, matching original implementation)
        for name, param in trainable_params.items():
            if name in update_accumulator:
                # update_accumulator already contains einsum(...) / num_envs
                # Scale by sqrt(valid_samples) as in JAX version (line 125: * jnp.sqrt(fitnesses.size))
                # This gives: einsum(...) * sqrt(valid_samples) / valid_samples
                update = update_accumulator[name] * np.sqrt(valid_samples)

                # Clip update if needed
                if grad_clip is not None:
                    update_norm = torch.norm(update)
                    if update_norm > grad_clip:
                        update = update * (grad_clip / update_norm)

                # Apply update: µ += α * sqrt(N) * (1/N) * Σ E_i * f_i = α * (1/sqrt(N)) * Σ E_i * f_i
                # Check for NaN/Inf in update before applying
                if torch.isfinite(update).all():
                    param.data.add_(learning_rate * update)
                else:
                    print(f"Warning: Non-finite update detected for parameter {name}, skipping update")

    # Compute current loss with updated parameters
    current_loss = compute_model_loss(model, batch, device, use_weighted_loss)
    metrics = {
        'loss': current_loss.item(),
        'valid_samples': valid_samples,
        'mean_sample_loss': np.mean(sample_losses) if sample_losses else float('inf'),
    }

    # Compute update norm (equivalent to gradient norm)
    # This should be computed BEFORE applying learning_rate, matching the actual update magnitude
    total_update_norm = 0.0
    for name in update_accumulator:
        if name in trainable_params:
            # Compute final update that was applied (with sqrt scaling, before learning_rate)
            update = update_accumulator[name] * np.sqrt(valid_samples)  # Match the scaling applied above
            if grad_clip is not None:
                update_norm = torch.norm(update)
                if update_norm > grad_clip:
                    update = update * (grad_clip / update_norm)
            # Compute norm of the update (before learning_rate multiplication)
            total_update_norm += torch.norm(update).item() ** 2
    metrics['grad_norm'] = np.sqrt(total_update_norm)

    return current_loss.item(), metrics


class EGGROLLTrainer:
    """
    Trainer class for EGGROLL optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        rank: int = 30,
        sigma: float = 0.01,
        learning_rate: float = 0.001,
        n_workers: int = 30,
        grad_clip: Optional[float] = 1.0,
        use_weighted_loss: bool = False,
        log_steps: int = 100,
        save_steps: Optional[int] = None,
        results_dir: Optional[str] = None,
        normalize_fitness: bool = True,
        base_seed: int = 0,
        optimizer: Optional[Union[str, torch.optim.Optimizer]] = None,
        optimizer_kwargs: Optional[Dict] = None,
        noise_reuse: int = 0,
        group_size: int = 0,
    ):
        """
        Initialize EGGROLL trainer.

        Args:
            model: PyTorch model to train
            device: Device to train on
            rank: Rank of low-rank perturbation
            sigma: Perturbation scale
            learning_rate: Learning rate for weight updates
            n_workers: Number of perturbation samples per iteration (N_workers in EGGROLL paper)
            grad_clip: Gradient clipping threshold
            use_weighted_loss: Whether to use affinity-based weights (not used in standalone version)
            log_steps: Frequency of logging
            save_steps: Frequency of checkpoint saving (None to disable)
            results_dir: Directory to save checkpoints and logs
            normalize_fitness: Whether to normalize fitness values using z-score (default: True, as in official EGGROLL)
            base_seed: Base random seed for deterministic perturbation generation (default: 0)
            optimizer: Optimizer to use. Can be:
                - None: Use simple SGD-like update (default, matches original implementation)
                - str: Name of optimizer ('sgd', 'adam', 'adamw')
                - torch.optim.Optimizer: Pre-initialized optimizer instance
            optimizer_kwargs: Additional keyword arguments for optimizer (e.g., {'betas': (0.9, 0.999)} for AdamW)
            noise_reuse: Number of epochs to reuse the same noise (0 = no reuse, default).
                         If > 0, epochs are grouped into blocks of size noise_reuse, and all epochs
                         in the same block use identical perturbations. For example, noise_reuse=2 means
                         epochs 0-1 share perturbations, epochs 2-3 share perturbations, etc.
            group_size: Size of groups for group-wise fitness normalization (0 = global normalization, default).
                         If > 0, fitness values are grouped and normalized within each group. Each group's
                         fitness values are normalized by subtracting the group mean, but divided by the
                         global standard deviation (matching JAX behavior). Must divide n_workers evenly.
                         For example, if n_workers=8 and group_size=4, there will be 2 groups of 4 samples each.
        """
        self.model = model
        self.device = device
        self.rank = rank
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_workers = n_workers
        self.grad_clip = grad_clip
        self.use_weighted_loss = use_weighted_loss
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.results_dir = results_dir
        self.normalize_fitness = normalize_fitness
        self.base_seed = base_seed
        self.noise_reuse = noise_reuse
        self.group_size = group_size

        # Validate group_size
        if group_size > 0 and n_workers % group_size != 0:
            raise ValueError(f"group_size ({group_size}) must divide n_workers ({n_workers}) evenly")

        self.step = 0
        self.loss_history = []
        self.metrics_history = []

        # Move model to device
        self.model.to(device)

        # Initialize optimizer
        if optimizer is None:
            # Default: no optimizer (simple SGD-like update)
            self.optimizer = None
        elif isinstance(optimizer, str):
            # Create optimizer from string name
            optimizer_kwargs = optimizer_kwargs or {}
            optimizer_name = optimizer.lower()

            if optimizer_name == 'sgd':
                self.optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    **optimizer_kwargs
                )
            elif optimizer_name == 'adam':
                self.optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    **optimizer_kwargs
                )
            elif optimizer_name == 'adamw':
                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    **optimizer_kwargs
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}. Supported: 'sgd', 'adam', 'adamw'")
        elif isinstance(optimizer, torch.optim.Optimizer):
            # Use provided optimizer instance
            self.optimizer = optimizer
        else:
            raise TypeError(f"optimizer must be None, str, or torch.optim.Optimizer, got {type(optimizer)}")

    def train_step(self, batch) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary of metrics
        """
        loss, metrics = eggroll_step(
            self.model,
            batch,
            self.device,
            self.n_workers,
            self.rank,
            self.sigma,
            self.learning_rate,
            self.grad_clip,
            self.use_weighted_loss,
            self.normalize_fitness,
            self.base_seed,
            self.step,  # Use step as epoch for deterministic generation
            optimizer=self.optimizer,
            noise_reuse=self.noise_reuse,
            group_size=self.group_size,
        )

        self.step += 1
        self.loss_history.append(loss)
        self.metrics_history.append(metrics)

        return metrics

    def save_checkpoint(self, checkpoint_name: str = None) -> str:
        """
        Save model checkpoint.

        Args:
            checkpoint_name: Name for checkpoint (default: checkpoint-{step})

        Returns:
            Path to saved checkpoint
        """
        if self.results_dir is None:
            return None

        import os
        os.makedirs(self.results_dir, exist_ok=True)

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint-{self.step}"

        checkpoint_path = os.path.join(self.results_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model state
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)

        # Save training state
        # Convert numpy arrays to lists for serialization compatibility
        loss_history = [float(x) if isinstance(x, (np.ndarray, np.generic)) else x for x in self.loss_history]
        metrics_history = []
        for metrics in self.metrics_history:
            metrics_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (np.ndarray, np.generic)):
                    metrics_dict[k] = float(v) if v.ndim == 0 else v.tolist()
                else:
                    metrics_dict[k] = v
            metrics_history.append(metrics_dict)

        training_state = {
            'step': self.step,
            'loss_history': loss_history,
            'metrics_history': metrics_history,
            'rank': self.rank,
            'sigma': float(self.sigma),
            'learning_rate': float(self.learning_rate),
        }

        # Save optimizer state if using optimizer
        if self.optimizer is not None:
            training_state['optimizer_state'] = self.optimizer.state_dict()

        state_path = os.path.join(checkpoint_path, "training_state.pt")
        torch.save(training_state, state_path)

        return checkpoint_path

