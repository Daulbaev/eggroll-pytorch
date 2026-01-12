"""
EGGROLL: Evolution Guided General Optimization via Low-rank Learning

Unofficial PyTorch implementation of the EGGROLL algorithm.
"""

from .trainer import EGGROLLTrainer, eggroll_step, fold_in_seed, generate_low_rank_perturbation

__version__ = "0.1.0"

__all__ = ["EGGROLLTrainer", "eggroll_step", "fold_in_seed", "generate_low_rank_perturbation"]

