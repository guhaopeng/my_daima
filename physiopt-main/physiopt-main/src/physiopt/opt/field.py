import torch
import numpy as np


def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def occ_kernel(
    sdf: torch.Tensor | np.ndarray,
    res: int,
    beta: float = 2.0,
):
    """Converts the SDF to occupancy using a sigmoid kernel (Equation 2)"""
    if isinstance(sdf, torch.Tensor):
        # Multiply by res to bring to scale, then this is just an inverse sigmoid scaled by beta
        return torch.sigmoid(-1.0 * sdf * res * beta)
    elif isinstance(sdf, np.ndarray):
        return sigmoid(-1.0 * sdf * res * beta)
    else:
        return TypeError()
