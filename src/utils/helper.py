"""
Helper functions that are usable across the whole application.
"""
from typing import Union
import numpy as np

import torch

def set_device() -> str:
    """
    Sets CUDA device to GPU if available. Returns a string of the device type.
    """
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA available. Device set to GPU.")
    else:
        device = "cpu"
        print("CUDA unavailable. Device set to CPU.")
    return device

def to_tensor(x: Union[list, np.array, torch.Tensor]) -> torch.Tensor:
    """Converts a list or numpy array to a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.array(x))

def to_numpy(x: torch.Tensor, detach: bool = False) -> np.array:
    """Converts a torch tensor to a numpy array."""
    if detach:
        return x.cpu().detach().numpy()
    return x.cpu().numpy()

def normalize_states(states: Union[list, np.array, torch.Tensor]) -> Union[np.array, torch.Tensor]:
    """Normalize a given list of states."""
    if not isinstance(states, torch.Tensor):
        states = np.asarray(states)
    return (1.0 / 255) * states
