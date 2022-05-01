"""
Helper functions that are usable across the whole application.
"""
import math
from typing import Union
import numpy as np

import torch

def get_primary_device(count: int) -> str:
    """Gets the primary CUDA device name for single GPU, CPU or multiple GPUs."""
    if count == 0:
        print("CUDA unavailable. Device set to CPU.")
        return "cpu"
    elif count == 1:
        print(f'Single CUDA device available. Device set to GPU.')
        return "cuda:0"
    else:
        print(f'{count} CUDA devices available. Device set to second GPU.')
    return "cuda:1"

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

def human_format_number(num: int) -> str:
    """Returns a number as a string in human readable format. For example: 1000 -> 1k."""
    letters = ['', 'K', 'M', 'G', 'T', 'P']
    condition = 0 if num == 0 else math.log10(abs(num)) / 3
    idx = max(0, min(len(letters)-1, int(math.floor(condition))))
    num /= 10 ** (3 * idx)
    return [num, letters[idx]]
