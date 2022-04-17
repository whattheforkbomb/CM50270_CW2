"""
Utility functions that are usable across the whole application.
"""
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

def set_multi_processors(x: torch.Tensor, y: torch.Tensor) -> None:
    """Splits data across multiple GPU cores."""
    pass

def save_model():
    """Saves a models parameters with the given filepath."""
    pass

def load_model():
    """Loads a saved model based on the given filepath."""
    pass