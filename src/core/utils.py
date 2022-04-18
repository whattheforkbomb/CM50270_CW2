"""
Utility functions that are usable across the whole application.
"""
import torch
import torch.nn as nn

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

def save_model(params_dict: dict, filename: str) -> None:
    """
    Saves a models parameters with the given filepath.

    Parameters:
    - params_dict (dict) - additional parameters to store
    - filename (str) - name of the saved model
    """
    torch.save(params_dict, f'saved_models/{filename}.pt')

def load_model(model: nn.Module, filename: str) -> None:
    """
    Loads a saved model based on the given filepath and stores its parameters to the model.

    Parameters:
    - model (nn.Module) - model to store parameters to
    - filename (str) - name of the model to load    
    """
    checkpoint = torch.load(f'saved_models/{filename}.pt')

    # Store variables as model attributes
    for key, val in checkpoint.items():
        if key != 'parameters':
            setattr(model, key, val)

    # Load model parameters
    model.load_state_dict(checkpoint['parameters'])

    print("Model loaded. Utility variables available:")
    print("  advantages, state_values, rewards, policy,") 
    print("  losses_policy, losses_sv, losses_entropy, losses_total")

def set_multi_processors(x: torch.Tensor, y: torch.Tensor) -> None:
    """Splits data across multiple GPU cores."""
    pass