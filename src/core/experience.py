from collections import namedtuple
import torch

class ExperienceBuffer():
    """
    A basic experience buffer that holds information useful information for sampling.
    
    Parameters:
    - buffer_size (int) - number of samples to store in the buffer
    """
    def __init__(self, buffer_size: int) -> None:
        self.keys = ['state', 'action', 'reward', 'mask', 'v', 'q', 'pi', 'log_pi', 'entropy', 'advantage', 'returns', 'q_a', 'log_pi_a', 'mean', 'next_state']
        self.buffer_size = buffer_size
        self.empty()

    def add(self, data: dict) -> None:
        """Add data to the buffer."""
        for key, val in data.items():
            getattr(self, key).append(val)

    def initalise_values(self) -> None:
        """Initalise key values to empty lists."""
        for key in self.keys:
            val = getattr(self, key)

            if len(val) == 0:
                setattr(self, key, [None] * self.buffer_size)
    
    def empty(self) -> None:
        """Empty the buffer."""
        self.pos = 0
        self._size = 0
        for key in self.keys():
            setattr(self, key, [])
    
    def sample(self, keys) -> namedtuple:
        """Extract data from the buffer."""
        data = [getattr(self, key)[:self.buffer_size] for key in keys]
        data = map(lambda x: torch.cat(x, dim=0), data)
        Entry = namedtuple('Entry', keys)
        return Entry(*list(data))
