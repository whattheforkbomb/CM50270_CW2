from collections import namedtuple

import torch

class Storage():
    """A storage container for agent experiences and other useful metrics."""
    def __init__(self, size: int) -> None:
        self.exp_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.other_keys = ['returns', 'log_probs', 'entropys', 'next_returns', 'advantages']
        self.all_keys = self.exp_keys + self.other_keys
        self.size = size
        self.reset() # initalize key values

    def add(self, **kwargs) -> None:
        """Adds a tuple of experience to the storage."""
        for key, val in kwargs.items():
            if key not in self.all_keys:
                raise RuntimeError('Undefined key')
            
            getattr(self, key).append(val)

    def placeholder(self) -> None:
        """Sets key list data with 'None' values of the storage size."""
        for key in self.all_keys:
            data = getattr(self, key)
            if len(data) == 0:
                setattr(self, key, [None] * self.size)

    def reset(self) -> None:
        """Empties storage."""
        for key in self.all_keys:
            setattr(self, key, [])

    def retrieve(self, keys: list) -> namedtuple:
        """Retrieves a batch of data from the storage based on the provided keys."""
        data = [getattr(self, key)[:self.size] for key in keys]
        data = map(lambda x: torch.cat(x, dim=0), data)
        Entry = namedtuple('Entry', keys)
        return Entry(*list(data))
