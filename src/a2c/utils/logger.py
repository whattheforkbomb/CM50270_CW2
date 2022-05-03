
class Logger():
    """Stores information for each episode iteration."""
    def __init__(self):
        self.keys = ['actions', 'log_probs', 'entropys', 'env_info', 'avg_advantages', 'avg_returns', 'total_losses', 'policy_losses', 'value_losses', 'entropy_losses', 'save_batch_stats']
        self.set_defaults()

    def add(self, **kwargs) -> None:
        """Add episode items to respective lists in the logger."""
        for key, val in kwargs.items():
            if key not in self.keys:
                raise RuntimeError('Undefined key')
            
            getattr(self, key).append(val)

    def set_defaults(self) -> None:
        """Creates empty list values for all keys."""
        for key in self.keys:
            setattr(self, key, [])
