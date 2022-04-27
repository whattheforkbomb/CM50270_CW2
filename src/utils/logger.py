class Logger():
    """Stores information for each episode iteration."""
    def __init__(self):
        self.actions = []
        self.avg_returns = []
        self.total_losses = []

    def add_actions(self, actions: list) -> None:
        """Add episode actions to the logger."""
        self.actions.append(actions)

    def add_return(self, avg_return: float) -> None:
        """Add the average return for the episode to the logger."""
        self.avg_returns.append(avg_return)

    def add_loss(self, total_loss: float) -> None:
        """Add the total loss of an episode to the logger."""
        self.total_losses.append(total_loss)
