class Config():
    """A class that stores core configuration variables."""
    def add(self, **kwargs) -> None:
        """Add parameters to the class."""
        for arg, val in kwargs.items():
            setattr(self, arg, val)

    def set_env_params(self) -> None:
        """Sets crucial environment variables."""
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.input_shape = self.observation_space.shape
        self.n_actions = self.action_space.n
