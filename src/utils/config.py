from a2c.env_wrappers import SingleVecEnv

class Config():
    """A class that stores core configuration variables."""
    def add(self, **kwargs) -> None:
        """Add parameters to the class."""
        for arg, val in kwargs.items():
            setattr(self, arg, val)

    def set_env_params(self) -> None:
        """Sets crucial environment variables."""
        self.env = SingleVecEnv(self.env, self.env_name)
        self.input_shape = self.env.input_shape
        self.state_dim = self.env.state_dim
        self.action_space = self.env.action_space
        self.n_actions = self.env.n_actions
