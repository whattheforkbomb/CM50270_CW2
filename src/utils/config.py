
class Config():
    """A class that stores core configuration variables."""
    def add(self, **kwargs) -> None:
        """Add parameters to the class."""
        for arg, val in kwargs.items():
            setattr(self, arg, val)
