from a2c.model import ACNetwork
from a2c.agent import A2CAgent
from utils.config import Config
from utils.helper import set_device

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import torch.optim as optim

# Set hyperparameters
ENV_NAME = 'SuperMarioBros-v0'
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1e-3
ENTROPY_WEIGHT = 0.01
N_STEPS = 4 # TD bootstrapping
GRAD_CLIP = 0.1 # Prevents gradients from being too large
NUM_EPISODES = 100
SAVE_MODEL_FILENAME = 'a2c'

# Create environment
env = gym_super_mario_bros.make(ENV_NAME)
env = JoypadSpace(env, RIGHT_ONLY)

# Set config instance
config = Config()

def main() -> None:
    """Runs the main application."""
    # Set cuda device
    device = set_device()

    # Add core items to config
    config = Config()
    config.add(
        env=env,
        env_name=ENV_NAME,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        epsilon=EPSILON,
        entropy_weight=ENTROPY_WEIGHT,
        rollout_size=N_STEPS,
        grad_clip=GRAD_CLIP,
        device=device,
        num_episodes=NUM_EPISODES,
        filename=SAVE_MODEL_FILENAME
    )

    # Setup environment parameters
    config.set_env_params()

    # Create network
    a2c = ACNetwork(config.input_shape, config.n_actions).to(device)

    # Add optimizer and network to config
    config.add(
        optimizer_fn=lambda params: optim.Adam(
            params,
            lr=config.lr,
            eps=config.epsilon
        ),
        network_fn=lambda: a2c
    )

    # Train agent
    agent = A2CAgent(config)
    agent.train()

if __name__ == "__main__":
    main()
