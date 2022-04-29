from a2c.agent import A2CAgent
from a2c.model import ACNetwork
from a2c.wrappers import ResizeObservation, SkipFrame
from utils.config import Config
from utils.helper import set_device

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace

import torch.optim as optim

# Set hyperparameters
ENV_NAME = 'SuperMarioBros-v3'

GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1e-3
ENTROPY_WEIGHT = 0.01
VALUE_LOSS_WEIGHT = 1.0

N_STEPS = 10 # TD bootstrapping
GRAD_CLIP = 5 # Prevents gradients from being too large
NUM_EPISODES = 1000

# Create environment
env = gym_super_mario_bros.make(ENV_NAME)
env = JoypadSpace(env, RIGHT_ONLY)

# Apply wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False) # Grayscale images
env = ResizeObservation(env, shape=84) # image dim: [84, 84]
env = FrameStack(env, num_stack=4) # 4 frames at a time

# Set config instance
config = Config()

def main() -> None:
    """Runs the main application."""
    # Set cuda device
    device = set_device()

    # Add core items to config
    config.add(
        env=env,
        env_name=ENV_NAME,
        discount=GAMMA,
        entropy_weight=ENTROPY_WEIGHT,
        value_loss_weight=VALUE_LOSS_WEIGHT,
        rollout_size=N_STEPS,
        grad_clip=GRAD_CLIP,
        device=device,
        num_episodes=NUM_EPISODES
    )

    # Setup environment parameters
    config.set_env_params()

    # Create networks
    network = ACNetwork(config.input_shape, config.n_actions).to(device)

    # Add networks and optimizers to config
    config.add(
        network=network,
        optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=EPSILON)
    )

    # Train agent
    agent = A2CAgent(config)
    agent.train()

if __name__ == "__main__":
    main()
