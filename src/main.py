from a2c.agent import A2CAgent
from a2c.model import ACNetwork
from utils.wrappers import ResizeObservation, SkipFrame
from utils.config import Config
from utils.helper import get_cuda_device_names

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace

import torch
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

def tuning(config: Config, rollout_sizes: list, print_every: int, save_count: int) -> None:
    """Trains the model on multiple rollout sizes."""
    # Get device names
    device_count = torch.cuda.device_count()
    device_names = get_cuda_device_names(device_count)

    for device in device_names:
        for size in rollout_sizes:
            agent = A2CAgent(config, device=device)
            agent.config.rollout_size = size
            agent.train(print_every=print_every, save_count=save_count)

def main() -> None:
    """Runs the main application."""
    # Create environment
    env = gym_super_mario_bros.make(ENV_NAME)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Apply wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False) # Grayscale images
    env = ResizeObservation(env, shape=84) # image dim: [84, 84]
    env = FrameStack(env, num_stack=4) # 4 frames at a time

    # Set config instance
    config = Config(env=env, env_name=ENV_NAME)

    # Create network and optimizer
    network = ACNetwork(config.input_shape, config.n_actions)
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=EPSILON)

    # Add hyperparameters to config
    config.add(
        discount=GAMMA,
        entropy_weight=ENTROPY_WEIGHT,
        value_loss_weight=VALUE_LOSS_WEIGHT,
        rollout_size=N_STEPS,
        grad_clip=GRAD_CLIP,
        num_episodes=NUM_EPISODES,
        network=network,
        optimizer=optimizer
    )

    # Train agent
    tuning(config=config, rollout_sizes=[50, 100], print_every=5000, save_count=20000)

if __name__ == "__main__":
    main()
