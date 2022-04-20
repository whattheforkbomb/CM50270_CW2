import argparse

from a2c.model import ACNetwork
from a2c.agent import A2CAgent
from a2c.multi_envs import MultiEnvs
from core.tuning import Tuning
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
VALUE_LOSS_WEIGHT = 0.01
BATCH_SIZE = 128
NUM_AGENTS = 5
N_STEPS = 4 # TD bootstrapping
GRAD_CLIP = 0.1 # Prevents gradients from being too large

# Create environment
env = gym_super_mario_bros.make(ENV_NAME)
env = JoypadSpace(env, RIGHT_ONLY)

# Set config instance
config = Config()

def parser_settings() -> argparse.Namespace:
    """Enables argument parsing in console and returns passed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-eps", "--episode_count", required=True, help="Episode count (default=5000)", type=int, default=5000)
    return parser.parse_args()

def main() -> None:
    """Runs the main application."""
    # Get console arguments and device
    args = parser_settings()
    device = set_device()

    # Add core items to config
    config.add(
        game=ENV_NAME,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        epsilon=EPSILON,
        entropy_weight=ENTROPY_WEIGHT,
        value_loss_weight=VALUE_LOSS_WEIGHT,
        batch_size=BATCH_SIZE,
        num_agents=NUM_AGENTS,
        grad_clip=GRAD_CLIP,
        rollout_size=N_STEPS,
        device=device,
        num_episodes=args.episode_count,
        task_fn=lambda: MultiEnvs(ENV_NAME, num_envs=NUM_AGENTS)
    )

    # Set core classes
    a2c = ACNetwork(
        env.observation_space.shape,
        env.action_space.n
    ).to(device)

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
