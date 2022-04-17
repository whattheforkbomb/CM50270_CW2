import argparse

from a2c.model import ACNetwork
from a2c.agent import A2C
from core.experience import FirstLastExpBuffer
from core.utils import set_device, set_multi_processors

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch.optim as optim

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1e-3
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

# Threshold for gradient clipping
# Prevents gradients from being too large
CLIP_GRAD = 0.1

# Steps ahead to approx reward for every action
REWARD_STEPS = 4

def parser_settings():
    """Enables argument parsing in console and returns passed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-eps", "--episode_count", required=True, help="Episode count (default=5000)", type=int, default=5000)
    return parser.parse_args()

def run(episode_count: int) -> None:
    done = True
    for step in range(episode_count):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample()) # pick random action on each step, no learning
        env.render() # Disable for faster execution (e.g. don't need to render the game output).
        # If dependencies correctly setup, you shuold see a window pop-up with the Super Mario Bros world in it.
    env.close()

def main() -> None:
    """Runs the main application."""
    # Get console arguments and device
    args = parser_settings()
    device = set_device()

    # Set core classes
    net = ACNetwork(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    ).to(device)

    agent = A2C(
        gamma=GAMMA, 
        reward_steps=REWARD_STEPS, 
        net=lambda x: net(x)[0], # policy only
        device=device
    )

    exp_buffer = FirstLastExpBuffer(
        env=env,
        agent=agent,
        gamma=GAMMA,
        buffer_size=REWARD_STEPS
    )

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=EPSILON)

    run(args.episode_count)

if __name__ == "__main__":
    main()
