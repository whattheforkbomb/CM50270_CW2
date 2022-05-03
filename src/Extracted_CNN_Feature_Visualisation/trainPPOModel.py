from subprocess import call
import commonPPOModel
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, StopTrainingOnMaxEpisodes, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import gym
import torch.nn as nn
import torch
import os

# Derived from: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        print(f"local: {self.locals.keys()}")
        print(f"rewards: {self.locals['rewards']}")
        print(f"rollout_buffer: {self.locals['rollout_buffer']}")
        print(f"values: {self.locals['values']}")
        return True

    # def _on_rollout_end(self) -> None:
    #     print(f"local: {self.locals.keys()}")
    #     pass

def create_model(training_env, log_dir, model_path):
    # hard-coded to different mount
    tensorboard_path = os.path.join(log_dir, "tensorboard")
    if not os.path.isdir(tensorboard_path):
        os.mkdir(tensorboard_path)
    print("Tensorboard logs will be saved to: {}".format(tensorboard_path))

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    if model_path is not None:
        assert os.path.isfile(model_path)
        print("Loading model from: {}".format(model_path))
        return PPO.load(model_path, training_env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)
    else:
        # Using Pre-implemented algo
        return PPO("CnnPolicy", training_env, batch_size=128, n_steps=4096, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)

def train_model(model, eval_env, total_timesteps, save_freq, src_dir, log_dir, model_name):
    assert os.path.isdir(src_dir)
    model_path = os.path.join(src_dir, "models/ppo")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    model_path = os.path.join(model_path, model_name)
    print("Model will be saved to: {}".format(model_path))

    checkpoint_path = os.path.join(log_dir, "checkpoints")
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    print("Checkpoints will be saved to: {}".format(checkpoint_path))

    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(log_dir, "best"),
                             log_path=log_dir, eval_freq=500,
                             deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                            save_path=checkpoint_path,
                                            name_prefix=model_name)
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=2000000, verbose=1)
    additional_tensorboard_values_callback = TensorboardCallback()
    callbacks = [
        checkpoint_callback, 
        callback_max_episodes, 
        # additional_tensorboard_values_callback, 
        # eval_callback
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(model_path)

def process_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument("-a", dest="action_space", choices=[*commonPPOModel.ACTION_SPACE], default="ro")
    parser.add_argument("-i", dest="input_type", choices=[*commonPPOModel.INPUT_TYPE], default="rect")
    parser.add_argument("-n", dest="env_count", default=1, type=int)
    parser.add_argument("-t", dest="timesteps", default=int(2e6), type=int)
    parser.add_argument("-r", dest="random_worlds", action="store_true")
    parser.add_argument("--log_dir", dest="log_dir", required=True)
    parser.add_argument("--src_dir", dest="src_dir", required=False, default=os.getcwd())
    parser.add_argument("--model_name", dest="model_name", required=False)

    return parser.parse_args()

args = process_args()

training_env, env_config = commonPPOModel.create_vec_env(action_space=args.action_space, version=args.input_type, random=args.random_worlds, n=args.env_count, monitor=True)

eval_env, _ = commonPPOModel.create_vec_env(action_space=args.action_space, version=args.input_type, random=args.random_worlds, n=1, monitor=True)

log_dir = args.log_dir
assert os.path.isdir(log_dir)
model_name = f"SMB_{env_config[0]}_{env_config[1]}_{env_config[2]}_{env_config[3]}_{args.timesteps}"
log_dir = os.path.join(log_dir, model_name)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

model = create_model(training_env, log_dir, None if args.model_name is None else os.path.join(args.src_dir, "models/ppo", args.model_name))

try:
    train_model(model, eval_env, args.timesteps, int((args.timesteps / 10) / args.env_count), args.src_dir, log_dir, model_name)
finally:
    training_env.close()
