import commonPPOModel
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines3 import PPO
import numpy as np
from PIL import Image
import os

def process_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument("-a", dest="action_space", choices=[*commonPPOModel.ACTION_SPACE], default="sm")
    parser.add_argument("-i", dest="input_type", choices=[*commonPPOModel.INPUT_TYPE], default="down")
    parser.add_argument("-t", dest="timesteps", default=int(1e3), type=int)
    parser.add_argument("-r", dest="random_worlds", action="store_true")
    parser.add_argument("--video_dir", dest="video_dir", required=True)
    parser.add_argument("--model_path", dest="model_path", required=True)

    return parser.parse_args()

def load_model(model_path, env):
    # Make ability to pick model?
    assert os.path.isfile(model_path)
    print("Loading model from: {}".format(model_path))
    return PPO.load(model_path, env)

def setup_recorder(env, limit, video_dir, checkpoint_name):
    # save to e drive (/mnt/e/smb_agent_training)
    print("Setting up recorder")
    assert os.path.isdir(video_dir)
    from stable_baselines3.common.vec_env import VecVideoRecorder
    return VecVideoRecorder(env, video_folder=video_dir, record_video_trigger=lambda step: step % int(1e3), video_length=limit, name_prefix=checkpoint_name)

def run_model(model, env, obs_path, delay = 0):
    import time
    obs = env.reset()
    attempts = 0
    frame = 0
    while attempts < 5:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        image = obs
        if len(image.shape) == 4:
            image = np.concatenate(image, axis=1)
        if image.shape[2] == 1:
            image = image.reshape((image.shape[0], image.shape[1]))
        Image.fromarray(image).save(os.path.join(obs_path, f"attempt_{attempts}_frame_{frame}.png"))
        frame += 1
        # save OBS as shape
        env.render()
        if done:
            obs = env.reset()
            attempts = attempts - 1
        if delay > 0:
            time.sleep(delay)
    env.close()

args = process_args()
obs_path = os.path.join(args.video_dir, "raw_obs_frames")

training_env, env_config = commonPPOModel.create_vec_env(action_space=args.action_space, version=args.input_type, random=args.random_worlds, n=1)

training_env = setup_recorder(training_env, args.timesteps, args.video_dir, os.path.splitext(os.path.basename(args.model_path))[0])

model = load_model(args.model_path, training_env)
print(model.policy)

try:
    run_model(model, training_env, obs_path, 0.01)
finally:
    training_env.close()
