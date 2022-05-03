import commonPPOModel
from PIL import Image
import numpy as np
import scipy.ndimage as nd
import torch
import os
from matplotlib import pyplot as plt

from lucent.optvis import render, param, transform, objectives
from lucent.misc.io import show

from stable_baselines3 import PPO

# Code derived from: https://colab.research.google.com/github/greentfrapp/lucent-notebooks/blob/master/notebooks/feature_inversion.ipynb#scrollTo=d47pkOPKvNjs

def process_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument("-a", dest="action_space", choices=[*commonPPOModel.ACTION_SPACE], default="sm")
    parser.add_argument("-i", dest="input_type", choices=[*commonPPOModel.INPUT_TYPE], default="down")
    parser.add_argument("-t", dest="timesteps", default=512, type=int)
    parser.add_argument("-r", dest="random_worlds", action="store_true")
    parser.add_argument("--src_img", dest="src_img", required=True)
    parser.add_argument("--img_path", dest="img_path", required=False, default=os.getcwd())
    parser.add_argument("--model_path", dest="model_path", required=True)


    return parser.parse_args()



def load_model(model_path, env = None):
    # Load torch model???
    # Make ability to pick model?
    assert os.path.isfile(model_path)
    print("Loading model from: {}".format(model_path))
    return PPO.load(model_path, env)

@objectives.wrap_objective()
def dot_compare(layer, batch=1, cossim_pow=0):
  def inner(T):
    dot = (T(layer)[batch] * T(layer)[0]).sum()
    mag = torch.sqrt(torch.sum(T(layer)[0]**2))
    cossim = dot/(1e-6 + mag)
    return -dot * cossim ** cossim_pow
  return inner

def get_param_f(img):
  img = torch.tensor(np.transpose(img, [2, 0, 1])).to(device)
  # Initialize parameterized input and stack with target image
  # to be accessed in the objective function
  params, image_f = param.image(img.shape[1], channels=img.shape[0])
  def stacked_param_f():
    return params, lambda: torch.stack([image_f()[0], img])

  return stacked_param_f

def feature_inversion(img, layer, img_path, n_steps=512, cossim_pow=0.0):  
  obj = objectives.Objective.sum([
    1.0 * dot_compare(layer, cossim_pow=cossim_pow),
    objectives.blur_input_each_step(),
  ])

  param_f = get_param_f(img)

  transforms = [
    transform.pad(8, mode='constant', constant_value=.5),
    transform.jitter(8),
    transform.random_scale([0.9, 0.95, 1.05, 1.1] + [1]*4),
    transform.random_rotate(list(range(-5, 5)) + [0]*5),
    transform.jitter(2),
  ]

  images = render.render_vis(model, obj, param_f, transforms=transforms, preprocess=False, thresholds=(n_steps,), show_image=False)#show_image=True, progress=True, save_image=True, image_name=img_path, verbose=True)

  # show(_[0][0])
  return images

args = process_args()

training_env, _ = commonPPOModel.create_vec_env(action_space=args.action_space, version=args.input_type, random=args.random_worlds, n=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model(args.model_path)
model = model.policy
model = model.features_extractor
model = model.cnn
print(model)
model = model.to(device).eval()

# Go through multiple images (hand-picked or derive from when performed well / poorly?)
# Do both CNN take same input?

# frame stacking? can this be run with 4 images together, or only work with 1 at a time?
# Does frame stacking get applied after the fact, e.g. CNN run 4 times and output fed into next layer?

# Run models to identify when stuck, and when reward goes up?
# Use this to extract images

# image = np.array(np.random.uniform(size=84*84), np.float32).reshape((84, 84, 1))
image = np.array(Image.open(args.src_img), np.float32)
if len(image.shape) == 2:
  image = image.reshape((image.shape[0], image.shape[1], 1))

# # Channel wide visualizations
layers = ['0', '2', '4']
images = []
for layer in layers:
  print(layer)
  images = images + feature_inversion(image, layer, os.path.join(args.img_path, f"frame_36_layer_{layer}.png"), n_steps=args.timesteps)
  print()
print([len(img) for img in images])
images = [images[0][1]] + [cnn_act_img[0] for cnn_act_img in images]
_, axs = plt.subplots(1, len(images))
for ax, image in zip(axs, images):
    ax.imshow(image)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
axs[0].set_title("Input")
plt.savefig(os.path.join(args.img_path, "frame_36.png"))

# Neuron Specific
# neurons = range(0, 32, 2)
# for neuron in neurons:
#   print(neuron)
#   param_f = get_param_f(image)
#   obj = objectives.channel("4", neuron, batch=0)
#   _ = render.render_vis(model, obj, param_f, preprocess=False, thresholds=(args.timesteps,), show_image=True, progress=True, save_image=True, image_name=os.path.join(args.img_path, f"frame_36_layer_3_neuron_{neuron}.png"))
#   print()

##############################################################
## Old Tensorflow / Lucid Code - couldn't get it working :( ##
##############################################################

# import gym_super_mario_bros
# # from stable_baselines3 import PPO
# import os
# from understanding_rl_vision import rl_clarity
# # from mpi4py import MPI

# def process_args():
#     from argparse import ArgumentParser
#     parser = ArgumentParser()

#     return parser.parse_args()


# args = process_args()

# # 'coinrun_old' / 'SuperMarioBros-v3'
# rl_clarity.train(env_kind="atari", env_id='SuperMarioBros-v3', save_dir='./feature_viz', frame_stack=4, num_envs=1)
# layer_kwargs = {'discard_first_n': 2}
# trajectories_kwargs = {'num_envs': 2, 'num_steps': 64}
# # load_kwargs = {
# #     'resample': False,
# #     'model_path': "feature_viz/rl-clarity/checkpoint.model.pb",
# #     'metadata_path': "feature_viz/rl-clarity/checkpoint.metadata.jd",
# #     'trajectories_path': "feature_viz/rl-clarity/checkpoint.trajectories.jd",
# #     'observations_path': "feature_viz/rl-clarity/checkpoint.observations.jd",
# # }
# # rl_clarity.run('feature_viz/checkpoint.jd', output_dir='feature_viz/rl_clarity_out',     trajectories_kwargs=trajectories_kwargs, layer_kwargs=layer_kwargs, attr_single_channels=False, load_kwargs=load_kwargs, observations_kwargs=trajectories_kwargs)
# # rl_clarity.run('./feature_viz/checkpoint.jd', output_dir='./feature_viz/rl_clarity_out', trajectories_kwargs=trajectories_kwargs, layer_kwargs=layer_kwargs, attr_single_channels=False)
