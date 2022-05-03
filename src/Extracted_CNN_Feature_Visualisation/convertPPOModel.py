import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import onnx
from onnx_tf.backend import prepare

# from: https://stable-baselines3.readthedocs.io/en/master/guide/export.html
# class OnnxablePolicy(torch.nn.Module):
#   def __init__(self, extractor, action_net, value_net):
#       super(OnnxablePolicy, self).__init__()
#       self.extractor = extractor
#       self.action_net = action_net
#       self.value_net = value_net

#   def forward(self, observation):
#       # NOTE: You may have to process (normalize) observation in the correct
#       #       way before using this. See `common.preprocessing.preprocess_obs`
#       action_hidden, value_hidden = self.extractor(observation)
#       return self.action_net(action_hidden), self.value_net(value_hidden)

def process_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument("-s", dest="pytorch_model_path")
    parser.add_argument("-o", dest="onnx_model_path")
    parser.add_argument("-d", dest="tensorflow_model_path")

    return parser.parse_args()

def load_pytorch_model(pytorch_model_path):
    from stable_baselines3 import PPO

    assert os.path.isfile(pytorch_model_path)
    print("Loading model from: {}".format(pytorch_model_path))
    return PPO.load(pytorch_model_path)
    
def convert_to_onnx(pytorch_model, onnx_model_path):
    onnxable_model = pytorch_model.policy
    dummy_input = Variable(torch.randn(1, 84, 84, 1))
    torch.onnx.export(onnxable_model, dummy_input, onnx_model_path, opset_version=9)

def load_onnx_model(onnx_model_path):
    import onnxruntime as ort
    import numpy as np
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    onnx_model.

    observation = np.zeros((1, (1, 84, 84, 1))).astype(np.float32)
    ort_sess = ort.InferenceSession(onnx_model_path)
    action, value = ort_sess.run(None, {'input.1': observation})

    return onnx_model

def convert_to_tensorflow(onnx_model):
    # Import the ONNX model to Tensorflow
    return prepare(onnx_model)

def save_tensorflow_model(tensorflow_model, tensorflow_model_path):
    tensorflow_model.export_graph(f'{tensorflow_model_path}.pb')


args = process_args()

p_mod = load_pytorch_model(args.pytorch_model_path)
print(p_mod.policy.get_shape())
convert_to_onnx(p_mod, args.onnx_model_path)
o_mod = load_onnx_model(args.onnx_model_path)
t_mod = convert_to_tensorflow(o_mod)
save_tensorflow_model(t_mod, args.tensorflow_model_path)

