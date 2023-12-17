# https://github.com/christophschuhmann/improved-aesthetic-predictor
# https://github.com/LAION-AI/aesthetic-predictor
# https://github.com/grexzen/SD-Chad
# from https://github.com/Xerxemi/auto-MBW-rt/tree/master/scripts/classifiers/laion_v1
#
# use safetensors, minimize VRAM usage by wkpark

import clip
import math
import numpy as np
import os
import safetensors
import torch
import torch.nn as nn

from modules import devices
from modules import sd_disable_initialization
from ..laion.laion import model as clip_model
from ..laion.laion import preprocess as clip_preprocess


dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "sa_0_4_vit_l_14_linear.safetensors")

# CLIP embedding dim is 768 for CLIP ViT L 14
predictor = nn.Linear(768, 1)
# load the model you trained previously or the model available in this repo
predictor.load_state_dict(safetensors.torch.load_file(aesthetic_path))
predictor.eval()
predictor.to("cpu")


def get_image_features(image, model=clip_model, preprocess=clip_preprocess, use_cuda=True):
    if use_cuda:
        model.to("cuda")
    else:
        model.float()
        model.to("cpu")

    image = preprocess(image).unsqueeze(0).float()
    if use_cuda:
        image = image.to("cuda")

    with torch.no_grad():
        image_features = model.encode_image(image)
        # l2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()

    if use_cuda:
        model.to("cpu")
    devices.torch_gc()

    return image_features


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def score(image, prompt="", use_cuda=True):
    image_features = get_image_features(image, use_cuda=use_cuda)
    features = torch.from_numpy(image_features).float()
    if use_cuda:
        features = features.to("cuda")
        predictor.to("cuda")
    else:
        predictor.to("cpu")
    score_origin = predictor(features).item() - 5.6

    if use_cuda:
        predictor.to("cpu")
    devices.torch_gc()
    print(" > score_origin =", score_origin)

    return sigmoid(score_origin)
