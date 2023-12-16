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


dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "sa_0_4_vit_l_14_linear.safetensors")

# CLIP embedding dim is 768 for CLIP ViT L 14
predictor = nn.Linear(768, 1)
# load the model you trained previously or the model available in this repo
predictor.load_state_dict(safetensors.torch.load_file(aesthetic_path))
predictor.eval()
predictor.to("cpu")

clip_model, clip_preprocess = clip.load("ViT-L/14")
clip_model.to("cpu")


def get_image_features(image, model=clip_model, preprocess=clip_preprocess, use_cuda=False):
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

    model.to("cpu")
    devices.torch_gc()

    return image_features


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def score(image, prompt="", use_cuda=True):
    image_features = get_image_features(image)
    features = torch.from_numpy(image_features).float()
    if use_cuda:
        features = features.to("cuda")
        predictor.to("cuda")
    score_origin = predictor(features).item() - 5.6

    predictor.to("cpu")
    devices.torch_gc()
    print(" > score_origin =", score_origin)

    return sigmoid(score_origin)
