import math
import ImageReward as reward

from modules import devices
from modules import sd_disable_initialization


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


model = None


def score(image, prompt="", use_cuda=True):
    global model

    if model == None:
        with sd_disable_initialization.DisableInitialization(disable_clip=False):
            model = reward.load("ImageReward-v1.0", device="cpu")
    if use_cuda:
        model.to("cuda")
        model.device = "cuda"
    else:
        model.to("cpu")
        model.device = "cpu"

    score_origin = model.score(prompt, image)
    score = sigmoid(score_origin)

    if use_cuda:
        model.to("cpu")
    devices.torch_gc()
    return score
