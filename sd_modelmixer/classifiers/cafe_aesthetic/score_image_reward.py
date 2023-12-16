import math
import ImageReward as reward

from modules import devices


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


model = None


def score(image, prompt="", use_cuda=True):
    global model

    if model == None:
        model = reward.load("ImageReward-v1.0")
    if use_cuda:
        model.to("cuda")
        model.device = "cuda"
    else:
        model.to("cpu")
        model.device = "cpu"

    score_origin = model.score(prompt, image)
    score = sigmoid(score_origin)

    model.to("cpu")
    devices.torch_gc()
    return score
