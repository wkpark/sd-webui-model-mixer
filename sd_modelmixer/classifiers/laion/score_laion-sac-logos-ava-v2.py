import math
import os
import torch
import safetensors
from modules import devices
from modules import sd_disable_initialization

from .laion import image_embeddings_direct_laion, MLP


dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "laion-sac-logos-ava-v2.safetensors")
state_dict = safetensors.torch.load_file(aesthetic_path)
aes_model = MLP(768)
with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device="cpu"):
    aes_model.load_state_dict(state_dict)
aes_model.eval()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def score(image, prompt="", use_cuda=True):
    image_embeds = image_embeddings_direct_laion(image)
    #image_embeds = image_embeddings_direct_laion(image, use_cuda=use_cuda)
    embeds = torch.from_numpy(image_embeds).float()
    if use_cuda:
        embeds = embeds.to("cuda")
        aes_model.to("cuda")
    else:
        aes_model.to("cpu")

    score_origin = aes_model(embeds).item() - 5.6
    if use_cuda:
        aes_model.to("cpu")
    print(" > origin score =", score_origin)

    devices.torch_gc()

    return sigmoid(score_origin)
