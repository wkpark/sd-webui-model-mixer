import math
import os
import torch
import safetensors
from modules import devices

from .laion import image_embeddings_direct_laion, MLP


dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "laion-sac-logos-ava-v2.safetensors")
aes_model = MLP(768)
aes_model.load_state_dict(safetensors.torch.load_file(aesthetic_path))
aes_model.eval()
aes_model.to("cpu")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def score(image, prompt="", use_cuda=True):
    image_embeds = image_embeddings_direct_laion(image, use_cuda=use_cuda)
    embeds = torch.from_numpy(image_embeds).float()
    if use_cuda:
        embeds = embeds.to("cuda")
        aes_model.to("cuda")

    score_origin = aes_model(embeds).item() - 5.6
    aes_model.to("cpu")
    print(" > origin score =", score_origin)

    devices.torch_gc()

    return sigmoid(score_origin)
