import os
import torch
import safetensors
from modules import devices

from .laion import image_embeddings_direct_laion, MLP


dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "laion-sac-logos-ava-v2.safetensors")
aes_model = MLP(768).eval()
aes_model.load_state_dict(safetensors.torch.load_file(aesthetic_path))


def score(image, prompt="", use_cuda=True):
    image_embeds = image_embeddings_direct_laion(image)
    embeds = torch.from_numpy(image_embeds).float()
    if use_cuda:
        embeds = embeds.to("cuda")
        aes_model.to("cuda")

    prediction = aes_model(embeds)
    aes_model.to("cpu")

    devices.torch_gc()

    return prediction.item()
