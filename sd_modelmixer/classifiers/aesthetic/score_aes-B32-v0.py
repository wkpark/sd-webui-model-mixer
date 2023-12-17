import os
import torch
import safetensors
from modules import devices
from modules import sd_disable_initialization
from transformers import CLIPModel, CLIPProcessor
from .aesthetic import image_embeddings_direct, Classifier

with sd_disable_initialization.DisableInitialization(disable_clip=False):
    clip_name = 'openai/clip-vit-base-patch32'
    clipprocessor = CLIPProcessor.from_pretrained(clip_name)
    clipmodel = CLIPModel.from_pretrained(clip_name)
clipmodel.eval()
clipmodel.to("cpu")

dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "aes-B32-v0.safetensors")
state_dict = safetensors.torch.load_file(aesthetic_path)
aes_model = Classifier(512, 256, 1)
with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device="cpu"):
    aes_model.load_state_dict(state_dict)


def score(image, prompt="", use_cuda=True):
    if use_cuda:
        clipmodel.to('cuda')
    else:
        clipmodel.to('cpu')
    image_embeds = image_embeddings_direct(image, clipmodel, clipprocessor, use_cuda)
    embeds = torch.from_numpy(image_embeds)
    if use_cuda:
        embeds = embeds.to('cuda')
        aes_model.to('cuda')
    else:
        aes_model.to('cpu')

    prediction = aes_model(embeds)

    aes_model.to('cpu')
    clipmodel.to('cpu')

    devices.torch_gc()

    return prediction.item()
