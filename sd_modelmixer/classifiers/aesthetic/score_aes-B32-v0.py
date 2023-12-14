import os
import torch
import safetensors
from modules import devices
from transformers import CLIPModel, CLIPProcessor
from .aesthetic import image_embeddings_direct, Classifier


dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "aes-B32-v0.safetensors")
clip_name = 'openai/clip-vit-base-patch32'
clipprocessor = CLIPProcessor.from_pretrained(clip_name)
clipmodel = CLIPModel.from_pretrained(clip_name)

clipmodel.eval()

aes_model = Classifier(512, 256, 1)
aes_model.load_state_dict(safetensors.torch.load_file(aesthetic_path))


def score(image, prompt="", use_cuda=True):
    if use_cuda:
        clipmodel.to('cuda')
    image_embeds = image_embeddings_direct(image, clipmodel, clipprocessor, use_cuda)
    embeds = torch.from_numpy(image_embeds)
    if use_cuda:
        embeds = embeds.to('cuda')
        aes_model.to('cuda')

    prediction = aes_model(embeds)

    aes_model.to('cpu')
    clipmodel.to('cpu')

    devices.torch_gc()

    return prediction.item()
