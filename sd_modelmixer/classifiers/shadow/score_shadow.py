# based on https://github.com/WhiteWipe/sd-webui-bayesian-merger/blob/main/sd_webui_bayesian_merger/models/ShadowScore.py
import os
import safetensors
import torch

from huggingface_hub import hf_hub_download
from modules import devices
from PIL import Image
from transformers import pipeline, AutoConfig, AutoProcessor, ViTForImageClassification


pathname = hf_hub_download(repo_id="shadowlilac/aesthetic-shadow", filename="model.safetensors")

statedict = safetensors.torch.load_file(pathname)

config = AutoConfig.from_pretrained(pretrained_model_name_or_path="shadowlilac/aesthetic-shadow")
model = ViTForImageClassification.from_pretrained(pretrained_model_name_or_path=None, state_dict=statedict, config=config)
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path="shadowlilac/aesthetic-shadow")


def score(image, prompt="", use_cuda=True):
    if use_cuda:
        model.to("cuda")
    else:
        model.float()
        model.to("cpu")

    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, str):
        if os.path.isfile(image):
            pil_image = Image.open(image)
    else:
        pil_image = image

    pipe = pipeline("image-classification", model=model, image_processor=processor, device="cpu" if not use_cuda else "cuda:0")

    score = pipe(images=[pil_image])[0]
    score = [p for p in score if p['label'] == 'hq'][0]['score']

    if use_cuda:
        model.to("cpu")
    print(" > score =", score)

    devices.torch_gc()

    return score
