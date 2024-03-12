# from https://huggingface.co/yuvalkirstain/PickScore_v1
import math
import os
import safetensors
import torch

from modules import devices
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor, AutoConfig


pathname = hf_hub_download(repo_id="yuvalkirstain/PickScore_v1", filename="model.safetensors")

statedict = safetensors.torch.load_file(pathname)
config = AutoConfig.from_pretrained(pretrained_model_name_or_path="yuvalkirstain/PickScore_v1")
model = AutoModel.from_pretrained(pretrained_model_name_or_path=None, state_dict=statedict, config=config)
preprocessor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

model.to("cpu")
model.eval()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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

    image_inputs = preprocessor(
        images=pil_image,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to("cpu" if not use_cuda else "cuda")

    text_inputs = preprocessor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to("cpu" if not use_cuda else "cuda")


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = torch.sum(torch.mul(text_embs, image_embs), dim=1, keepdim=True)

    score = scores.cpu().tolist()[0][0]
    print(" > origin score =", score)
    score += 1
    score *= 5
    score = sigmoid(score)

    if use_cuda:
        model.to("cpu")
    print(" > score =", score)

    devices.torch_gc()

    return score
