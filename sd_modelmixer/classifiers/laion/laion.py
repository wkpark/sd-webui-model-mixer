#
# from https://github.com/Xerxemi/sdweb-auto-MBW
#
# reduce VRAM usage by wkpark
#
import torch
import torch.nn as nn
import numpy as np
import clip

from modules import devices

model, preprocess = clip.load("ViT-L/14")
model.to("cpu")

def image_embeddings_direct(image, model, processor, use_cuda=True):
    if use_cuda:
        model.to("cuda")
    inputs = processor(images=image, return_tensors='pt')['pixel_values'].float()
    if use_cuda:
        inputs = inputs.to('cuda')
    result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()

    model.to("cpu")
    devices.torch_gc()

    return (result / np.linalg.norm(result)).squeeze(axis=0)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def image_embeddings_direct_laion(pil_image, use_cuda=True):
    if use_cuda:
        model.to("cuda")
    else:
        model.float()
        model.to("cpu")

    image = preprocess(pil_image).unsqueeze(0).float()
    if use_cuda:
        image = image.to("cuda")

    with torch.no_grad():
        image_features = model.encode_image(image)
    im_emb_arr = normalized(image_features.cpu().detach().numpy())

    model.to("cpu")
    devices.torch_gc()
    return im_emb_arr


class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)
