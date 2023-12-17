import os
from transformers import pipeline
from modules import devices
from modules import sd_disable_initialization

aesthetics = {}  # name: pipeline


def model_check(name):
    if name not in aesthetics:
        with sd_disable_initialization.DisableInitialization(disable_clip=False):
            if name == "aesthetic":
                aesthetics["aesthetic"] = pipeline(
                    "image-classification", model="cafeai/cafe_aesthetic"
                )
            elif name == "style":
                aesthetics["style"] = pipeline(
                    "image-classification", model="cafeai/cafe_style"
                )
            elif name == "waifu":
                aesthetics["waifu"] = pipeline(
                    "image-classification", model="cafeai/cafe_waifu"
                )


def judge_aesthetic(image, use_cuda=False):
    model_check("aesthetic")
    if use_cuda:
        aesthetics["aesthetic"].model.to("cuda")
    else:
        aesthetics["aesthetic"].model.to("cpu")
    data = aesthetics["aesthetic"](image, top_k=2)

    result = {}
    for d in data:
        result[d["label"]] = d["score"]
    if use_cuda:
        aesthetics["aesthetic"].to("cpu")
    devices.torch_gc()
    return result


def judge_style(image, use_cuda=False):
    model_check("style")
    if use_cuda:
        aesthetics["style"].model.to("cuda")
    else:
        aesthetics["style"].model.to("cpu")
    data = aesthetics["style"](image, top_k=5)
    result = {}
    for d in data:
        result[d["label"]] = d["score"]
    if use_cuda:
        aesthetics["style"].to("cpu")
    devices.torch_gc()
    return result


def judge_waifu(image, use_cuda=False):
    model_check("waifu")
    if use_cuda:
        aesthetics["waifu"].model.to("cuda")
    else:
        aesthetics["waifu"].model.to("cpu")
    data = aesthetics["waifu"](image, top_k=5)
    result = {}
    for d in data:
        result[d["label"]] = d["score"]
    if use_cuda:
        aesthetics["waifu"].to("cpu")
    devices.torch_gc()
    return result


def judge(image, use_cuda=False):
    if image is None:
        return None, None, None
    aesthetic = judge_aesthetic(image, use_cuda)
    style = judge_style(image, use_cuda)
    waifu = judge_waifu(image, use_cuda)
    return aesthetic, style, waifu
