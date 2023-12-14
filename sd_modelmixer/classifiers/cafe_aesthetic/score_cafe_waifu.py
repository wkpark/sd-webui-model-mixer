from .aesthetic import judge


def score(image, prompt=""):
    _, _, waifu = judge(image)
    return waifu["waifu"]
