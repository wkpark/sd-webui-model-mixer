from .aesthetic import judge_waifu


def score(image, prompt=""):
    waifu = judge_waifu(image)
    return waifu["waifu"]
