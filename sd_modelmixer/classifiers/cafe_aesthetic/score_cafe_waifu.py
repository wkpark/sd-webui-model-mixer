from .aesthetic import judge_waifu


def score(image, prompt="", use_cuda=False):
    waifu = judge_waifu(image, use_cuda)
    return waifu["waifu"]
