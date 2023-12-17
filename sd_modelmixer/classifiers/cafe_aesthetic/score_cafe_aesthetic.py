from .aesthetic import judge_aesthetic


def score(image, prompt="", use_cuda=False):
    aesthetic= judge_aesthetic(image, use_cuda)
    return aesthetic["aesthetic"]
