from .aesthetic import judge_aesthetic


def score(image, prompt=""):
    aesthetic= judge_aesthetic(image)
    return aesthetic["aesthetic"]
