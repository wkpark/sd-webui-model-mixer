from .aesthetic import judge


def score(image, prompt=""):
    aesthetic, _, _ = judge(image)
    return aesthetic["aesthetic"]
