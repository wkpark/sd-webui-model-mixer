from .aesthetic import judge


def score(image, prompt=""):
    _, style, _ = judge(image)
    return style["real_life"]
