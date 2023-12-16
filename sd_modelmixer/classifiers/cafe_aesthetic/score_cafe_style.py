from .aesthetic import judge_style


def score(image, prompt=""):
    style = judge_style(image):
    return style["real_life"]
