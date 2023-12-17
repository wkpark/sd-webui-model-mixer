from .aesthetic import judge_style


def score(image, prompt="", use_cuda=False):
    style = judge_style(image, use_cuda)
    return style["real_life"]
