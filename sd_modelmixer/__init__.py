import os
import sys

from functools import lru_cache
from .__version__ import __version__
from .utils import load_module, module_name

@lru_cache(maxsize=None)
def get_classifiers():
    plugins = {}
    d = os.path.abspath(os.path.dirname(os.path.join(os.path.dirname(__file__), "classifiers")))
    path = os.path.join(d, "classifiers")
    for _, dirs, _ in os.walk(path):
        for dir in dirs:
            dir_path = os.path.join(path, dir)
            # exclude __pycache__ dir
            if dir_path.endswith('__pycache__'):
                continue
            for module in os.listdir(dir_path):
                if module.startswith('score_') and module.endswith('.py'):
                    module_name = os.path.splitext(module)[0]
                    module_path = os.path.join(dir_path, module)
                    plugins[module_name] = module_path

    return plugins


def classifier_score(module_path, image, prompt):
    name = module_name(module_path)
    if name in sys.modules:
        module = sys.modules[name]
    else:
        module = load_module(module_path)
    return module.score(image, prompt)


__all__ = [
    "__version__",
    "load_module",
    "module_name",
]
