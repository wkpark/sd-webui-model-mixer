"""
misc utils for sd_modelmixer

"""
import importlib
import os
import sys

from modules import paths
from modules.scripts import basedir
from pathlib import Path


scriptdir = basedir()


def all_blocks(isxl=False):
    BLOCKLEN = 12 if not isxl else 9
    # return all blocks
    blocks = [ "BASE" ]
    for i in range(0, BLOCKLEN):
        blocks.append(f"IN{i:02d}")
    blocks.append("M00")
    for i in range(0, BLOCKLEN):
        blocks.append(f"OUT{i:02d}")

    blocks += [ "TIME_EMBED", "OUT" ]
    return blocks


def _all_blocks(isxl=False):
    BLOCKLEN = 12 - (0 if not isxl else 3)
    # return all blocks
    base_prefix = "cond_stage_model." if not isxl else "conditioner."
    blocks = [ base_prefix ]
    for i in range(0, BLOCKLEN):
        blocks.append(f"input_blocks.{i}.")
    blocks.append("middle_block.")
    for i in range(0, BLOCKLEN):
        blocks.append(f"output_blocks.{i}.")

    blocks += [ "time_embed.", "out." ]
    return blocks


def module_name(path):
    p = Path(path)
    if "sd_modelmixer" in p.parts:
        i = p.parts.index("sd_modelmixer")
        name = ".".join(p.parts[i:])
        if name.endswith(".py"):
            name = name[:-3]
    else:
        name = os.path.basename(path)
    return name


def load_module(path):
    """dynamic module loader"""
    # step 1: if path[0] == "scripts", search current scriptdir
    # step 2: strip "extensions" or "extensions-builtin and check "foobar" extension
    #   extensions/foobar/... => foobar/foo/bar
    #   extensions-builtin/foobar/... => foobar/foo/bar
    #
    name = None
    if type(path) in (list, tuple):
        path = [*path]
        if "scripts" == path[0]:
            name = ".".join(path)
            # make fullpath
            path = os.path.join(scriptdir, *path)
        elif "extensions" in path:
            i = path.index("extensions")
            path = path[i+1:]
            path = os.path.join(paths.extensions_dir, *path)
        elif "extensions-builtin" in path:
            i = path.index("extensions-builtin")
            path = path[i+1:]
            path = os.path.join(paths.extensions_builtin_dir, *path)
        else:
            if os.path.isdir(os.path.join(scriptdir, path[0])):
                # check path[0] is a current scripts' module name
                name = ".".join(path)
                path = os.path.join(scriptdir, *path)
            elif os.path.isdir(os.path.join(paths.extensions_dir, path[0])):
                # check path[0] is a external extension
                name = ".".join(path[1:])
                path = os.path.join(paths.extensions_dir, *path)
            elif os.path.isdir(os.path.join(paths.extensions_builtin_dir, path[0])):
                # check path[0] is a builtin extension
                name = ".".join(path[1:])
                path = os.path.join(paths.extensions_builtindir, *path)
            else:
                raise RuntimeError(f"Not a valid module name components {path}")
        if name is not None and name.endswith(".py"):
            name = name[:-3]

    if name is None:
        # given full path
        p = Path(path)
        if "extensions" in p.parts:
            i = p.parts.index("extensions")
            module_name = p.parts[i+1]
            if os.path.isdir(os.path.join(paths.extensions_dir, p.parts[i+1])):
                name = ".".join(p.parts[i+2:])
            else:
                raise RuntimeError(f"Not a valid module name {path}")
            if name.endswith(".py"):
                name = name[:-3]
        else:
            name = os.path.basename(path)

    if os.path.isdir(path):
        module_spec = importlib.util.spec_from_file_location(name, os.path.join(path, "__init__.py"))
    else:
        module_spec = importlib.util.spec_from_file_location(name, path)

    if module_spec is None:
        raise RuntimeError(f"Fail to load {path}.")
        return None

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    # register sys.modules as import_module() does
    sys.modules[name] = module
    if name.count(".") >= 2:
        parent =  name.rsplit(".", 1)[0]
        if "scripts" != parent and parent not in sys.modules:
            # register parent package as import_module() does
            load_module(os.path.dirname(path))
    return module
