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



def all_blocks(sdversion):
    """simple BLOCKIDS"""

    if type(sdversion) is bool:
        # for old behavior called by all_blocks(isxl)
        sdversion = "XL" if sdversion else "v1"

    blocks = [ 'BASE' ]
    if sdversion in ["v1", "v2", "XL"]:
        BLOCKLEN = 12 - (0 if sdversion != "XL" else 3)
        # return all blocks
        for i in range(0, BLOCKLEN):
            blocks.append(f"IN{i:02d}")
        blocks.append("M00")
        for i in range(0, BLOCKLEN):
            blocks.append(f"OUT{i:02d}")

    elif sdversion == "v3":
        for i in range(0, 24):
            blocks.append(f"IN{i:02d}")

    elif sdversion == "FLUX":
        for i in range(0, 19):
            blocks.append(f"DOUBLE{i:02d}")
        for i in range(0, 38):
            blocks.append(f"SINGLE{i:02d}")

    return blocks


def _all_blocks(sdversion):
    """1:1 mapping BLOCKIDS to tensor keys"""

    if type(sdversion) is bool:
        # for old behavior called by all_blocks(isxl)
        sdversion = "XL" if sdversion else "v1"

    if sdversion is True:
        # for old behavior called by _all_blocks(isxl)
        sdversion = "XL"

    if sdversion in ["v1", "v2", "XL"]:
        BLOCKLEN = 12 - (0 if sdversion != "XL" else 3)
        # return all blocks
        base_prefix = "cond_stage_model." if sdversion != "XL" else "conditioner."
        blocks = [ base_prefix ]
        for i in range(0, BLOCKLEN):
            blocks.append(f"input_blocks.{i}.")
        blocks.append("middle_block.")
        for i in range(0, BLOCKLEN):
            blocks.append(f"output_blocks.{i}.")

        blocks += [ "time_embed.", "out." ]
        if sdversion == "XL":
            blocks += [ "label_emb." ]

    elif sdversion == "v3":
        #blocks = [ "text_encoders.clip_l.", "text_encoders.clip_g.", "text_encoders.t5xxl." ]
        blocks = [ "text_encoders." ]
        for i in range(0, 24):
            blocks.append(f"joint_blocks.{i}.")

        blocks += [ "x_embedder.", "t_embedder.", "y_embedder.", "context_embedder.", "pos_embed", "final_layer." ]

    elif sdversion == "FLUX":
        #blocks = [ "text_encoders.clip_l.", "text_encoders.t5xxl." ]
        blocks = [ "text_encoders." ]
        for i in range(0, 19):
            blocks.append(f"double_blocks.{i}.")
        for i in range(0, 38):
            blocks.append(f"single_blocks.{i}.")

        blocks += [ "img_in.", "time_in.", "vector_in.", "guidance_in.", "txt_in.", "final_layer." ]

    return blocks


def normalize_blocks(blocks, sdv):
    """Normalize Merge Block Weights"""

    if type(sdv) is bool:
        # for old behavior
        sdv = "XL" if sdv else "v1"

    # no mbws blocks selected or have 'ALL' alias
    if len(blocks) == 0 or 'ALL' in blocks:
        # select all blocks
        blocks = [ 'BASE', 'INP*', 'MID', 'OUT*' ]

    # fix alias
    if 'MID' in blocks:
        i = blocks.index('MID')
        blocks[i] = 'M00'

    if sdv in ["v1", "v2", "XL"]:
        isxl = sdv == "XL"
        BLOCKLEN = 12 - (0 if not isxl else 3)

        # expand some aliases
        if 'INP*' in blocks:
            for i in range(0, BLOCKLEN):
                name = f"IN{i:02d}"
                if name not in blocks:
                    blocks.append(name)
        if 'OUT*' in blocks:
            for i in range(0, BLOCKLEN):
                name = f"OUT{i:02d}"
                if name not in blocks:
                    blocks.append(name)

    elif sdv == "FLUX":
        # expand some aliases
        if 'INP*' in blocks or 'DOUBLE*' in blocks:
            for i in range(0, 19):
                name = f"DOUBLE{i:02d}"
                if name not in blocks:
                    blocks.append(name)
        if 'OUT*' in blocks or 'SINGLE*' in blocks:
            for i in range(0, 38):
                name = f"SINGLE{i:02d}"
                if name not in blocks:
                    blocks.append(name)

    elif sdv == "v3":
        # expand some aliases
        if 'INP*' in blocks:
            for i in range(0, 24):
                name = f"IN{i:02d}"
                if name not in blocks:
                    blocks.append(name)

    blocks = list(set(blocks))

    # filter valid blocks
    BLOCKIDS = all_blocks(sdv)
    MAXLEN = len(BLOCKIDS)
    selected = [False]*MAXLEN

    normalized = []
    for i, name in enumerate(BLOCKIDS):
        if name in blocks:
            selected[i] = True
            normalized.append(name)

    return normalized, selected


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
