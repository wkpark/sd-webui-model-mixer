#
# Checkpoint Model Mixer extension for sd-webui
#
# Copyright 2023 wkpark at gmail.com
# License: AGPL
#
import collections
import gc
import io
import os
import sys
import gradio as gr
import hashlib
import importlib
import json
import pickle
from pathlib import Path
import re
import shutil
import time
import tqdm
from tqdm import tqdm
import torch
import traceback
from functools import partial, lru_cache
from safetensors.torch import save_file
from typing import Dict, Union
from zipfile import ZipFile, is_zipfile
import numpy as np

from PIL import Image

from copy import copy, deepcopy
from modules import script_callbacks, sd_hijack, sd_models, sd_vae, shared, ui_settings, ui_common
from modules import scripts, cache, devices, lowvram, deepbooru, images, paths
from modules import sd_unet
from modules.generation_parameters_copypaste import parse_generation_parameters
from modules.sd_models import model_hash, model_path, checkpoints_loaded
from modules.scripts import basedir
from modules.timer import Timer
from modules.ui import create_refresh_button
from ldm.modules.attention import CrossAttention

from scripts.vxa import generate_vxa, default_hidden_layer_name, get_layer_names
from scripts.vxa import tokenize

dump_cache = cache.dump_cache
cache = cache.cache

scriptdir = basedir()

save_symbol = "\U0001f4be"  # 💾
delete_symbol = "\U0001f5d1\ufe0f"  # 🗑️
refresh_symbol = "\U0001f504"  # 🔄

BLOCKID  =["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKIDXL=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08",                     "M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08",                       ]
ISXLBLOCK=[  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, True,   True,   True,   True,   True,   True,   True,   True,   True,   True,  False,  False,  False]

elemental_blocks = None

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

def gr_enable(interactive=True):
    return {"interactive": interactive, "__type__": "update"}

def gr_open(open=True):
    return {"open": open, "__type__": "update"}

def slider2text(isxl, *slider):
    if isxl:
        selected = []
        for i,v in enumerate(slider):
            if ISXLBLOCK[i]:
                selected.append(slider[i])
    else:
        selected = slider
    return gr.update(value = ",".join([str(x) for x in selected]))

parsed_mbwpresets = {}
def _load_mbwpresets():
    raw = None
    userfilepath = os.path.join(scriptdir, "data","mbwpresets.tsv")
    if os.path.isfile(userfilepath):
        try:
            with open(userfilepath) as f:
                raw = f.read()
                filepath = userfilepath
        except OSError as e:
            print(e)
            pass
    else:
        if not os.path.exists(os.path.join(scriptdir, "data")):
            os.makedirs(os.path.join(scriptdir, "data"))

        filepath = os.path.join(scriptdir, "scripts", "mbwpresets.tsv.in")
        try:
            with open(filepath) as f:
                raw = f.read()
                shutil.copyfile(filepath, userfilepath)
        except OSError as e:
            print(e)
            pass

    return raw

def _mbwpresets(raw=None):
    if raw is None:
        raw = _load_mbwpresets()
        if raw is None:
            return {}

    lines = raw.splitlines()
    presets = {}
    for l in lines[1:]:
        w = None
        if ":" in l:
            k, w = l.split(":", 1)
        elif "\t" in l:
            k, w = l.split("\t", 1)
        elif "," in l:
            k, w = l.split(",", 1)
        if w is not None:
            k = k.strip()
            w = [w for w in w.split(",")]
            if len(w) == 26:
                presets[k] = w
            elif len(w) == 25: # weights without BASE element
                presets[k] = [0.0] + w
            elif len(w) == 20: # SDXL weights
                presets[f"{k}.XL"] = w
            elif len(w) == 19: # SDXL weights without BASE element
                presets[f"{k}.XL"] = [0.0] + w

    return presets

def mbwpresets(reload=False):
    global parsed_mbwpresets
    if reload or len(parsed_mbwpresets) == 0:
        parsed_mbwpresets = _mbwpresets()

    return parsed_mbwpresets

def find_preset_by_name(preset, presets=None, reload=False):
    if presets is None:
        presets = mbwpresets(reload=reload)

    if preset in presets:
        return presets[preset]

    return None


def get_selected_blocks(mbw_blocks, isxl=False):
    MAXLEN = 26 - (0 if not isxl else 6)
    BLOCKLEN = 12 - (0 if not isxl else 3)
    BLOCKOFFSET = 13 if not isxl else 10
    selected = [False]*MAXLEN
    BLOCKIDS = BLOCKID if not isxl else BLOCKIDXL

    # no mbws blocks selected or have 'ALL' alias
    if 'ALL' in mbw_blocks:
        # select all blocks
        mbw_blocks += [ 'BASE', 'INP*', 'MID', 'OUT*' ]

    # fix alias
    if 'MID' in mbw_blocks:
        i = mbw_blocks.index('MID')
        mbw_blocks[i] = 'M00'

    # expand some aliases
    if 'INP*' in mbw_blocks:
        for i in range(0, BLOCKLEN):
            name = f"IN{i:02d}"
            if name not in mbw_blocks:
                mbw_blocks.append(name)
    if 'OUT*' in mbw_blocks:
        for i in range(0, BLOCKLEN):
            name = f"OUT{i:02d}"
            if name not in mbw_blocks:
                mbw_blocks.append(name)

    for i, name in enumerate(BLOCKIDS):
        if name in mbw_blocks:
            if name[0:2] == 'IN':
                num = int(name[2:])
                selected[num + 1] = True
            elif name[0:3] == 'OUT':
                num = int(name[3:])
                selected[num + BLOCKOFFSET + 1] = True
            elif name == 'M00':
                selected[BLOCKOFFSET] = True
            elif name == 'BASE':
                selected[0] = True

    all_blocks = _all_blocks(isxl)
    selected_blocks = []
    for i, v in enumerate(selected):
        if v:
            selected_blocks.append(all_blocks[i])
    return selected_blocks


def calc_mbws(mbw, mbw_blocks, isxl=False):
    weights = [t.strip() for t in mbw.split(",")]
    expect = 0
    MAXLEN = 26 - (0 if not isxl else 6)
    BLOCKLEN = 12 - (0 if not isxl else 3)
    BLOCKOFFSET = 13 if not isxl else 10
    selected = [False]*MAXLEN
    compact_blocks = []
    BLOCKIDS = BLOCKID if not isxl else BLOCKIDXL

    # no mbws blocks selected or have 'ALL' alias
    if len(mbw_blocks) == 0 or 'ALL' in mbw_blocks:
        # select all blocks
        mbw_blocks = [ 'BASE', 'INP*', 'MID', 'OUT*' ]

    # fix alias
    if 'MID' in mbw_blocks:
        i = mbw_blocks.index('MID')
        mbw_blocks[i] = 'M00'

    # expand some aliases
    if 'INP*' in mbw_blocks:
        for i in range(0, BLOCKLEN):
            name = f"IN{i:02d}"
            if name not in mbw_blocks:
                mbw_blocks.append(name)
    if 'OUT*' in mbw_blocks:
        for i in range(0, BLOCKLEN):
            name = f"OUT{i:02d}"
            if name not in mbw_blocks:
                mbw_blocks.append(name)

    for i, name in enumerate(BLOCKIDS):
        if name in mbw_blocks:
            if name[0:2] == 'IN':
                expect += 1
                num = int(name[2:])
                selected[num + 1] = True
                compact_blocks.append(f'inp.{num}.')
            elif name[0:3] == 'OUT':
                expect += 1
                num = int(name[3:])
                selected[num + BLOCKOFFSET + 1] = True
                compact_blocks.append(f'out.{num}.')
            elif name == 'M00':
                expect += 1
                selected[BLOCKOFFSET] = True
                compact_blocks.append('mid.1.')
            elif name == 'BASE':
                expect +=1
                selected[0] = True
                compact_blocks.append('base')

    if len(weights) > MAXLEN:
        weights = weights[:MAXLEN]
    elif len(weights) > expect:
        for i in range(len(weights), MAXLEN):
            weights.append(weights[len(weights)-1]) # fill up last weight
    elif len(weights) < expect:
        for i in range(len(weights), expect):
            weights.append(weights[len(weights)-1]) # fill up last weight

    if len(weights) == MAXLEN:
        # full weights given
        mbws = [0.0]*len(weights)
        compact_mbws = []
        for i,f in enumerate(weights):
            try:
                f = float(f)
                weights[i] = f
            except:
                pass # ignore invalid entries
            if selected[i]:
                mbws[i] = weights[i]
                compact_mbws.append(mbws[i])
    else:
        # short weights given
        compact_mbws = [0.0]*len(weights)
        mbws = [0.0]*MAXLEN
        for i,f in enumerate(weights):
            try:
                f = float(f)
                compact_mbws[i] = f
            except:
                pass # ignore invalid entries

            block = compact_blocks[i]
            if 'base' == block:
                off = 0
                num = 0
            else:
                block, num, _ = compact_blocks[i].split(".")
                num = int(num)
                if 'inp' == block:
                    off = 1
                elif 'mid' == block:
                    off = BLOCKOFFSET
                    num = 0
                elif 'out' == block:
                    off = BLOCKOFFSET + 1

            mbws[off + num] = compact_mbws[i]

    return mbws, compact_mbws, selected

def get_mbws(mbw, mbw_blocks, isxl=False):
    mbws, compact_mbws, selected = calc_mbws(mbw, mbw_blocks, isxl=isxl)
    if isxl:
        j = 0
        ret = []
        for i, v in enumerate(ISXLBLOCK):
            if v:
                ret.append(gr.update(value = mbws[j]))
                j += 1
            else:
                ret.append(gr.update())
        return ret + [gr.update(open=True)]

    return [gr.update(value = v) for v in mbws] + [gr.update(open=True)]

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

def print_blocks(blocks):
    str = []
    for i,x in enumerate(blocks):
        if "input_blocks." in x:
            n = int(x[13:len(x)-1])
            block = f"IN{n:02d}"
            str.append(block)
        elif "middle_block." in x:
            block = "M00"
            str.append(block)
        elif "output_blocks." in x:
            n = int(x[14:len(x)-1])
            block = f"OUT{n:02d}"
            str.append(block)
        elif "cond_stage_model" in x or "conditioner." in x:
            block = f"BASE"
            str.append(block)
        elif "time_embed." in x:
            block = "TIME_EMBED"
            str.append(block)
        elif "out." in x:
            block = "OUT"
            str.append(block)
    return ','.join(str)

def _selected_blocks_and_weights(mbw, isxl=False):
    if type(mbw) is str:
        weights = [t.strip() for t in mbw.split(",")]
    else:
        weights = mbw
    # get all blocks
    all_blocks = _all_blocks(isxl)

    sel_blocks = []
    sel_mbws = []
    for i, w in enumerate(weights):
        v = float(w)
        if v != 0.0:
            sel_blocks.append(all_blocks[i])
            sel_mbws.append(v)
    return sel_blocks, sel_mbws

def _weight_index(key, isxl=False):
    num = -1
    offset = [ 0, 1, 13, 14 ] if not isxl else [ 0, 1, 10, 11 ]
    base_prefix = "cond_stage_model." if not isxl else "conditioner."
    for k, s in enumerate([ base_prefix, "input_blocks.", "middle_block.", "output_blocks." ]):
        if s in key:
            if k == 0: return 0 # base
            if k == 2: return offset[2] # middle_block

            i = key.find(s)
            j = key.find(".", i+len(s))
            num = int(key[i+len(s):j]) + offset[k]
    return num

def prune_model(model, isxl=False):
    keys = list(model.keys())
    base_prefix = "conditioner." if isxl else "cond_stage_model."
    for k in keys:
        if "diffusion_model." not in k and "first_stage_model." not in k and base_prefix not in k:
            model.pop(k, None)
    return model

def to_half(tensor, enable):
    if enable and type(tensor) in [dict, collections.OrderedDict]:
        for key in tensor.keys():
            if 'model' in key and tensor[key].dtype == torch.float:
                tensor[key] = tensor[key].half()
    elif enable and tensor.dtype == torch.float:
        return tensor.half()

    return tensor

def read_metadata_from_safetensors(filename):
    """read metadata from safetensor - from sd-webui modules/sd_models.py"""
    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass

        return res


def get_ckpt_header(file):
    """Load ckpt state_dict with keys and sizes only"""

    # modified vesion of _dtype_to_storage_type_map() from torch/storage.py
    def _dtype_to_storage_type_map():
        return {
            "double": 'DoubleStorage',
            "float": 'FloatStorage',
            "half": 'HalfStorage',
            "long": 'LongStorage',
            "int": 'IntStorage',
            "int16": 'ShortStorage',
            "int8": 'CharStorage',
            "uint8": 'ByteStorage',
            "bool": 'BoolStorage',
            "bfloat16": 'BFloat16Storage',
            "cdouble": 'ComplexDoubleStorage',
            "cfloat": 'ComplexFloatStorage',
            "qint8": 'QInt8Storage',
            "qint32": 'QInt32Storage',
            "quint8": 'QUInt8Storage',
            "quint4x2": 'QUInt4x2Storage',
            "quint2x4": 'QUInt2x4Storage',
        }

    @lru_cache(maxsize=None)
    def _storage_type_to_dtype_map():
        dtype_map = {
            val: key for key, val in _dtype_to_storage_type_map().items()}
        return dtype_map

    def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
        try:
            return _storage_type_to_dtype_map()[pickle_storage_type]
        except KeyError as e:
            raise KeyError(
                f'pickle storage type "{pickle_storage_type}" is not recognized') from e

    class StorageType():
        def __init__(self, name):
            self.dtype = _get_dtype_from_pickle_storage_type(name)

        def __str__(self):
            return f'StorageType(dtype={self.dtype})'

    # from torch/serialization.py
    def dummy_persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        assert typename == 'storage', \
            f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        data = saved_id[1:]

        storage_type, key, location, numel = data
        dtype = storage_type.dtype
        # return dtype only
        return dtype

    load_module_mapping: Dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        'torch.tensor': 'torch._tensor',
    }

    def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
        if isinstance(bytes_str, bytes):
            return bytes_str.decode('ascii')
        return bytes_str

    def _rebuild_hook_tensor(storage, storage_offset, size, stride, *_args):
        # storage is returned by dummy_persistent_load()
        # in this case, storage == dtype
        return {"shape": list(size), "type": str(storage)}

    class UnpicklerWrapper(pickle.Unpickler):  # type: ignore[name-defined]
        # from https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
        # Lets us override the imports that pickle uses when unpickling an object.
        # This is useful for maintaining BC if we change a module path that tensor instantiation relies on.
        def find_class(self, mod_name, name):
            if type(name) is str and 'Storage' in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            mod_name = load_module_mapping.get(mod_name, mod_name)
            if mod_name == "torch._utils" and name in ["_rebuild_tensor_v2", "_rebuild_tensor"]:
                return _rebuild_hook_tensor

            return super().find_class(mod_name, name)

    with open(file, 'rb') as opened_file:
        if is_zipfile(opened_file):
            with ZipFile(opened_file) as opened_zipfile:
                x = opened_zipfile.namelist()
                for n in x:
                    if "data.pkl" in n:
                        with opened_zipfile.open(n, 'r') as datazip:
                            byte = datazip.read()
                            byteio = io.BytesIO(byte)

                            unpickler = UnpicklerWrapper(byteio, encoding="utf-8")
                            unpickler.persistent_load = dummy_persistent_load
                            result = unpickler.load()
                            if "state_dict" in result:
                                result = result["state_dict"]
                            return result

                        break
    return None


def get_safetensors_header(filename):
    if not os.path.exists(filename):
        return None

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        if metadata_len > 2 and json_start in (b'{"', b"{'"):
            json_data = json_start + file.read(metadata_len-2)
            return json.loads(json_data)

        # invalid safetensors
        return None

def is_xl(modelname):
    checkpointinfo = sd_models.get_closet_checkpoint_match(modelname)
    if checkpointinfo is None:
        return None

    if checkpointinfo.is_safetensors:
        header = get_safetensors_header(checkpointinfo.filename)
    elif checkpointinfo.filename.endswith(".ckpt"):
        header = get_ckpt_header(checkpointinfo.filename)
    else:
        return None

    if header is not None:
        if "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in header:
            return True
        return False
    return None


def sdversion(modelname):
    checkpointinfo = sd_models.get_closet_checkpoint_match(modelname)
    if checkpointinfo is None:
        return None

    if checkpointinfo.is_safetensors:
        header = get_safetensors_header(checkpointinfo.filename)
    elif checkpointinfo.filename.endswith(".ckpt"):
        header = get_ckpt_header(checkpointinfo.filename)
    else:
        return None

    if header is not None:
        if "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in header:
            return 'XL'

        v2 = False
        if 'model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight' in header:
            v2 = header['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight']["shape"][1] == 1024
        return 'v1' if not v2 else 'v2'
    return None


def get_valid_checkpoint_title():
    checkpoint_info = shared.sd_model.sd_checkpoint_info if shared.sd_model is not None else None
    # check validity of current checkpoint_info
    if checkpoint_info is not None:
        filename = checkpoint_info.filename
        name = os.path.basename(filename)
        info = sd_models.get_closet_checkpoint_match(name)
        if info is None:
            # no matched found.
            return sd_models.checkpoint_tiles()[0]
        if info != checkpoint_info:
            # this is a fake checkpoint_info
            # return original title
            return info.title

        return checkpoint_info.title
    return ""

def mm_list_models():
    # save current checkpoint_info and call register() again to restore
    checkpoint_info = shared.sd_model.sd_checkpoint_info if shared.sd_model is not None else None
    sd_models.list_models()
    if checkpoint_info is not None:
        # register again
        checkpoint_info.register()


permutation_spec = None
def get_rebasin_perms(mbws, isxl):
    """all blocks permutations of selected blocks"""
    global permutation_spec

    if "scripts.rebasin.weight_matching" not in sys.modules:
        print(" - Dynamic loading rebasin module...")
        load_module(os.path.join(scriptdir, "scripts", "rebasin", "weight_matching.py"))
    from scripts.rebasin.weight_matching import sdunet_permutation_spec

    if permutation_spec is None:
        # one time initializer
        permutation_spec = sdunet_permutation_spec()

    if True in mbws or False in mbws: # already have selected
        _selected = mbws
        all_blocks = _all_blocks(isxl)
        selected = []
        for i, v in enumerate(_selected):
            if v:
                selected.append(all_blocks[i])
    else:
        selected = get_selected_blocks(mbws, isxl)

    if len(selected) > 0:
        axes = []
        perms = []
        for block in selected:
            if block not in ["cond_stage_model.", "conditioner."]:
                block = f"model.diffusion_model.{block}"
            for axe, perm in permutation_spec.axes_to_perm.items():
                if block in axe:
                    axes.append(axe)
                    perms += list(perm)

        perms = sorted(list(set(perms) - set([None])))

        return perms
    return None


def get_rebasin_axes(mbws, isxl):
    """select all blocks correspond their permutation groups"""

    perms = get_rebasin_perms(mbws, isxl)
    if perms is None:
        return None

    # get all axes and corresponde blocks
    blocks = []
    axes = []
    for perm in perms:
        axes += [axes[0] for axes in permutation_spec.perm_to_axes[perm]]
    axes = list(set(axes))

    return axes


def _get_rebasin_blocks(mbws, isxl):
    """select all blocks correspond their permutation groups"""

    perms = get_rebasin_perms(mbws, isxl)
    if perms is None:
        return None

    # get all axes and corresponde blocks
    blocks = []
    axes = []
    for perm in perms:
        axes += [axes[0] for axes in permutation_spec.perm_to_axes[perm]]
    axes = list(set(axes))

    # get all block representations to show gr.Dropdown
    MAXLEN = 26 - (0 if not isxl else 6)
    BLOCKLEN = 12 - (0 if not isxl else 3)
    BLOCKOFFSET = 13 if not isxl else 10
    selected = [False]*MAXLEN
    BLOCKIDS = BLOCKID if not isxl else BLOCKIDXL

    all_blocks = _all_blocks(isxl)
    for j, block in enumerate(all_blocks[:MAXLEN]):
        if block not in ["cond_stage_model.", "conditioner."]:
            block = f"model.diffusion_model.{block}"
        if any(block in axe for axe in axes):
            selected[j] = True

    return selected


def get_device():
    device_id = shared.cmd_opts.device_id
    if device_id is not None:
        cuda_device = f"cuda:{device_id}"
    else:
        cuda_device = "cpu"
    return cuda_device


def unet_blocks_map(diffusion_model, isxl=False):
    block_map = {}
    block_map['time_embed.'] = diffusion_model.time_embed

    BLOCKLEN = 12 - (0 if not isxl else 3)
    for j in range(BLOCKLEN):
        block_name = f"input_blocks.{j}."
        block_map[block_name] = diffusion_model.input_blocks[j]

    block_map["middle_block."] = diffusion_model.middle_block

    for j in range(BLOCKLEN):
        block_name = f"output_blocks.{j}."
        block_map[block_name] = diffusion_model.output_blocks[j]

    block_map["out."] = diffusion_model.out

    return block_map


def load_module(path):
    p = Path(path)
    if "extensions" in p.parts:
        i = p.parts.index("extensions")
        if "scripts" in p.parts[i:]:
            j = p.parts[i:].index("scripts")
            name = ".".join(p.parts[i+j:])
        else:
            name = ".".join(p.parts[i:])
        if name.endswith(".py"):
            name = name[:-3]
    else:
        name = os.path.basename(path)

    if os.path.isdir(path):
        module_spec = importlib.util.spec_from_file_location(name, os.path.join(path, "__init__.py"))
    else:
        module_spec = importlib.util.spec_from_file_location(name, path)

    if module_spec is None:
        return None

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    # register sys.modules as import_module() does
    sys.modules[name] = module
    if name.count(".") >= 2:
        parent =  name.rsplit(".", 1)[0]
        if parent not in sys.modules:
            # register parent package as import_module() does
            load_module(os.path.dirname(path))
    return module


class ModelMixerScript(scripts.Script):
    global elemental_blocks
    elemental_blocks = None

    init_on_after_callback = False

    components = {}

    def __init__(self):
        super().__init__()

    def title(self):
        return "Model Mixer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def _model_option_ui(self, n, isxl):
        name = chr(66+n)

        with gr.Row():
            mm_alpha = gr.Slider(label=f"Multiplier for Model {name}", minimum=-1.0, maximum=2, step=0.001, value=0.5)
        with gr.Group():
            with gr.Row():
                with gr.Group(Visible=True) as mbw_advanced:
                    mm_usembws = gr.Dropdown(["ALL","BASE","INP*","MID","OUT*"]+BLOCKID[1:], value=[], multiselect=True, label="Merge Block Weights", show_label=False, info="or use Merge Block Weights for selected blocks")
                with gr.Group(visible=False) as mbw_simple:
                    mm_usembws_simple = gr.CheckboxGroup(["BASE","INP*","MID","OUT*"], value=[], label="Merge Block Weights", show_label=False, info="or use Merge Block Weights for selected blocks")
            with gr.Row():
                mbw_use_advanced = gr.Checkbox(label="Use advanced MBW mode", value=True, visible=True)
        with gr.Row():
            mm_explain = gr.HTML("")
        with gr.Group() as mbw_ui:
            with gr.Row():
                mm_weights = gr.Textbox(label="Merge Block Weights: BASE,IN00,IN02,...IN11,M00,OUT00,...,OUT11", show_copy_button=True, elem_classes=["mm_mbw"],
                    value="0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")
            with gr.Row():
                mm_setalpha = gr.Button(elem_classes=["mm_mbw_set"], value="↑ set alpha")
                mm_readalpha = gr.Button(elem_classes=["mm_mbw_read"], value="↓ read alpha")
        with gr.Column():
            with gr.Row():
                mm_use_elemental = gr.Checkbox(label="Use Elemental merge", value=False, visible=True)
        with gr.Group(visible=False) as elemental_ui:
            with gr.Row():
                mm_elemental = gr.Textbox(label="Blocks:Elements:Ratio,Blocks:Elements:Ratio,...", lines=2, max_lines=4, value="", show_copy_button=True)
            with gr.Row():
                mm_set_elem = gr.Button(value="↑ Set elemental merge weights")

        # some interactions between controls
        mm_usembws.change(fn=lambda a,b: [gr.update(visible=len(a)>0 or len(b)>0), gr_enable(len(a)==0 and len(b)==0)],
            inputs=[mm_usembws, mm_usembws_simple], outputs=[mbw_ui, mm_alpha], show_progress=False)
        mm_usembws_simple.change(fn=lambda a,b: [gr.update(visible=len(a)>0 or len(b)>0), gr_enable(len(a)==0 and len(b)==0)],
            inputs=[mm_usembws, mm_usembws_simple], outputs=[mbw_ui, mm_alpha], show_progress=False)
        mm_use_elemental.change(fn=lambda u: gr.update(visible=u), inputs=[mm_use_elemental], outputs=[elemental_ui])
        mbw_use_advanced.change(fn=lambda mode: [gr.update(visible=True), gr.update(visible=False)] if mode==True else [gr.update(visible=False),gr.update(visible=True)],
            inputs=[mbw_use_advanced], outputs=[mbw_advanced, mbw_simple])

        return mm_alpha, mm_usembws, mm_usembws_simple, mbw_use_advanced, mbw_advanced, mbw_simple, mm_explain, mm_weights, mm_use_elemental, mm_elemental, mm_setalpha, mm_readalpha, mm_set_elem


    def after_component(self, component, **_kwargs):
        MM = ModelMixerScript

        elem_id = getattr(component, "elem_id", None)
        if elem_id is None:
            return

        if elem_id in [ "txt2img_generate", "img2img_generate" ]:
            MM.components[elem_id] = component
            return


    def ui(self, is_img2img):
        import modules.ui
        MM = ModelMixerScript

        num_models = shared.opts.data.get("mm_max_models", 2)
        mm_use = [None]*num_models
        mm_models = [None]*num_models
        mm_modes = [None]*num_models
        mm_calcmodes = [None]*num_models
        mm_alpha = [None]*num_models
        mm_usembws = [None]*num_models
        mm_usembws_simple = [None]*num_models
        mm_weights = [None]*num_models
        mm_elementals = [None]*num_models

        mm_setalpha = [None]*num_models
        mm_readalpha = [None]*num_models
        mm_explain = [None]*num_models

        model_options = [None]*num_models
        default_use = [False]*num_models
        mbw_advanced = [None]*num_models
        mbw_simple = [None]*num_models
        mbw_use_advanced = [None]*num_models
        mm_use_elemental = [None]*num_models
        mm_set_elem = [None]*num_models

        default_use[0] = True

        def initial_checkpoint():
            if shared.sd_model is not None and shared.sd_model.sd_checkpoint_info is not None:
                return shared.sd_model.sd_checkpoint_info.title
            return sd_models.checkpoint_tiles()[0]

        with gr.Accordion("Checkpoint Model Mixer", open=False, elem_id="mm_main_" + ("txt2img" if not is_img2img else "img2img")):
            with gr.Row():
                mm_information = gr.HTML("Merge multiple models and load it for image generation.")
            with gr.Row():
                enabled = gr.Checkbox(label="Enable Model Mixer", value=False, visible=True, elem_classes=["mm-enabled"])
            with gr.Row():
                default_vals = []
                config_lock = shared.opts.data.get("mm_default_config_lock", False)
                shared.config_lock = config_lock
                if config_lock:
                    default_vals.append("lock")
                basic_settings = gr.CheckboxGroup(label="Basic Settings", show_label=False, choices=[("Lock current settings", "lock")], value=default_vals, visible=True)
            with gr.Row():
                recipe_all = gr.HTML("<h3></h3>")

            with gr.Row():
                model_a = gr.Dropdown(sd_models.checkpoint_tiles(), value=initial_checkpoint, elem_id="model_mixer_model_a", label="Model A", interactive=True)
                create_refresh_button(model_a, mm_list_models,lambda: {"choices": sd_models.checkpoint_tiles(), "value": get_valid_checkpoint_title()},"refresh_checkpoint_Z")

                base_model = gr.Dropdown(["None"]+sd_models.checkpoint_tiles(), elem_id="model_mixer_model_base", value="None", label="Base Model used for Add-Difference mode", interactive=True)
                create_refresh_button(base_model, mm_list_models,lambda: {"choices": ["None"]+sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
            with gr.Row():
                enable_sync = gr.Checkbox(label="Sync with Default SD checkpoint", value=False, visible=True)
                is_sdxl = gr.Checkbox(label="is SDXL", value=False, visible=True)
            with gr.Row():
                calc_settings = gr.CheckboxGroup(label=f"Calculation options", info="Optional paramters for calculation if needed. e.g.) Rebasin",
                    choices=[("Use GPU", "GPU"), ("Use CPU", "CPU"), ("Fast Rebasin", "fastrebasin")], value=["GPU", "fastrebasin"])


            def update_basic_settings(basic_settings):
                shared.config_lock = "lock" in basic_settings


            basic_settings.change(
                fn=update_basic_settings,
                inputs=[basic_settings],
                outputs=[],
            )

            def check_calc_settings(calc_settings):
                last = calc_settings.pop()
                if "CPU" == last:
                    if "GPU" in calc_settings:
                        idx = calc_settings.index("GPU")
                        calc_settings.pop(idx)
                    calc_settings.append(last)
                    return gr.update(value=calc_settings)
                elif "GPU" == last:
                    if "CPU" in calc_settings:
                        idx = calc_settings.index("CPU")
                        calc_settings.pop(idx)
                    calc_settings.append(last)
                    return gr.update(value=calc_settings)
                else:
                    calc_settings.append(last)
                    if all(s not in calc_settings for s in ["CPU", "GPU"]):
                        # no calc device
                        calc_settings.append("GPU")
                        return gr.update(value=calc_settings)

                return gr.update()

            calc_settings.change(
                fn=check_calc_settings,
                inputs=[calc_settings],
                outputs=[calc_settings],
                show_progress=False,
            )

            mm_max_models = gr.Number(value=num_models, precision=0, visible=False)
            merge_method_info = [{}] * num_models
            with gr.Group(), gr.Tabs():
                mm_states = gr.State({})
                for n in range(num_models):
                    name_a = chr(66+n-1) if n == 0 else f"merge_{n}"
                    name = chr(66+n)
                    lowername = chr(98+n)
                    merge_method_info[n] = {"Sum": f"Weight sum: {name_a}×(1-alpha)+{name}×alpha", "Add-Diff": f"Add difference:{name_a}+({name}-model_base)×alpha"}
                    default_merge_info = merge_method_info[n]["Sum"]
                    tabname = f"Merge Model {name}" if n == 0 else f"Model {name}"
                    with gr.Tab(tabname, elem_classes=["mm_model_tab"]):
                        with gr.Row():
                            mm_use[n] = gr.Checkbox(label=f"Model {name}", value=default_use[n], visible=True)
                        with gr.Row():
                            mm_models[n] = gr.Dropdown(["None"]+sd_models.checkpoint_tiles(), value="None", elem_id=f"model_mixer_model_{lowername}", label=f"Merge {name}", show_label=False, interactive=True)
                            create_refresh_button(mm_models[n], mm_list_models, lambda: {"choices": ["None"]+sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")

                        with gr.Group(visible=False) as model_options[n]:
                            with gr.Row():
                                mm_modes[n] = gr.Radio(label=f"Merge Mode for Model {name}", info=default_merge_info, choices=["Sum", "Add-Diff"], value="Sum")
                            with gr.Row():
                                mm_calcmodes[n] = gr.Radio(label=f"Calcmode for Model {name}", info="Calculation mode (rebasin will not work for SDXL)", choices=["Normal", "Rebasin"], value="Normal")
                            mm_alpha[n], mm_usembws[n], mm_usembws_simple[n], mbw_use_advanced[n], mbw_advanced[n], mbw_simple[n], mm_explain[n], mm_weights[n], mm_use_elemental[n], mm_elementals[n], mm_setalpha[n], mm_readalpha[n], mm_set_elem[n] = self._model_option_ui(n, is_sdxl)

            with gr.Accordion("Merge Block Weights", open=False, elem_classes=["model_mixer_mbws_control"]) as mbw_controls:

                with gr.Row():
                    advanced_range_mode = gr.Checkbox(label="Enable Advanced block range", value=False, visible=True, interactive=True)
                with gr.Row():
                    with gr.Group(), gr.Tabs():
                        with gr.Tab("Presets"):
                            with gr.Row():
                                preset_weight = gr.Dropdown(label="Select preset", choices=mbwpresets().keys(), interactive=True, elem_id="model_mixer_presets")
                                create_refresh_button(preset_weight, lambda: None, lambda: {"choices": list(mbwpresets(True).keys())}, "mm_refresh_presets")
                                preset_save = gr.Button(value=save_symbol, elem_classes=["tool"])
                        with gr.Tab("Helper"):
                            with gr.Column():
                                resetval = gr.Slider(label="Value", show_label=False, info="Value to set/add/mul", minimum=0, maximum=2, step=0.001, value=0)
                                resetopt = gr.Radio(label="Pre defined", show_label=False, choices = ["0", "0.25", "0.5", "0.75", "1"], value = "0", type="value")
                            with gr.Column():
                                resetblockopt = gr.CheckboxGroup(["BASE","INP*","MID","OUT*"], value=["INP*","OUT*"], label="Blocks", show_label=False, info="Select blocks to change")
                            with gr.Column():
                                with gr.Row():
                                    resetweight = gr.Button(elem_classes=["reset"], value="Set")
                                    addweight = gr.Button(elem_classes=["reset"], value="Add")
                                    mulweight = gr.Button(elem_classes=["reset"], value="Mul")

                    with gr.Box(elem_id=f"mm_preset_edit_dialog", elem_classes="popup-dialog") as preset_edit_dialog:
                        with gr.Row():
                            preset_edit_select = gr.Dropdown(label="Presets", elem_id="mm_preset_edit_edit_select", choices=mbwpresets().keys(), interactive=True, value=[], allow_custom_value=True, info="Preset editing allow you to add custom weight presets.")
                            create_refresh_button(preset_edit_select, lambda:  None , lambda: {"choices": list(mbwpresets(True).keys())}, "mm_refresh_presets")
                            preset_edit_read = gr.Button(value="↓", elem_classes=["tool"])
                        with gr.Row():
                            preset_edit_weight = gr.Textbox(label="Weights", show_label=True, elem_id=f"mm_preset_edit", lines=1)

                        with gr.Row():
                            preset_edit_overwrite = gr.Checkbox(label="Overwrite", value=False, interactive=True, visible=False) # XXX checkbox is not working with dialog
                            preset_edit_save = gr.Button('Save', variant='primary', elem_id=f'mm_preset_edit_save')
                            preset_edit_delete = gr.Button('Delete', variant='primary', elem_id=f'mm_preset_edit_delete')
                            preset_edit_close = gr.Button('Close', variant='secondary', elem_id=f'mm_preset_edit_close')

                    ui_common.setup_dialog(button_show=preset_save, dialog=preset_edit_dialog, button_close=preset_edit_close)
                    # preset_save.click() will be redefined later

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)
                    with gr.Column(scale=2, min_width=200):
                        base = gr.Slider(label="BASE", minimum=0, maximum=1, step=0.0001, value=0.5)
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)

                with gr.Row():
                    with gr.Column(scale=2, min_width=200):
                        in00 = gr.Slider(label="IN00", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in01 = gr.Slider(label="IN01", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in02 = gr.Slider(label="IN02", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in03 = gr.Slider(label="IN03", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in04 = gr.Slider(label="IN04", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in05 = gr.Slider(label="IN05", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in06 = gr.Slider(label="IN06", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in07 = gr.Slider(label="IN07", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in08 = gr.Slider(label="IN08", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in09 = gr.Slider(label="IN09", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in10 = gr.Slider(label="IN10", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        in11 = gr.Slider(label="IN11", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                    with gr.Column(scale=2, min_width=200):
                        ou11 = gr.Slider(label="OUT11", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou10 = gr.Slider(label="OUT10", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou09 = gr.Slider(label="OUT09", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou08 = gr.Slider(label="OUT08", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou07 = gr.Slider(label="OUT07", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou06 = gr.Slider(label="OUT06", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou05 = gr.Slider(label="OUT05", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou04 = gr.Slider(label="OUT04", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou03 = gr.Slider(label="OUT03", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou02 = gr.Slider(label="OUT02", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou01 = gr.Slider(label="OUT01", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                        ou00 = gr.Slider(label="OUT00", minimum=0.0, maximum=1, step=0.0001, value=0.5)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)
                    with gr.Column(scale=2):
                        mi00 = gr.Slider(label="M00", minimum=0.0, maximum=1, step=0.0001, value=0.5)
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)

                    dtrue =  gr.Checkbox(value = True, visible = False)
                    dfalse =  gr.Checkbox(value = False, visible = False)

            with gr.Row(elem_classes="accordions"):
              with gr.Accordion("Adjust settings", open=False, elem_classes=["input-accordion", "model_mixer_adjust_control"]):
                with gr.Row():
                    mm_finetune = gr.Textbox(label="IN,OUT,OUT2,Contrast,Bright,COL1,COL2,COL3", visible=True, value="", lines=1, show_copy_button=True)
                    finetune_write = gr.Button(value="↖", elem_classes=["tool"])
                    finetune_read = gr.Button(value="↓", elem_classes=["tool", "mm_adjust_read"])
                    finetune_reset = gr.Button(value="\U0001f5d1\ufe0f", elem_classes=["tool"])
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        detail1 = gr.Slider(label="IN", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                    with gr.Column(scale=1, min_width=100):
                        detail2 = gr.Slider(label="OUT", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                    with gr.Column(scale=1, min_width=100):
                        detail3 = gr.Slider(label="OUT2", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        contrast = gr.Slider(label="Contrast", minimum=-10, maximum=10, step=0.01, value=0, info="Contrast/\U0000200BDetail")
                    with gr.Column(scale=1, min_width=100):
                        bri = gr.Slider(label="Brightness", minimum=-10, maximum=10, step=0.01, value=0, info="Dark(-)-Bright(+)")
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        col1 = gr.Slider(label="Cyan-Red", minimum=-10, maximum=10, step=0.01, value=0, info="Color1, Cyan(-)-Red(+)")
                    with gr.Column(scale=1, min_width=100):
                        col2 = gr.Slider(label="Magenta-Green", minimum=-10, maximum=10, step=0.01, value=0, info="Color2, Magenta(-)-Green(+)")
                    with gr.Column(scale=1, min_width=100):
                        col3 = gr.Slider(label="Yellow-Blue", minimum=-10, maximum=10, step=0.01, value=0, info="Color3, Yellow(-)-Blue(+)")

              with gr.Accordion("Elemental Merge",open = False, elem_classes=["input-accordion"]):
                with gr.Row():
                    mm_elemental_main = gr.Textbox(label="Blocks:Elements:Ratio,Blocks:Elements:Ratio,...",lines=1,max_lines=4,value="", show_copy_button=True)
                    elemental_write = gr.Button(value="↖", elem_classes=["tool"])
                    elemental_read = gr.Button(value="↓", elem_classes=["tool"])
                    elemental_reset = gr.Button(value="\U0001f5d1\ufe0f", elem_classes=["tool"])
                with gr.Row():
                    with gr.Column(variant="compact"):
                        with gr.Row():
                            not_elemblks = gr.Checkbox(value=False, label="Not Blocks", show_label=False, info="NOT", scale=1, min_width=30, elem_classes=["not-button"])
                            elemblks = gr.Dropdown(BLOCKID, value=[], label="Blocks", show_label=False, multiselect=True, info="Select Blocks", scale=7)
                    with gr.Column(variant="compact"):
                        with gr.Row():
                            not_elements = gr.Checkbox(value=False, label="Not Elements", show_label=False, info="NOT", scale=1, min_width=30, elem_classes=["not-button"])
                            elements = gr.Dropdown(["time_embed", "time_embed.0", "time_embed.2", "out", "out.0", "out.2"], values=[], label="Elements", show_label=False, multiselect=True, info="Select Elements", elem_id="mm_elemental_elements", scale=7)
                    with gr.Column(variant="compact"):
                        with gr.Row():
                            elemental_ratio = gr.Slider(minimum=0, maximum=2, value=0.5, step=0.01, label="Ratio", scale=8)

            with gr.Accordion("Cross-Attention Visualizer", open=False):
                with gr.Row():
                    with gr.Column(variant="compact"):
                        if is_img2img:
                            input_image = gr.Image(elem_id="mm_vxa_input_image", visible=False, type="pil")
                        else:
                            input_image = gr.Image(elem_id="mm_vxa_input_image", type="pil")
                        with gr.Row():
                            import_image = gr.Button(value="Import image", visible=False if is_img2img else True)
                            deepbooru_image = gr.Button(value="Interrogate Deepbooru")
                        vxa_prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Prompt to be visualized")
                        stripped = gr.Textbox(label="Prompt", value="", visible=False)
                        ignore_webui = gr.Checkbox(label="Ignore sd-webui grammar", value=True, interactive=True)
                        with gr.Row():
                            current_step = gr.Slider(label='Current steps', minimum=1, maximum=150, value=1, step=1, interactive=True)
                            total_step = gr.Slider(label='Total steps', minimum=1, maximum=150, value=28, step=1, interactive=True)
                            and_block = gr.Number(label='Prompt block (separated by AND)', value=0, step=1, interactive=True)
                        go = gr.Button(value="Tokenize")
                        with gr.Row():
                            with gr.Tabs():
                                with gr.Tab("Text"):
                                    tokenized_text = gr.HTML(elem_id="mm_tokenized_text")

                                with gr.Tab("Tokens"):
                                    tokens = gr.HTML(elem_id="mm_tokenized_tokens")
                        tokens_checkbox = gr.CheckboxGroup(label="Select words", choices=[], value=[], interactive=True)

                        vxa_token_indices = gr.Textbox(value="", label="Indices of tokens to be visualized", lines=2, placeholder="Example: 1, 3 means the sum of the first and the third tokens. 1 is suggected for a single token. Leave blank to visualize all tokens.")
                        vxa_time_embedding = gr.Textbox(value="1.0", label="Time embedding")

                        with gr.Row():
                            hidden_layer_select = gr.Dropdown(value=default_hidden_layer_name, label="Cross-attention layer", choices=get_layer_names())
                            create_refresh_button(hidden_layer_select, lambda: None, lambda: {"choices": get_layer_names()},"imm_refresh_vxa_layer_names")
                        vxa_output_mode = gr.CheckboxGroup(choices=["masked", "gray", "resize"], value=["masked", "resize"], interactive=True, label="Visualize options")
                        vxa_generate = gr.Button(value="Visualize Cross-Attention", elem_id="mm_vxa_gen_btn", variant="primary")

                        #output_gallery = gr.Image(label="Output", show_label=False)
                        #output_gallery = gr.Gallery(label="Output", columns=[1], rows=[1], height="auto", show_label=False)

                def interrogate_deepbooru(image):
                    if image is None:
                        return
                        #return gr.update(), gr.update()

                    # deepboru
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(np.uint8(image)).convert('RGB')

                    prompt = deepbooru.model.tag(image)
                    return prompt, ''

                def get_prompt(image):
                    prompt = ''
                    if image is None:
                        return gr.update(), gr.update()

                    geninfo, _ = images.read_info_from_image(image)

                    if geninfo is not None:
                        params = parse_generation_parameters(geninfo)
                        prompt = params.get("Prompt", "")

                    return prompt, ''

                if not is_img2img:
                    deepbooru_image.click(
                        fn=interrogate_deepbooru,
                        inputs=[input_image],
                        outputs=[vxa_prompt, stripped],
                    )

                go.click(
                    fn=tokenize,
                    inputs=[vxa_prompt, current_step, total_step, and_block, ignore_webui],
                    outputs=[tokenized_text, tokens, stripped, tokens_checkbox],
                )
                tokens_checkbox.select(
                    fn=lambda n: ",".join([str(x) for x in sorted(n)]),
                    inputs=[tokens_checkbox],
                    outputs=vxa_token_indices,
                    show_progress=False,
                )
                input_image.change(
                    fn=get_prompt,
                    inputs=[input_image],
                    outputs=[vxa_prompt, stripped],
                    show_progress=False,
                )

                def change_mode(mode):
                    resize = True if "resize" in mode else False
                    if resize: mode.pop(mode.index("resize"))
                    if len(mode) == 0:
                        return ["masked"] + (["resize"] if resize else [])
                    if len(mode) == 1:
                        return mode + (["resize"] if resize else [])
                    mode.pop(0)
                    return mode + (["resize"] if resize else [])

                vxa_output_mode.change(
                    fn=change_mode,
                    inputs=[vxa_output_mode],
                    outputs=[vxa_output_mode],
                    show_progress=False,
                )

            with gr.Accordion("Save the current merged model", open=False):
                with gr.Row():
                    logging = gr.Textbox(label="Message", lines=1, value="", show_label=False, info="log message")
                with gr.Group(), gr.Tabs():
                    with gr.Tab("Save current checkpoint"):
                        with gr.Row():
                            save_settings = gr.CheckboxGroup(["overwrite","safetensors","prune","fp16", "with LoRAs", "fix CLIP ids"], value=["fp16","prune","safetensors"], label="Select settings")
                        with gr.Row():
                            with gr.Column(min_width = 50):
                                with gr.Row():
                                    custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="model_mixer_custom_name")

                            with gr.Column():
                                with gr.Row():
                                    bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", label="Bake in VAE", elem_id="model_mixer_bake_in_vae")
                                    create_refresh_button(bake_in_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["None"] + list(sd_vae.vae_dict)}, "model_mixer_refresh_bake_in_vae")
                        with gr.Row():
                            save_current = gr.Button("Save current model")

                    with gr.Tab("Save as LoRA/LyCORIS"):
                        with gr.Row():
                            save_lora_settings = gr.CheckboxGroup(["LoRA", "LyCORIS"], value=["LyCORIS"], label="Extract settings")
                        with gr.Row():
                            save_lora_mode = gr.Radio([("Extract merged LoRAs", "extract"), ("Difference between base and current", "diff")],
                                value="extract", label="Extract method")
                        with gr.Group(visible=False) as model_orig_options:
                            tabname = "img2img" if is_img2img else "txt2img"
                            with gr.Row():
                                model_orig = gr.Dropdown(sd_models.checkpoint_tiles(), value="", elem_id=f"model_mixer_model_orig_{tabname}", label="Base model",
                                    info="If selected, 'Tuned model' - 'current model' will be performed to extract LoRA.",
                                    interactive=True)
                                create_refresh_button(model_orig, mm_list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "mm_refresh_model_orig")
                                model_tuned = gr.Dropdown(["Current model"], value="Current model", elem_id=f"model_mixer_model_tuned_{tabname}", label="Tuned model",
                                    info="If selected, LoRA will be extracted from 'Tuned model' - 'base model'.",
                                    interactive=False)
                                create_refresh_button(model_tuned, mm_list_models, lambda: {"choices": ["Current model"]+sd_models.checkpoint_tiles()}, "mm_refresh_model_tuned")

                            model_orig.select(
                                fn=lambda model: gr.update(interactive=True, choices=["Current model"]+sd_models.checkpoint_tiles()) if model != "" else gr.update(),
                                inputs=[model_orig],
                                outputs=[model_tuned],
                                show_progress=False,
                            )

                            with gr.Row():
                                diff_model_mode = gr.Radio([("without LoRAs", "None"), ("with LoRAs", "lora")],
                                    value="lora", label="Current model options", info="If any LoRAs used in the prompt, the current model includes LoRA weights.")
                        with gr.Group() as lycoris_options:
                            extract_info = {
                                "Fixed": "Fixed Layer Dimension mode",
                                "Threshold": "DIM = sum(S>threshold)",
                                "Ratio": "DIM = sum(S>max(S)*ratio)",
                                "Quantile": "DIM = sum(cumsum(S) < sum(S)*quantile)",
                            }
                            with gr.Row():
                                extract_mode = gr.Radio(label="Extraction Mode", info=extract_info["Fixed"], choices=["Fixed", "Threshold", "Ratio", "Quantile"], value="Fixed")

                            extract_mode.change(
                                fn=lambda mode: gr.update(info=extract_info[mode]),
                                inputs=[extract_mode],
                                outputs=[extract_mode],
                                show_progress=False,
                            )

                            with gr.Group() as fixed_options:
                                with gr.Row():
                                    lin_dim = gr.Radio(label="Linear Layer Dimension", choices=[1, 4, 8, 16, 32, 64, 128, 256, 512, 768], value=64)
                                with gr.Row():
                                    conv_dim = gr.Radio(label="Conv2D Layer Dimension", choices=[1, 4, 8, 16, 32, 64, 128, 256, 512, 768], value=64)
                            with gr.Column(visible=False) as variable_options:
                                with gr.Accordion("Variable Layer Dimension Options", open=False):
                                    with gr.Row():
                                        lin_slider = gr.Slider(label="Linear Layer", info="Singular value", minimum=0., maximum=1., value=0., step=0.001)
                                        conv_slider = gr.Slider(label="Conv2D Layer", info="Singular value", minimum=0., maximum=1., value=0., step=0.001)
                                rank_info = {
                                    "Threshold": "Threshold to determine rank dim",
                                    "Ratio": "Ratio value to determine rank dim",
                                    "Quantile": "Quantile value to determine rank dim",
                                }
                            extract_mode.change(
                                fn=lambda mode: [
                                    gr.update(info=f"{rank_info[mode]}") if mode != "Fixed" else gr.update(),
                                    gr.update(info=f"{rank_info[mode]}") if mode != "Fixed" else gr.update(),
                                ],
                                inputs=[extract_mode],
                                outputs=[lin_slider, conv_slider],
                                show_progress=False,
                            )
                        with gr.Group(visible=False) as lora_options:
                            with gr.Row():
                                lora_dim = gr.Radio(label="Conv Dimension", choices=[4, 8, 16, 32, 64, 128, 256, 512, 768, 1024], value=64)

                            with gr.Accordion("LoRA Extraction Options", open=False):
                                with gr.Row():
                                    min_diff = gr.Number(label="Textencoder Minimum Diff", info="Minimum difference to consider different Textencoder", minimum=0., maximum=1.0, step=0.000001, value=0.0001)
                                    clamp_quantile = gr.Slider(label="Clamp Quantile", info="Quantile clamping value", minimum=0., maximum=1.0, step=0.001, value=0.99)
                        with gr.Row():
                            precision = gr.Radio(label="Save precision", info="Select weight type", choices=[("fp16", "fp16"), ("fp32 (32bit)", "fp32"), ("bf16", "bf16")], value="fp16")
                            calc_device = gr.Radio(label="Calculation device", choices=[("auto", "auto"), ("GPU", "cuda"), ("CPU", "cpu")], value="cuda",
                                info="SVD calculation will be performed on this device.")
                        with gr.Row():
                            extra_settings = gr.CheckboxGroup([("overwrite", "overwrite"), ("safetensors", "safetensors"), ("No half (more RAM needed)", "nohalf")],
                                value=["safetensors"], label="Additional settings")

                            prec_info = {
                                'fp16': 'float16 type (half precision. default)',
                                'fp32': 'float32 type (single precision)',
                                'bf16': 'bfloat16 type (less memory but 32bit range)',
                            }
                            precision.change(
                                fn=lambda t: gr.update(info=prec_info[t]),
                                inputs=[precision],
                                outputs=[precision],
                                show_progress=False,
                            )
                            device_info = {
                                'auto': 'Device will be selected by system',
                                'cpu': 'SVD calc using CPU (higher precision but slow)',
                                'cuda': 'SVD calc using GPU (high speed)',
                            }
                            calc_device.change(
                                fn=lambda d: gr.update(info=device_info[d]),
                                inputs=[calc_device],
                                outputs=[calc_device],
                                show_progress=False,
                            )

                        with gr.Row():
                            custom_lora_name = gr.Textbox(label="Custom LoRA Name", placeholder="Name your LoRA", elem_id="model_mixer_custom_lora_name")

                        save_lora_mode.change(
                            fn=lambda save_mode: gr.update(visible=save_mode != "extract"),
                            inputs=[save_lora_mode],
                            outputs=[model_orig_options],
                            show_progress=True,
                        )

                        def check_extract_mode(extract_mode):
                            if extract_mode == "Fixed":
                                return gr.update(visible=True), gr.update(visible=False), gr.update(), gr.update()

                            if extract_mode == 'Quantile':
                                return gr.update(visible=False), gr.update(visible=True), gr.update(value=1.0), gr.update(value=1.0)

                            return gr.update(visible=False), gr.update(visible=True), gr.update(value=0.0), gr.update(value=0.0)


                        def check_lora_settings(save_lora_settings):
                            last = save_lora_settings.pop()
                            if "LoRA" == last:
                                if "LyCORIS" in save_lora_settings:
                                    idx = save_lora_settings.index("LyCORIS")
                                    save_lora_settings.pop(idx)
                                save_lora_settings.append(last)
                                return gr.update(value=save_lora_settings), gr.update(visible=False), gr.update(visible=True)
                            elif "LyCORIS" == last:
                                if "LoRA" in save_lora_settings:
                                    idx = save_lora_settings.index("LoRA")
                                    save_lora_settings.pop(idx)
                                save_lora_settings.append(last)
                                return gr.update(value=save_lora_settings), gr.update(visible=True), gr.update(visible=False)
                            else:
                                save_lora_settings.append(last)
                                if all(s not in save_lora_settings for s in ["LoRA", "LyCORIS"]):
                                    # no LoRA type
                                    save_lora_settings.append("LyCORIS")
                                    return gr.update(value=save_lora_settings), gr.update(visible=True), gr.update(visible=False)

                            return gr.update(), gr.update(), gr.update()


                        with gr.Row():
                            extract_lora = gr.Button("Save as LoRA/LyCORIS")

                        extract_mode.change(
                            fn=check_extract_mode,
                            inputs=[extract_mode],
                            outputs=[fixed_options, variable_options, lin_slider, conv_slider],
                            show_progress=False,
                        )

                        save_lora_settings.change(
                            fn=check_lora_settings,
                            inputs=[save_lora_settings],
                            outputs=[save_lora_settings, lycoris_options, lora_options],
                            show_progress=False,
                        )

                    with gr.Tab("Save model to Diffusers"):
                        with gr.Row():
                            custom_diffusers_name = gr.Textbox(label="Custom Diffusers Path", placeholder="Name of Diffusers dump path", elem_id="model_mixer_custom_diffusers_path")
                        with gr.Row():
                            convert_diffusers = gr.Button("Save as Diffusers")

                    with gr.Row():
                        metadata_settings = gr.CheckboxGroup(["merge recipe"], value=["merge recipe"], label="Metadata settings")

                with gr.Accordion("Metadata Viewer", open=False):
                    metadata_json = gr.TextArea('{}', label="Metadata in JSON format", lines=4)
                    with gr.Row():
                        read_metadata = gr.Button("Read current metadata")
                        read_model_a_metadata = gr.Button("model A metadata")
                        read_model_b_metadata = gr.Button("model B metadata")

            use_model_dl = shared.opts.data.get("mm_use_model_dl", False)
            with gr.Accordion("Download Model Helper", open=False, visible=False) as download_helper:
                with gr.Row():
                    download_status = gr.Textbox(visible=True, label="message")

                def downloader(fileinfo): # progress=gr.Progress(track_tqdm=False)):
                    import asyncio
                    use_model_dl = shared.opts.data.get("mm_use_model_dl", False)

                    try:
                        fileinfo = json.loads(fileinfo)
                    except Exception as e:
                        print("Error:", e)
                        return gr.update(), ""

                    destdir = os.path.join(paths.models_path, 'Stable-diffusion')
                    fname = os.path.join(destdir, fileinfo['filename'])

                    if use_model_dl:
                        import requests

                        resp = requests.get(fileinfo['downloadUrl'], stream=True)
                        total = int(resp.headers.get('content-length', 0))
                        try:
                            with open(fname, 'wb') as file:
                                bar = tqdm(desc=fileinfo['filename'], total=total, unit='iB', unit_scale=True, unit_divisor=1024)
                                bar.update(0)
                                for data in resp.iter_content(chunk_size=10 * 1024 * 1024):
                                    size = file.write(data)
                                    bar.update(size)
                                bar.close()
                        except:
                            pass

                    if os.path.exists(fname):
                        downloaded = True
                        # update model list
                        mm_list_models()
                    else:
                        downloaded = False

                    return gr.update(value=downloaded, visible=True), fileinfo['filename'] + " downloaded!"


                def download_ui(i):
                    with gr.Row(visible=False) as wrapper:
                        downloadfile = gr.Textbox(visible=False, label=f"Download Info {i}", show_label=False)
                        with gr.Column(scale=1, min_width=10):
                            checkbox = gr.Checkbox(label=f"", show_label=False, value=False, visible=use_model_dl, elem_classes=["downloaded"], interactive=False, container=False)
                        with gr.Column(scale=10, min_width=160):
                            downfile = gr.HTML(label=f"Download File {i}", show_label=False, value="<p></p>")
                        with gr.Column(scale=1, label="Download Button", visible=use_model_dl, show_label=False, min_width=10):
                            downbtn = gr.Button(' ', elem_classes=["download"])

                    downbtn.click(fn=downloader, inputs=[downloadfile], outputs=[checkbox, download_status], show_progress=False)
                    return wrapper, checkbox, downloadfile, downfile, downbtn

                def setup_download_ui(fileinfo):
                    use_model_dl = shared.opts.data.get("mm_use_model_dl", False)

                    try:
                        s = json.loads(fileinfo)
                    except Exception as e:
                        print("Error:", e)
                        return gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update()

                    return [gr.update(visible=True), gr.update(visible=True),
                        gr.update(value=False, visible=use_model_dl), gr.update(visible=use_model_dl), gr.update(value=f"""<p>Link: <a target="_blank" href="{s['url']}">{s["name"]}</a><br>
Direct Download: <a href="{s['downloadUrl']}" target="_blank">{s["filename"]} [{s["hash"]}]</a></p>""")]

                with gr.Column(variant="compact"):
                    downloadfiles = [[]]*5
                    checkboxes = [[]]*5
                    downfiles = [[]]*5
                    downbtns = [[]]*5
                    wrappers = [[]]*5

                    for i in range(5):
                        wrappers[i], checkboxes[i], downloadfiles[i], downfiles[i], downbtns[i] = download_ui(i)

                        downloadfiles[i].change(
                            fn=setup_download_ui,
                            inputs=[downloadfiles[i]],
                            outputs=[download_helper, wrappers[i], checkboxes[i], downbtns[i], downfiles[i]],
                            show_progress=False,
                        )

            with gr.Row(variant="compact"):
                unload_sd_model = gr.Button("Unload model to free VRAM")
                reload_sd_model = gr.Button("Reload model back to VRAM")

            def call_func_and_return_text(func, text):
                def handler():
                    t = Timer()
                    func()
                    t.record(text)

                    return f'{text} in {t.total:.1f}s'

                return handler

            unload_sd_model.click(
                fn=call_func_and_return_text(lambda: sd_models.send_model_to_cpu(shared.sd_model), 'Unloaded the checkpoint'),
                inputs=[],
                outputs=[logging]
            )

            reload_sd_model.click(
                fn=call_func_and_return_text(lambda: sd_models.send_model_to_device(shared.sd_model), 'Reload the checkpoint'),
                inputs=[],
                outputs=[logging]
            )

        def addblockweights(val, blockopt, *blocks):
            if val == "none":
                val = 0

            value = float(val)

            if "BASE" in blockopt:
                vals = [blocks[0] + value]
            else:
                vals = [blocks[0]]

            if "INP*" in blockopt:
                inp = [blocks[i + 1] + value for i in range(12)]
            else:
                inp = [blocks[i + 1] for i in range(12)]
            vals = vals + inp

            if "MID" in blockopt:
                mid = [blocks[13] + value]
            else:
                mid = [blocks[13]]
            vals = vals + mid

            if "OUT*" in blockopt:
                out = [blocks[i + 14] + value for i in range(12)]
            else:
                out = [blocks[i + 14] for i in range(12)]
            vals = vals + out

            return setblockweights(vals, blockopt)

        def mulblockweights(val, blockopt, *blocks):
            if val == "none":
                val = 0

            value = float(val)

            if "BASE" in blockopt:
                vals = [blocks[0] * value]
            else:
                vals = [blocks[0]]

            if "INP*" in blockopt:
                inp = [blocks[i + 1] * value for i in range(12)]
            else:
                inp = [blocks[i + 1] for i in range(12)]
            vals = vals + inp

            if "MID" in blockopt:
                mid = [blocks[13] * value]
            else:
                mid = [blocks[13]]
            vals = vals + mid

            if "OUT*" in blockopt:
                out = [blocks[i + 14] * value for i in range(12)]
            else:
                out = [blocks[i + 14] for i in range(12)]
            vals = vals + out

            return setblockweights(vals, blockopt)

        def resetblockweights(val, blockopt):
            if val == "none":
                val = 0
            vals = [float(val)] * 26
            return setblockweights(vals, blockopt)

        def setblockweights(vals, blockopt):
            if "BASE" in blockopt:
                ret = [gr.update(value = vals[0])]
            else:
                ret = [gr.update()]

            if "INP*" in blockopt:
                inp = [gr.update(value = vals[i + 1]) for i in range(12)]
            else:
                inp = [gr.update() for _ in range(12)]
            ret = ret + inp

            if "MID" in blockopt:
                mid = [gr.update(value = vals[13])]
            else:
                mid = [gr.update()]
            ret = ret + mid

            if "OUT*" in blockopt:
                out = [gr.update(value = vals[i + 14]) for i in range(12)]
            else:
                out = [gr.update() for _ in range(12)]
            ret = ret + out

            return ret

        def resetvalopt(opt):
            if opt == "none":
                value = 0.0
            else:
                value = float(opt)

            return gr.update(value = value)

        def sync_main_checkpoint(enable_sync, model):
            ret = [gr.update(value=True if is_xl(model) else False, interactive=True if is_xl(model) is None else False)]
            ret.append(gr.update(value=model))
            # load checkpoint
            if enable_sync:
                shared.opts.data['sd_model_checkpoint'] = model
                ret.append(gr.update(value=model))
                modules.sd_models.reload_model_weights()
            else:
                ret.append(gr.update())

            prepare_model(model)

            return ret

        def import_image_from_gallery(gallery):
            prompt = ""
            if len(gallery) == 0:
                return gr.update(), gr.update()
            if isinstance(gallery[0], dict) and gallery[0].get("name", None) is not None:
                print("Import ", gallery[0]["name"])
                image = Image.open(gallery[0]["name"])
                geninfo, _ = images.read_info_from_image(image)
                if geninfo is not None:
                    params = parse_generation_parameters(geninfo)
                    prompt = params.get("Prompt", "")
                return image, prompt
            elif isinstance(gallery[0], np.ndarray):
                return gallery[0], prompt
            else:
                print("Invalid gallery image {type(gallery[0]}")
            return gr.update(), prompt

        def on_after_components(component, **kwargs):
            nonlocal input_image
            self.init_on_after_callback = True
            MM = ModelMixerScript

            elem_id = getattr(component, "elem_id", None)
            if elem_id is None:
                return

            if elem_id == "mm_elemental_elements":
                # one time initializer
                prepare_model(get_valid_checkpoint_title())
                return

            if elem_id == "img2img_image":
                if MM.components.get(elem_id, None) is None:
                    MM.components[elem_id] = component
                    return

            if elem_id == "img2img_gallery":
                if MM.components.get(elem_id, None) is None and MM.components.get("img2img_image", None) is not None:
                    MM.components[elem_id] = component

            if is_img2img and (MM.components.get("__vxa_generate_img2img", None) is None and
                    MM.components.get("img2img_gallery", None) is not None and MM.components.get("img2img_image", None) is not None):
                # HACK
                MM.components["__vxa_generate_img2img"] = True
                vxa_generate.click(
                    fn=lambda *a: [generate_vxa(*a, is_img2img)],
                    inputs=[MM.components["img2img_image"], vxa_prompt, stripped, vxa_token_indices, vxa_time_embedding, hidden_layer_select, vxa_output_mode],
                    outputs=[MM.components["img2img_gallery"]]
                )
                deepbooru_image.click(
                    fn=interrogate_deepbooru,
                    inputs=[MM.components["img2img_image"]],
                    outputs=[vxa_prompt, stripped],
                )

            if elem_id == "txt2img_gallery":
                if MM.components.get(elem_id, None) is None:
                    MM.components[elem_id] = component
                    vxa_generate.click(
                        fn=lambda *a: [generate_vxa(*a, is_img2img)],
                        inputs=[input_image, vxa_prompt, stripped, vxa_token_indices, vxa_time_embedding, hidden_layer_select, vxa_output_mode],
                        outputs=[MM.components[elem_id]]
                    )

                    import_image.click(
                        fn=import_image_from_gallery,
                        inputs=[component],
                        outputs=[input_image, vxa_prompt]
                    )

            if elem_id == "setting_sd_model_checkpoint":
                if MM.components.get(elem_id, None) is None:
                    MM.components[elem_id] = component
                    # component is the setting_sd_model_checkpoint
                    model_a.select(fn=sync_main_checkpoint,
                        inputs=[enable_sync, model_a],
                        outputs=[is_sdxl, model_a, component],
                        show_progress=False,
                    )

                    enable_sync.select(fn=sync_main_checkpoint,
                        inputs=[enable_sync, model_a],
                        outputs=[is_sdxl, model_a, component],
                        show_progress=False,
                    )

        def current_metadata():
            if shared.sd_model and shared.sd_model.sd_checkpoint_info:
                metadata = shared.sd_model.sd_checkpoint_info.metadata
                data = json.dumps(metadata, indent=4, ensure_ascii=False)

                return gr.update(value=data)

        def model_metadata(model):
            """read metadata"""
            checkpoint = sd_models.get_closet_checkpoint_match(model)
            if checkpoint is not None:
                metadata = read_metadata_from_safetensors(checkpoint.filename)
                data = json.dumps(metadata, indent=4, ensure_ascii=False)

                return gr.update(value=data)

        # set callback only once
        if self.init_on_after_callback is False:
            script_callbacks.on_after_component(on_after_components)

        members = [base,in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,mi00,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11]

        def update_slider_range(advanced_mode):
            if advanced_mode:
                return [gr.update(minimum=-1, maximum=2) for _ in range(len(members))]
            return [gr.update(minimum=0, maximum=1) for _ in range(len(members))]

        # fix block range (gradio bug?)
        advanced_range_mode.change(fn=update_slider_range, inputs=[advanced_range_mode], outputs=[*members])

        # update block text using js
        block_args = dict(_js="slider_to_text", fn=lambda *args: None, inputs=[is_sdxl, *members], outputs=[], show_progress=False)
        for block in members:
            block.release(**block_args)

        self.infotext_fields = (
            (model_a, "ModelMixer model a"),
            (base_model, "ModelMixer base model"),
            (mm_max_models, "ModelMixer max models"),
            (mm_finetune, "ModelMixer adjust"),
        )

        for n in range(num_models):
            name = f"{chr(98+n)}"
            self.infotext_fields += (
                (mm_use[n], f"ModelMixer use model {name}"),
                (mm_models[n], f"ModelMixer model {name}"),
                (mm_modes[n], f"ModelMixer merge mode {name}"),
                (mm_calcmodes[n], f"ModelMixer calcmode {name}"),
                (mm_alpha[n], f"ModelMixer alpha {name}"),
                (mbw_use_advanced[n], f"ModelMixer mbw mode {name}"),
                (mm_usembws[n], f"ModelMixer mbw {name}"),
                (mm_usembws_simple[n], f"ModelMixer simple mbw {name}"),
                (mm_weights[n], f"ModelMixer mbw weights {name}"),
                (mm_use_elemental[n], f"ModelMixer use elemental {name}"),
                (mm_elementals[n], f"ModelMixer elemental {name}"),
            )

        # download helper
        for n in range(5):
            self.infotext_fields += (
                (downloadfiles[n], f"Download Model {n+1}"),
            )

        # load settings
        read_metadata.click(fn=current_metadata, inputs=[], outputs=[metadata_json])
        read_model_a_metadata.click(fn=model_metadata, inputs=[model_a], outputs=[metadata_json])
        read_model_b_metadata.click(fn=model_metadata, inputs=[mm_models[0]], outputs=[metadata_json])
        save_current.click(fn=save_current_model, inputs=[custom_name, bake_in_vae, save_settings, metadata_settings], outputs=[logging])

        extract_lora.click(
            fn=extract_lora_from_current_model,
            inputs=[save_lora_mode, model_orig, model_tuned, diff_model_mode,
                custom_lora_name, extract_mode, lin_dim, conv_dim, lin_slider, conv_slider, lora_dim, min_diff, clamp_quantile,
                precision, calc_device, save_lora_settings, metadata_settings, extra_settings],
            outputs=[logging]
        )

        convert_diffusers.click(
            fn=save_as_diffusers,
            inputs=[custom_diffusers_name, save_settings, metadata_settings],
            outputs=[logging],
        )

        def recipe_update(num_models, *_args):
            uses = [False]*num_models
            modes = [None]*num_models
            models = [None]*num_models
            j = 0
            recipe = "A"
            for n in range(num_models):
                uses[n] = _args[n]
                modes[n] = _args[num_models + n]
                models[n] = None if _args[num_models*2 + n] == "None" else _args[num_models*2 + n]
                modelname = chr(66+n)
                if uses[n] and models[n] is not None:
                    if "Sum" in modes[n]:
                        recipe = f"({recipe})" if recipe.find(" ") != -1 else recipe
                        recipe = f"{recipe} × (1 - α<sub>{n}</sub>) + {modelname} × α<sub>{n}</sub>"
                    elif "Add-Diff" in modes[n]:
                        recipe = f"{recipe} + ({modelname} - base) × α<sub>{n}</sub>"

            if recipe == "A":
                recipe = "<h3></h3>"
            else:
                recipe = f"<h3>Recipe: {recipe}</h3>"
            return gr.update(value=recipe)

        # recipe all
        recipe_all.change(fn=recipe_update, inputs=[mm_max_models, *mm_use, *mm_modes, *mm_models], outputs=recipe_all, show_progress=False)

        def finetune_update(finetune, detail1, detail2, detail3,contrast, bri, col1, col2, col3):
            arr = [detail1, detail2, detail3, contrast, bri, col1, col2, col3]
            tmp = ",".join(map(lambda x: str(int(x)) if x == 0.0 else str(x), arr))
            if finetune != tmp:
                return gr.update(value=tmp)
            return gr.update()

        def finetune_reader(finetune):
            tmp = [t.strip() for t in finetune.split(",")]
            ret = [gr.update()]*8
            for i, f in enumerate(tmp[0:8]):
                try:
                    f = float(f)
                    ret[i] = gr.update(value=f)
                except:
                    pass
            return ret

        # update finetune
        finetunes = [detail1, detail2, detail3, contrast, bri,  col1, col2, col3]
        finetune_reset.click(fn=lambda: [gr.update(value="")]+[gr.update(value=0.0)]*8, inputs=[], outputs=[mm_finetune, *finetunes])
        finetune_read.click(fn=finetune_reader, inputs=[mm_finetune], outputs=[*finetunes])
        finetune_write.click(fn=finetune_update, inputs=[mm_finetune, *finetunes], outputs=[mm_finetune])
        detail1.release(fn=finetune_update, inputs=[mm_finetune, *finetunes], outputs=mm_finetune, show_progress=False)
        detail2.release(fn=finetune_update, inputs=[mm_finetune, *finetunes], outputs=mm_finetune, show_progress=False)
        detail3.release(fn=finetune_update, inputs=[mm_finetune, *finetunes], outputs=mm_finetune, show_progress=False)
        contrast.release(fn=finetune_update, inputs=[mm_finetune, *finetunes], outputs=mm_finetune, show_progress=False)
        bri.release(fn=finetune_update, inputs=[mm_finetune, *finetunes], outputs=mm_finetune, show_progress=False)
        col1.release(fn=finetune_update, inputs=[mm_finetune, *finetunes], outputs=mm_finetune, show_progress=False)
        col2.release(fn=finetune_update, inputs=[mm_finetune, *finetunes], outputs=mm_finetune, show_progress=False)
        col3.release(fn=finetune_update, inputs=[mm_finetune, *finetunes], outputs=mm_finetune, show_progress=False)

        def config_sdxl(isxl, num_models):
            if isxl:
                BLOCKS = BLOCKIDXL
            else:
                BLOCKS = BLOCKID
            ret = [gr.update(choices=BLOCKS)]
            ret += [gr.update(visible=True) for _ in range(26)] if not isxl else [gr.update(visible=ISXLBLOCK[i]) for i in range(26)]
            choices = ["ALL","BASE","INP*","MID","OUT*"]+BLOCKID[1:] if not isxl else ["ALL","BASE","INP*","MID","OUT*"]+BLOCKIDXL[1:]
            ret += [gr.update(choices=choices) for _ in range(num_models)]
            last = 11 if not isxl else 8
            info = f"Merge Block Weights: BASE,IN00,IN02,...IN{last:02d},M00,OUT00,...,OUT{last:02d}"
            ret += [gr.update(label=info) for _ in range(num_models)]
            return ret

        def select_block_elements(blocks):
            # change choices for selected blocks
            elements = []
            if elemental_blocks is None or len(elemental_blocks) == 0:
                return gr.update(choices=["time_embed", "time_embed.0", "time_embed.2", "out", "out.0", "out.2"])

            for b in blocks:
                elements += elemental_blocks.get(b, [])

            elements = list(set(elements))
            elements = sorted(elements)
            return gr.update(choices=elements)

        def write_elemental(not_blocks, not_elements, blocks, elements, ratio, elemental):
            # update elemental information
            if len(blocks) == 0 and len(elements) == 0:
                return gr.update()

            # newly added
            info = ("NOT " if not_blocks else "") + " ".join(zipblocks(blocks, BLOCKID))
            info += ":" + ("NOT " if not_elements else "") + " ".join(elements) + ":" + str(ratio)

            # old
            tmp = elemental.strip().replace(",", "\n").strip().split("\n")
            tmp = list(filter(None, tmp))
            tmp = [f.strip() for f in tmp]

            edit = [info]

            # add newly added entries only
            newtmp = tmp + [l for l in edit if l not in tmp]
            info = "\n".join(newtmp) + "\n"
            return gr.update(value=info)

        def read_elemental(elemental):
            tmp = elemental.strip()
            if len(tmp) == 0:
                return [gr.update()]*5
            lines = tmp.splitlines()
            sel = lines[len(lines) - 1]
            tmp = sel.split(":")
            if len(tmp) != 3:
                return [gr.update()]*5
            ratio = float(tmp[2])

            blks = tmp[0].strip().split(" ")
            blks = list(filter(None, blks))
            not_blks = False
            if len(blks) > 0 and blks[0].upper() == "NOT":
                not_blks = True
                blks = blks[1:]
            # expand any block ranges
            blks = prepblocks(blks, BLOCKID)

            elem = tmp[1].strip().split(" ")
            elem = list(filter(None, elem))
            not_elem = False
            if len(elem) > 0 and elem[0].upper() == "NOT":
                not_elem = True
                elem = elem[1:]

            return [gr.update(value=not_blks), gr.update(value=not_elem), gr.update(value=blks), gr.update(value=elem), gr.update(value=ratio)]

        elemblks.change(fn=select_block_elements, inputs=[elemblks], outputs=[elements])
        elemental_reset.click(fn=lambda: [gr.update(value=False)]*2 + [gr.update(value=[])]*2+[gr.update(value=0.5)], inputs=[], outputs=[not_elemblks, not_elements, elemblks, elements, elemental_ratio])
        elemental_write.click(fn=write_elemental, inputs=[not_elemblks, not_elements, elemblks, elements, elemental_ratio, mm_elemental_main], outputs=mm_elemental_main)
        elemental_read.click(fn=read_elemental, inputs=mm_elemental_main, outputs=[not_elemblks, not_elements, elemblks, elements, elemental_ratio])

        is_sdxl.change(fn=config_sdxl, inputs=[is_sdxl, mm_max_models], outputs=[elemblks, *members, *mm_usembws, *mm_weights])

        resetopt.change(fn=resetvalopt, inputs=[resetopt], outputs=[resetval])
        resetweight.click(fn=resetblockweights, inputs=[resetval,resetblockopt], outputs=members)
        addweight.click(fn=addblockweights, inputs=[resetval, resetblockopt, *members], outputs=members)
        mulweight.click(fn=mulblockweights, inputs=[resetval, resetblockopt, *members], outputs=members)

        # for weight presets
        def on_change_preset_weight(preset):
            weights = find_preset_by_name(preset)
            if weights is not None:
                if len(weights) == 26:
                    return [gr.update(value = float(w)) for w in weights]
                elif len(weights) == 20:
                    j = 0
                    ret = []
                    for i, v in enumerate(ISXLBLOCK):
                        if v:
                            ret.append(gr.update(value = float(weights[j])))
                            j += 1
                        else:
                            ret.append(gr.update())
                    return ret

        def save_preset_weight(preset, weight, overwrite=False):
            # check already have
            w = find_preset_by_name(preset)
            if w is not None and not overwrite:
                raise gr.Error("Preset exists. Please enable overwrite or rename it before save a new preset")

            filepath = os.path.join(scriptdir, "data", "mbwpresets.tsv")
            if os.path.isfile(filepath):
                # prepare weight
                arr = [f.strip() for f in weight.split(",")]
                if len(arr) != 26 and len(arr) != 20:
                    raise gr.Error("Invalid weight")
                for i, a in enumerate(arr):
                    try:
                        a = float(a)
                        arr[i] = a
                    except:
                        arr[i] = 0.0
                        pass

                if preset.find(".XL") > 0:
                    # get the original preset name "foobar.XL" -> "foobar"
                    preset = preset[0:-3]
                newpreset = preset + "\t" + ",".join(map(lambda x: str(int(x)) if x == int(x) else str(x), arr))
                if w is not None:
                    replaced = False
                    # replace preset entry
                    with open(filepath, "r+") as f:
                        raw = f.read()

                        lines = raw.splitlines()
                        for i, l in enumerate(lines[1:], start=1):
                            if ":" in l:
                                k, ws = l.split(":", 1)
                            elif "\t" in l:
                                k, ws = l.split("\t", 1)
                            elif "," in l:
                                k, ws = l.split(",", 1)
                            ws = ws.split(",")
                            if k.strip() == preset and len(ws) == len(w):
                                lines[i] = newpreset
                                replaced = True
                                break

                        if replaced:
                            f.seek(0)
                            raw = "\n".join(lines) + "\n"
                            f.write(raw)
                            f.truncate()

                    if not replaced:
                        raise gr.Error("Fail to save. Preset not found or mismatched")
                else:
                    # append preset entry
                    try:
                        with open(filepath, "a") as f:
                            f.write(newpreset + "\n")
                    except OSError as e:
                        print(e)
                        pass

            gr.Info(f"Successfully save preset {preset}")

            # update dropdown
            updated = list(mbwpresets(True).keys())
            return gr.update(choices=updated)

        def delete_preset_weight(preset, weight, confirm=True):
            w = find_preset_by_name(preset)
            if w is None:
                raise gr.Error("Preset not exists")
            if not confirm:
                raise gr.Error("Please confirm before delete entry")

            filepath = os.path.join(scriptdir, "data", "mbwpresets.tsv")
            deleted = False
            if os.path.isfile(filepath):
                if preset.find(".XL") > 0: # to original name "foobar.XL" -> "foobar"
                    preset = preset[0:-3]
                # search preset entry
                with open(filepath, "r+") as f:
                    raw = f.read()

                    lines = raw.splitlines()
                    newlines = [lines[0]]
                    for l in lines[1:]:
                        if ":" in l:
                            k, ws = l.split(":", 1)
                        elif "\t" in l:
                            k, ws = l.split("\t", 1)
                        elif "," in l:
                            k, ws = l.split(",", 1)
                        ws = ws.split(",")
                        # check weight length
                        if k.strip() == preset and len(ws) == len(w):
                            # remove.
                            deleted = True
                            continue
                        if len(l) > 0:
                            newlines.append(l)

                    if deleted:
                        raw = "\n".join(newlines) + "\n"
                        f.seek(0)
                        f.write(raw)
                        f.truncate()

            if not deleted:
                raise gr.Error("Fail to delete. Preset not found or mismatched.")

            gr.Info(f"Successfully deleted preset {preset}")

            # update dropdown
            updated = list(mbwpresets(True).keys())
            return gr.update(choices=updated)

        def select_preset(name, weight=""):
            weights = find_preset_by_name(name)
            if weights is not None and weight == "":
                return gr.update(value=",".join(weights))

            return gr.update()

        def set_elemental(elemental, elemental_edit):
            tmp = elemental.strip().replace(",", "\n").strip().split("\n")
            tmp = [f.strip() for f in tmp]
            edit = elemental_edit.strip().replace(",", "\n").strip().split("\n")
            edit = [f.strip() for f in edit]

            # add newly added entries only
            newtmp = tmp + [l for l in edit if l not in tmp]

            return gr.update(value="\n".join(newtmp))

        preset_weight.change(fn=on_change_preset_weight, inputs=[preset_weight], outputs=members)
        preset_save.click(
            fn=lambda isxl, *mem: [slider2text(isxl, *mem), gr.update(visible=True)],
            inputs=[is_sdxl, *members],
            outputs=[preset_edit_weight, preset_edit_dialog],
            show_progress=False,
        ).then(fn=None, _js="function(){ popupId('" + preset_edit_dialog.elem_id + "'); }")

        preset_edit_select.change(fn=select_preset, inputs=[preset_edit_select, preset_edit_weight], outputs=preset_edit_weight, show_progress=False)
        preset_edit_read.click(fn=select_preset, inputs=[preset_edit_select], outputs=preset_edit_weight, show_progress=False)
        preset_edit_save.click(fn=save_preset_weight, inputs=[preset_edit_select, preset_edit_weight, preset_edit_overwrite], outputs=[preset_edit_select])
        preset_edit_delete.click(fn=delete_preset_weight, inputs=[preset_edit_select, preset_edit_weight], outputs=[preset_edit_select])

        def check_model_b(model_a, model_b):
            sdv = sdversion(model_a)
            sdv_b = sdversion(model_b)
            if sdv != sdv_b and sdv_b != "None":
                gr.Warning(f"model_a is SD{sdv} but model_b is SD{sdv_b}")
                return gr_show(False), gr.update(value="None")

            return gr_show(model_b != "None"), gr.update()

        for n in range(num_models):
            mm_setalpha[n].click(fn=slider2text,inputs=[is_sdxl, *members],outputs=[mm_weights[n]])
            mm_set_elem[n].click(fn=set_elemental, inputs=[mm_elementals[n], mm_elemental_main], outputs=[mm_elementals[n]])

            mm_readalpha[n].click(fn=get_mbws, inputs=[mm_weights[n], mm_usembws[n], is_sdxl], outputs=[*members, mbw_controls], show_progress=False)
            mm_models[n].change(fn=lambda modelname: [gr_show(modelname != "None"), gr.update(value="<h3>...</h3>")], inputs=[mm_models[n]], outputs=[model_options[n], recipe_all], show_progress=False)
            mm_models[n].select(fn=check_model_b, inputs=[model_a, mm_models[n]], outputs=[model_options[n], mm_models[n]], show_progress=False)
            mm_modes[n].change(fn=(lambda nd: lambda mode: [gr.update(info=merge_method_info[nd][mode]), gr.update(value="<h3>...</h3>")])(n), inputs=[mm_modes[n]], outputs=[mm_modes[n], recipe_all], show_progress=False)
            mm_use[n].change(fn=lambda use: gr.update(value="<h3>...</h3>"), inputs=mm_use[n], outputs=recipe_all, show_progress=False)

        def prepare_states(states, save_settings, calc_settings, *calcmodes):
            states["calcmodes"] = calcmodes
            states["save_settings"] = save_settings
            states["calc_settings"] = calc_settings

            return states

        generate_button = MM.components["img2img_generate" if is_img2img else "txt2img_generate"]
        generate_button.click(
            fn=prepare_states,
            inputs=[mm_states, save_settings, calc_settings, *mm_calcmodes],
            outputs=[mm_states],
            show_progress=False,
            queue=False,
        )

        return [enabled, model_a, base_model, mm_max_models, mm_finetune, mm_states, *mm_use, *mm_models, *mm_modes, *mm_alpha,
            *mbw_use_advanced, *mm_usembws, *mm_usembws_simple, *mm_weights, *mm_use_elemental, *mm_elementals]

    def modelmixer_extra_params(self, model_a, base_model, mm_max_models, mm_finetune, mm_states, *args_):
        num_models = int(mm_max_models)
        params = {
            "ModelMixer model a": model_a,
            "ModelMixer max models": mm_max_models,
        }

        if mm_finetune != "":
            params.update({"ModelMixer adjust": mm_finetune})
        if base_model is not None and len(base_model) > 0:
            params.update({"ModelMixer base model": base_model})

        _calcmodes = mm_states["calcmodes"]
        for j in range(num_models):
            name = f"{chr(98+j)}"
            params.update({f"ModelMixer use model {name}": args_[j]})

            if args_[num_models+j] != "None" and len(args_[num_models+j]) > 0:
                use_elemental = args_[num_models*8+j]
                if type(use_elemental) == str:
                    use_elemental = True if use_elemental == "True" else False
                params.update({
                    f"ModelMixer model {name}": args_[num_models+j],
                    f"ModelMixer merge mode {name}": args_[num_models*2+j],
                    f"ModelMixer alpha {name}": args_[num_models*3+j],
                    f"ModelMixer mbw mode {name}": args_[num_models*4+j],
                    f"ModelMixer use elemental {name}": use_elemental,
                    f"ModelMixer calcmode {name}": _calcmodes[j],
                })
                if len(args_[num_models*5+j]) > 0:
                    params.update({f"ModelMixer mbw {name}": ",".join(args_[num_models*5+j])})
                if len(args_[num_models*6+j]) > 0:
                    params.update({f"ModelMixer simple mbw {name}": ",".join(args_[num_models*6+j])})
                if len(args_[num_models*7+j]) > 0:
                    params.update({f"ModelMixer mbw weights {name}": args_[num_models*7+j]})

                if use_elemental:
                    elemental = args_[num_models*9+j]
                    elemental = elemental.strip()
                    if elemental != "":
                        elemental = elemental.replace(",", "\n").strip().split("\n")
                        elemental = [f.strip() for f in elemental]
                        elemental = ",".join(elemental)
                        params.update({f"ModelMixer elemental {name}": elemental})

        return params

    def before_process(self, p, enabled, model_a, base_model, mm_max_models, mm_finetune, mm_states, *args_):
        global permutation_spec

        if not enabled:
            return
        debugs = shared.opts.data.get("mm_debugs", ["elemental merge"])
        print("debugs = ", debugs)
        use_extra_elements = shared.opts.data.get("mm_use_extra_elements", True)
        print("use_extra_elements = ", use_extra_elements)

        base_model = None if base_model == "None" else base_model
        # extract model infos
        num_models = int(mm_max_models)
        mm_use = ["False"]*num_models
        mm_models = []
        mm_modes = []
        mm_calcmodes = []
        mm_alpha = []
        mbw_use_advanced = []
        mm_usembws = []
        mm_weights = []
        mm_use_elemental = []
        mm_elementals = []

        # save given model index
        modelindex = [None]*num_models

        FINETUNES = [ "IN","OUT", "OUT2", "CONT", "BRI", "COL1", "COL2", "COL3" ]

        # save xyz pinpoint block info.
        xyz_pinpoint_blocks = [{}]*num_models

        # cleanup mm_finetune
        mm_finetune = mm_finetune.strip()
        mm_finetune = "" if mm_finetune.rstrip(",0") == "" else mm_finetune

        # check xyz grid and override args_
        if hasattr(p, "modelmixer_xyz"):
            # prepare some variables
            fines = mm_finetune.split(",")
            if len(fines) < 7:
                fines += [0]*(7-len(fines))
            fines = fines[0:7]

            args = list(args_)
            for k, v in p.modelmixer_xyz.items():
                print("XYZ:",k,"->",v)
                if k == "model a":
                    model_a = p.modelmixer_xyz[k]
                elif k == "base model":
                    base_model = p.modelmixer_xyz[k]
                elif k == "adjust":
                    mm_finetune = p.modelmixer_xyz[k]
                    fines = mm_finetune.strip().split(",")[0:7]
                elif k == "pinpoint adjust":
                    pinpoint = p.modelmixer_xyz[k]
                    if pinpoint in FINETUNES:
                        idx = FINETUNES.index(pinpoint)
                        if "pinpoint alpha" in p.modelmixer_xyz:
                            alpha = p.modelmixer_xyz["pinpoint alpha"]
                        else:
                            raise RuntimeError(f"'Pinpoint adjust' needs 'pinpoint alpha' in another axis")
                        fines[idx] = str(alpha)
                    else:
                        raise RuntimeError(f"Invalid pinpoint adjust {pinpoint}")
                elif "pinpoint block" in k:
                    pinpoint = p.modelmixer_xyz[k]
                    j = k.rfind(" ")
                    name = k[j+1:] # model name: model b -> get "b"
                    idx = ord(name) - 98 # model index: model b -> get 0
                    if pinpoint in BLOCKID:
                        if f"pinpoint alpha {name}" in p.modelmixer_xyz:
                            alpha = p.modelmixer_xyz[f"pinpoint alpha {name}"]
                        else:
                            raise RuntimeError(f"Pinpoint block' needs 'pinpoint alpha {name}' in another axis")
                        # save pinpoint alpha to use later.
                        xyz_pinpoint_blocks[idx][pinpoint] = alpha

                        # insert pinpoint block into selected mbws blocks
                        mbw_use_advanced = args[num_models*4+idx]
                        usembws = args[num_models*5+idx]
                        usembws_simple = args[num_models*6+idx]
                        if not mbw_use_advanced:
                            usembws = usembws_simple
                        if pinpoint not in usembws:
                            usembws += [pinpoint]
                    else:
                        raise RuntimeError(f"Pinpoint block' name {pinpoint} not found")

                elif "pinpoint" not in k:
                    # extract model index, field name etc.
                    j = k.rfind(" ")
                    name = k[j+1:] # model name: model b -> get "b"
                    field = k[:j] # field name
                    idx = ord(name) - 98 # model index: model b -> get 0

                    if idx >= 0 and idx < num_models:
                        if field == "model":
                            args[num_models+idx] = v
                        elif field == "alpha":
                            args[num_models*3+idx] = v
                        elif field == "mbw alpha":
                            args[num_models*7+idx] = v
                        elif field == "elemental":
                            args[num_models*9+idx] = v
            # restore
            args_ = tuple(args)
            mm_finetune = ",".join([str(int(float(x))) if float(x) == int(float(x)) else str(x) for x in fines])

        for n in range(num_models):
            use = args_[n]
            if type(use) is str:
                use = True if use == "True" else False
            mm_use[n] = use

        if True not in mm_use and mm_finetune == "":
            # no selected merge models
            print("No selected models found")
            return

        _calcmodes = mm_states["calcmodes"]
        for j in range(num_models):
            if mm_use[j]:
                model = args_[num_models+j]
                mode = args_[num_models*2+j]
                alpha = args_[num_models*3+j]
                if type(alpha) == str: alpha = float(alpha)
                mbw_use_advanced = args_[num_models*4+j]
                usembws = args_[num_models*5+j]
                usembws_simple = args_[num_models*6+j]
                weights = args_[num_models*7+j]
                use_elemental = args_[num_models*8+j]
                if type(use_elemental) == str:
                    use_elemental = True if use_elemental == "True" else False
                elemental = args_[num_models*9+j]
                elemental = elemental.strip()
                if elemental != "":
                    elemental = elemental.replace(",", "\n").strip().split("\n")
                    elemental = [f.strip() for f in elemental]
                    elemental = ",".join(elemental)

                calcmode = _calcmodes[j]

                if not mbw_use_advanced:
                    usembws = usembws_simple

                model = None if model == "None" else model
                # ignore some cases
                if alpha == 0.0 and len(usembws) == 0:
                    continue
                if model is None:
                    continue

                # save original model index
                modelindex[len(mm_models)] = j

                mm_models.append(model)
                mm_modes.append(mode)
                mm_alpha.append(alpha)
                mm_usembws.append(usembws)
                mm_weights.append(weights)
                mm_use_elemental.append(use_elemental)
                mm_elementals.append(elemental)
                mm_calcmodes.append(calcmode)

        calc_settings = mm_states.pop("calc_settings", {})
        # exclude calc_settings from fake hash

        # extra_params
        extra_params = self.modelmixer_extra_params(model_a, base_model, mm_max_models, mm_finetune, mm_states, *args_)
        p.extra_generation_params.update(extra_params)

        # make hash different for xyz-grid
        xyz = None
        if hasattr(p, "modelmixer_xyz"):
            xyz = p.modelmixer_xyz
        # make a hash to cache results
        confighash = hashlib.sha256(json.dumps([model_a, base_model, mm_finetune, mm_elementals, mm_use_elemental, mm_models, mm_modes, mm_states, mm_alpha, mm_usembws, mm_weights, xyz]).encode("utf-8")).hexdigest()
        print("config hash = ", confighash)
        current = getattr(shared, "modelmixer_config", None)

        if shared.sd_model is not None and shared.sd_model.sd_checkpoint_info is not None:
            if shared.sd_model.sd_checkpoint_info.sha256 == confighash and sd_models.get_closet_checkpoint_match(shared.sd_model.sd_checkpoint_info.title) is not None:
                # already mixed
                print(f"  - use current mixed model {confighash}")
                return
            elif getattr(shared.sd_model.sd_checkpoint_info, "fake", False) and current and current.get("confighash", None) == confighash:
                print(f"  - use current mixed model {confighash}")
                return

        timer = Timer()
        print("  - mm_use", mm_use)
        print("  - model_a", model_a)
        print("  - base_model", base_model)
        print("  - max_models", mm_max_models)
        print("  - models", mm_models)
        print("  - modes", mm_modes)
        print("  - calcmodes", mm_calcmodes)
        print("  - usembws", mm_usembws)
        print("  - weights", mm_weights)
        print("  - alpha", mm_alpha)
        print("  - adjust", mm_finetune)
        print("  - use elemental", mm_use_elemental)
        print("  - elementals", mm_elementals)

        # parse elemental weights
        if "elemental merge" in debugs: print("  - Parse elemental merge...")
        all_elemental_blocks = []
        for j in range(len(mm_models)):
            elemental_ws = None
            if mm_use_elemental[j]:
                elemental_ws = parse_elemental(mm_elementals[j])
                if "elemental merge" in debugs: print(" Elemental merge wegiths = ", elemental_ws)
                if elemental_ws is not None:
                    mm_elementals[j] = elemental_ws
                    all_elemental_blocks = all_elemental_blocks + list(elemental_ws.keys())
            if elemental_ws is None:
                mm_elementals[j] = None

        # get all elemental blocks
        if len(all_elemental_blocks) > 0:
            all_elemental_blocks = set(all_elemental_blocks)
            if "elemental merge" in debugs: print(" Elemental: all elemental blocks = ", all_elemental_blocks)

        def selected_elemental_blocks(blocks, isxl):
            max_blocks = 26 - (0 if not isxl else 6)
            BLOCKIDS = BLOCKID if not isxl else BLOCKIDXL
            elemental_selected = [False] * max_blocks
            for j, b in enumerate(BLOCKIDS):
                if b in blocks:
                    elemental_selected[j] = True
            return elemental_selected

        mm_weights_orig = mm_weights

        # load model_a
        # check model_a
        checkpoint_info = sd_models.get_closet_checkpoint_match(model_a)
        if checkpoint_info is None:
            print(f"ERROR: Fail to get {model_a}")
            return
        model_a = checkpoint_info.model_name
        print(f"model_a = {model_a}")

        # load models
        models = {}

        def load_state_dict(checkpoint_info):
            # is it already loaded model?
            already_loaded = shared.sd_model.sd_checkpoint_info if shared.sd_model is not None else None
            if already_loaded is not None and already_loaded.title == checkpoint_info.title:
                # model is already loaded
                print(f"Loading {checkpoint_info.title} from loaded model...")

                return get_current_state_dict()[0]

            # get cached state_dict
            if shared.opts.sd_checkpoint_cache > 0:
                # call sd_models.get_checkpoint_state_dict() without calculate_shorthash() for fake checkpoint_info
                if checkpoint_info in checkpoints_loaded:
                    # use checkpoint cache
                    print(f"Loading weights {checkpoint_info.title} from cache")
                    state_dict = checkpoints_loaded[checkpoint_info]
                    # check validity of cached state_dict
                    keylen = len(state_dict.keys())
                    if keylen < 686: # for SD-v1, SD-v2
                        print(f"Invalid cached state_dict...")
                    else:
                        return {k: v.cpu() for k, v in state_dict.items()}

            if not os.path.exists(checkpoint_info.filename):
                # this is a fake checkpoint_info
                raise RuntimeError(f"No cached checkpoint found for {checkpoint_info.title}")

            # read state_dict from file
            print(f"Loading from file {checkpoint_info.filename}...")
            return sd_models.read_state_dict(checkpoint_info.filename, map_location = "cpu").copy()

        models['model_a'] = load_state_dict(checkpoint_info)

        # check SDXL
        isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in models['model_a']
        recheck_xl = "model.diffusion_model.input_blocks.11.0.out_layers.3.weight" not in models['model_a']
        if recheck_xl and not isxl:
            print(f"WARN: Loaded SDXL from shared.sd_model has size mismatch. Loading again from file {checkpoint_info.filename}...")
            del models['model_a']
            models['model_a'] = sd_models.read_state_dict(checkpoint_info.filename, map_location = "cpu").copy()
            isxl = True

        print("isxl =", isxl)

        # get all selected elemental blocks
        elemental_selected = []
        if len(all_elemental_blocks) > 0:
            elemental_selected = selected_elemental_blocks(all_elemental_blocks, isxl)

        # prepare for merges
        compact_mode = None
        mm_selected = [[]] * num_models
        for j, model in enumerate(mm_models):
            if len(mm_usembws[j]) > 0:
                # normalize Merge block weights
                mm_weights[j], compact_mbws, mm_selected[j] = calc_mbws(mm_weights[j], mm_usembws[j], isxl=isxl)
                compact_mode = True if compact_mode is None else compact_mode
            else:
                compact_mode = False
                max_blocks = 26 - (0 if not isxl else 6)
                mm_weights[j] = [mm_alpha[j]] * max_blocks

        # fix mm_weights to use poinpoint blocks xyz
        for j in range(len(mm_models)):
            # get original model index
            n = modelindex[j]
            BLOCKS = BLOCKID if not isxl else BLOCKIDXL
            if len(xyz_pinpoint_blocks[n]) > 0:
                for pin in xyz_pinpoint_blocks[n].keys():
                    if pin in BLOCKS:
                        mm_weights[j][BLOCKS.index(pin)] = xyz_pinpoint_blocks[n][pin]
                    else:
                        print("WARN: No pinpoint block found. ignore...")

        # get overall selected blocks
        if compact_mode:
            max_blocks = 26 - (0 if not isxl else 6)
            selected_blocks = []
            mm_selected_all = [False] * max_blocks
            for j in range(len(mm_models)):
                for k in range(max_blocks):
                    mm_selected_all[k] = mm_selected_all[k] or mm_selected[j][k]

            # add elemental_selected blocks
            if len(elemental_selected) > 0:
                for k in range(max_blocks):
                    mm_selected_all[k] = mm_selected_all[k] or elemental_selected[k]

            all_blocks = _all_blocks(isxl)
            BLOCKIDS = BLOCKID if not isxl else BLOCKIDXL

            # get all blocks affected by same perm groups by rebasin merge
            if not isxl and "Rebasin" in mm_calcmodes:
                print("check affected permutation blocks by rebasin merge...")
                jj = 0
                while True:
                    xx_selected_all = _get_rebasin_blocks(mm_selected_all, isxl)
                    changed = [BLOCKIDS[i] for i, v in enumerate(mm_selected_all) if v != xx_selected_all[i]]
                    if len(changed) > 0:
                        print(f" - [{jj+1}] {changed} block{'s' if len(changed) > 1 else ''} added")
                        mm_selected_all = xx_selected_all
                        jj += 1
                    else:
                        break

            for k in range(max_blocks):
                if mm_selected_all[k]:
                    selected_blocks.append(all_blocks[k])
        else:
            # no compact mode, get all blocks
            selected_blocks = _all_blocks(isxl)

        print("compact_mode = ", compact_mode)

        # check base_model
        model_base = {}
        if True in mm_use and "Add-Diff" in mm_modes:
            if base_model is None:
                # check SD version
                if not isxl:
                    w = models['model_a']["model.diffusion_model.input_blocks.1.1.proj_in.weight"]
                    if len(w.shape) == 4:
                        candidates = [ "v1-5-pruned-emaonly", "v1-5-pruned-emaonly.safetensors [6ce0161689]", "v1-5-pruned-emaonly.ckpt [cc6cb27103]",
                                    "v1-5-pruned.safetensors [1a189f0be6]", "v1-5-pruned.ckpt [e1441589a6]" ]
                    else:
                        candidates = [ "v2-1_768-nonema-pruned", "v2-1_768-nonema-pruned.safetensors [ff144a4984]", "v2-1_768-nonema-pruned.ckpt [4711ff4dd2]",
                                    "v2-1_768-ema-pruned.safetensors [dcd690123c]", "v2-1_768-ema-pruned.ckpt [ad2a33c361]"]

                    for a in candidates:
                        check = sd_models.get_closet_checkpoint_match(a)
                        if check is not None:
                            base_model = a
                            break
                else:
                    candidates = [ "sd_xl_base_1.0", "sd_xl_base_1.0_0.9vae",
                        "sd_xl_base_1.0.safetensors [31e35c80fc4]", "sd_xl_base_1.0_0.9vae.safetensors [e6bb9ea85b]", "sdXL_v10VAEFix.safetensors [e6bb9ea85b]" ]
                    for a in candidates:
                        check = sd_models.get_closet_checkpoint_match(a)
                        if check is not None:
                            base_model = a
                            break
                if base_model is not None:
                    print(f"base_model automatically detected as {base_model}")
            if base_model is None:
                raise Exception('No base model selected and automatic detection failed')

            checkpointinfo = sd_models.get_closet_checkpoint_match(base_model)
            model_base = sd_models.read_state_dict(checkpointinfo.filename, map_location = "cpu")

        # setup selected keys
        theta_0 = {}
        keys = []
        keyremains = []
        if compact_mode:
            # get keylist of all selected blocks
            base_prefix = "cond_stage_model." if not isxl else "conditioner."
            for k in models['model_a'].keys():
                keyadded = False
                for s in selected_blocks:
                    if s not in ["cond_stage_model.", "conditioner."]:
                        s = f"model.diffusion_model.{s}"
                    if s in k:
                        # ignore all non block releated keys
                        if "diffusion_model." not in k and base_prefix not in k:
                            continue
                        keys.append(k)
                        theta_0[k] = models['model_a'][k]
                        keyadded = True
                if not keyadded:
                    keyremains.append(k)

            # add some missing extra_elements
            last_block = "output_blocks.11." if not isxl else "output_blocks.8."
            if use_extra_elements and (last_block in selected_blocks) or ("" in all_elemental_blocks):
                selected_blocks += [ "time_embed.", "out." ]
                for el in [ "time_embed.0.bias", "time_embed.0.weight", "time_embed.2.bias", "time_embed.2.weight", "out.0.bias", "out.0.weight", "out.2.bias", "out.2.weight" ]:
                    k = f"model.diffusion_model.{el}"
                    keys.append(k)
                    theta_0[k] = models['model_a'][k]
                    if k in keyremains:
                        j = keyremains.index(k)
                        del keyremains[j]

        else:
            # get all keys()
            keys = list(models['model_a'].keys())
            theta_0 = models['model_a'].copy()

        # check finetune
        if mm_finetune.rstrip(",0") != "":
            fines = fineman(mm_finetune, isxl)
            if fines is not None:
                for tune_block in [ "input_blocks.0.", "out."]:
                    if tune_block not in selected_blocks:
                        selected_blocks += [ tune_block ]

                for key in tunekeys:
                    if key not in keys:
                        keyremains.append(key)

        # prepare metadata
        metadata = { "format": "pt" }
        merge_recipe = {
            "type": "sd-webui-model-mixer",
            "blocks": print_blocks(selected_blocks),
            "mbw": compact_mode,
            "weights_alpha": mm_weights,
            "weights_alpha_orig": mm_weights_orig,
            "alpha": mm_alpha,
            "model_a": model_a,
            "mode": mm_modes,
            "calcmode": mm_calcmodes,
        }
        metadata["sd_merge_models"] = {}

        def add_model_metadata(checkpoint_name):
            checkpointinfo = sd_models.get_closet_checkpoint_match(checkpoint_name)
            if checkpointinfo is None:
                return
            metadata["sd_merge_models"][checkpointinfo.sha256] = {
                "name": checkpoint_name,
                "legacy_hash": checkpointinfo.hash,
                "sd_merge_recipe": checkpointinfo.metadata.get("sd_merge_recipe", None)
            }

        if model_a:
            add_model_metadata(model_a)
        if base_model is not None:
            add_model_metadata(base_model)

        # non block section U-Net elements
        extra_elements = {
            "time_embed.0.bias":-1,
            "time_embed.0.weight":-1,
            "time_embed.2.bias":-1,
            "time_embed.2.weight":-1,
            "out.0.bias":-1,
            "out.0.weight":-1,
            "out.2.bias":-1,
            "out.2.weight":-1,
        }

        # merge functions
        def _weighted_sum(theta0, theta1, alpha):
            return (1 - alpha) * theta0 + alpha * theta1

        def _torch_lerp(theta0, theta1, alpha):
            return torch.lerp(theta0.to(torch.float32), theta1.to(torch.float32), alpha).to(theta0.dtype)

        def add_difference(theta0, theta1, base, alpha):
            return theta0 + (theta1 - base) * alpha

        # merge main
        weight_start = 0
        # total stage = number of models + key uninitialized stage + key remains stage
        stages = len(mm_models) + 1 + (1 if len(keyremains) > 0 else 0)
        modes = mm_modes
        calcmodes = mm_calcmodes

        # model info
        modelinfos = [ model_a ]
        if checkpoint_info.shorthash is None:
            # this is a newly added checkpoint file.
            checkpoint_info.calculate_shorthash()
        modelhashes = [ checkpoint_info.shorthash ]
        alphas = []

        # get current model config and check difference to support partial update
        partial_update = False
        changed_keys = None

        if current is None and shared.sd_model is not None:
            # no current mixed model but model_a == shared.sd_model case.
            info = shared.sd_model.sd_checkpoint_info
            if info == checkpoint_info:
                current = { "hashes": [ checkpoint_info.shorthash ], "weights": [], "adjust": "" }

        use_unet_partial_update = shared.opts.data.get("mm_use_unet_partial_update", False)
        if use_unet_partial_update and current is not None:
            # check same models used
            hashes = current["hashes"]
            # only the first model need to be checked
            first_model_is_the_same = False

            # check model_a
            info = sd_models.get_closet_checkpoint_match(model_a)
            if info is not None:
                if info.shorthash is None:
                    info.calculate_shorthash()
                if hashes[0] == info.shorthash:
                    first_model_is_the_same = True

                if first_model_is_the_same:
                    print(" - check possible UNet partial update...")
                    max_blocks = 26 - (0 if not isxl else 6)

                    # check changed weights
                    weights = current["weights"]
                    changed = [False] * max_blocks

                    for j in range(max(len(mm_models), len(hashes[1:]))):
                        if len(hashes[1:]) > j and len(mm_models) > j:
                            model = mm_models[j]
                        else: # some models are changed
                            model = None

                        same_model = True
                        if model is None:
                            same_model = False
                        else:
                            info = sd_models.get_closet_checkpoint_match(model)
                            if info is not None:
                                if info.shorthash is None:
                                    info.calculate_shorthash()

                                # is it different model?
                                if hashes[j + 1] != info.shorthash:
                                    same_model = False
                            else:
                                same_model = False

                        if same_model and (current["calcmode"][j] != mm_calcmodes[j] or current["mode"][j] != mm_modes[j]):
                            # calc method or calc mode changed
                            same_model = False

                        if same_model:
                            # check modified block weighs
                            changed |= np.array(weights[j][:max_blocks]) != np.array(mm_weights[j][:max_blocks])
                        else:
                            # check all non zero blocks
                            if len(weights) > j:
                                changed |= np.array(weights[j][:max_blocks]) != np.array([0.0]*max_blocks)
                            if len(mm_weights) > j:
                                changed |= np.array(mm_weights[j][:max_blocks]) != np.array([0.0]*max_blocks)

                    BLOCKIDS = BLOCKID if not isxl else BLOCKIDXL
                    print(" - partial changed blocks = ", [BLOCKIDS[k] for k, b in enumerate(changed) if b])
                    all_blocks = _all_blocks(isxl)
                    weight_changed_blocks = []
                    for j, b in enumerate(changed):
                        # recalculate all elemental blocks
                        if len(elemental_selected) > 0:
                            b |= elemental_selected[j]
                        if b:
                            weight_changed_blocks.append(all_blocks[j])

                    # check ".out.", ".time_embed." elemental blocks
                    if "" in all_elemental_blocks:
                        # always update elemental blocks
                        weight_changed_blocks += ["time_embed.", "out."]

                    # check finetune
                    finetune_changed = current["adjust"] != mm_finetune

                    if finetune_changed:
                        # add "input_blocks.0.", "out." for finetune
                        weight_changed_blocks.append("out.")
                        if "input_blocks.0." not in weight_changed_blocks:
                            weight_changed_blocks.append("input_blocks.0.")

                    weight_changed = {}
                    if len(weight_changed_blocks) > 0:
                        # get changed keys
                        for k in keys:
                            for s in weight_changed_blocks:
                                if s not in ["cond_stage_model.", "conditioner."]:
                                    ss = f"model.diffusion_model.{s}"
                                else:
                                    ss = s
                                if ss in k:
                                    weight_changed[s] = weight_changed.get(s, [])
                                    weight_changed[s].append(k)
                                    break

                        tmp_keys = [[*weight_changed[s]] for s in weight_changed.keys()]
                        changed_keys = []
                        for k in tmp_keys: changed_keys += k

                        # partial updatable case
                        partial_update = True
                        print(" - UNet partial update mode")
                    else:
                        print(" - No change blocks detected.")

        # check Rebasin mode
        if not isxl and "Rebasin" in calcmodes:
            fullmatching = "fastrebasin" not in calc_settings
            print(" - Dynamic loading rebasin module...")
            load_module(os.path.join(scriptdir, "scripts", "rebasin", "weight_matching.py"))
            from scripts.rebasin.weight_matching import weight_matching, apply_permutation, sdunet_permutation_spec

            print("Rebasin mode")

            if permutation_spec is None:
                # one time initializer
                permutation_spec = sdunet_permutation_spec()

            device = get_device()
            usefp16 = True
            if "CPU" in calc_settings:
                device = "cpu"
            elif "GPU" in calc_settings:
                device = "cuda"

            print(" - Calulation device for Rebasin is ", device)

            laplib = shared.opts.data.get("mm_laplib", "lap")
            print(" - LAP library is", laplib)

        # check to use torch.lerp()
        use_fast_weighted_sum = shared.opts.data.get("mm_use_fast_weighted_sum", True)
        if use_fast_weighted_sum:
            weighted_sum = _torch_lerp
        else:
            weighted_sum = _weighted_sum


        # set job_count
        save_jobcount = None
        if shared.state.job_count == -1:
            save_jobcount = shared.state.job_count
            shared.state.job_count = 0
            if len(mm_models) > 0 and len(keys) > 0:
                shared.state.job_count += len(mm_models)
                if not isxl and "Rebasin" in calcmodes:
                    shared.state.job_count += 1

        sel_keys = changed_keys if partial_update else keys

        # save some dicts
        checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids", "conditioner.embedders.1.model.transformer.text_model.embeddings.position_ids" ]
        for k in checkpoint_dict_skip_on_merge:
            if k in sel_keys:
                sel_keys.remove(k)
                item = theta_0.pop(k)
                keyremains.append(k)

        timer.record("prepare")
        stage = 1
        theta_1 = None
        checkpointinfo = checkpoint_info # checkpointinfo of model_a
        for n, file in enumerate(mm_models,start=weight_start):
            checkpointinfo1 = sd_models.get_closet_checkpoint_match(file)
            if checkpointinfo1 is None:
                raise RuntimeError(f"No checkpoint found for {file}")

            model_name = checkpointinfo1.model_name
            if checkpointinfo != checkpointinfo1:
                print(f"Loading model {model_name}...")
                theta_1 = load_state_dict(checkpointinfo1)
                checkpointinfo = checkpointinfo1
            else:
                print("use already loaded model...")
                theta_1 = theta_1 if theta_1 is not None else models["model_a"]

            model_b = f"model_{chr(97+n+1-weight_start)}"
            merge_recipe[model_b] = model_name
            modelinfos.append(model_name)
            if checkpointinfo.shorthash is None:
                checkpointinfo.calculate_shorthash()
            modelhashes.append(checkpointinfo.shorthash)

            # add metadata
            add_model_metadata(model_name)

            usembw = len(mm_usembws[n]) > 0
            if not usembw:
                alpha = mm_alpha[n]
                alphas.append([alpha])
                print(f"mode = {modes[n]}, alpha = {alpha}")
            else:
                alphas.append(mm_weights[n])
                print(f"mode = {modes[n]}, mbw mode, alpha = {mm_weights[n]}")

            # main routine
            shared.state.sampling_steps = len(sel_keys)
            shared.state.sampling_step = 0
            for key in (tqdm(sel_keys, desc=f"Stage #{stage}/{stages}")):
                shared.state.sampling_step += 1
                if "model_" in key:
                    continue
                if key in checkpoint_dict_skip_on_merge:
                    continue
                if "model" in key and key in theta_1:
                    if usembw:
                        i = _weight_index(key, isxl=isxl)
                        if i == -1:
                            if use_extra_elements and any(s in key for s in extra_elements.keys()):
                                # FIXME
                                i = -1
                            else:
                                continue # not found
                        alpha = mm_weights[n][i]

                        # check elemental merge weights
                        if mm_elementals[n] is not None:
                            if i < 0:
                                name = "" # empty block name -> extra elements
                            else:
                                name = BLOCKID[i] if not isxl else BLOCKIDXL[i]
                            ws = mm_elementals[n].get(name, None)
                            new_alpha = None
                            if ws is not None:
                                for j, w in enumerate(ws):
                                    flag = w["flag"]
                                    elem = w["elements"]
                                    if flag and any(item in key for item in w["elements"]):
                                        new_alpha = w['ratio']
                                        if "elemental merge" in debugs: print(f' - Elemental: merge weights[{j}] - {key} -', key, w["elements"], new_alpha)
                                    elif not flag and all(item not in key for item in w["elements"]):
                                        new_alpha = w['ratio']
                                        if "elemental merge" in debugs: print(f' - Elemental: merge weights[{j}] - {key} - NOT', key, w["elements"], new_alpha)

                            # apply elemental merging weight ratio
                            if new_alpha is not None:
                                alpha = new_alpha

                    if "Sum" in modes[n]:
                        if alpha == 1.0:
                            theta_0[key] = theta_1[key]
                        elif alpha != 0.0:
                            theta_0[key] = weighted_sum(theta_0[key], theta_1[key], alpha)
                    else:
                        if alpha != 0.0:
                            theta_0[key] = add_difference(theta_0[key], theta_1[key], model_base[key], alpha)

            shared.state.nextjob()

            if n == weight_start:
                stage += 1
                for key in (tqdm(sel_keys, desc=f"Check uninitialized #{n+2-weight_start}/{stages}")):
                    if "model" in key:
                        for s in selected_blocks:
                            if s not in ["cond_stage_model.", "conditioner."]:
                                s = f"model.diffusion_model.{s}"
                            if s in key and key not in theta_0 and key not in checkpoint_dict_skip_on_merge:
                                print(f" +{k}")
                                theta_0[key] = theta_1[key]

            if not isxl and "Rebasin" in calcmodes[n]:
                print("Rebasin calc...")
                # rebasin mode
                # Replace theta_0 with a permutated version using model A and B
                first_permutation, y = weight_matching(permutation_spec, models["model_a"], theta_0, usefp16=usefp16, device=device, full=fullmatching, lap=laplib)
                theta_0 = apply_permutation(permutation_spec, first_permutation, theta_0)
                #second_permutation, z = weight_matching(permutation_spec, theta_1, theta_0, usefp16=usefp16, device=device, full=fullmatching, lap=laplib)
                #theta_3= apply_permutation(permutation_spec, second_permutation, theta_0)
                shared.state.nextjob()

            stage += 1

        del theta_1
        timer.record("merging")

        def make_recipe(modes, model_a, models):
            weight_start = 0
            recipe_all = model_a
            for n, file in enumerate(models, start=weight_start):
                checkpointinfo = sd_models.get_closet_checkpoint_match(file)
                model_name = checkpointinfo.model_name
                if model_name.find(" ") > -1: model_name = f"({model_name})"

                # recipe string
                if "Sum" in modes[n]:
                    if recipe_all.find(" ") > -1: recipe_all = f"({recipe_all})"
                    recipe_all = f"{recipe_all} * (1 - alpha_{n}) + {model_name} * alpha_{n}"
                elif modes[n] in [ "Add-Diff" ]:
                    recipe_all = f"{recipe_all} + ({model_name} - {base_model}) * alpha_{n}"

            return recipe_all

        if save_jobcount is not None and save_jobcount == -1:
            # restore job_countm job_no
            shared.state.job_count = -1
            shared.state.job_no = 0

        # full recipe
        recipe_all = make_recipe(modes, model_a, mm_models)

        # store unmodified remains
        for key in (tqdm(keyremains, desc=f"Save unchanged weights #{stages}/{stages}")):
            theta_0[key] = models['model_a'][key]

        # check for partial update
        if partial_update and len(weight_changed_blocks) > 0:
            # get empty changed blocks. it means, these keys are in the keyremains.
            remains_blocks = []
            for s in weight_changed_blocks:
                weight_changed[s] = weight_changed.get(s, [])
                if len(weight_changed[s]) == 0:
                    remains_blocks.append(s)

            # FIXME ineffective method. search all keys.
            if len(remains_blocks) > 0:
                for k in keyremains:
                    for remain in remains_blocks:
                        if remain not in ["cond_stage_model.", "conditioner."]:
                            r = f"model.diffusion_model.{remain}"
                        else:
                            r = remain
                        if r in k:
                            weight_changed[remain] = weight_changed.get(remain, [])
                            weight_changed[remain].append(k)
                            break

        # apply finetune
        if mm_finetune.rstrip(",0") != "":
            fines = fineman(mm_finetune, isxl)
            if fines is not None:
                old_finetune = shared.opts.data.get("mm_use_old_finetune", False)
                print(f"Apply fine tune {fines}")
                if old_finetune: print(" - Old adjust")
                for i, key in enumerate(tunekeys):
                    if i == 5:
                        theta_0[key] = theta_0[key] + torch.tensor(fines[5])
                    elif fines[i] != 1.0:
                        theta_0[key] = theta_0[key] * fines[i]

        # save recipe
        alphastr = ','.join(['(' + ','.join(map(lambda x: str(int(x)) if x == int(x) else str(x), sub)) + ')' for sub in alphas])
        full_recipe_str = f"{recipe_all}{alphastr}"
        if mm_finetune != "":
            full_recipe_str += '@' + mm_finetune
        merge_recipe["recipe"] = full_recipe_str
        metadata["sd_merge_recipe"] = merge_recipe
        model_name = full_recipe_str.replace("*", "x")

        # load theta_0, checkpoint_info was used for model_a
        # XXX HACK make a FAKE checkpoint_info
        def fake_checkpoint(checkpoint_info, metadata, model_name, sha256, fake=True):
            # XXX HACK
            # change model name (name_for_extra field used webui internally)
            if fake:
                checkpoint_info = deepcopy(checkpoint_info)
            checkpoint_info.name_for_extra = model_name

            checkpoint_info.sha256 = sha256
            checkpoint_info.name = f"{model_name}.safetensors"
            checkpoint_info.model_name = checkpoint_info.name_for_extra.replace("/", "_").replace("\\", "_")
            checkpoint_info.title = f"{checkpoint_info.name} [{sha256[0:10]}]"
            # without set checkpoint_info.shorthash, load_model() will call calculate_shorthash() and register()
            # simply ignore legacy hash
            checkpoint_info.hash = None
            # use new metadata
            checkpoint_info.metadata = metadata

            # add new attribute
            checkpoint_info.fake = True

            # XXX HACK use any valid filename to trick checkpoint_info.calculate_shorthash()
            if not os.path.exists(checkpoint_info.filename):
                for title in sd_models.checkpoint_tiles():
                    info = sd_models.get_closet_checkpoint_match(title)
                    if info is not None and os.path.exists(info.filename):
                        checkpoint_info.filename = info.filename
                        break

            # XXX add a fake checkpoint_info
            # force to set with a new sha256 hash
            hashes = cache("hashes")
            hashes[f"checkpoint/{checkpoint_info.name}"] = {
                "mtime": os.path.getmtime(checkpoint_info.filename),
                "sha256": sha256,
            }
            dump_cache()

            # XXX hack. set ids for a fake checkpoint info
            checkpoint_info.ids = [checkpoint_info.model_name, checkpoint_info.name, checkpoint_info.name_for_extra]
            return checkpoint_info

        # fix/check bad CLIP ids
        fixclip(theta_0, mm_states["save_settings"], isxl)

        timer.record("merging")
        print(f' - merge processing in {timer.summary()}.')
        sha256 = None
        t = Timer()
        if not partial_update and shared.opts.data.get("mm_use_precalculate_hash", False):
            sha256 = precalculate_safetensors_hashes(theta_0, metadata.copy(), mm_states["save_settings"], isxl)
            print(" - precalculated hash = ", sha256)
            t.record("precalculate hash")

        if not partial_update and "save model" in debugs:
            save_settings = shared.opts.data.get("mm_save_model", ["safetensors", "fp16"])
            save_filename = shared.opts.data.get("mm_save_model_filename", "modelmixer-[hash].safetensors")
            save_filename = save_filename.replace("[hash]", f"{sha256[0:10] if sha256 else confighash[0:10]}").replace("[model_name]", f"{model_name}")
            save_current_model(save_filename, "None", save_settings, ["merge recipe"], state_dict=theta_0, metadata=metadata.copy())
            t.record("save model")
        if t.total > 0:
            print(f' - post process in {t.summary()}.')

        # partial update
        state_dict = theta_0.copy()
        if partial_update:
            # in this case, use sd_model's checkpoint_info
            checkpoint_info = shared.sd_model.sd_checkpoint_info
            # copy old aliases ids
            old_ids = checkpoint_info.ids.copy()
            # change info without using deepcopy() if checkpoint.filename already exists
            if os.path.exists(checkpoint_info.filename):
                make_fake = True
            else:
                make_fake = False
            checkpoint_info = fake_checkpoint(checkpoint_info, metadata, model_name, sha256 if sha256 else confighash, make_fake)

            # check lora_patches
            lora_patch = False
            try:
                print(" - loading scripts.patches...")
                load_module(os.path.join(scriptdir, "scripts", "patches.py"))
                from scripts.patches import StateDictPatches

                patch = StateDictPatches()
                print(" - lora patch")
                lora_patch = True
            except Exception:
                print(" - Fail to patch lora")
                pass
            # to cpu ram
            sd_unet.apply_unet("None")
            sd_models.send_model_to_cpu(shared.sd_model)
            sd_hijack.model_hijack.undo_hijack(shared.sd_model)

            if "cond_stage_model." in weight_changed_blocks or "conditioner." in weight_changed_blocks:
                # Textencoder(BASE)
                if isxl:
                    prefix = "conditioner."
                else:
                    prefix = "cond_stage_model."
                base_dict = {}
                for k in weight_changed[prefix]:
                    # remove prefix, 'cond_stage_model.' or 'conditioner.' will be removed
                    key = k[len(prefix):]
                    base_dict[key] = theta_0[k]
                if isxl:
                    shared.sd_model.conditioner.load_state_dict(base_dict, strict=False)
                else:
                    shared.sd_model.cond_stage_model.load_state_dict(base_dict, strict=False)
                print(" - \033[92mTextencoder(BASE) has been successfully updated\033[0m")

            # get unet_blocks_map
            unet_map = unet_blocks_map(shared.sd_model.model.diffusion_model, isxl)

            # partial update unet blocks state_dict
            unet_updated = 0
            for s in weight_changed_blocks:
                if s in ["cond_stage_model.", "conditioner."]:
                    # Textencoder(BASE)
                    continue
                print(" - update UNet block", s)
                unet_dict = unet_map[s].state_dict()
                prefix = f"model.diffusion_model.{s}"
                for k in weight_changed[s]:
                    # remove block prefix, 'model.diffusion_model.input_blocks.0.' will be removed
                    key = k[len(prefix):]
                    unet_dict[key] = theta_0[k]
                unet_map[s].load_state_dict(unet_dict)
                unet_updated += 1
            if unet_updated > 0:
                print(" - \033[92mUNet partial blocks have been successfully updated\033[0m")

            # textencoder partial update does not work as expected. read state_dict() and set state_dict.
            if "cond_stage_model." in weight_changed_blocks or "conditioner." in weight_changed_blocks:
                print(" - \033[93mReload full state_dict...\033[0m")
                state_dict = shared.sd_model.state_dict().copy()
            else:
                state_dict = None

                # restore to gpu
                sd_models.send_model_to_device(shared.sd_model)
                sd_hijack.model_hijack.hijack(shared.sd_model)

                sd_models.model_data.set_sd_model(shared.sd_model)
                sd_unet.apply_unet()

            if lora_patch:
                patch.undo()
                del patch

            if state_dict is None:
                # after partial updated
                # update checkpoint_aliases, normally done in the load_model()
                # HACK, FIXME
                # manually remove old aliasses
                for id in old_ids:
                    sd_models.checkpoint_aliases.pop(id, None)

                # set shorthash
                checkpoint_info.shorthash = sha256[0:10] if sha256 else confighash[0:10]
                # manually add aliasses
                checkpoint_info.ids += [checkpoint_info.title]
                checkpoint_info.register()

                # update shared.*
                shared.sd_model.sd_checkpoint_info = checkpoint_info

                shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
                shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256
                shared.sd_model.sd_model_hash = checkpoint_info.shorthash

        if state_dict is not None:
          checkpoint_info = fake_checkpoint(checkpoint_info, metadata, model_name, sha256 if sha256 else confighash)
          if shared.sd_model is not None and hasattr(shared.sd_model, 'lowvram') and shared.sd_model.lowvram:
            print("WARN: lowvram/medvram load_model() with minor workaround")
            sd_models.unload_model_weights()
            #sd_models.model_data.__init__()

          if sd_models.model_data.sd_model:
            sd_models.send_model_to_cpu(sd_models.model_data.sd_model)
            sd_models.model_data.sd_model = None

            # the follow procedure normally called at the reuse_model_from_already_loaded()
            # manully check shared.opts.sd_checkpoints_limit here before call load_model()
            if len(sd_models.model_data.loaded_sd_models) > shared.opts.sd_checkpoints_limit > 0:
                print(f"Unloading model {len(sd_models.model_data.loaded_sd_models)} over the limit of {shared.opts.sd_checkpoints_limit}...")
                while len(sd_models.model_data.loaded_sd_models) > shared.opts.sd_checkpoints_limit:
                    loaded_model = sd_models.model_data.loaded_sd_models.pop()
                    print(f" - model {len(sd_models.model_data.loaded_sd_models)}: {loaded_model.sd_checkpoint_info.title}")
                    sd_models.send_model_to_trash(loaded_model)
            devices.torch_gc()

          sd_models.load_model(checkpoint_info=checkpoint_info, already_loaded_state_dict=state_dict)
          del state_dict

        devices.torch_gc()

        # XXX fix checkpoint_info.filename
        filename = os.path.join(model_path, f"{model_name}.safetensors")
        shared.sd_model.sd_model_checkpoint = checkpoint_info.filename = filename

        if not partial_update and shared.opts.sd_checkpoint_cache > 0:
            # FIXME for partial updated case
            # check checkponts_loaded bug
            # unload cached merged model
            saved_state_dict = checkpoints_loaded[checkpoint_info]
            if len(saved_state_dict.keys()) < 600:
                print("sd-webui bug workaround!")
                checkpoints_loaded[checkpoint_info] = theta_0.copy()

        del theta_0

        # update merged model info.
        shared.modelmixer_config = {
            "hash": sha256 if sha256 else confighash,
            "confighash": confighash,
            "models" : modelinfos,
            "hashes" : modelhashes,
            "model_a": model_a,
            "weights": mm_weights,
            "alpha": mm_alpha,
            "mode": mm_modes,
            "calcmode": mm_calcmodes,
            "selected": selected_blocks,
            "adjust": mm_finetune,
            "elemental": mm_elementals,
            "recipe": recipe_all + alphastr,
        }
        return

def precalculate_safetensors_hashes(state_dict, metadata, save_settings, isxl, fixheader=True):
    import safetensors

    if "sd_merge_models" in metadata:
        metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"], separators=(',', ':'))
    if "sd_merge_recipe" in metadata:
        metadata["sd_merge_recipe"] = json.dumps(metadata["sd_merge_recipe"], separators=(',', ':'))

    if "fp16" in save_settings:
        state_dict = to_half(state_dict, True)
    if "prune" in save_settings:
        state_dict = prune_model(state_dict, isxl)

    hash_sha256 = hashlib.sha256()
    bytes = safetensors.torch.save(state_dict, metadata)
    b = io.BytesIO(bytes)

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    if n > 2 and fixheader:
        # fix header for old safetensors
        start = b.read(2)
        if start in (b'{"', b"{'"):
            l = 0
            meta = start
            metatag = b.read(12)
            if metatag == b"__metadata__":
                l += 2 + 12
                meta += metatag
                tag = 0;

                while True:
                    c = b.read(1)
                    l += 1
                    meta += c
                    if c == b"{":
                        tag += 1
                    elif c == b"}":
                        tag -= 1
                        c = b.read(1)
                        l += 1
                        if tag == 0 and c == b",":
                            break
                        elif c == b"}":
                            tag -= 1
                        meta += c

                readlen = l - 2 - 12 - 2 - 1
                parsed = json.loads(meta + b"}")
                assert(metadata == parsed["__metadata__"])
                # fix metadata
                b.seek(8 + 2 + 12 + 2)
                s = json.dumps(metadata, separators=(',', ':'))
                b.write(s.encode("utf-8"))
                if readlen > len(s):
                    print("fill remains")
                    # fill possible remains with spaces
                    fill = b" " * (readlen - len(s))
                    b.write(fill)

    # calculate sha256 hash
    b.seek(0)
    blksize = 1024 * 1024

    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def fixclip(theta_0, settings, isxl):
    """fix/check bad CLIP ids"""
    base_prefix = "cond_stage_model." if not isxl else "conditioner."
    position_id_key = f"{base_prefix}transformer.text_model.embeddings.position_ids"
    if position_id_key in theta_0:
        correct = torch.tensor([list(range(77))], dtype=torch.int64, device="cpu")
        current = theta_0[position_id_key].to(torch.int64)
        broken = correct.ne(current)
        broken = [i for i in range(77) if broken[0][i]]
        if len(broken) != 0:
            if "fix CLIP ids" in settings:
                theta_0[position_id_key] = correct
                print(f"Fixed broken clip\n{broken}")
            else:
                print(f"Broken clip!\n{broken}")
        else:
            print("Clip is fine")


# fine tune (from supermerger)
tunekeys = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.input_blocks.0.0.bias",

    "model.diffusion_model.out.0.weight",
    "model.diffusion_model.out.0.bias",

    "model.diffusion_model.out.2.weight",
    "model.diffusion_model.out.2.bias",
]

# from supermerger, CD-Tunner's method
COLS = [[-1, 1/3, 2/3], [1, 1, 0], [0, -1, -1], [1, 0, 1]]
COLSXL = [[0, 0, 1], [1, 0, 0], [-1, -1, 0], [-1, 1, 0]]

def colorcalc(cols, isxl):
    old_finetune = shared.opts.data.get("mm_use_old_finetune", False)
    if not isxl and old_finetune:
        # old adjust method
        return [x * 0.02 for x in cols[1:4]]

    colors = COLSXL if isxl else COLS
    outs = [[y * cols[i] * 0.02 for y in x] for i,x in enumerate(colors)]
    return [sum(x) for x in zip(*outs)]

# parse finetune: IN,OUT1,OUT2,CONTRAST,BRI,COL1,COL2,COL3
def fineman(fine, isxl):
    if fine.find(",") != -1:
        tmp = [t.strip() for t in fine.split(",")]
        fines = [0.0]*8
        for i,f in enumerate(tmp[0:8]):
            try:
                f = float(f)
                fines[i] = f
            except Exception:
                pass

        fine = [
            1 - fines[0] * 0.01,
            1 + fines[0] * 0.02,
            1 - fines[1] * 0.01,
            1 + fines[1] * 0.02,
            1 - fines[2] * 0.01,
            [fines[3] * 0.02] + colorcalc(fines[4:8], isxl)
        ]
        return fine
    return None


def extract_lora_from_current_model(save_lora_mode, model_orig, model_tuned, diff_model_mode,
        custom_name, extract_mode, lin_dim, conv_dim, lin_slider, conv_slider, lora_dim, min_diff, clamp_quantile,
        precision, calc_device, save_settings, metadata_settings, extra_settings): #, progress=gr.Progress(track_tqdm=False)):

    if shared.sd_model and shared.sd_model.sd_checkpoint_info:
        metadata = shared.sd_model.sd_checkpoint_info.metadata.copy()
    else:
        return gr.update(value="Have no valid current model")

    current = getattr(shared, "modelmixer_config", None)
    if current is None:
        print(" - Current model is not a merged model. ignored...")

    else:
        sha256 = current["hash"]
        if shared.sd_model.sd_checkpoint_info.sha256 != sha256:
            err_msg = " - Current model is not a merged model. ignored..."
            print(err_msg)

    if "sd_merge_recipe" not in metadata or "sd_merge_models" not in metadata:
        print(" - No merged recipe found")

    elif metadata_settings is not None and "merge recipe" in metadata_settings:
        metadata["sd_merge_recipe"] = json.dumps(metadata["sd_merge_recipe"], separators=(',', ':'))
    else:
        del metadata["sd_merge_recipe"]
    if "sd_merge_models" in metadata:
        metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"], separators=(',', ':'))

    # check save_lora_mode
    state_dict_base = None
    state_dict_trained = None
    if save_lora_mode == "diff" and model_orig != "None":
        checkpointinfo = sd_models.get_closet_checkpoint_match(model_orig)
        if checkpointinfo:
            print(f" - load original base model {model_orig}...")
            state_dict_base = sd_models.read_state_dict(checkpointinfo.filename, map_location = "cpu")
        if state_dict_base is not None and model_tuned != "Current model":
            checkpointinfo = sd_models.get_closet_checkpoint_match(model_tuned)
            if checkpointinfo:
                print(f" - load tuned model {model_tuned}...")
                state_dict_trained = sd_models.read_state_dict(checkpointinfo.filename, map_location = "cpu")
    if save_lora_mode == "diff" and state_dict_base is None:
        save_lora_mode = "extract"
        print("No base model selected. Use extract mode to extract any LoRAs used in the prompt...")

    # setup file, imported from supermerger
    ext = ".safetensors" if "safetensors" in extra_settings else ".ckpt"

    if not custom_name or custom_name == "":
        return gr.update(value="No LoRA name given!")
    else:
        fname = custom_name if ext in custom_name else custom_name + ext

    lora_dir = "Lora"
    lora_path = os.path.abspath(os.path.join(paths.models_path, lora_dir))
    fname = os.path.join(lora_path, fname)

    if len(fname) > 255:
       fname.replace(ext, "")
       fname = fname[:240] + ext

    # check if output file already exists
    if os.path.isfile(fname) and not "overwrite" in extra_settings:
        err_msg = f"Output file ({fname}) exists. not saved."
        print(err_msg)
        return gr.update(value=err_msg)

    if save_lora_mode == "diff":
        if state_dict_trained is not None:
            pass
        elif diff_model_mode == "lora":
            print(" - \033[92mget the merged model with LoRAs weighted\033[0m, if any LoRAs have been used in the prompt...")
            state_dict_trained = get_current_state_dict(lora=True, base=False)[0]
        else:
            print(" - \033[92mget the merged model\033[0m...")
            state_dict_trained = get_current_state_dict(lora=False, base=True)[0]
    else: # extract mode
        print(" - \033[92mget the base state_dict and the state_dict with LoRAs weighted\033[0m, if any LoRAs have been used in the prompt...")
        state_dict_trained, state_dict_base = get_current_state_dict(lora=True, base=True)

    # some possible ommitted keys
    possible_keys = [ "conditioner.embedders.1.model.transformer.text_model.embeddings.position_ids" ]
    is_equal = True
    #progress.track_tqdm=False
    checkbar = tqdm(state_dict_base.keys(), desc="check difference")
    for key in checkbar:
        if "model" not in key:
            continue
        if key not in state_dict_trained:
            if key not in possible_keys:
                print("key ", key, " could be ommited")
            else:
                print("key ", key, " not found")
            continue
        if torch.any(torch.ne(state_dict_base[key], state_dict_trained[key])):
            is_equal = False
            break
        else:
            continue
    checkbar.close()

    if is_equal:
        return gr.update(value="No difference found")

    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in state_dict_base

    gc.collect()
    devices.torch_gc()
    #progress.track_tqdm=False

    device = None
    if calc_device == "auto":
        calc_device = get_device()
    elif calc_device == "cuda":
        pass
    elif calc_device != "cpu":
        calc_device = "cpu"
    print(f" - svd calc device = {calc_device}")
    no_half = "nohalf" in extra_settings
    if "LyCORIS" in save_settings:
        try:
            load_module(os.path.join(scriptdir, "scripts", "kohya"))
            # for dev purpose
            load_module(os.path.join(scriptdir, "scripts", "kohya", "lycoris_utils.py"))
            load_module(os.path.join(scriptdir, "scripts", "kohya", "model_utils.py"))
            from scripts.kohya.lycoris_utils import extract_diff
            if isxl:
                sdxl_model_util = load_module(os.path.join(scriptdir, "scripts", "kohya", "sdxl_model_util.py"))
            from scripts.kohya.model_utils import load_models_from_stable_diffusion_checkpoint
        except Exception as e:
            print(f"No lycoris module found {e}")
            return gr.update(value="LyCORIS module not found")

        if not isxl:
            base = load_models_from_stable_diffusion_checkpoint(None, dict(state_dict_base), no_half=no_half)
            lora = load_models_from_stable_diffusion_checkpoint(None, dict(state_dict_trained), no_half=no_half)
        else:
            from scripts.kohya.sdxl_model_util import load_models_from_sdxl_checkpoint, MODEL_VERSION_SDXL_BASE_V1_0
            base = load_models_from_sdxl_checkpoint(MODEL_VERSION_SDXL_BASE_V1_0, dict(state_dict_base), "cpu", no_half=no_half)
            lora = load_models_from_sdxl_checkpoint(MODEL_VERSION_SDXL_BASE_V1_0, dict(state_dict_trained), "cpu", no_half=no_half)

        metadata = {
            "ss_network_module": "lycoris.kohya",
            "ss_output_name": custom_name,
        }

        if extract_mode == 'Fixed':
            linear_mode_param = lin_dim
            conv_mode_param = lin_dim
            metadata["ss_network_args"] = json.dumps({
                "conv_dim": str(conv_dim),
                "conv_alpha": str(float(conv_dim)),
            })
        else:
            linear_mode_param = lin_slider
            conv_mode_param = conv_slider
            mode = extract_mode.lower()
            metadata["ss_network_args"] = json.dumps({
                "method": extract_mode.lower(),
                f"linear_{mode}": str(conv_slider),
                f"conv_{mode}": str(lin_slider),
            })

        lora_state_dict = extract_diff(
            base, lora,
            extract_mode.lower(),
            linear_mode_param, conv_mode_param,
            calc_device,
            False, #args.use_sparse_bias,
            0.98, # args.sparsity,
            True #not args.disable_cp
        )
        gc.collect()
        devices.torch_gc()

    else: # LoRA
        try:
            load_module(os.path.join(scriptdir, "scripts", "kohya"))
            load_module(os.path.join(scriptdir, "scripts", "kohya", "lora.py"))
            load_module(os.path.join(scriptdir, "scripts", "kohya", "model_utils.py"))
            if isxl:
                load_module(os.path.join(scriptdir, "scripts", "kohya", "sdxl_model_util.py"))
            load_module(os.path.join(scriptdir, "scripts", "kohya", "extract_lora_from_models.py"))
            from scripts.kohya.extract_lora_from_models import svd
        except Exception as e:
            print(f"No scripts.kohya.* modules found. ERROR: {e}")
            return gr.update(value="No scripts.kohya.* modules found")

        # sanitiy check
        if min_diff < 0.:
            min_diff = 1e-6
        if clamp_quantile < 0. or clamp_quantile > 1.0:
            min_diff = 1.0
        if lora_dim <= 0:
            return gr.update(value="Invalid LoRA DIM")

        extracted_lora = svd(dict(state_dict_base), dict(state_dict_trained), None, lora_dim, min_diff=min_diff, clamp_quantile=clamp_quantile, device=calc_device,
            no_half=no_half
        )
        gc.collect()
        devices.torch_gc()
        lora_state_dict = extracted_lora.state_dict()
        metadata = {
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(lora_dim),
            "ss_network_alpha": str(float(lora_dim)),
            "ss_output_name": custom_name,
        }
    v2 = False
    if 'model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight' in state_dict_base:
        v2 = state_dict_base['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight'].shape[1] == 1024
    metadata["ss_v2"] = str(v2)
    v_parameterization = v2
    try:
        from scripts.kohya.sai_model_spec import build_metadata

        sai_metadata = build_metadata(
            None, v2, v_parameterization, isxl, True, False, time.time(), title=custom_name
        )
        metadata.update(sai_metadata)
    except Exception as e:
        pass

    if isxl:
        # fix for SDXL LoRA
        metadata["ss_base_model_version"] = "sdxl_base_v1-0"
    else:
        version_str = "sd"
        if v2:
            version_str += "_v2"
        else:
            version_str += "_v1"
        if v_parameterization:
            version_str += "_v"
        metadata["ss_base_model_version"] = version_str

    target_dtype = torch.float16
    if precision == "fp32":
        target_dtype = torch.float
    elif precision == "bf16":
        target_dtype = torch.bfloat16

    for key in lora_state_dict.keys():
        v = lora_state_dict[key]
        v = v.detach().to("cpu").to(target_dtype)
        lora_state_dict[key] = v

    try:
        if ext == ".safetensors":
            save_file(lora_state_dict, fname, metadata=metadata)
        else:
            torch.save(lora_state_dict, fname)
    except Exception as e:
        print(f"ERROR: Couldn't saved:{fname},ERROR is {e}")
        return gr.update(value=f"ERROR: Couldn't saved:{fname},ERROR is {e}")

    info = "Extracted LoRA saved in " + fname
    print(info)
    return gr.update(value=info)


def save_as_diffusers(custom_name, save_settings, metadata_settings, state_dict=None, metadata=None):
    from modules import sd_models_config

    try:
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
    except Exception as e:
        return f"{e}"

    if state_dict is None:
        current = getattr(shared, "modelmixer_config", None)
        if current is None:
            print(" - Current model is not a merged model. ignored...")

        if shared.sd_model and shared.sd_model.sd_checkpoint_info:
            metadata = shared.sd_model.sd_checkpoint_info.metadata.copy()
        else:
            return gr.update(value="No model loaded")

        if current and shared.sd_model.sd_checkpoint_info.sha256 == current["hash"]:
            print("use current merged checkpoint.")

        print(" - \033[92mget the merged model\033[0m...")
        state_dict = get_current_state_dict(lora=False, base=True)[0]

    v2 = False
    if 'model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight' in state_dict:
        v2 = state_dict['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight'].shape[1] == 1024
    print(" v2 = ", v2)

    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in state_dict
    print(" isxl = ", isxl)

    pipeline_type = None
    scheduler_type = "pndm"
    image_size = 512
    prediction_type = "epsilon"
    upcast_attention = False
    extract_ema = False
    device = "cuda"
    stable_unclip_prior = False
    clip_stats_path = None
    controlnet = None
    pipeline_class = None
    stable_unclip = None
    stable_unclip_prior = None
    clip_stats_path = None
    if v2:
        image_size = 768
        prediction_type = "v_prediction"
        upcast_attention = True

    if not custom_name or custom_name == "":
        dump_path = shared.sd_model.sd_checkpoint_info.name_for_extra.replace(" ","").replace(",","_").replace("(","_").replace(")","_")
        if dump_path[0] == "_":
            dump_path = dump_path[1:]
    else:
        dump_path = custom_name

    dump_path = os.path.join(model_path, dump_path)

    if len(dump_path) > 255:
       dump_path = dump_path[:240]


    # fix/check bad CLIP ids
    fixclip(state_dict, save_settings, isxl)

    # for safetensors contiguous error
    print(" - check contiguous...")
    for key in state_dict.keys():
        v = state_dict[key]
        v = v.detach().cpu().contiguous()
        state_dict[key] = v

    original_config_file = sd_models_config.find_checkpoint_config(state_dict, None)

    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=state_dict,
        original_config_file=original_config_file,
        config_files=None,
        image_size=image_size,
        prediction_type=prediction_type,
        model_type=pipeline_type if pipeline_type else None,
        extract_ema=extract_ema,
        scheduler_type=scheduler_type,
        num_in_channels=None,
        upcast_attention=upcast_attention,
        from_safetensors=None,
        device=device,
        stable_unclip=stable_unclip,
        stable_unclip_prior=stable_unclip_prior,
        clip_stats_path=clip_stats_path,
        controlnet=controlnet,
        vae_path=None,
        pipeline_class=pipeline_class,
    )

    pipe.save_pretrained(dump_path, safe_serialization=True)
    log = f"Done! diffusers saved at {dump_path}"
    print(log)

    return log


def save_current_model(custom_name, bake_in_vae, save_settings, metadata_settings, state_dict=None, metadata=None):
    if state_dict is None:
        current = getattr(shared, "modelmixer_config", None)
        if current is None:
            return gr.update(value="No merged model found")

        if shared.sd_model and shared.sd_model.sd_checkpoint_info:
            metadata = shared.sd_model.sd_checkpoint_info.metadata.copy()
        else:
            return gr.update(value="Not a valid merged model")

        sha256 = current["hash"]
        if shared.sd_model.sd_checkpoint_info.sha256 != sha256:
            err_msg = "Current checkpoint is not a merged one."
            print(err_msg)
            return gr.update(value=err_msg)

    if "sd_merge_recipe" not in metadata or "sd_merge_models" not in metadata:
        return gr.update(value="Not a valid merged model")

    if metadata_settings is not None and "merge recipe" in metadata_settings:
        metadata["sd_merge_recipe"] = json.dumps(metadata["sd_merge_recipe"], separators=(',', ':'))
    else:
        del metadata["sd_merge_recipe"]
    if "sd_merge_models" in metadata:
        metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"], separators=(',', ':'))

    if state_dict is None and shared.sd_model is not None:
        print("Load state_dict from shared.sd_model..")

        if "with LoRAs" in save_settings:
            print(" - \033[92mwith LoRAs\033[0m, if any LoRAs have been used in the prompt...")
        state_dict = get_current_state_dict("with LoRAs" in save_settings, "with LoRAs" not in save_settings)[0]
    elif state_dict is None:
        print("No loaded model found")
        return gr.update(value="No loaded model found")

    # setup file, imported from supermerger
    if "fp16" in save_settings:
        pre = ".fp16"
    else:
        pre = ""
    ext = ".safetensors" if "safetensors" in save_settings else ".ckpt"

    if not custom_name or custom_name == "":
        fname = shared.sd_model.sd_checkpoint_info.name_for_extra.replace(" ","").replace(",","_").replace("(","_").replace(")","_") + pre + ext
        if fname[0] == "_":
            fname = fname[1:]
    else:
        fname = custom_name if ext in custom_name else custom_name + pre + ext

    fname = os.path.join(model_path, fname)

    if len(fname) > 255:
       fname.replace(ext, "")
       fname = fname[:240] + ext

    # check if output file already exists
    if os.path.isfile(fname) and not "overwrite" in save_settings:
        err_msg = f"Output file ({fname}) exists. not saved."
        print(err_msg)
        return gr.update(value=err_msg)

    # bake in VAE
    bake_in_vae_filename = sd_vae.vae_dict.get(bake_in_vae, None)
    if bake_in_vae_filename is not None:
        print(f"Baking in VAE from {bake_in_vae_filename}")
        vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename, map_location='cpu')
        for key in (tqdm(vae_dict.keys(), desc=f"Bake in VAE...")):
            key_name = 'first_stage_model.' + key
            state_dict[key_name] = deepcopy(vae_dict[key])
        del vae_dict

    print("Saving...")
    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in state_dict
    print("isxl = ", isxl)
    if isxl:
        # prune share memory tensors, "cond_stage_model." prefixed base tensors are share memory with "conditioner." prefixed tensors
        for i, key in enumerate(state_dict.keys()):
            if "cond_stage_model." in key:
                del state_dict[key]

    if "fp16" in save_settings:
        state_dict = to_half(state_dict, True)
    if "prune" in save_settings:
        state_dict = prune_model(state_dict, isxl)

    # fix/check bad CLIP ids
    fixclip(state_dict, save_settings, isxl)

    # for safetensors contiguous error
    print(" - check contiguous...")
    for key in state_dict.keys():
        v = state_dict[key]
        v = v.detach().cpu().contiguous()
        state_dict[key] = v

    try:
        if ext == ".safetensors":
            save_file(state_dict, fname, metadata=metadata)
        else:
            torch.save(state_dict, fname)
    except Exception as e:
        print(f"ERROR: Couldn't saved:{fname},ERROR is {e}")
        return gr.update(value=f"ERROR: Couldn't saved:{fname},ERROR is {e}")
    print(f"Successfully model saved! - {fname}")
    print("Done!")

    data = "Merged model saved in " + fname
    return gr.update(value=data)


def get_current_state_dict(lora=False, base=True):
    if "scripts.patches" not in sys.modules:
        print(" - loading script.patches...")
        patches = load_module(os.path.join(scriptdir, "scripts", "patches.py"))
        #sys.modules["scripts.patches"] = patches

    from scripts.patches import StateDictPatches, StateDictLoraPatches

    # save to cpu
    sd_models.send_model_to_cpu(shared.sd_model)
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)

    ret = []
    # HACK patch nn.Module 'state_dict' to fix lora extension bug
    if base:
        lora_patch = False
        try:
            patch = StateDictPatches()
            lora_patch = True
            print(" - base lora_patch ")
        except Exception:
            print("Please enable extension-builtin lora")
            pass

        state_dict = shared.sd_model.state_dict()
        ret.append(state_dict.copy())
        del state_dict

        if lora_patch:
            patch.undo()
            del patch

    if lora:
        lora_patch = False
        try:
            patch = StateDictLoraPatches()
            lora_patch = True
            print(" - lora patch ")
        except Exception:
            print("Please enable extension-builtin lora")
            pass

        if base:
            # withot following two line patch does not work correctly
            sd_models.send_model_to_device(shared.sd_model)
            sd_models.send_model_to_cpu(shared.sd_model)

        state_dict_with_lora = shared.sd_model.state_dict()
        ret.insert(0, state_dict_with_lora.copy())
        del state_dict_with_lora

        if lora_patch:
            patch.undo()
            del patch

    # restore to gpu
    sd_hijack.model_hijack.hijack(shared.sd_model)
    sd_models.send_model_to_device(shared.sd_model)

    return ret


def prepare_model(model):
    global elemental_blocks
    if elemental_blocks is None:
        elemental_blocks = {}
        elemental_blocks = prepare_elemental_blocks(model)
    else:
        # check settings again
        isxl = is_xl(model)
        if isxl:
            if elemental_blocks.get("IN09", None) is not None:
                # read elements-xl.json
                elemental_blocks = prepare_elemental_blocks(model)
        else:
            if elemental_blocks.get("IN09", None) is None:
                # read elements.json
                elemental_blocks = prepare_elemental_blocks(model)

def prepare_elemental_blocks(model=None, force=False):
    if model is not None:
        isxl = is_xl(model)
    else:
        isxl = False
    elemdata = "elements.json" if not isxl else "elements-xl.json"
    elempath = os.path.join(scriptdir, "data", elemdata)
    if not os.path.exists(os.path.join(scriptdir, "data")):
        os.makedirs(os.path.join(scriptdir, "data"))
    if os.path.isfile(elempath):
        try:
            with open(elempath) as f:
                data = json.load(f)
        except OSError as e:
            print(f"Fail to load {elempath}, e = {e}")
            pass

        return data

    # try to load any valid safetenors
    res = {}
    if model is None and sd_models.checkpoints_list is not None:
        for checkpoint in sd_models.checkpoints_list:
            if checkpoint.is_safetensors:
                abspath = os.path.abspath(checkpoint.filename)
                if os.path.exists(abspath):
                    res = get_safetensors_header(abspath)
                    if len(res) > 0:
                        break
        if len(res) == 0:
            return None
    elif model is not None:
        checkpoint = sd_models.get_closet_checkpoint_match(model)
        if checkpoint.is_safetensors:
            abspath = os.path.abspath(checkpoint.filename)
            if os.path.exists(abspath):
                res = get_safetensors_header(abspath)
                if len(res) == 0:
                    return None
        else:
            return None
    else:
        return None

    elements = get_blocks_elements(res)
    try:
        with open(elempath, 'w') as f:
            json.dump(elements, f, indent=4, ensure_ascii=False)
    except OSError as e:
        print(f"Fail to save {elempath}, e = {e}")
        pass

    return elements

def get_blocks_elements(res):
    import collections

    blockmap = { "input_blocks": "IN", "output_blocks": "OUT", "middle_block": "M" }

    key_re = re.compile(r"^(?:\d+\.)?(.*?)(?:\.\d+)?$")
    key_split_re = re.compile(r"\.\d+\.")

    elements = {}
    for key in res:
        if ".bias" in key:
            # ignore bias keys if .weight exists
            k = key.replace(".bias", ".weight")
            if res.get(k, None) is not None:
                continue
        k = key.replace(".weight", "") # strip .weight
        k = k.replace("model.diffusion_model.", "")
        k = k.replace("cond_stage_model.transformer.text_model.", "BASE.")

        name = None
        # only for block level keys
        if any(item in k for item in ["input_blocks.", "output_blocks.", "middle_block."]):
            tmp = k.split(".",2)
            num = int(tmp[1])
            name = f"{blockmap[tmp[0]]}{num:02d}"
            if name in [ "M00", "M01", "M02" ]:
                name = "M00" # supermerger does not distinguish M01 and M02
            last = tmp[2]
        elif "BASE" in k:
            if "position_ids" in k: continue
            tmp = k.split(".",1)
            name = "BASE"
            last = tmp[1]
            last = last.replace("encoder.layers.", "")

        if name and last != "":
            m = key_re.match(last) # trim out some numbering: 0.foobar.1 => foobar
            if m:
                elem = elements.get(name, None)
                if elem is None:
                    elements[name] = {}
                    elem = elements[name]
                b = m.group(1)
                tmp = key_split_re.split(b) # split foo.1.bar => foo, bar
                if len(tmp)>0:
                    for e in tmp:
                        if e == "0": # for IN00 case, only has 0.bias and 0.weight -> "0" remain
                            continue
                        elem[e] = 1
                        if e.find(".") != -1: # split attn1.to_q -> attn1, to_q
                            tmp1 = e.split(".")
                            if len(tmp1) > 0:
                                for e1 in tmp1:
                                    elem[e1] = 1
                                    e2 = e1.rstrip("12345") # attn1 -> attn
                                    if e1 != e2:
                                        elem[e2] = 1

    # sort elements
    sorted_elements = collections.OrderedDict()
    sort = sorted(elements)
    for name in sort:
        elems = sorted(elements[name])
        sorted_elements[name] = elems

    return sorted_elements

def prepblocks(blocks, blockids, select=True):
    #blocks = sorted(set(blocks)) # one liner block sorter
    if len(blocks) == 0 or (len(blocks) == 1 and blocks[0] == '*'):
        return []

    expands = []
    for br in blocks:
        if "-" in br:
            bs = [b.strip() for b in br.split('-')]
            try:
                si, ei = blockids.index(bs[0]), blockids.index(bs[1])
            except:
                # ignore
                print(f" - WARN: Invalid block range {br} was ignored...")
                continue
            if si > ei:
                si, ei = ei, si
            expands += blockids[si:ei+1]
        else:
            if br in blockids:
                expands.append(br)
            else:
                print(f" - WARN: Invalid block {br} was ignored...")

    selected = [not select]*len(blockids)

    for b in expands:
        selected[blockids.index(b)] = select

    out = []
    for i, s in enumerate(selected):
        if s:
            out.append(blockids[i])
    return out

def zipblocks(blocks, blockids):
    """zip blocks to block ranges"""
    if len(blocks) == 0:
        return ""

    selected = [False]*len(blockids)
    for b in blocks:
        if b in blockids:
            selected[blockids.index(b)] = True
        else:
            # ignore
            print(f" - WARN: Invalid block {b} was ignored...")

    i = 0
    out = []
    while i < len(selected):
        try:
            start = selected.index(True, i)
        except:
            break

        if selected[start+1] is True:
            try:
                end = selected.index(False, start+2)
            except:
                end = len(selected)

            if end - start == 2:
                out += blockids[start:end]
            else:
                out.append(f"{blockids[start]}-{blockids[end-1]}")
            i = end + 1
        else:
            out.append(blockids[start])
            i = start + 1
    return out

def parse_elemental(elemental):
    if len(elemental) > 0:
        elemental = elemental.replace(",","\n").strip().split("\n")
        elemental = [f.strip() for f in elemental]

    elemental_weights = {}
    if len(elemental) > 0:
        for d in elemental:
            if d.count(":") != 2:
                # invalid case
                continue
            dbs, dws, dr = [f.strip() for f in d.split(":")]
            try:
                dr = float(dr)
            except:
                print(f"Invalid elemental entry - {d}")
                pass

            dbs = dbs.split(" ")
            dbs = list(filter(None, dbs))
            if len(dbs) > 0:
                dbn, dbs = (False, dbs[1:]) if dbs[0].upper() == "NOT" else (True, dbs)
                dbs = prepblocks(dbs, BLOCKID, select=dbn)
            else:
                dbn = True

            dws = dws.split(" ")
            dws = list(filter(None, dws))
            if len(dbs) == 0 and len(dws) == 0:
                # invalid case
                print(f"Invalid elemental entry - {d}")
                continue

            if len(dws) > 0 and len(dbs) == 0:
                dbs = [""] # empty blocks case
            dws = [""] if len(dws) == 0 else dws # empty elements case
            dwn, dws = (False, dws[1:]) if dws[0].upper() == "NOT" else (True, dws)

            for block in dbs:
                weights = elemental_weights.get(block, [])
                was_empty = len(weights) == 0
                weights.append({"flag": dwn, "elements": dws, "ratio": dr})
                if was_empty:
                    elemental_weights[block] = weights

        return elemental_weights
    return None

# from modules/generation_parameters_copypaste.py
re_param_code = r'\s*([\w ]+):\s*("(?:\\"[^,]|\\"|\\|[^\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)

def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)

def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text

    try:
        return json.loads(text)
    except Exception:
        return text

def parse(lastline):
    """from parse_generation_parameters(x: str)"""
    res = {}
    for k, v in re_param.findall(lastline):
        try:
            if v[0] == '"' and v[-1] == '"':
                v = unquote(v)

            res[k] = v
        except Exception:
            print(f"Error parsing \"{k}: {v}\"")

    return res

def on_image_save(params):
    if 'parameters' not in params.pnginfo: return

    # load mixed model info
    model = getattr(shared, "modelmixer_config", None)
    if model is None: return
    sha256 = model["hash"]
    if shared.sd_model is None or shared.sd_model.sd_checkpoint_info is None or shared.sd_model.sd_checkpoint_info.sha256 != sha256:
        return

    modelinfos = model["models"]
    modelhashes = model["hashes"]
    recipe = model.get("recipe", None)
    # filterout empty shorthash
    modelhashes = list(filter(None, modelhashes))

    lines = params.pnginfo['parameters'].split('\n')
    generation_params = lines.pop()
    prompt_parts = '\n'.join(lines).split('Negative prompt:')
    prompt, negative_prompt = [s.strip() for s in prompt_parts[:2] + ['']*(2-len(prompt_parts))]

    # add multiple "Model hash:" entries and change "Model:" name
    res = parse(generation_params)
    res["Model hash"] = modelhashes[0]
    res["Model"] = " + ".join(modelinfos)
    generation_params = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in res.items() if v is not None])

    # add Model hash a: xxx, Model a: yyy, Model hash b: zzz, Model b: uuu...
    for j,v in enumerate(modelhashes):
        n = chr(97+j)
        model = modelinfos[j].replace(",", " ")
        generation_params += f", Model hash {n}: {v}, Model {n}: {model}"

    # add recipe
    if recipe is not None:
        generation_params += ", Model recipe: " + recipe.replace(","," ")

    params.pnginfo['parameters'] = prompt + "\nNegative prompt:" + negative_prompt + "\n" + generation_params

def on_ui_settings():
    section = ("Model Mixer", "Model Mixer")
    shared.opts.add_option(
        "mm_max_models",
        shared.OptionInfo(
            default=3,
            label="Maximum Merge models",
            component=gr.Slider,
            component_args={"minimum": 1, "maximum": 5, "step": 1},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_debugs",
        shared.OptionInfo(
            default=["elemental merge"],
            label="Debug Infos",
            component=gr.CheckboxGroup,
            component_args={"choices": ["elemental merge", "merge", "adjust", "save model"]},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_save_model",
        shared.OptionInfo(
            default=["safetensors", "fp16", "prune", "overwrite"],
            label="Auto save merged model",
            component=gr.CheckboxGroup,
            component_args={"choices": ["overwrite", "safetensors", "fp16", "prune", "fix CLIP ids"]},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_save_model_filename",
        shared.OptionInfo(
            default="modelmixer-[hash]",
            label="Filename of auto save model",
            component=gr.Textbox,
            component_args={"interactive": True, "placeholder": "save model filename e.g) [model_name]-[hash]"},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_use_extra_elements",
        shared.OptionInfo(
            default=True,
            label="Merge Extra Elements (.time_embed.*, .out.*)",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_use_old_finetune",
        shared.OptionInfo(
            default=False,
            label="Use old Adjust method",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_use_unet_partial_update",
        shared.OptionInfo(
            default=True,
            label="Use UNet block partial update",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_laplib",
        shared.OptionInfo(
            default="lap",
            label="Select LAP library for Rebasin calc. (Linear Assignment Problem maximum weight matching)",
            component=gr.Radio,
            component_args={"choices": ["lap", "lapjv", "scipy"]},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_use_fast_weighted_sum",
        shared.OptionInfo(
            default=True,
            label="Use Fast weighted sum using torch.lerp()",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_use_precalculate_hash",
        shared.OptionInfo(
            default=False,
            label="Use precalculating model hash",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_use_model_dl",
        shared.OptionInfo(
            default=False,
            label="Use Model downloader",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_default_config_lock",
        shared.OptionInfo(
            default=False,
            label="Default Config Lock to keep current config",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "mm_civitai_api_key",
        shared.OptionInfo(
            default="",
            label="Civitai API Key",
            section=section,
        ),
    )


def on_infotext_pasted(infotext, results):
    updates = {}

    config_lock = getattr(shared, "config_lock", False)
    excludes = []

    models = {}
    modelnames = {}
    modelhashes = {}
    hashes = set()
    for k, v in results.items():
        # check all "Model a:...,  Model hash a:,... Model b:..."
        if k.find("Model ") == 0:
            name = k.rsplit(" ", 1)[-1]
            if len(k) not in [7, 12]: # valid keys are 'Model a', 'Model hash a'
                continue
            if k.find(" hash ") > 0:
                h = v.strip().upper()[0:10] # AutoV2
                modelhashes[name] = h
            else:
                modelnames[name] = v + (".safetensors" if not v.endswith(".safetensors") and not v.endswith(".ckpt") else "")

            if modelnames.get(name, None) is not None and modelhashes.get(name, None) is not None:
                h = modelhashes[name]
                models[h] = modelnames[name] + " [" + h + "]"
                hashes.add(h)
                continue
            else:
                continue

        elif not k.startswith("ModelMixer"):
            continue

        if config_lock:
            excludes.append(k)
            continue

        if k.find(" merge mode ") > 0 and v == "Sum(lerp)":
            updates[k] = "Sum"

        if k.find(" model ") > 0:
            # ModelMixer Model a:... params
            if (j:= v.rfind("[")) > 0 and v.find("]", j) > 0:
                h = v[j+1:-1].upper() # AutoV2
                models[h] = v[:j].strip()
                hashes.add(h)

        if k.find(" elemental ") > 0:
            if v in [ "True", "False"]:
                continue
            if v[0] == '"' and v[-1] == '"':
                v = v[1:-1]

            arr = v.strip().split(",")
            updates[k] = "\n".join(arr)

        if k.find(" mbw ") > 0:
            if v in [ "True", "False"]:
                continue
            elif k.find(" weights ") > 0:
                continue

            if v[0] == '"' and v[-1] == '"':
                v = v[1:-1]
            arr = v.split(",")
            updates[k] = arr

        # fix for old adjust value
        if k.find(" adjust") > 0:
            tmp = v.split(",")
            if len(tmp) == 7: # old
                tmp.insert(4, "0")
                v = ",".join(tmp)
                updates[k] = v

    if len(hashes) > 0:
        notfound = []
        for h in hashes:
            info = sd_models.get_closet_checkpoint_match(models[h])
            if info is not None:
                print(info.title, "found")
            else:
                print(models[h], "not found")
                notfound.append(h)

        if len(notfound) > 0:
            count = 1
            for h in notfound:
                model = get_civitai_model_by_hash(h, models[h])
                if model is not None and len(model) > 0:
                    updates[f"Download Model {count}"] = model
                    count += 1
        else:
            for j in range(5):
                # reset Download Model form
                updates[f"Download Model {j+1}"] = ''

    if config_lock and len(excludes) > 0:
        for k in excludes:
            results.pop(k, None)

    results.update(updates)


# civitai utils
def get_civitai_model_version_by_hash(hash: str):
    response = civitai_req(f"/model-versions/by-hash/{hash}")
    return response

def civitai_req(endpoint, method='GET', data=None, params=None, headers=None):
    import requests

    base_url = 'https://civitai.com/api/v1'
    user_agent = 'CivitaiLink:Automatic1111'

    """Make a request to the Civitai API."""
    if headers is None:
        headers = {}
    headers['User-Agent'] = user_agent
    api_key = shared.opts.data.get("civitai_api_key", None)
    if api_key is None:
        api_key = shared.opts.data.get("mm_civitai_api_key", None)
    if api_key is not None:
        headers['Authorization'] = f'Bearer {api_key}'
    if data is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(data)
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint
    if params is None:
        params = {}
    response = requests.request(method, base_url+endpoint, data=data, params=params, headers=headers)
    if response.status_code != 200:
        print(f'Error: {response.status_code} {response.text}')
        return None
    return response.json()


def get_civitai_model_by_hash(h, filename=None):
    """get model info as json"""

    r = {'files': []}
    ret = None
    try:
        r = get_civitai_model_version_by_hash(h)
    except Exception as e:
        print(e)
    else:
        if r is None:
            return None

    for file in r['files']:
        if not 'hashes' in file or not 'SHA256' in file['hashes']:
            continue

        autov2 = file['hashes']['AutoV2']
        if autov2.upper() == h.upper():
            downloadUrl = file['downloadUrl']
            print(downloadUrl)
            modelurl = f"https://civitai.com/models/{r['modelId']}"
            ret = json.dumps({
                'filename': filename if filename else file['name'],
                'type': file['type'], # Model
                'hash': autov2,
                'name': r['model'].get('name', r['name']),
                'url': modelurl,
                'downloadUrl': downloadUrl,
            }, ensure_ascii=False)
            break
    return ret


# xyz support
def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    model_list = ["None"]+sd_models.checkpoint_tiles()

    num_models = shared.opts.data.get("mm_max_models", 2)

    def set_value(p, x, xs, *, field: str):
        if not hasattr(p, "modelmixer_xyz"):
            p.modelmixer_xyz = {}

        p.modelmixer_xyz[field] = x

    def format_weights_add_label(p, opt, x):
        if type(x) == str:
            x = x.replace(" ", ",")
        return f"{opt.label}: {x}"

    def format_elemental_add_label(p, opt, x):
        x = x.replace(";", ",")
        return f"{opt.label}: {x}"

    axis = [
        xyz_grid.AxisOption(
            "[Model Mixer] Model A",
            str,
            partial(set_value, field="model a"),
            choices=lambda: model_list,
        ),
        xyz_grid.AxisOption(
            "[Model Mixer] Base model",
            str,
            partial(set_value, field="base model"),
            choices=lambda: model_list,
        ),
        xyz_grid.AxisOption(
            "[Model Mixer] Adjust",
            str,
            partial(set_value, field="adjust"),
            format_value=format_weights_add_label
        ),
        xyz_grid.AxisOption(
            "[Model Mixer] Pinpoint Adjust",
            str,
            partial(set_value, field="pinpoint adjust"),
            choices=lambda: ["IN", "OUT", "OUT2", "CONT", "BRI", "COL1", "COL2", "COL3"],
        ),
        xyz_grid.AxisOption(
            "[Model Mixer] Pinpoint alpha",
            float,
            partial(set_value, field="pinpoint alpha"),
            choices=lambda: [-6,-4,-2,0,2,4,6],
        ),
    ]

    for n in range(num_models):
        name = chr(98+n)
        Name = chr(66+n)
        entries = [
            xyz_grid.AxisOption(
                f"[Model Mixer] Model {Name}",
                str,
                partial(set_value, field=f"model {name}"),
                choices=lambda: model_list,
            ),
            xyz_grid.AxisOption(
                f"[Model Mixer] alpha {Name}",
                float,
                partial(set_value, field=f"alpha {name}"),
            ),
            xyz_grid.AxisOption(
                f"[Model Mixer] MBW alpha {Name}",
                str,
                partial(set_value, field=f"mbw alpha {name}"),
                format_value=format_weights_add_label
            ),
            xyz_grid.AxisOption(
                f"[Model Mixer] elemental merge {Name}",
                str,
                partial(set_value, field=f"elemental {name}"),
                format_value=format_elemental_add_label,
            ),
            xyz_grid.AxisOption(
                f"[Model Mixer] Pinpoint block {Name}",
                str,
                partial(set_value, field=f"pinpoint block {name}"),
                choices=lambda: BLOCKID,
            ),
            xyz_grid.AxisOption(
                f"[Model Mixer] Pinpoint alpha {Name}",
                float,
                partial(set_value, field=f"pinpoint alpha {name}"),
                choices=lambda: [0.2,0.4,0.6,0.8,1.0],
            ),
        ]
        axis += entries

    if not any(x.label.startswith("[Model Mixer]") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)

def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f" - Model Mixer: xyz_grid error:\n{error}",
            file=sys.stderr,
        )


def on_model_loaded(model):
    shared.modelmixer_config = None


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(on_image_save)
script_callbacks.on_infotext_pasted(on_infotext_pasted)
script_callbacks.on_before_ui(on_before_ui)
script_callbacks.on_model_loaded(on_model_loaded)
