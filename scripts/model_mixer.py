#
# Checkpoint Model Mixer extension for sd-webui
#
# Copyright 2023 wkpark at gmail.com
# License: AGPL
#
import collections
import os
import gradio as gr
import hashlib
import json
from pathlib import Path
import time
import tqdm
from tqdm import tqdm
import torch
from safetensors.torch import save_file

from copy import copy, deepcopy
from modules import script_callbacks, sd_hijack, sd_models, sd_vae, shared, ui_settings
from modules import scripts, cache
from modules.sd_models import model_hash, model_path, checkpoints_loaded
from modules.timer import Timer
from modules.ui import create_refresh_button

dump_cache = cache.dump_cache
cache = cache.cache

BLOCKID=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

def gr_enable(interactive=True):
    return {"interactive": interactive, "__type__": "update"}

def gr_open(open=True):
    return {"open": open, "__type__": "update"}

def slider2text(*slider):
    return gr.update(value = ",".join([str(x) for x in slider]))

def calc_mbws(mbw, mbw_blocks):
    weights = [t.strip() for t in mbw.split(",")]
    expect = 0
    MAXLEN = 26
    selected = [False]*MAXLEN
    compact_blocks = []

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
        for i in range(0, 12):
            name = f"IN{i:02d}"
            if name not in mbw_blocks:
                mbw_blocks.append(name)
    if 'OUT*' in mbw_blocks:
        for i in range(0, 12):
            name = f"OUT{i:02d}"
            if name not in mbw_blocks:
                mbw_blocks.append(name)

    for i, name in enumerate(BLOCKID):
        if name in mbw_blocks:
            if name[0:2] == 'IN':
                expect += 1
                num = int(name[2:])
                selected[num + 1] = True
                compact_blocks.append(f'inp.{num}.')
            elif name[0:3] == 'OUT':
                expect += 1
                num = int(name[3:])
                selected[num + 14] = True
                compact_blocks.append(f'out.{num}.')
            elif name == 'M00':
                expect += 1
                selected[13] = True
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
                    off = 13
                    num = 0
                elif 'out' == block:
                    off = 14

            mbws[off + num] = compact_mbws[i]

    return mbws, compact_mbws, selected

def get_mbws(mbw, mbw_blocks):
    mbws, compact_mbws, selected = calc_mbws(mbw, mbw_blocks)
    return [gr.update(value = v) for v in mbws]

def _all_blocks():
    # return all blocks
    blocks = [ "cond_stage_model." ]
    for i in range(0,12):
        blocks.append(f"input_blocks.{i}.")
    blocks.append("middle_block.1.")
    for i in range(0,12):
        blocks.append(f"output_blocks.{i}.")
    return blocks

def print_blocks(blocks):
    str = []
    for i,x in enumerate(blocks):
        if "input_blocks." in x:
            n = int(x[13:len(x)-1])
            block = f"IN{n:02d}"
            str.append(block)
        elif "middle_block." in x:
            block = "MID00"
            str.append(block)
        elif "output_blocks." in x:
            n = int(x[14:len(x)-1])
            block = f"OUT{n:02d}"
            str.append(block)
        elif "cond_stage_model" in x:
            block = f"BASE"
            str.append(block)
    return ','.join(str)

def _selected_blocks_and_weights(mbw):
    if type(mbw) is str:
        weights = [t.strip() for t in mbw.split(",")]
    else:
        weights = mbw
    # get all blocks
    all_blocks = _all_blocks()

    sel_blocks = []
    sel_mbws = []
    for i, w in enumerate(weights):
        v = float(w)
        if v != 0.0:
            sel_blocks.append(all_blocks[i])
            sel_mbws.append(v)
    return sel_blocks, sel_mbws

def _get_selected(mbw, model):
    sel_blocks, sel_mbws = _selected_blocks_and_weights(mbw)
    sel_keys = []
    for i, k in enumerate(sel_blocks):
        if k in model:
            sel_keys.append(k)

    return sel_keys, sel_blocks

def _get_selected_keys(sel_blocks, model):
    sel_keys = []
    for i, k in enumerate(sel_blocks):
        if k in model:
            sel_keys.append(k)

    return sel_keys

def _weight_index(key):
    num = -1
    offset = [ 0, 1, 13, 14 ]
    for k, s in enumerate([ "cond_stage_model.", "input_blocks.", "middle_block.", "output_blocks." ]):
        if s in key:
            if k == 0: return 0 # base
            if k == 2: return 13 # middle_block

            i = key.find(s)
            j = key.find(".", i+len(s))
            num = int(key[i+len(s):j]) + offset[k]
    return num

def prune_model(model):
    keys = list(model.keys())
    for k in keys:
        if "diffusion_model." not in k and "first_stage_model." not in k and "cond_stage_model." not in k:
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

def get_valid_checkpoint_title():
    checkpoint_info = shared.sd_model.sd_checkpoint_info if shared.sd_model is not None else None
    # check validity of current checkpoint_info
    if checkpoint_info is not None:
        filename = checkpoint_info.filename
        name = os.path.basename(filename)
        info = sd_models.get_closet_checkpoint_match(name)
        if info != checkpoint_info:
            # this is a fake checkpoint_info
            # return original title
            return info.title

        return checkpoint_info.title
    return ""

class ModelMixerScript(scripts.Script):
    init_model_a_change = False

    def __init__(self):
        super().__init__()

    def title(self):
        return "Model Mixer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        import modules.ui
        num_models = shared.opts.data.get("mm_max_models", 2)
        mm_use = [None]*num_models
        mm_models = [None]*num_models
        mm_modes = [None]*num_models
        mm_alpha = [None]*num_models
        mm_usembws = [None]*num_models
        mm_usembws_simple = [None]*num_models
        mm_weights = [None]*num_models

        mm_setalpha = [None]*num_models
        mm_readalpha = [None]*num_models
        mm_explain = [None]*num_models

        model_options = [None]*num_models
        default_use = [False]*num_models
        mbw_advanced = [None]*num_models
        mbw_simple = [None]*num_models
        mbw_use_advanced = [None]*num_models

        default_use[0] = True

        with gr.Accordion("Checkpoint Model Mixer", open=False):
            with gr.Row():
                mm_information = gr.HTML("Merge multiple models and load it for image generation.")
            with gr.Row():
                enabled = gr.Checkbox(label="Enable Model Mixer", value=False, visible=True)

            with gr.Row():
                model_a = gr.Dropdown(sd_models.checkpoint_tiles(), value=shared.sd_model.sd_checkpoint_info.title, elem_id="model_mixer_model_a", label="Model A", interactive=True)
                create_refresh_button(model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles(), "value": get_valid_checkpoint_title()},"refresh_checkpoint_Z")

                base_model = gr.Dropdown(["None"]+sd_models.checkpoint_tiles(), elem_id="model_mixer_model_base", value="None", label="Base Model used for Add-Difference mode", interactive=True)
                create_refresh_button(base_model, sd_models.list_models,lambda: {"choices": ["None"]+sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
            with gr.Row():
                enable_sync = gr.Checkbox(label="Sync with Default SD checkpoint", value=False, visible=True)

            mm_max_models = gr.Number(value=num_models, precision=0, visible=False)
            merge_method_info = [{}] * num_models
            with gr.Group(), gr.Tabs():
                for n in range(num_models):
                    name_a = chr(66+n-1) if n == 0 else f"merge_{n}"
                    name = chr(66+n)
                    lowername = chr(98+n)
                    merge_method_info[n] = {"Sum": f"Weight sum: {name_a}×(1-alpha)+{name}×alpha", "Add-Diff": f"Add difference:{name_a}+({name}-model_base)×alpha"}
                    default_merge_info = merge_method_info[n]["Sum"]
                    with gr.Tab(f"Merge Model {name}"):
                        with gr.Row():
                            mm_use[n] = gr.Checkbox(label=f"Model {name}", value=default_use[n], visible=True)
                        with gr.Row():
                            mm_models[n] = gr.Dropdown(["None"]+sd_models.checkpoint_tiles(), value="None", elem_id=f"model_mixer_model_{lowername}", label=f"Merge {name}", show_label=False, interactive=True)
                            create_refresh_button(mm_models[n], sd_models.list_models, lambda: {"choices": ["None"]+sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")

                        with gr.Group(visible=False) as model_options[n]:
                            with gr.Row():
                                mm_modes[n] = gr.Radio(label=f"Merge Method for Model {name}", info=default_merge_info, choices=["Sum", "Add-Diff"], value="Sum")
                            with gr.Row():
                                mm_alpha[n] = gr.Slider(label=f"Multiplier for Model {name}", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                            with gr.Row():
                                with gr.Column(scale=3):
                                    with gr.Group(Visible=True) as mbw_advanced[n]:
                                        mm_usembws[n] = gr.Dropdown(["ALL","BASE","INP*","MID","OUT*"]+BLOCKID[1:], value=[], multiselect=True, label="Merge Block Weights", show_label=False, info="or use Merge Block Weights for selected blocks")
                                    with gr.Group(visible=False) as mbw_simple[n]:
                                        mm_usembws_simple[n] = gr.CheckboxGroup(["BASE","INP*","MID","OUT*"], value=[], label="Merge Block Weights", show_label=False, info="or use Merge Block Weights for selected blocks")
                                with gr.Column(scale=1):
                                    with gr.Row():
                                        mbw_use_advanced[n] = gr.Checkbox(label="Use advanced MBW mode", value=True, visible=True)
                            with gr.Row():
                                mm_explain[n] = gr.HTML("")
                            with gr.Row():
                                mm_weights[n] = gr.Textbox(label="Block Level Weights: BASE,IN00,IN02,...IN11,M00,OUT00,...,OUT11", show_copy_button=True,
                                    value="0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")
                            with gr.Row():
                                mm_setalpha[n] = gr.Button(elem_id="copytogen", value="↑ set alpha")
                                mm_readalpha[n] = gr.Button(elem_id="copytogen", value="↓ read alpha")

            with gr.Accordion("Block Level Weights", open=False):

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)
                    with gr.Column(scale=2, min_width=200):
                        base = gr.Slider(label="BASE", minimum=0, maximum=1, step=0.001, value=0.5)
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)

                with gr.Row():
                    with gr.Column(scale=2, min_width=200):
                        in00 = gr.Slider(label="IN00", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in01 = gr.Slider(label="IN01", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in02 = gr.Slider(label="IN02", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in03 = gr.Slider(label="IN03", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in04 = gr.Slider(label="IN04", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in05 = gr.Slider(label="IN05", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in06 = gr.Slider(label="IN06", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in07 = gr.Slider(label="IN07", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in08 = gr.Slider(label="IN08", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in09 = gr.Slider(label="IN09", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in10 = gr.Slider(label="IN10", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        in11 = gr.Slider(label="IN11", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                    with gr.Column(scale=2, min_width=200):
                        ou11 = gr.Slider(label="OUT11", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou10 = gr.Slider(label="OUT10", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou09 = gr.Slider(label="OUT09", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou08 = gr.Slider(label="OUT08", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou07 = gr.Slider(label="OUT07", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou06 = gr.Slider(label="OUT06", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou05 = gr.Slider(label="OUT05", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou04 = gr.Slider(label="OUT04", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou03 = gr.Slider(label="OUT03", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou02 = gr.Slider(label="OUT02", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou01 = gr.Slider(label="OUT01", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        ou00 = gr.Slider(label="OUT00", minimum=-1.0, maximum=2, step=0.001, value=0.5)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)
                    with gr.Column(scale=2):
                        mi00 = gr.Slider(label="M00", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                    with gr.Column(scale=1, min_width=100):
                        gr.Slider(visible=False)

                with gr.Row():
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

                    dtrue =  gr.Checkbox(value = True, visible = False)
                    dfalse =  gr.Checkbox(value = False, visible = False)

            with gr.Accordion("Save the current merged model", open=False):
                with gr.Row():
                    logging = gr.Textbox(label="Message", lines=1, value="", show_label=False, info="log message")
                with gr.Row():
                    save_settings = gr.CheckboxGroup(["overwrite","safetensors","prune","fp16"], value=["fp16","prune","safetensors"], label="Select settings")
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

                with gr.Row():
                    metadata_settings = gr.CheckboxGroup(["merge recipe","Copy metadata from merged models"], value=["merge recipe"], label="Metadata settings")

                metadata_json = gr.TextArea('{}', label="Metadata in JSON format")
                read_metadata = gr.Button("Read current metadata")

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
            ret = [gr.update(value=model)]
            # load checkpoint
            if enable_sync:
                shared.opts.data['sd_model_checkpoint'] = model
                ret.append(gr.update(value=model))
                modules.sd_models.reload_model_weights()
            else:
                ret.append(gr.update())

            return ret

        def on_after_components(component, **kwargs):
            if self.init_model_a_change is True:
                return

            # shared.settings_components not initialized yet.
            # set onchange after_components
            if shared.settings_components is not None:
                if shared.settings_components is not None:
                    checkpoint = shared.settings_components.get('sd_model_checkpoint', None)
                else:
                    checkpoint = None
                if checkpoint is not None:
                    model_a.change(fn=sync_main_checkpoint,
                        inputs=[enable_sync, model_a],
                        outputs=[model_a, checkpoint]
                    )
                    self.init_model_a_change = True

                    enable_sync.change(fn=sync_main_checkpoint,
                        inputs=[enable_sync, model_a],
                        outputs=[model_a, checkpoint]
                    )

        def current_metadata():
            current = shared.opts.data.get("sd_webui_model_mixer_model", None)
            if current is None:
                return gr.update()

            data = json.dumps(current["metadata"], indent=4, ensure_ascii=False)

            return gr.update(value=data)

        def save_current_model(custom_name, bake_in_vae, save_settings, metadata_settings):
            current = shared.opts.data.get("sd_webui_model_mixer_model", None)
            if current is None:
                return gr.update(value="No merged model found")

            metadata = current["metadata"].copy()
            if  "merge recipe" in metadata_settings:
                metadata["sd_merge_recipe"] = json.dumps(metadata["sd_merge_recipe"])
            else:
                del metadata["sd_merge_recipe"]
            metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])

            if shared.sd_model is not None:
                print("load from shared.sd_model..")
                state_dict = shared.sd_model.state_dict()
            else:
                print("No loaded model found")
                return gr.update(value="No loaded model found")

            # setup file, imported from supermerger
            if "fp16" in save_settings:
                pre = ".fp16"
            else:
                pre = ""
            ext = ".safetensors" if "safetensors" in save_settings else ".ckpt"

            sha256 = current["hash"]
            if shared.sd_model.sd_checkpoint_info is None:
                err_msg = "No checkpoint file loaded."
                print(err_msg)
                return gr.update(value=err_msg)

            if shared.sd_model.sd_checkpoint_info.sha256 != sha256:
                err_msg = "Current checkpoint is not a merged one."
                print(err_msg)
                return gr.update(value=err_msg)

            if not custom_name or custom_name == "":
                fname = shared.sd_model.sd_checkpoint_info.model_name.replace(" ","").replace(",","_").replace("(","_").replace(")","_") + pre + ext
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
                    state_dict[key_name] = copy.deepcopy(vae_dict[key])
                del vae_dict

            print("Saving...")

            if "fp16" in save_settings:
                state_dict = to_half(state_dict, True)
            if "prune" in save_settings:
                state_dict = prune_model(state_dict)

            try:
                if ext == ".safetensors":
                    save_file(state_dict, fname, metadata=metadata)
                else:
                    torch.save(state_dict, fname)
            except Exception as e:
                print(f"ERROR: Couldn't saved:{fname},ERROR is {e}")
                return gr.update(value=f"ERROR: Couldn't saved:{fname},ERROR is {e}")
            print("Done!")

            data = "Merged model saved in " + fname
            return gr.update(value=data)

        # set callback
        if self.init_model_a_change is False:
            script_callbacks.on_after_component(on_after_components)

        members = [base,in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,mi00,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11]

        self.infotext_fields = (
            (model_a, "ModelMixer model a"),
            (base_model, "ModelMixer base model"),
            (mm_max_models, "ModelMixer max models")
        )

        for n in range(num_models):
            name = f"{chr(98+n)}"
            self.infotext_fields += (
                (mm_use[n], f"ModelMixer use model {name}"),
                (mm_models[n], f"ModelMixer model {name}"),
                (mm_modes[n], f"ModelMixer merge mode {name}"),
                (mm_alpha[n], f"ModelMixer alpha {name}"),
                (mbw_use_advanced[n], f"ModelMixer mbw mode {name}"),
                (mm_usembws[n], f"ModelMixer mbw {name}"),
                (mm_usembws_simple[n], f"ModelMixer simple mbw {name}"),
                (mm_weights[n], f"ModelMixer mbw weights {name}"),
            )

        # load settings
        print("checkpoint title = ", shared.sd_model.sd_checkpoint_info.title)

        read_metadata.click(fn=current_metadata, inputs=[], outputs=[metadata_json])
        save_current.click(fn=save_current_model, inputs=[custom_name, bake_in_vae, save_settings, metadata_settings], outputs=[logging])

        resetopt.change(fn=resetvalopt, inputs=[resetopt], outputs=[resetval])
        resetweight.click(fn=resetblockweights, inputs=[resetval,resetblockopt], outputs=members)
        addweight.click(fn=addblockweights, inputs=[resetval, resetblockopt, *members], outputs=members)
        mulweight.click(fn=mulblockweights, inputs=[resetval, resetblockopt, *members], outputs=members)

        for n in range(num_models):
            mm_setalpha[n].click(fn=slider2text,inputs=[*members],outputs=[mm_weights[n]])

            mm_readalpha[n].click(fn=get_mbws, inputs=[mm_weights[n], mm_usembws[n]], outputs=members)
            mm_usembws[n].change(fn=lambda mbws: gr_enable(len(mbws) == 0), inputs=[mm_usembws[n]], outputs=[mm_alpha[n]], show_progress=False)
            mm_models[n].change(fn=lambda modelname: gr_show(modelname != "None"), inputs=[mm_models[n]], outputs=[model_options[n]])
            mm_modes[n].change(fn=(lambda nd: lambda mode: gr.update(info=merge_method_info[nd][mode]))(n), inputs=[mm_modes[n]], outputs=[mm_modes[n]], show_progress=False)
            mbw_use_advanced[n].change(fn=lambda mode: [gr.update(visible=True), gr.update(visible=False)] if mode==True else [gr.update(visible=False),gr.update(visible=True)], inputs=[mbw_use_advanced[n]], outputs=[mbw_advanced[n], mbw_simple[n]])

        return [enabled, model_a, base_model, mm_max_models, *mm_use, *mm_models, *mm_modes, *mm_alpha, *mbw_use_advanced, *mm_usembws, *mm_usembws_simple, *mm_weights]

    def modelmixer_extra_params(self, model_a, base_model, mm_max_models, *args_):
        num_models = int(mm_max_models)
        params = {
            "ModelMixer model a": model_a,
            "ModelMixer max models": mm_max_models,
        }
        if base_model is not None and len(base_model) > 0:
            params.update({"ModelMixer base model": base_model})

        for j in range(num_models):
            name = f"{chr(98+j)}"
            params.update({f"ModelMixer use model {name}": args_[j]})

            if args_[num_models+j] != "None" and len(args_[num_models+j]) > 0:
                params.update({
                    f"ModelMixer model {name}": args_[num_models+j],
                    f"ModelMixer merge mode {name}": args_[num_models*2+j],
                    f"ModelMixer alpha {name}": args_[num_models*3+j],
                    f"ModelMixer mbw mode {name}": args_[num_models*4+j]
                })
                if len(args_[num_models*5+j]) > 0:
                    params.update({f"ModelMixer mbw {name}": ",".join(args_[num_models*5+j])})
                if len(args_[num_models*6+j]) > 0:
                    params.update({f"ModelMixer simple mbw {name}": ",".join(args_[num_models*6+j])})
                if len(args_[num_models*7+j]) > 0:
                    params.update({f"ModelMixer mbw weights {name}": args_[num_models*7+j]})

        return params

    def before_process(self, p, enabled, model_a, base_model, mm_max_models, *args_):
        if not enabled:
            return

        base_model = None if base_model == "None" else base_model
        # extract model infos
        num_models = int(mm_max_models)
        mm_use = ["False"]*num_models
        mm_models = []
        mm_modes = []
        mm_alpha = []
        mbw_use_advanced = []
        mm_usembws = []
        mm_weights = []

        for n in range(num_models):
            use = args_[n]
            if type(use) is str:
                use = True if use == "True" else False
            mm_use[n] = use

        if True not in mm_use:
            # no selected merge models
            print("No selected models found")
            return

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

                if not mbw_use_advanced:
                    usembws = usembws_simple

                model = None if model == "None" else model
                # ignore some cases
                if alpha == 0.0 and len(usembws) == 0:
                    continue
                if model is None:
                    continue

                mm_models.append(model)
                mm_modes.append(mode)
                mm_alpha.append(alpha)
                mm_usembws.append(usembws)
                mm_weights.append(weights)

        # extra_params
        extra_params = self.modelmixer_extra_params(model_a, base_model, mm_max_models, *args_)
        p.extra_generation_params.update(extra_params)

        # make a hash to cache results
        sha256 = hashlib.sha256(json.dumps([model_a, base_model, mm_models, mm_modes, mm_alpha, mm_usembws, mm_weights]).encode("utf-8")).hexdigest()
        print("config hash = ", sha256)
        if shared.sd_model.sd_checkpoint_info is not None and shared.sd_model.sd_checkpoint_info.sha256 == sha256:
            # already mixed
            print(f"  - use current mixed model {sha256}")
            return

        print("  - mm_use", mm_use)
        print("  - model_a", model_a)
        print("  - base_model", base_model)
        print("  - max_models", mm_max_models)
        print("  - models", mm_models)
        print("  - modes", mm_modes)
        print("  - usembws", mm_usembws)
        print("  - weights", mm_weights)
        print("  - alpha", mm_alpha)

        mm_weights_orig = mm_weights
        # prepare for merges
        compact_mode = None
        mm_selected = [[]] * num_models
        for j, model in enumerate(mm_models):
            if len(mm_usembws[j]) > 0:
                # normalize Merge block weights
                mm_weights[j], compact_mbws, mm_selected[j] = calc_mbws(mm_weights[j], mm_usembws[j])
                compact_mode = True if compact_mode is None else compact_mode
            else:
                compact_mode = False

        # get overall selected blocks
        if compact_mode:
            selected_blocks = []
            mm_selected_all = [False] * 26
            for j in range(len(mm_models)):
                for k in range(26):
                    mm_selected_all[k] = mm_selected_all[k] or mm_selected[j][k]
            all_blocks = _all_blocks()
            for k in range(26):
                if mm_selected_all[k]:
                    selected_blocks.append(all_blocks[k])
        else:
            # no compact mode, get all blocks
            selected_blocks = _all_blocks()

        print("compact_mode = ", compact_mode)
        # prepare theta_0 from model_a
        # check model_a
        checkpoint_info = sd_models.get_closet_checkpoint_match(model_a)
        if checkpoint_info is None:
            print(f"ERROR: Fail to get {model_a}")
            return
        model_a = checkpoint_info.model_name
        print(f"model_a = {model_a}")

        # load models
        models = {}
        timer = Timer()

        sd_models.load_model(checkpoint_info)
        models['model_a'] = None
        # get cached state_dict
        if shared.opts.sd_checkpoint_cache > 0:
            models['model_a'] = sd_models.get_checkpoint_state_dict(checkpoint_info, timer)
            # check validity of cached state_dict
            keylen = len(models['model_a'].keys())
            if keylen < 686: # for SD-v1, SD-v2
                models['model_a'] = None
                print(f"Invalid cached state_dict...")

        if models['model_a'] is None:
            # read state_dict from file
            print(f"Loading {checkpoint_info.filename}...")
            models['model_a'] = sd_models.read_state_dict(checkpoint_info.filename, map_location = "cpu")

        # check base_model
        model_base = {}
        if "Add-Diff" in mm_modes:
            if base_model is None:
                # check SD version
                w = models['model_a']["model.diffusion_model.input_blocks.1.1.proj_in.weight"]
                if len(w.shape) == 4:
                    base_model = "v1-5-pruned-emaonly"
                else:
                    base_model = "v2-1_768-nonema-pruned"
                print(f"base_model automatically detected as {base_model}")

            checkpointinfo = sd_models.get_closet_checkpoint_match(base_model)
            model_base = sd_models.read_state_dict(checkpointinfo.filename, map_location = "cpu")

        # setup selected keys
        theta_0 = {}
        keys = []
        keyremains = []
        if compact_mode:
            keys = _get_selected_keys(selected_blocks, models['model_a'])

            # get keylist of all selected blocks
            for k in models['model_a'].keys():
                keyadded = False
                for s in selected_blocks:
                    if s in k:
                        keys.append(k)
                        theta_0[k] = models['model_a'][k]
                        keyadded = True
                if not keyadded:
                    keyremains.append(k)

        else:
            # get all keys()
            keys = list(models['model_a'].keys())
            theta_0 = models['model_a'].copy()

        # save some dicts
        checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]
        for k in checkpoint_dict_skip_on_merge:
            if k in keys:
                keys.remove(k)
                item = theta_0.pop(k)
                keyremains.append(k)

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
            "calcmode": "normal",
        }
        metadata["sd_merge_recipe"] = merge_recipe
        metadata["sd_merge_models"] = {}

        # full recipe
        recipe_all = None

        def add_model_metadata(checkpoint_name):
            checkpointinfo = sd_models.get_closet_checkpoint_match(checkpoint_name)
            checkpointinfo.calculate_shorthash()
            metadata["sd_merge_models"][checkpointinfo.sha256] = {
                "name": checkpoint_name,
                "legacy_hash": checkpointinfo.hash
            }

        if model_a:
            add_model_metadata(model_a)
        if base_model is not None:
            add_model_metadata(base_model)

        # merge main
        weight_start = 0
        # total stage = number of models + key uninitialized stage + key remains stage
        stages = len(mm_models) + 1 + (1 if len(keyremains) > 0 else 0)
        modes = mm_modes

        # model info
        modelinfos = [ model_a ]
        modelhashes = [ checkpoint_info.calculate_shorthash() ]
        alphas = []
        # XXX HACK
        checkpoint_info = deepcopy(checkpoint_info)

        stage = 1
        for n, file in enumerate(mm_models,start=weight_start):
            checkpointinfo = sd_models.get_closet_checkpoint_match(file)
            model_name = checkpointinfo.model_name
            print(f"Loading model {model_name}...")
            theta_1 = sd_models.read_state_dict(checkpointinfo.filename, map_location = "cpu")

            model_b = f"model_{chr(97+n+1-weight_start)}"
            merge_recipe[model_b] = model_name
            modelinfos.append(model_name)
            modelhashes.append(checkpointinfo.calculate_shorthash())

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
            for key in (tqdm(keys, desc=f"Stage #{stage}/{stages}")):
                if "model_" in key:
                    continue
                if key in checkpoint_dict_skip_on_merge:
                    continue
                if "model" in key and key in theta_0:
                    if usembw:
                        i = _weight_index(key)
                        if i == -1: continue # not found
                        alpha = mm_weights[n][i]

                    if modes[n] == "Sum":
                        theta_0[key] = (1 - alpha) * (theta_0[key]) + alpha * theta_1[key]
                    else:
                        theta_0[key] = theta_0[key] + (theta_1[key] - model_base[key]) * alpha

            # recipe string
            if modes[n] == "Sum":
                if recipe_all is None:
                    recipe_all = f"{model_a} * (1 - alpha_{n}) + {model_name} * alpha_{n}"
                else:
                    recipe_all = f"({recipe_all}) * (1 - alpha_{n}) + {model_name} * alpha_{n}"
            elif modes[n] in [ "Add-Diff" ]:
                if recipe_all is None:
                    recipe_all = f"{model_a} + ({model_name} - {base_model}) * alpha_{n}"
                else:
                    recipe_all = f"{recipe_all} + ({model_name} - {base_model}) * alpha_{n}"

            if n == weight_start:
                stage += 1
                for key in (tqdm(keys, desc=f"Check uninitialized #{n+2-weight_start}/{stages}")):
                    if "model" in key:
                        for s in selected_blocks:
                            if s in key and key not in theta_0 and key not in checkpoint_dict_skip_on_merge:
                                print(f" +{k}")
                                theta_0[key] = theta_1[key]

            stage += 1
            del theta_1

        # store unmodified remains
        for key in (tqdm(keyremains, desc=f"Save unchanged weights #{stages}/{stages}")):
            theta_0[key] = models['model_a'][key]

        # load theta_0, checkpoint_info was used for model_a
        # XXX make a FAKE checkpoint_info
        # change model name (name_for_extra field used webui internally)
        checkpoint_info.name_for_extra = " + ".join(modelinfos)

        checkpoint_info.sha256 = sha256
        checkpoint_info.name = checkpoint_info.name_for_extra + ".safetensors"
        checkpoint_info.model_name = checkpoint_info.name_for_extra.replace("/", "_").replace("\\", "_")
        checkpoint_info.title = f"{checkpoint_info.name} [{sha256[0:10]}]"

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

        sd_models.load_model(checkpoint_info=checkpoint_info, already_loaded_state_dict=theta_0)
        if shared.opts.sd_checkpoint_cache > 0:
            # unload cached merged model
            checkpoints_loaded.popitem()

        del theta_0

        # update merged model info.
        alphastr = ','.join(['(' + ','.join(map(lambda x: str(int(x)) if x == 0.0 else str(x), sub)) + ')' for sub in alphas])
        shared.opts.data["sd_webui_model_mixer_model"] = {
            "hash": sha256,
            "models" : modelinfos,
            "hashes" : modelhashes,
            "recipe": recipe_all + alphastr,
            "metadata": metadata,
        }
        return

def on_image_save(params):
    if 'parameters' not in params.pnginfo: return

    # load mixed model info
    model = shared.opts.data.get("sd_webui_model_mixer_model", None)
    if model is None: return
    sha256 = model["hash"]
    if shared.sd_model.sd_checkpoint_info is not None and shared.sd_model.sd_checkpoint_info.sha256 != sha256:
        return

    modelinfos = model["models"]
    modelhashes = model["hashes"]
    recipe = model.get("recipe", None)

    lines = params.pnginfo['parameters'].split('\n')
    generation_params = lines.pop()
    prompt_parts = '\n'.join(lines).split('Negative prompt:')
    prompt, negative_prompt = [s.strip() for s in prompt_parts[:2] + ['']*(2-len(prompt_parts))]

    lines = generation_params.split(",")
    for i,x in enumerate(lines):
        if "Model:" in x:
            lines[i] = " Model: " + " + ".join(modelinfos).replace(","," ")
        elif "Model hash:" in x:
            lines[i] = " Model hash: " + ", Model hash: ".join(modelhashes)
    generation_params = ",".join(lines)
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

def on_infotext_pasted(infotext, results):
    updates = {}
    for k, v in results.items():
        if not k.startswith("ModelMixer"):
            continue

        if k.find(" mbw ") > 0:
            if v in [ "True", "False"]:
                continue
            elif k.find(" weights ") > 0:
                continue

            if v[0] == '"' and v[-1] == '"':
                v = v[1:-1]
            arr = v.split(",")
            updates[k] = arr

    results.update(updates)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(on_image_save)
script_callbacks.on_infotext_pasted(on_infotext_pasted)
