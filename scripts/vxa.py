import html
import os
import gradio as gr
from PIL import Image
import numpy as np
import open_clip.tokenizer
import re
import torch
from modules import devices, shared, extra_networks, prompt_parser
from torch import nn, einsum
from einops import rearrange
import math
from ldm.modules.attention import CrossAttention
from sgm.modules.attention import CrossAttention as CrossAttention2
from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
from modules.ui import create_refresh_button

hidden_layer_names = []
default_hidden_layer_name = "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2"

# from https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer

class VanillaClip:
    def __init__(self, clip):
        self.clip = clip

    def vocab(self):
        return self.clip.tokenizer.get_vocab()

    def byte_decoder(self):
        return self.clip.tokenizer.byte_decoder

class OpenClip:
    def __init__(self, clip):
        self.clip = clip
        self.tokenizer = open_clip.tokenizer._tokenizer

    def vocab(self):
        return self.tokenizer.encoder

    def byte_decoder(self):
        return self.tokenizer.byte_decoder

def tokenize(text, input_is_ids=False):
    if shared.sd_model is None:
        raise gr.Error("Model not loaded...")

    text, res = extra_networks.parse_prompt(text)
    #_, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
    #prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, 20)

    conditioner = getattr(shared.sd_model, 'conditioner', None)
    if conditioner:
        # SDXL, only the first embedder selected
        cond_stage_model = conditioner.embedders[0]
    else:
        # SD1.5, SD2.1
        cond_stage_model = shared.sd_model.cond_stage_model

    typename = type(cond_stage_model.wrapped).__name__
    if typename == "FrozenCLIPEmbedder":
        clip = VanillaClip(cond_stage_model.wrapped)
    elif typename == "FrozenOpenCLIPEmbedder":
        clip = OpenClip(cond_stage_model.wrapped)
    else:
        raise gr.Error(f'Unknown CLIP model: {typename}')

    if input_is_ids:
        tokens = [int(x.strip()) for x in text.split(",")]
    else:
        tokens = cond_stage_model.tokenize([text])[0]

    vocab = {v: k for k, v in clip.vocab().items()}

    code = ''
    ids = []
    choices = []

    current_ids = []
    class_index = 0

    byte_decoder = clip.byte_decoder()

    def dump(last=False):
        nonlocal code, ids, current_ids
        nonlocal choices

        words = [vocab.get(x, "") for x in current_ids]

        def wordscode(n, ids, word):
            nonlocal class_index
            w = html.escape(word)
            if bool(re.match("[-_,.0-9:;()\[\]]", w)):
                class_name = 'mm-vxa-punct'
                n = ''
            else:
                class_name = f"mm-vxa-token mm-vxa-token-{class_index%4}"
                class_index += 1
                choices.append((w, n+1))
                if n is not None: n = f"({n+1})"
            res = f"""<span class='{class_name}' title='{html.escape(", ".join([str(x) for x in ids]))}'>{w}{n}</span>"""
            return res

        try:
            word = bytearray([byte_decoder[x] for x in ''.join(words)]).decode("utf-8")
        except UnicodeDecodeError:
            if last:
                word = "❌" * len(current_ids)
            elif len(current_ids) > 4:
                id = current_ids[0]
                ids += [id]
                local_ids = current_ids[1:]
                code += wordscode([id], "❌")

                current_ids = []
                for id in local_ids:
                    current_ids.append(id)
                    dump()

                return
            else:
                return

        word = word.replace("</w>", " ")

        code += wordscode(len(ids), current_ids, word)
        ids += current_ids

        current_ids = []

    for j, token in enumerate(tokens):
        token = int(token)
        current_ids.append(token)

        dump()

    if len(current_ids) > 0:
        dump(last=True)

    ids_html = f"""
<p>
Token count: {len(ids)}<br>
{", ".join([str(x) for x in ids])}
</p>
"""

    return code, ids_html, gr.update(choices=choices, value=[])

def get_layer_names(model=None):
    if model is None:
        if shared.sd_model is None:
            return [default_hidden_layer_name]
        else:
            model = shared.sd_model

    hidden_layers = []
    for n, m in model.named_modules():
        if isinstance(m, CrossAttention) or isinstance(m, CrossAttention2):
            hidden_layers.append(n)
    return list(filter(lambda s : "attn2" in s, hidden_layers))

def get_attn(emb, ret):
    def hook(self, sin, sout):
        h = self.heads
        q = self.to_q(sin[0])

        if type(emb) is dict:
            context = emb.get("crossattn", sin[0])
        else:
            context = emb

        k = self.to_k(context)
        q, k = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q, k))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        ret["out"] = attn
    return hook

def generate_vxa(image, prompt, idx, time, layer_name, output_mode):
    if isinstance(image, str):
        print("Invalid str image", image)
        return None
    elif isinstance(image, Image.Image):
        image = np.asarray(image)
    elif not isinstance(image, np.ndarray):
        print("Invalid image #")
        return image
    output = image.copy()
    image = image.astype(np.float32) / 255.0
    image = np.moveaxis(image, 2, 0)
    image = torch.from_numpy(image).unsqueeze(0)

    model = shared.sd_model
    layer = None
    print(f"Search {layer_name}...")
    for n, m in model.named_modules():
        if (isinstance(m, CrossAttention) or isinstance(m, CrossAttention2)) and n == layer_name:
            print("layer found = ", n)
            layer = m
            break
    if layer is None:
        print("Layer not found")
        return image

    with torch.no_grad(), devices.autocast():
        image = image.to(devices.device, dtype=devices.dtype_vae)
        latent = model.get_first_stage_encoding(model.encode_first_stage(image))
        try:
            t = torch.tensor([float(time)]).to(devices.device)
        except:
            print(f"Not a valid timesteps {time}")
            return output

        attn_out = {}
        if hasattr(model, 'conditioner'):
            emb = model.get_learned_conditioning([prompt])
        else:
            emb = model.cond_stage_model([prompt])

        handle = layer.register_forward_hook(get_attn(emb, attn_out))
        try:
            model.apply_model(latent, t, emb)
        except:
            print("Error apply_model()")
        finally:
            handle.remove()

    if (idx == ""):
        img = attn_out["out"][:,:,1:].sum(-1).sum(0)
    else:
        try:
            idxs = [int(x) for x in idx.strip().split(',') if x.strip().isdigit()]
            img = attn_out["out"][:,:,idxs].sum(-1).sum(0)
        except:
            print("Fail to get attn_out")
            return output

    scale = round(math.sqrt((image.shape[2] * image.shape[3]) / img.shape[0]))
    h = image.shape[2] // scale
    w = image.shape[3] // scale
    img = img.reshape(h, w) / img.max()
    img = img.to("cpu").numpy()
    output = output.astype(np.float64)
    if output_mode == "masked":
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i][j] *= img[i // scale][j // scale]
    elif output_mode == "grey":
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i][j] = [img[i // scale][j // scale] * 255.0] * 3
    output = output.astype(np.uint8)

    devices.torch_gc()

    print("Done")
    return output
