"""
lora networks state_dict bug patch
"""
import torch

from typing import Union
from modules import devices, patches


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    import networks

    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)

    if weights_backup is None and bias_backup is None:
        return

    current_names = getattr(self, "network_current_names", ())
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in networks.loaded_networks)

    # save lora weights
    weights_lora = getattr(self, "network_weights_lora", None)
    bias_lora = getattr(self, "network_bias_lora", None)

    if weights_lora is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_lora = (self.in_proj_weight.to(devices.cpu, copy=True), self.out_proj.weight.to(devices.cpu, copy=True))
        else:
            weights_lora = self.weight.to(devices.cpu, copy=True)

        self.network_weights_lora = weights_lora

    if bias_lora is None:
        if isinstance(self, torch.nn.MultiheadAttention) and self.out_proj.bias is not None:
            bias_lora = self.out_proj.bias.to(devices.cpu, copy=True)
        elif getattr(self, 'bias', None) is not None:
            bias_lora = self.bias.to(devices.cpu, copy=True)
        else:
            bias_lora = None
        self.network_bias_lora = bias_lora

    # weights from backup
    if weights_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.in_proj_weight.copy_(weights_backup[0])
            self.out_proj.weight.copy_(weights_backup[1])
        else:
            self.weight.copy_(weights_backup)

    if bias_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.out_proj.bias.copy_(bias_backup)
        else:
            self.bias.copy_(bias_backup)
    else:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.out_proj.bias = None
        else:
            self.bias = None


def network_restore_weights_from_lora(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    import networks

    weights_backup = getattr(self, "network_weights_lora", None)
    bias_backup = getattr(self, "network_bias_lora", None)

    if weights_backup is None and bias_backup is None:
        return

    # weights from lora
    if weights_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.in_proj_weight.copy_(weights_backup[0])
            self.out_proj.weight.copy_(weights_backup[1])
        else:
            self.weight.copy_(weights_backup)

    if bias_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.out_proj.bias.copy_(bias_backup)
        else:
            self.bias.copy_(bias_backup)
    else:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.out_proj.bias = None
        else:
            self.bias = None

    # reset
    self.network_weights_lora = None
    self.network_bias_lora = None


class StateDictPatches:
    def __init__(self):
        import networks

        def network_Linear_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                network_restore_weights_from_backup(module)

            return self.Linear_state_dict(module, *args, **kwargs)


        def network_Conv2d_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                network_restore_weights_from_backup(module)

            return self.Conv2d_state_dict(module, *args, **kwargs)


        def network_GroupNorm_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                network_restore_weights_from_backup(module)

            return self.GroupNorm_state_dict(module, *args, **kwargs)


        def network_LayerNorm_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                network_restore_weights_from_backup(module)

            return self.LayerNorm_state_dict(module, *args, **kwargs)


        def network_MultiheadAttention_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                network_restore_weights_from_backup(module)

            return self.MultiheadAttention_state_dict(module, *args, **kwargs)


        self.Linear_state_dict = patches.patch(__name__, torch.nn.Linear, 'state_dict', network_Linear_state_dict)
        self.Conv2d_state_dict = patches.patch(__name__, torch.nn.Conv2d, 'state_dict', network_Conv2d_state_dict)
        self.GroupNorm_state_dict = patches.patch(__name__, torch.nn.GroupNorm, 'state_dict', network_GroupNorm_state_dict)
        self.LayerNorm_state_dict = patches.patch(__name__, torch.nn.LayerNorm, 'state_dict', network_LayerNorm_state_dict)
        self.MultiheadAttention_state_dict = patches.patch(__name__, torch.nn.MultiheadAttention, 'state_dict', network_MultiheadAttention_state_dict)


    def undo(self):
        self.Linear_state_dict = patches.undo(__name__, torch.nn.Linear, 'state_dict')
        self.Conv2d_state_dict = patches.undo(__name__, torch.nn.Conv2d, 'state_dict')
        self.GroupNorm_state_dict = patches.undo(__name__, torch.nn.GroupNorm, 'state_dict')
        self.LayerNorm_state_dict = patches.undo(__name__, torch.nn.LayerNorm, 'state_dict')
        self.MultiheadAttention_state_dict = patches.undo(__name__, torch.nn.MultiheadAttention, 'state_dict')


class StateDictLoraPatches:
    def __init__(self):
        import networks

        def network_Linear_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                network_restore_weights_from_lora(module)

            return self.Linear_state_dict(module, *args, **kwargs)


        def network_Conv2d_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                network_restore_weights_from_lora(module)

            return self.Conv2d_state_dict(module, *args, **kwargs)


        def network_GroupNorm_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                network_restore_weights_from_lora(module)

            return self.GroupNorm_state_dict(module, *args, **kwargs)


        def network_LayerNorm_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                network_restore_weights_from_lora(module)

            return self.LayerNorm_state_dict(module, *args, **kwargs)


        def network_MultiheadAttention_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                network_restore_weights_from_lora(module)

            return self.MultiheadAttention_state_dict(module, *args, **kwargs)


        self.Linear_state_dict = patches.patch(__name__, torch.nn.Linear, 'state_dict', network_Linear_state_dict)
        self.Conv2d_state_dict = patches.patch(__name__, torch.nn.Conv2d, 'state_dict', network_Conv2d_state_dict)
        self.GroupNorm_state_dict = patches.patch(__name__, torch.nn.GroupNorm, 'state_dict', network_GroupNorm_state_dict)
        self.LayerNorm_state_dict = patches.patch(__name__, torch.nn.LayerNorm, 'state_dict', network_LayerNorm_state_dict)
        self.MultiheadAttention_state_dict = patches.patch(__name__, torch.nn.MultiheadAttention, 'state_dict', network_MultiheadAttention_state_dict)


    def undo(self):
        self.Linear_state_dict = patches.undo(__name__, torch.nn.Linear, 'state_dict')
        self.Conv2d_state_dict = patches.undo(__name__, torch.nn.Conv2d, 'state_dict')
        self.GroupNorm_state_dict = patches.undo(__name__, torch.nn.GroupNorm, 'state_dict')
        self.LayerNorm_state_dict = patches.undo(__name__, torch.nn.LayerNorm, 'state_dict')
        self.MultiheadAttention_state_dict = patches.undo(__name__, torch.nn.MultiheadAttention, 'state_dict')
