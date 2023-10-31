"""
lora networks state_dict bug patch
"""
import torch

from modules import patches, script_callbacks


class StateDictPatches:
    def __init__(self):
        import networks

        def network_Linear_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                networks.network_restore_weights_from_backup(module)

            return self.Linear_state_dict(module, *args, **kwargs)


        def network_Conv2d_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                networks.network_restore_weights_from_backup(module)

            return self.Conv2d_state_dict(module, *args, **kwargs)


        def network_GroupNorm_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                networks.network_restore_weights_from_backup(module)

            return self.GroupNorm_state_dict(module, *args, **kwargs)


        def network_LayerNorm_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                networks.network_restore_weights_from_backup(module)

            return self.LayerNorm_state_dict(module, *args, **kwargs)


        def network_MultiheadAttention_state_dict(module, *args, **kwargs):
            with torch.no_grad():
                networks.network_restore_weights_from_backup(module)

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
