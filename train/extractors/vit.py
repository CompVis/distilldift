import torch
import torch.nn as nn
import gc
import copy

import timm

class HookExtractor(nn.Module):
    def __init__(self, name, half_precision=False):
        super(HookExtractor, self).__init__()

        self.name = name
        if name == 'IJEPA':
            self.model = timm.create_model("vit_huge_patch16_gap_448.in1k_ijepa")
            ckpt = torch.load(".cache/torch/hub/checkpoints/IN1K-vit.h.16-448px-300e.pth.tar")

            pretrained_dict = ckpt['encoder']

            for k, v in pretrained_dict.items():
                self.model.state_dict()[k[len("module."):]].copy_(v)
        elif name == 'EfficientViT':
            self.model = timm.create_model('efficientvit_b3.r224_in1k')

        self.model.eval()
        gc.collect()
    
    def save_fn(self, layer_idx):
        def hook(module, input, output):
            if len(output.shape) == 3:
                b, n, c = output.shape
                self.features[layer_idx] = output.permute(0, 2, 1).reshape(b, c, int(n**0.5), int(n**0.5))
            elif len(output.shape) == 4:
                b, n, h, w = output.shape
                self.features[layer_idx] = output.permute(0, 2, 3, 1).reshape(b, h, w, n)
        return hook

    def __call__(self, images, layers=[5]):
        if self.name == 'IJEPA':
            blocks = self.model.blocks
        elif self.name == 'EfficientViT':
            blocks = [b for stage in self.model.stages for b in stage.blocks]

        # Set hooks at the specified layers
        self.features = {}
        hooks = []
        layer_counter = 0
        for block in blocks:
            if layer_counter in layers:
                hooks.append(block.register_forward_hook(self.save_fn(layer_counter)))
            layer_counter += 1

        # Run model
        self.model(images)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return self.features # copy.deepcopy(self.features)