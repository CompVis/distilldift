import torch
from .base import CacheModel

class ZoeDepth(CacheModel):
    """
    ZoeDepth model.

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(ZoeDepth, self).__init__(config)
        
        self.patch_size = 16 # BeiT
        self.layers = config["layers"]
        self.version = config["version"]
        self.extractor = torch.hub.load("isl-org/ZoeDepth", 'ZoeD_' + self.version, pretrained=True)
        self.extractor.eval()

        # Set hooks at the specified layers
        layer_counter = 0
        self.features = {}

        # BeiT encoder layers 0-23
        for block in self.extractor.core.core.pretrained.model.blocks:
            if layer_counter in self.layers:
                block.register_forward_hook(self.save_fn(layer_counter))
            layer_counter += 1
        
        # Postprocess layers 24-27
        self.extractor.core.core.pretrained.act_postprocess1.register_forward_hook(self.save_fn(layer_counter))
        layer_counter += 1
        self.extractor.core.core.pretrained.act_postprocess2.register_forward_hook(self.save_fn(layer_counter))
        layer_counter += 1
        self.extractor.core.core.pretrained.act_postprocess3.register_forward_hook(self.save_fn(layer_counter))
        layer_counter += 1
        self.extractor.core.core.pretrained.act_postprocess4.register_forward_hook(self.save_fn(layer_counter))
        layer_counter += 1

        # Residual decoder blocks (refinenet) 28-31
        self.extractor.core.core.scratch.refinenet1.register_forward_hook(self.save_fn(layer_counter))
        layer_counter += 1
        self.extractor.core.core.scratch.refinenet2.register_forward_hook(self.save_fn(layer_counter))
        layer_counter += 1
        self.extractor.core.core.scratch.refinenet3.register_forward_hook(self.save_fn(layer_counter))
        layer_counter += 1
        self.extractor.core.core.scratch.refinenet4.register_forward_hook(self.save_fn(layer_counter))
        layer_counter += 1

    def save_fn(self, layer_idx):
        def hook(module, input, output):
            self.features[layer_idx] = output
        return hook
    
    def get_features(self, image, category=None):
        self.features = {}
        _ = self.extractor.infer(image)
        b = image.shape[0]
        h = image.shape[2] // self.patch_size
        w = image.shape[3] // self.patch_size
        return [l[:, 1:].permute(0, 2, 1).reshape(b, -1, h, w) if len(l.shape) == 3 else l for l in self.features.values()]
