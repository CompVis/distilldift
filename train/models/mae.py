import torch
from .base import CacheModel

import sys
sys.path.append('./thirdparty/mae')

from torchvision.transforms import Normalize

class MAE(CacheModel):
    """
    MAE model.

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(MAE, self).__init__(config)
        
        self.patch_size = config["patch_size"]
        self.layers = config["layers"]
        self.model_path = config["model_path"]
        self.arch = config["arch"]
        
        # Load model
        import models_mae
        
        self.extractor = getattr(models_mae, self.arch)()
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.extractor.load_state_dict(checkpoint['model'], strict=False)

        # Set hooks at the specified layers
        layer_counter = 0
        self.features = {}

        # Encoder blocks 0-23
        for block in self.extractor.blocks:
            if layer_counter in self.layers:
                block.register_forward_hook(self.save_fn(layer_counter))
            layer_counter += 1

        # Decoder layers 24-31
        for layer in self.extractor.decoder_blocks:
            if layer_counter in self.layers:
                layer.register_forward_hook(self.save_fn(layer_counter))
            layer_counter += 1

    def save_fn(self, layer_idx):
        def hook(module, input, output):
            self.features[layer_idx] = output
        return hook
    
    def get_features(self, image, category=None):
        self.features = {}
        image = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(image) # important
        _ = self.extractor(image, mask_ratio=0.0)
        b = image.shape[0]
        h = image.shape[2] // self.patch_size
        w = image.shape[3] // self.patch_size
        return [l[:, 1:].permute(0, 2, 1).reshape(b, -1, h, w) if len(l.shape) == 3 else l for l in self.features.values()]
