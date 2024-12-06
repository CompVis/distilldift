import torch
from .base import CacheModel
from extractors.vit import HookExtractor
from torchvision.transforms import Normalize

class EfficientViT(CacheModel):
    """
    EfficientViT model.

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(EfficientViT, self).__init__(config)
        
        self.layers = config["layers"]
        self.half_precision = config.get("half_precision", False)

        self.extractor = HookExtractor('EfficientViT', self.half_precision)

        self.weights = config.get("weights", None)
        if self.weights is not None:
            self.load_state_dict(torch.load(self.weights, map_location="cpu"))

        self.params_to_optimize = self.extractor.model.parameters()
    
    def get_features(self, image, category=None):
        image = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(image)
        features = self.extractor(image, layers=self.layers)
        return list(features.values())
