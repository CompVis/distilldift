import torch

from .base import CacheModel
from extractors.diffusion import SDExtractor
from torchvision.transforms import RandomResizedCrop

class Ensemble(CacheModel):
    """
    Ensemble model.

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(Ensemble, self).__init__(config)
        
        self.model = config["model"]
        self.layers = config["layers"]
        self.steps = config["steps"]
        self.ensemble_size = config["ensemble_size"]
        self.random_cropping = config["random_cropping"]

        self.extractor = SDExtractor(self.model)

    def get_features(self, image, category):
        prompt = [f'a photo of a {c}' for c in category]
        if self.ensemble_size > 1 and len(self.steps) == 1:
            features = {}
            for k in range(self.ensemble_size):
                if self.random_cropping:
                    image_preprocessed = RandomResizedCrop(image.shape[-2:], scale=(0.95, 0.95), ratio=(1.0, 1.0))(image)
                else:
                    image_preprocessed = image
                features[k] = self.extractor(image_preprocessed, prompt=prompt, layers=self.layers, steps=self.steps)[self.steps[0]]
        else:
            features = self.extractor(image, prompt=prompt, layers=self.layers, steps=self.steps)

        features = list(zip(*[s.values() for s in features.values()])) # (steps, layers, b, c, H, W) -> (layers, steps, b, c, H, W)
        features = [torch.stack(l).mean(0) for l in features] # (layers, steps, b, c, H, W) -> (layers, b, c, H, W)
        return features
