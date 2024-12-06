import torch

from .base import CacheModel
from torch.nn.functional import interpolate

class Combination(CacheModel):
    """
    Diffusion model.

    Args:
        config (dict): Model config
        models (CacheModel): Models to combine
    """
    def __init__(self, config, models):
        super(Combination, self).__init__(config)
        
        self.models = models

        self.output_all = config.get('output_all', False)
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for model in self.models:
            model.to(*args, **kwargs)
        return self
    
    def get_features(self, image, category):
        features = [model.get_features(interpolate(image, model.config['image_size'], mode='bilinear'), category) for model in self.models]
        for i, f in enumerate(features):
            if len(f) == 1:
                features[i] = f[0]
            else:
                image_size = max([f.shape[-1] for f in f])
                features[i] = torch.cat([interpolate(feat, size=image_size, mode='bilinear') for feat in f], dim=1)

        # normalize
        features = [f / f.norm(dim=1, keepdim=True) for f in features]

        # interpolate and concatenate
        image_size = max([f.shape[-1] for f in features])
        features_int = [interpolate(f, size=image_size, mode='bilinear') for f in features]
        if self.output_all:
            return [*features, torch.cat(features_int, dim=1)]
        return torch.cat(features_int, dim=1)
