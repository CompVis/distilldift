from .base import CacheModel
from extractors.diffusion import SDExtractor

class Diffusion(CacheModel):
    """
    Diffusion model.

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(Diffusion, self).__init__(config)
        
        self.model = config["model"]
        self.layers = config["layers"]
        self.step = config["step"]
        self.half_precision = config.get("half_precision", False)

        self.extractor = SDExtractor(self.model, self.half_precision)
    
    def get_features(self, image, category):
        prompt = [f'a photo of a {c}' for c in category]
        if isinstance(self.step, int):
            features = self.extractor(image, prompt=prompt, layers=self.layers, steps=[self.step])[self.step]
            return list(features.values())
        else:
            features = self.extractor(image, prompt=prompt, layers=self.layers, steps=self.step)
            return [list(f.values())[0] for f in features.values()]
