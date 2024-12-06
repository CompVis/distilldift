from .base import CacheModel
from extractors.diffusion import SDHookExtractor

class DiffusionHook(CacheModel):
    """
    Diffusion model.

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(DiffusionHook, self).__init__(config)
        
        self.model = config["model"]
        self.layers = config["layers"]
        self.step = config["step"]

        self.extractor = SDHookExtractor("cuda", self.model)
    
    def get_features(self, image, prompt):
        if isinstance(self.step, int):
            features = self.extractor(image, prompt=prompt, layers=self.layers, steps=[self.step])[self.step]
            return list(features.values())
        else:
            features = self.extractor(image, prompt=prompt, layers=self.layers, steps=self.step)
            return [list(f.values())[0] for f in features.values()]
