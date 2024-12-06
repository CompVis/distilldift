import torch
from .base import CacheModel
from torch.nn.functional import interpolate

class DINO(CacheModel):
    """
    DINO models.

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(DINO, self).__init__(config)

        self.version = config["version"]
        self.model_size = config["model_size"]
        self.patch_size = config["patch_size"]
        self.registers = config["registers"]
        self.layers = config["layers"]

        if self.version == 1:
            repo = 'facebookresearch/dino:main'
            model = 'dino_vit' + self.model_size + str(self.patch_size)
        elif self.version == 2:
            repo = 'facebookresearch/dinov' + str(self.version)
            model = 'dinov2_vit' + self.model_size + str(self.patch_size)
            if self.registers:
                model += '_reg'
        
        self.extractor = torch.hub.load(repo, model)

    def get_features(self, image, category=None):
        b = image.shape[0]
        h = image.shape[2] // self.patch_size
        w = image.shape[3] // self.patch_size
        num_layers_from_bottom = len(self.extractor.blocks) - min(self.layers)

        if self.version == 1:
            features = [f[:, 1:] for f in self.extractor.get_intermediate_layers(image, num_layers_from_bottom)] # remove class token
        elif self.version == 2:
            features = self.extractor.get_intermediate_layers(image, num_layers_from_bottom, return_class_token=False)
        
        return [features[l - min(self.layers)].permute(0, 2, 1).reshape(b, -1, h, w) for l in self.layers]
    