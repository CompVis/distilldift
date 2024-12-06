import torch

from .base import CacheModel
from extractors.diffusion import SDExtractor
from torchvision.transforms import Normalize, Resize

from transformers import BlipProcessor, BlipForConditionalGeneration

class Prompt(CacheModel):
    """
    Prompt model.

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(Prompt, self).__init__(config)
        
        self.model = config["model"]
        self.layers = config["layers"]
        self.step = config["step"]
        self.prompt_mode = config["prompt_mode"]

        self.extractor = SDExtractor(self.model)

        if self.prompt_mode == 'caption':
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    def get_features(self, image, category):
        if self.prompt_mode == 'empty':
            prompt = [''] * len(category)
        elif self.prompt_mode == 'category':
            prompt = [f'a photo of a {c}' for c in category]
        elif self.prompt_mode == 'caption':
            blip_image = Resize((384, 384), interpolation=2)(image)
            blip_image = (blip_image + 1) / 2
            blip_image = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(blip_image)
            out = self.blip_model.generate(blip_image)
            prompt = [self.blip_processor.decode(o, skip_special_tokens=True) for o in out]

        features = self.extractor(image, prompt=prompt, layers=self.layers, steps=[self.step])[self.step]
        return list(features.values())
