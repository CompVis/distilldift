import torch
import torch.nn as nn
import gc
import copy

from transformers import T5EncoderModel
from diffusers import PixArtAlphaPipeline
from diffusers.schedulers import DDIMScheduler

class HookExtractor(nn.Module):
    def __init__(self, half_precision=False):
        super(HookExtractor, self).__init__()

        self.text_encoder = T5EncoderModel.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            subfolder="text_encoder",
            load_in_8bit=True,
        )
        self.pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            text_encoder=self.text_encoder,
            torch_dtype=torch.float16
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae.decoder = None

        self.pipe.enable_attention_slicing()
        self.pipe.enable_xformers_memory_efficient_attention()
        gc.collect()

    # Pipe is not a nn.Module, so it needs to be moved explicitly
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.pipe.to(*args, **kwargs)
        return self
    
    def save_fn(self, layer_idx):
        def hook(module, input, output):
            b, n, c = output.shape
            print(output.shape, layer_idx)
            self.features[layer_idx] = output.permute(0, 2, 1)[int(b/2):].reshape(int(b/2), c, 48, 48)
        return hook

    def __call__(self, images, prompt, layers=[5], steps=[101]):
        # Set hooks at the specified layers
        self.features = {}
        hooks = []
        layer_counter = 0
        for block in self.pipe.transformer.transformer_blocks:
            if layer_counter in layers:
                hooks.append(block.attn2.to_q.register_forward_hook(self.save_fn(layer_counter)))
            layer_counter += 1

        prompt_embeddings, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = self.pipe.encode_prompt(prompt)

        if images.dtype != prompt_embeddings.dtype:
            images = images.to(prompt_embeddings.dtype)

        # Check prompt
        if images.shape[0] != prompt_embeddings.shape[0]:
            if isinstance(prompt, str):
                if prompt is None:
                    raise ValueError("Prompt is not provided")
                else:
                    prompt_embeddings = prompt_embeddings.repeat(images.shape[0], 1, 1)
            else:
                raise ValueError("Batch-size does not match number of prompts")
        
        # TODO: remove
        self.prompt_embeddings = prompt_embeddings
        
        # Encode images
        latents = self.pipe.vae.encode(images).latent_dist.sample() * self.pipe.vae.config.scaling_factor

        # Run model
        features = {}
        for t in steps:
            # Add noise from scheduler
            noise = torch.randn_like(latents).to(images.device)
            ti = torch.tensor(t, dtype=torch.long, device=images.device)
            latents_noisy = self.pipe.scheduler.add_noise(latents, noise, ti)
            
            self.pipe(
                latents=latents_noisy,
                num_inference_steps=1,
                num_images_per_prompt=1,
                negative_prompt=None, 
                prompt_embeds=prompt_embeddings,
                negative_prompt_embeds=negative_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                output_type="latent"
            )
            features[t] = copy.deepcopy(self.features)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return features