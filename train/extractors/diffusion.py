import gc
import torch
import numpy as np
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from diffusers.models.unet_2d_blocks import UpBlock2D
from diffusers.models.unet_2d_blocks import CrossAttnUpBlock2D
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from diffusers.utils.import_utils import is_torch_version

def custom_forward_UpBlock2D(self,
    hidden_states: torch.FloatTensor,
    res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
    temb: Optional[torch.FloatTensor] = None,
    upsample_size: Optional[int] = None,
    store_intermediates=False # new parameter
):
    intermediates = []
    for resnet in self.resnets:
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            if is_torch_version(">=", "1.11.0"):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                )
            else:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
        else:
            hidden_states = resnet(hidden_states, temb)
        if store_intermediates:
            intermediates.append(hidden_states) #.detach())

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)
            if store_intermediates:
                intermediates.append(hidden_states) #.detach())
        
    return hidden_states, intermediates


def custom_forward_CrossAttnUpBlock2D(
    self,
    hidden_states: torch.FloatTensor,
    res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
    temb: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    upsample_size: Optional[int] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    store_intermediates=False # new parameter
):
    intermediates = []
    for resnet, attn in zip(self.resnets, self.attentions):
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        if store_intermediates:
            intermediates.append(hidden_states) #.detach())

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)
            if store_intermediates:
                intermediates.append(hidden_states) #.detach())

    return hidden_states, intermediates

class CustomUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        feature_indices, # New parameter
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        features = {}
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

            if 0 in feature_indices:
                features[0] = sample
        
        # 5. up
        layer_counter = 1
        for i, upsample_block in enumerate(self.up_blocks):
            if layer_counter > np.max(feature_indices):
                break
            
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            
            # Get features from each block
            if isinstance(upsample_block, CrossAttnUpBlock2D):
                sample, intermediates = custom_forward_CrossAttnUpBlock2D(
                    self=upsample_block,
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    store_intermediates=True
                )
            elif isinstance(upsample_block, UpBlock2D):
                sample, intermediates = custom_forward_UpBlock2D(
                    self=upsample_block,
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    store_intermediates=True
                )
            else:
                raise NotImplementedError(f"Missing forward for {type(upsample_block)}")
            
            for layer_sample in intermediates:
                if layer_counter in feature_indices:
                    features[layer_counter] = layer_sample
                layer_counter += 1

        return features

# This is used to perform a single timestep of the diffusion model, instead of the whole diffusion process.
# The normal StableDiffusionPipeline only takes a prompt and a number of timesteps as input.
# This way, steps that are not needed can be skipped, which saves a lot of time.
def custom_forward_OneStepSDPipeline(
    self,
    img_tensor,
    t,
    feature_indices=None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
):
    device = img_tensor.device
    latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
    t = torch.tensor(t, dtype=torch.long, device=device)
    noise = torch.randn_like(latents).to(device)
    latents_noisy = self.scheduler.add_noise(latents, noise, t)
    unet_args = {
        'sample': latents_noisy,
        'timestep': t,
        'encoder_hidden_states': prompt_embeds,
        'cross_attention_kwargs': cross_attention_kwargs
    }
    if feature_indices is not None:
        unet_args['feature_indices'] = feature_indices
    unet_output = self.unet(**unet_args)
    return unet_output


class SDExtractor(nn.Module):
    def __init__(self, model, half_precision=False):
        super(SDExtractor, self).__init__()

        unet = CustomUNet2DConditionModel.from_pretrained(model, subfolder="unet")
        self.pipe = AutoPipelineForText2Image.from_pretrained(model, unet=unet, safety_checker=None, torch_dtype=torch.bfloat16 if half_precision else torch.float32)
        self.pipe.scheduler = DDIMScheduler.from_pretrained(model, subfolder="scheduler")
        self.pipe.vae.decoder = None

        self.pipe.enable_attention_slicing()
        self.pipe.enable_xformers_memory_efficient_attention()
        gc.collect()
    
    # Pipe is not a nn.Module, so it needs to be moved explicitly
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.pipe.to(*args, **kwargs)
        return self

    def __call__(self, images, prompt, layers=[5], steps=[101]):
        """
        Args:
            images: should be a torch tensor in the shape of [b, c, h, w] in range [-1, 1]
            prompt: the prompt to use, a string or a list of strings (length must
                match the batch-size of img_tensor)
            steps: the time steps to use, should be an list of ints in the 
                range of [0, 1000]
            layers: which upsampling layers of the U-Net to extract features
                from. With input (1, 3, 512, 512) and SD1.5 you can choose
                ---- bottleneck
                middle block: (1, 1280, 8, 8)   # 0
                ---- upsample block 0
                ResNet: (1, 1280, 8, 8)         # 1
                ResNet: (1, 1280, 8, 8)         # 2
                ResNet: (1, 1280, 8, 8)         # 3
                Upsampler: (1, 1280, 16, 16)    # 4
                ---- upsample block 1
                ResNet: (1, 1280, 16, 16)       # 5
                ResNet: (1, 1280, 16, 16)       # 6
                ResNet: (1, 1280, 16, 16)       # 7
                Upsampler: (1, 1280, 32, 32)    # 8
                ---- upsample block 2
                ResNet: (1, 640, 32, 32)        # 9
                ResNet: (1, 640, 32, 32)        # 10
                ResNet: (1, 640, 32, 32)        # 11
                Upsampler: (1, 640, 64, 64)     # 12
                ---- upsample block 3
                ResNet: (1, 320, 64, 64)        # 13
                ResNet: (1, 320, 64, 64)        # 14
                ResNet: (1, 320, 64, 64)        # 15
        Return:
            features: a two-level dictionary with keys being timesteps, values again are
                dictionaries with keys being the layer number and values being the
                respective timestep-layer feature map of shape (bs, c, h, w). e.g.:
                {101: {
                    2: (bs, 1280, 8, 8),
                    9: (bs, 640, 32, 32)
                },
                201: {...}
                }
        """
        
        # Embed prompt
        prompt_embeddings = self.pipe.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            device=images.device
        )[0] # [1, 77, dim]
        
        # Check prompt
        if images.shape[0] != prompt_embeddings.shape[0]:
            if isinstance(prompt, str):
                if prompt is None:
                    raise ValueError("Prompt is not provided")
                else:
                    prompt_embeddings = prompt_embeddings.repeat(images.shape[0], 1, 1)
            else:
                raise ValueError("Batch-size does not match number of prompts")

        # Extract features
        features = {}
        for t in steps:
            features[t] = custom_forward_OneStepSDPipeline(
                self=self.pipe,
                img_tensor=images,
                t=t,
                feature_indices=layers,
                prompt_embeds=prompt_embeddings)
        return features


####################################################################################################
### This version uses hooks instead of altering the forward function of the UNet2DConditionModel ###
### It is slower than the above version, because the forward pass is not stopped, but it is more ###
### flexible to other models.                                                                    ###
####################################################################################################

import copy
from diffusers import UNet2DConditionModel
from diffusers import StableDiffusionPipeline

class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        feature_indices=None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_args = {
            'sample': latents_noisy,
            'timestep': t,
            'encoder_hidden_states': prompt_embeds,
            'cross_attention_kwargs': cross_attention_kwargs
        }
        if feature_indices is not None:
            unet_args['feature_indices'] = feature_indices
        unet_output = self.unet(**unet_args)
        return unet_output
    
class SDHookExtractor:
    def __init__(self, device, model='stabilityai/stable-diffusion-2-1'):
        self.device = device

        unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")
        self.pipe = OneStepSDPipeline.from_pretrained(model, unet=unet, safety_checker=None)
        self.pipe.scheduler = DDIMScheduler.from_pretrained(model, subfolder="scheduler")
        self.pipe.vae.decoder = None
        self.pipe = self.pipe.to(device)

        self.pipe.enable_attention_slicing()
        self.pipe.enable_xformers_memory_efficient_attention()
        gc.collect()

    def save_fn(self, layer_idx):
        def hook(module, input, output):
            self.features[layer_idx] = output
        return hook

    def __call__(self, images, prompt, layers=[5], steps=[101], cross_attention_maps=False):
        # Set hooks at the specified layers
        self.features = {}
        hooks = []
        if 0 in layers:
            hooks.append(self.pipe.unet.mid_block.register_forward_hook(self.save_fn(0)))
        layer_counter = 1
        for block in self.pipe.unet.up_blocks:
            for l in block.resnets:
                if layer_counter in layers:
                    hooks.append(l.register_forward_hook(self.save_fn(layer_counter)))
                layer_counter += 1
            if block.upsamplers is not None:
                for l in block.upsamplers:
                    if layer_counter in layers:
                        hooks.append(l.register_forward_hook(self.save_fn(layer_counter)))
                    layer_counter += 1

            if cross_attention_maps and isinstance(block, CrossAttnUpBlock2D):
                for l in block.attentions:
                    hooks.append(l.transformer_blocks[0].attn2.to_q.register_forward_hook(self.save_fn(layer_counter)))
                    hooks.append(l.transformer_blocks[0].attn2.to_k.register_forward_hook(self.save_fn(layer_counter+1)))
                    hooks.append(l.transformer_blocks[0].attn2.to_v.register_forward_hook(self.save_fn(layer_counter+2)))
                    layer_counter += 3

        # Embed prompt
        prompt_embeddings = self.pipe.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            device=images.device
        )[0] # [1, 77, dim]
        
        # Check prompt
        if images.shape[0] != prompt_embeddings.shape[0]:
            if isinstance(prompt, str):
                if prompt is None:
                    raise ValueError("Prompt is not provided")
                else:
                    prompt_embeddings = prompt_embeddings.repeat(images.shape[0], 1, 1)
            else:
                raise ValueError("Batch-size does not match number of prompts")

        # Run model
        features = {}
        for t in steps:
            self.pipe(
                img_tensor=images,
                t=t,
                prompt_embeds=prompt_embeddings)
            features[t] = copy.deepcopy(self.features)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return features