import torch
import torch.nn as nn
import math

from .base import CacheModel

# From https://github.com/JamesQFreeman/LoRA-ViT/blob/main/lora.py
class LoRALayer(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        dropout
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        qkv = self.dropout(qkv)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

class DistilledModel(CacheModel):
    """
    Distilled model.

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(DistilledModel, self).__init__(config)
        
        self.weights = config.get("weights", None)
        self.rank = config.get("rank", None)
        self.lora_layers = config.get("lora_layers", [])
        self.lora_dropout = config.get("lora_dropout", 0.0)
        self.optimize_modules = config.get("optimize_modules", [])

        self.patch_size = 14
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        self.params_to_optimize = []
        
        if self.rank is not None:
            self.lora_parameters = []
            self.add_lora(self.model, self.rank)

        # Load weights
        if self.weights is not None:
            self.load_state_dict(torch.load(self.weights), strict=False)
        
        # What to optimize: vit, lora, head, upsampler
        for m in self.optimize_modules:
            if m == "vit":
                self.params_to_optimize.extend(list(self.model.parameters()))
            elif m == "lora":
                self.params_to_optimize.extend(self.lora_parameters)

        for p in self.params_to_optimize:
            p.requires_grad = True

    def add_lora(self, model, rank):
        r = rank

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # Here, we do the surgery
        for i, blk in enumerate(model.blocks):
            if i not in self.lora_layers:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = LoRALayer(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                self.lora_dropout
            )

            for p in blk.attn.qkv.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                self.lora_parameters.append(p)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def get_features(self, image, category=None):
        b = image.shape[0]
        h = image.shape[2] // self.patch_size
        w = image.shape[3] // self.patch_size

        features = self.model(image, is_training=True)["x_norm_patchtokens"] # B, N, C
        features = features.permute(0, 2, 1).reshape(b, -1, h, w) # B, C, H, W
        return features
