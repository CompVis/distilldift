import yaml

from models.diffusion import Diffusion
from models.dino import DINO
from models.zoedepth import ZoeDepth
from models.mae import MAE
from models.clip import CLIP
from models.combination import Combination
from models.ensemble import Ensemble
from models.prompt import Prompt
from models.dit import DiT
from models.ijepa import IJEPA
from models.diffusion_hook import DiffusionHook
from models.efficientvit import EfficientViT

from models.distilled_model import DistilledModel

def read_model_config(config_path):
    """
    Read config from JSON file.

    Args:
        config_path (str): Path to config file
    
    Returns:
        dict: Config
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_name, config):
    """
    Load model from config.

    Args:
        model_name (str): Name of model
        config (dict): Model config
    
    Returns:
        BaseModel: Model
    """

    # Pretrained models
    if model_name.startswith('diff'):
        return Diffusion(config)
    if model_name.startswith('dit'):
        return DiT(config)
    if model_name.startswith('dino'):
        return DINO(config)
    if model_name.startswith('zoedepth'):
        return ZoeDepth(config)
    if model_name.startswith('mae'):
        return MAE(config)
    if model_name.startswith('clip'):
        return CLIP(config)
    if model_name.startswith('ijepa'):
        return IJEPA(config)
    if model_name.startswith('diff_hook'):
        return DiffusionHook(config)
    if model_name.startswith('efficientvit'):
        return EfficientViT(config)
    
    # Experimenatal models
    if model_name.startswith('combination'):
        return Combination(config, [load_model(config[key], config[key + '_config']) for key in config.keys() if key.startswith('model') and not key.endswith('_config')])
    if model_name.startswith('ensemble'):
        return Ensemble(config)
    if model_name.startswith('prompt'):
        return Prompt(config)
    
    # Distillation models
    if model_name.startswith('distilled'):
        return DistilledModel(config)

    raise ValueError('Model not recognized.')