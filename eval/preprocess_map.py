import os
import sys
import torch
import yaml
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.utils_correspondence import resize
from model_utils.extractor_dino import ViTExtractor
from model_utils.extractor_sd import load_model, process_features_and_mask

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def process_and_save_features(file_paths, sd_size, dino_size, layer, facet, model, aug, extractor_vit, flip=False, angle=0):
    for file_path in tqdm(file_paths, desc="Processing images (Flip: {})".format(flip)):
        img = Image.open(file_path).convert('RGB')
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        sd_input = resize(img, sd_size, resize=True, to_pil=True, padding=not NO_PADDING)
        dino_input = resize(img, dino_size, resize=True, to_pil=True, padding=not NO_PADDING)

        subdir_name = 'features' if NUM_ENSEMBLE == 1 else f'features_ensemble{NUM_ENSEMBLE}'
        output_subdir = file_path.replace('JPEGImages', subdir_name).rsplit('/', 1)[0]
        os.makedirs(output_subdir, exist_ok=True)
        suffix = '_flip' if flip else ''

        if not ONLY_DINO:
            accumulated_features = {}
            for _ in range(NUM_ENSEMBLE): 
                features = process_features_and_mask(model, aug, sd_input, mask=False, raw=True)
                del features['s2']
                for k in features:
                    accumulated_features[k] = accumulated_features.get(k, 0) + features[k]

            for k in accumulated_features:
                accumulated_features[k] /= NUM_ENSEMBLE

            output_path = os.path.join(output_subdir, os.path.splitext(os.path.basename(file_path))[0] + f'_sd{suffix}.pt')
            torch.save(accumulated_features, output_path)

        dino_batch = extractor_vit.preprocess_pil(dino_input)
        desc_dino = extractor_vit.extract_descriptors(dino_batch.cuda(), layer, facet).permute(0, 1, 3, 2).reshape(1, -1, NUM_PATCHES, NUM_PATCHES)
        output_path_dino = os.path.join(output_subdir, os.path.splitext(os.path.basename(file_path))[0] + f'_dino{suffix}.pt')
        torch.save(desc_dino, output_path_dino)

def load_config(config_path):
    """Load configurations from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == '__main__':
    # Parse the YAML config file
    parser = argparse.ArgumentParser(description="Process image features with ViT and SD models.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()
    config = load_config(args.config)

    # Extract values from the configuration
    DATASET_DIR = config.get('DATASET_DIR', 'data/SPair-71k/JPEGImages')
    ONLY_DINO = config.get('ONLY_DINO', False)
    ADAPT_FLIP = config.get('ADAPT_FLIP', False)
    NO_PADDING = config.get('NO_PADDING', False)
    NUM_ENSEMBLE = config.get('NUM_ENSEMBLE', 1)
    NUM_PATCHES = config.get('NUM_PATCHES_PRE', 31)

    sd_size = config.get('SD_SIZE', 960)
    dino_size = config.get('DINO_SIZE', 840)
    layer = config.get('LAYER', 11)
    facet = config.get('FACET', 'token')

    dino_version = config.get('DINO_VERSION', 'dinov2_vitb14_reg')
    patch_size = config.get('PATCH_SIZE', 14)
    lora_rank = config.get('LORA_RANK', None)
    lora_layers = config.get('LORA_LAYERS', None)
    weights = config.get('WEIGHTS', None)

    # Configuration
    set_seed()

    # Load models
    if not ONLY_DINO:
        model, aug = load_model(diffusion_ver='v1-5', image_size=sd_size, num_timesteps=50, block_indices=[2,5,8,11])
    else:
        model, aug = None, None
    extractor_vit = ViTExtractor(dino_version, patch_size, device='cuda', lora_rank=lora_rank, lora_layers=lora_layers, weights=weights)

    all_files = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(DATASET_DIR) for file in files if file.endswith('.jpg')]

    angles = [0] # angles for rotation
    for angle in angles:
        # Process and save features
        process_and_save_features(all_files, sd_size, dino_size, layer, facet, model, aug, extractor_vit, flip=False, angle=angle)
        if ADAPT_FLIP:
            process_and_save_features(all_files, sd_size, dino_size, layer, facet, model, aug, extractor_vit, flip=True, angle=angle)

    print("All processing done.")