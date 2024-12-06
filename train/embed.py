import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from utils.dataset import read_dataset_config, load_dataset, CorrespondenceDataset_to_ImageDataset, Preprocessor
from utils.correspondence import preprocess_image
from datasets.correspondence import CorrespondenceDataset

dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
dinov2_vitb14_reg = dinov2_vitb14_reg.cuda()
dinov2_vitb14_reg.eval()

# Parse the arguments
parser = argparse.ArgumentParser(description='Embed images using DINOv2')
parser.add_argument('--dataset_name', type=str, default='SPair-71k', help='Dataset name')
parser.add_argument('--image_size', type=int, default=434, help='Image size')
parser.add_argument('--config', type=str, default='configs/dataset_config.yaml', help='Path to the dataset config file')
parser.add_argument('--cache_dir', type=str, default='cache', help='Path to the cache directory')

args = parser.parse_args()
dataset_name = args.dataset_name
image_size = [args.image_size, args.image_size]
config_path = args.config
cache_dir = args.cache_dir

# Load the dataset
dataset_config = read_dataset_config(config_path)
config = dataset_config[dataset_name]
config['split'] = 'train'
random_sampling = config.get('random_sampling', False)
num_samples = config.get('num_samples', None)

preprocess = Preprocessor(image_size=image_size, image_range=[0, 1], rescale_data=False, flip_data=False, normalize_image=False)
dataset = load_dataset(dataset_name, config, preprocess)
if random_sampling:
    torch.manual_seed(42)
    dataset.data = [dataset.data[i] for i in torch.randperm(len(dataset))]
if num_samples is not None:
    dataset.data = dataset.data[:num_samples]
if hasattr(dataset, 'create_category_to_id'):
    dataset.create_category_to_id()

if issubclass(type(dataset), CorrespondenceDataset):
    dataset = CorrespondenceDataset_to_ImageDataset(dataset)

features = []
for sample in tqdm(dataset.data):
    image_path = sample['image_path']
    image = Image.open(image_path)
    image = preprocess_image(image, image_size, [0, 1], False)
    image = torch.unsqueeze(image, 0).cuda()
    with torch.no_grad():
        features.append(dinov2_vitb14_reg(image).cpu())

features = torch.cat(features, 0)
torch.save(features, os.path.join(cache_dir, f'{dataset_name}_embeddings.pth'))