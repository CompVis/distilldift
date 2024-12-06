import os
import yaml
import tqdm
import h5py
import copy
import torch
import imagesize
import numpy as np
from PIL import Image
import torch.utils.data as data
from torch.nn.functional import interpolate

from datasets.correspondence import SPair, PFWillow, CUB, S2K, CO3D
from datasets.image import ImageDataset, ImageNet, PASCALPart, COCO
from utils.correspondence import preprocess_image, flip_points, flip_bbox, rescale_points, rescale_bbox, normalize_features

def read_dataset_config(config_path):
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

def load_dataset(dataset_name, config, preprocess=None):
    """
    Load dataset from config.

    Args:
        dataset_name (str): Name of dataset
        config (dict): Dataset config
        preprocess (callable): Preprocess function

    Returns:
        CorrespondenceDataset: Dataset
    """

    if dataset_name == 'SPair-71k':
        return SPair(config, preprocess)
    if dataset_name == 'PF-WILLOW':
        return PFWillow(config, preprocess)
    if dataset_name == 'CUB-200-2011':
        return CUB(config, preprocess)
    if dataset_name == 'ImageNet':
        return ImageNet(config, preprocess)
    if dataset_name == 'PASCALPart':
        return PASCALPart(config, preprocess)
    if dataset_name == 'S2K':
        return S2K(config, preprocess)
    if dataset_name == 'COCO':
        return COCO(config, preprocess)
    if dataset_name == 'CO3D':
        return CO3D(config, preprocess)
    
    raise ValueError('Dataset not recognized.')

def cache_dataset(model, dataset, cache_path, reset_cache, batch_size, num_workers, device, half_precision):
    """
    Cache features from dataset.

    Args:
        model (CacheModel): Model
        dataset (CorrespondenceDataset): Dataset
        cache_path (str): Path to cache file
        reset_cache (bool): Whether to reset cache
        batch_size (int): Batch size
        num_workers (int): Number of workers for dataloader
        device (torch.device): Device
        half_precision (bool): Whether to use half precision
    """

    if os.path.exists(cache_path) and not reset_cache:
        print(f"Cache file {cache_path} already exists.")
        return CacheDataset(dataset, cache_path)

    print(f"Caching features to {cache_path}")

    with h5py.File(cache_path, 'w') as f:
        keys = []
        samples = []
        def process(image_path, category):
            key = os.path.basename(image_path)
            if key not in keys:
                image, _ = dataset.load_image(image_path)
                image = dataset.preprocess.process_image(image)
                samples.append((key, image, category))
                keys.append(key)
        
        for sample in dataset.data:
            if 'source_image_path' in sample:
                process(sample['source_image_path'], sample['source_category'])
                process(sample['target_image_path'], sample['target_category'])
            else:
                process(sample['image_path'], sample['category'])
        
        # Create dataloader
        dataloader = data.DataLoader(samples,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)

        # Create group for each layer if it doesn't exist
        if hasattr(model, 'layers') and model.layers is not None:
            layers = [f.create_group(str(l)) if str(l) not in f else f[str(l)] for l in model.layers]

        # Move model to device
        model.to(device)

        # Get features and save to cache file
        with torch.no_grad():
            for key, image, category in tqdm.tqdm(dataloader):
                image = image.to(device, dtype=torch.bfloat16 if half_precision else torch.float32)

                # Extend last batch if necessary
                if len(key) < batch_size:
                    image = torch.cat([image, image[-1].repeat(batch_size-len(key), 1, 1, 1)])
                    category = list(category) + [category[-1]] * (batch_size-len(key))
                
                # Get features
                features = model(image, category)

                # Save features
                def save_features(g, features):
                    for i, k in enumerate(key): # for each image and key in the batch
                        g.create_dataset(k, data=features[i].type(torch.float16 if half_precision else torch.float32)) # bfloat16 not supported

                if type(features) is list: # (l, b, c, h, w)
                    for l, layer in enumerate(layers):
                        save_features(layer, features[l].cpu())
                else: # (b, c, h, w)
                    save_features(f, features.cpu())
            
    print("Caching complete.")
    return CacheDataset(dataset, cache_path)

class CacheDataset(data.Dataset):
    """
    Wrapper that loads features from cache instead of images.
    """
    
    def __init__(self, dataset, cache_path):
        self.config = dataset.config
        self.data = dataset.data
        self.preprocess = dataset.preprocess
        self.image_sampling = self.config.get('image_sampling', 'ground_truth')
        self.top_k = self.config.get('top_k', 10)
        if isinstance(cache_path, list):
            self.file = [h5py.File(p, 'r') for p in cache_path]
        else:
            self.file = h5py.File(cache_path, 'r')
        self.cache = self.file
        self.load_images = False
        self.category_to_id = None

        if self.image_sampling == 'retrieval':
            embeddings = torch.load(self.config['embeddings_path'])
            embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min()) # min-max normalize
            self.weights = embeddings @ embeddings.transpose(0, 1)
            print("Embeddings loaded.")

        self.hdf5_file = dataset.hdf5_file

    def set_layer(self, layer):
        if isinstance(self.cache, list):
            self.cache = [f[str(l)] for f, l in zip(self.file, layer)]
        else:
            self.cache = self.file[str(layer)]

    def load_image(self, path):
        if self.hdf5_file is not None:
            group = self.hdf5_file[path]
            image = group['image'][()]
            size = group['size'][()]
        else:
            image = Image.open(path).convert('RGB')
            size = image.size
        return image, size
    
    def get_image_size(self, path):
        if self.hdf5_file is not None:
            group = self.hdf5_file[path]
            size = group['size'][()]
        else:
            size = imagesize.get(path)
        return size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # Prevent memory leak

        if self.image_sampling != 'ground_truth':
            if self.image_sampling == 'retrieval':
                weights = self.weights[idx]
                _, top_k_indices = torch.topk(weights, self.top_k+1) # +1 to exclude self
                match_id = top_k_indices[np.random.choice(self.top_k)+1]
            elif self.image_sampling == 'random_category':
                match_id = np.random.choice(self.category_to_id[sample['category']])
            elif self.image_sampling == 'random':
                match_id = np.random.choice(len(self.data))
            elif self.image_sampling == 'same':
                match_id = idx

            matching_sample = self.data[match_id]
            sample['source_image_path'] = sample['image_path']
            sample['target_image_path'] = matching_sample['image_path']
            sample['source_category'] = sample['category']
            sample['target_category'] = matching_sample['category']

        if self.load_images:
            # Load image
            sample['source_image'], sample['source_size'] = self.load_image(sample['source_image_path'])
            sample['target_image'], sample['target_size'] = self.load_image(sample['target_image_path'])
        else:
            # Get image size quickly
            sample['source_size'] = self.get_image_size(sample['source_image_path'])
            sample['target_size'] = self.get_image_size(sample['target_image_path'])

        # Load features from cache
        source_key = os.path.basename(sample['source_image_path'])
        target_key = os.path.basename(sample['target_image_path'])
        def load_features(cache, key):
            if len(cache) == 1:
                return torch.tensor(cache[0][key][()])
            features = [normalize_features(torch.tensor(c[key][()])) for c in cache]
            #max_wh = max([f.shape[-1] for f in features])
            #features = [interpolate(f.unsqueeze(0), size=max_wh, mode='bilinear').squeeze(0) for f in features]
            return torch.cat(features, dim=0)
        if isinstance(self.cache, list):
            sample['source_features'] = load_features(self.cache, source_key)
            sample['target_features'] = load_features(self.cache, target_key)
        else:
            sample['source_features'] = torch.tensor(self.cache[source_key][()])
            sample['target_features'] = torch.tensor(self.cache[target_key][()])
        
        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample
    
class Preprocessor:
    """
    Preprocess dataset samples.

    Args:
        image_size (tuple): (width, height) used for resizing images
        preprocess_image (bool): Whether to preprocess images (resize, normalize, etc.)
        rescale_data (bool): Whether to rescale points and bounding boxes (also sets source_size and target_size to image_size)
    """

    def __init__(self, image_size, preprocess_image=True, image_range=[-1, 1], rescale_data=True, flip_data=True, normalize_image=False):
        self.image_size = image_size
        self.preprocess_image = preprocess_image
        self.image_range = image_range
        self.rescale_data = rescale_data
        self.flip_data = flip_data
        self.normalize_image = normalize_image

    def process_image(self, image):
        return preprocess_image(image, self.image_size, range=self.image_range, norm=self.normalize_image)

    def __call__(self, sample):
        # Preprocess images
        if self.preprocess_image:
            sample['source_image'] = self.process_image(sample['source_image'])
            sample['target_image'] = self.process_image(sample['target_image'])

        # Rescale points and bounding boxes
        if self.rescale_data:
            source_size = sample['source_size']
            target_size = sample['target_size']
            sample['source_points'] = rescale_points(sample['source_points'], source_size, self.image_size)
            sample['target_points'] = rescale_points(sample['target_points'], target_size, self.image_size)
            sample['source_bbox'] = rescale_bbox(sample['source_bbox'], source_size, self.image_size)
            sample['target_bbox'] = rescale_bbox(sample['target_bbox'], target_size, self.image_size)
            sample['source_size'] = self.image_size
            sample['target_size'] = self.image_size
        
        # Flip x, y and w, h axis
        if self.flip_data:
            sample['source_points'] = flip_points(sample['source_points'])
            sample['target_points'] = flip_points(sample['target_points'])
            sample['source_bbox'] = flip_bbox(sample['source_bbox'])
            sample['target_bbox'] = flip_bbox(sample['target_bbox'])

        return sample

def CorrespondenceDataset_to_ImageDataset(dataset):
    """
    Convert CorrespondenceDataset to ImageDataset.
    
    Args:
        dataset (CorrespondenceDataset): Dataset
    
    Returns:
        ImageDataset: Dataset
    """
    unique_data = []
    unique_paths = {}
    category_to_id = {}
    for sample in dataset.data:
        if sample['source_image_path'] not in unique_paths:
            unique_paths[sample['source_image_path']] = True
            unique_data.append({
                'image_path': sample['source_image_path'],
                'category': sample['source_category']
            })
            if sample['source_category'] not in category_to_id:
                category_to_id[sample['source_category']] = []
            category_to_id[sample['source_category']].append(len(unique_data)-1)
        if sample['target_image_path'] not in unique_paths:
            unique_paths[sample['target_image_path']] = True
            unique_data.append({
                'image_path': sample['target_image_path'],
                'category': sample['target_category']
            })
            if sample['target_category'] not in category_to_id:
                category_to_id[sample['target_category']] = []
            category_to_id[sample['target_category']].append(len(unique_data)-1)

    image_dataset = ImageDataset(dataset.config, dataset.preprocess)
    image_dataset.data = unique_data
    image_dataset.category_to_id = category_to_id
    return image_dataset

def combine_caches(cache_paths, combined_path):
    """
    Combine multiple cache files into one.

    Args:
        cache_paths (list): List of cache file paths
        combined_path (str): Path to combined cache file

    Returns:
        str: Path to combined cache file
    """
    with h5py.File(combined_path, 'w') as f:
        for path in cache_paths:
            with h5py.File(path, 'r') as g:
                copy_items(g, f)

    # Remove original cache files
    for path in cache_paths:
        os.remove(path)

def copy_items(source_group, target_group):
    """
    Recursively copies items (groups and datasets) from source to target group.
    source_group: Source HDF5 group object.
    target_group: Target HDF5 group object.
    """
    for name, item in source_group.items():
        if isinstance(item, h5py.Dataset):
            target_group.create_dataset(name, data=item[()])
        elif isinstance(item, h5py.Group):
            if name not in target_group:
                target_group.create_group(name)
            copy_items(item, target_group[name])