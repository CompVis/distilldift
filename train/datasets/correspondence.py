import itertools
import torch
import torch.utils.data as data
import numpy as np
import json
import os
import csv
from PIL import Image
import copy
import h5py
from tqdm import tqdm

class CorrespondenceDataset(data.Dataset):
    """
    General dataset class for datasets with correspondence points.
    """

    def __init__(self, config, preprocess=None):
        self.config = config
        self.dataset_directory = config['path']
        self.image_pair = config.get('image_pair', False)
        self.preprocess = preprocess
        self.split = config.get('split', 'test')

        self.data = []
        self.hdf5_path = config.get('hdf5_path', None)
        self.hdf5_file = None
        if self.hdf5_path is not None:
            self.load_hdf5()
        else:
            self.load_data()

    def load_hdf5(self):
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        keys_order = [key.decode('utf-8') for key in self.hdf5_file['keys_order']]
        for key in keys_order:
            group = self.hdf5_file[key]
            self.data.append({
                "image_path": key,
                **{key: group[key][()] for key in group.keys() if '_image' not in key}
            })

    def load_image(self, path):
        if self.hdf5_file is not None:
            group = self.hdf5_file[path]
            image = group['image'][()]
            size = group['size'][()]
        else:
            image = Image.open(path).convert('RGB')
            size = image.size
        return image, size
    
    def load_data(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # prevent memory leak

        # Load image
        sample['source_image'], sample['source_size'] = self.load_image(sample['source_image_path'])
        sample['target_image'], sample['target_size'] = self.load_image(sample['target_image_path'])
    
        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample


class SPair(CorrespondenceDataset):
    """
    Dataset class for the SPair-71k dataset.
    """

    def load_data(self):
        images_dir = os.path.join(self.dataset_directory, 'JPEGImages')
        annotations_dir = os.path.join(self.dataset_directory, 'PairAnnotation', 'test' if self.split == 'test' else 'trn')
        annotations_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]

        for annotation_file in annotations_files:
            with open(os.path.join(annotations_dir, annotation_file), 'r') as file:
                annotation = json.load(file)

            category = annotation['category']
            source_image_path = os.path.join(images_dir, category, annotation['src_imname'])
            target_image_path = os.path.join(images_dir, category, annotation['trg_imname'])
            source_points = torch.tensor(annotation['src_kps'], dtype=torch.float16)
            target_points = torch.tensor(annotation['trg_kps'], dtype=torch.float16)
            source_bbox = torch.tensor(annotation['src_bndbox'], dtype=torch.float16)
            target_bbox = torch.tensor(annotation['trg_bndbox'], dtype=torch.float16)

            # Convert from (x, y, x+w, y+h) to (x, y, w, h)
            source_bbox[2:] -= source_bbox[:2]
            target_bbox[2:] -= target_bbox[:2]

            # Mutual visible keypoint ids
            mutual_visible_idx = [int(k) for k in annotation['kps_ids']]

            self.data.append({
                'source_image_path': source_image_path,
                'target_image_path': target_image_path,
                'source_points': source_points,
                'target_points': target_points,
                'source_bbox': source_bbox,
                'target_bbox': target_bbox,
                'source_category': category,
                'target_category': category,
                'mutual_visible_idx': mutual_visible_idx
            })


class PFWillow(CorrespondenceDataset):
    """
    Dataset class for the PF-Willow dataset.
    """
    
    def load_data(self):
        csv_file = os.path.join(self.dataset_directory, 'test_pairs_pf.csv')
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                source_image_path = os.path.join(self.dataset_directory, row[0].replace('PF-dataset/', ''))
                target_image_path = os.path.join(self.dataset_directory, row[1].replace('PF-dataset/', ''))

                row[2:] = list(map(float, row[2:]))
                source_points = torch.tensor(list(zip(row[2:12], row[12:22])), dtype=torch.float16) # X, Y
                target_points = torch.tensor(list(zip(row[22:32], row[32:])), dtype=torch.float16) # X, Y

                # use min and max to get the bounding box
                source_bbox = torch.tensor([source_points[:, 0].min(), source_points[:, 1].min(),
                                            source_points[:, 0].max(), source_points[:, 1].max()], dtype=torch.float16)
                target_bbox = torch.tensor([target_points[:, 0].min(), target_points[:, 1].min(),
                                            target_points[:, 0].max(), target_points[:, 1].max()], dtype=torch.float16)

                # Convert from (x, y, x+w, y+h) to (x, y, w, h)
                source_bbox[2:] -= source_bbox[:2]
                target_bbox[2:] -= target_bbox[:2]

                # Get category from image path
                category = source_image_path.split('(')[0]

                self.data.append({
                    'source_image_path': source_image_path,
                    'target_image_path': target_image_path,
                    'source_points': source_points,
                    'target_points': target_points,
                    'source_bbox': source_bbox,
                    'target_bbox': target_bbox,
                    'source_category': category,
                    'target_category': category
                })


class CUB(CorrespondenceDataset):
    """
    Dataset class for the CUB-200-2011 dataset.
    """

    def load_data(self):
        self.images_dir = os.path.join(self.dataset_directory, 'images')

        with open(os.path.join(self.dataset_directory, "images.txt"), "r") as f:
            images = [line.strip().split() for line in f.readlines()]

        with open(os.path.join(self.dataset_directory, "train_test_split.txt"), "r") as f:
            train_test_split = [line.strip().split() for line in f.readlines()]

        with open(os.path.join(self.dataset_directory, "parts/part_locs.txt"), "r") as f:
            part_locs = {}
            for line in f.readlines():
                img_id, _, x, y, visible = line.strip().split()
                if img_id not in part_locs:
                    part_locs[img_id] = []
                part_locs[img_id].append((x, y, visible == '1'))

        with open(os.path.join(self.dataset_directory, "image_class_labels.txt"), "r") as f:
            image_class_labels = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}

        with open(os.path.join(self.dataset_directory, "bounding_boxes.txt"), "r") as f:
            bounding_boxes = {line.split()[0]: list(map(float, line.strip().split()[1:])) for line in f.readlines()}

        # Filter images based on train/test split and class labels
        filtered_images = []
        for img_id, img_name in images:
            is_training_image = int(train_test_split[int(img_id) - 1][1])
            class_id = image_class_labels[img_id]
            if (self.split == 'train' and is_training_image) or (self.split == 'test' and not is_training_image):
                filtered_images.append((img_id, img_name))
        
        # Get class names
        with open(os.path.join(self.dataset_directory, "classes.txt"), "r") as f:
            classes = [line.strip().split('.')[1].replace('_', '') for line in f.readlines()]

        # Generate all pairs for each class
        for class_id in range(1, 201):
            class_images = [img for img in filtered_images if image_class_labels[img[0]] == class_id]
            for source, target in itertools.combinations(class_images, 2):
                source_image_path = os.path.join(self.images_dir, source[1])
                target_image_path = os.path.join(self.images_dir, target[1])
                
                source_points = np.array(part_locs[source[0]], dtype=float)
                target_points = np.array(part_locs[target[0]], dtype=float)

                # Filter out points that are not visible in either of the images
                visible_points = np.logical_and(source_points[:, 2], target_points[:, 2])
                source_points = torch.tensor(source_points[visible_points][:, :2], dtype=torch.float16)
                target_points = torch.tensor(target_points[visible_points][:, :2], dtype=torch.float16)

                source_bbox = torch.tensor(bounding_boxes[source[0]], dtype=torch.float16)
                target_bbox = torch.tensor(bounding_boxes[target[0]], dtype=torch.float16)
                
                # Get category from image class id
                category = classes[class_id - 1]

                self.data.append({
                    'source_image_path': source_image_path,
                    'target_image_path': target_image_path,
                    'source_points': source_points,
                    'target_points': target_points,
                    'source_bbox': source_bbox,
                    'target_bbox': target_bbox,
                    'source_category': category,
                    'target_category': category
                })
        
        return data


class S2K(CorrespondenceDataset):
    """
    Dataset class for the S2K dataset.
    """

    def load_data(self):
        annotation_file = os.path.join(self.dataset_directory, 's2k.json')
        with open(annotation_file, 'r') as file:
            annotations = json.load(file)

        for a in annotations:
            source_bbox = torch.tensor(a['source_bbox'], dtype=torch.float16) # is already in (x, y, w, h) format
            target_bbox = torch.tensor(a['target_bbox'], dtype=torch.float16) # is already in (x, y, w, h) format

            self.data.append({
                'source_image_path': os.path.join(self.dataset_directory, "VOC2010", "JPEGImages", a['source_image_path']),
                'target_image_path': os.path.join(self.dataset_directory, "VOC2010", "JPEGImages", a['target_image_path']),
                'source_annotation_path': os.path.join(self.dataset_directory, "Annotations_Part",  a['source_annotation_path']),
                'target_annotation_path': os.path.join(self.dataset_directory, "Annotations_Part", a['target_annotation_path']),
                'source_bbox': source_bbox,
                'target_bbox': target_bbox,
                'source_parts': a['source_parts'],
                'target_parts': a['target_parts'],
                'source_category': a['source_category'],
                'target_category': a['target_category']
            })

        return data
    
    def sample_points(self, sample):
        # Load parts
        source_annotation = np.load(sample['source_annotation_path'], allow_pickle=True).item()
        target_annotation = np.load(sample['target_annotation_path'], allow_pickle=True).item()

        sample['source_annotation'] = source_annotation
        sample['target_annotation'] = target_annotation
    
        # Sample center points of parts
        source_parts = source_annotation['parts']
        target_parts = target_annotation['parts']
        if self.config.get('only_non_unique', False):
            source_parts = [part for i, part in enumerate(source_parts) if i in sample['source_parts']]
            target_parts = [part for i, part in enumerate(target_parts) if i in sample['target_parts']]

        source_points = []
        for part in source_parts:
            y, x = np.where(part["mask"])
            source_points.append([np.mean(x), np.mean(y)])
        
        target_points = []
        for part in target_parts:
            y, x = np.where(part["mask"])
            target_points.append([np.mean(x), np.mean(y)])

        sample['source_points'] = torch.tensor(source_points, dtype=torch.float16)
        sample['target_points'] = torch.tensor(target_points, dtype=torch.float16)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # prevent memory leak

        # Load image
        sample['source_image'] = Image.open(sample['source_image_path'])
        sample['target_image'] = Image.open(sample['target_image_path'])

        # Save image size
        sample['source_size'] = sample['source_image'].size
        sample['target_size'] = sample['target_image'].size

        # Sample points
        self.sample_points(sample)

        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample
    
class CO3D(CorrespondenceDataset):
    """
    Dataset class for the CO3D dataset.
    """

    def load_data(self):
        if not os.path.exists(os.path.join(self.dataset_directory, "cache.h5")):
            with open(os.path.join(self.dataset_directory, 'metadata.json'), 'r') as file:
                metadata = json.load(file)

            h5_file = h5py.File(os.path.join(self.dataset_directory, 'co3d.h5'), 'r')
            annotations = h5_file['data']
            
            # Standard source matrix
            image_size = 128
            num_samples = self.config.get('num_samples', -1)

            for sequence_name in tqdm(list(metadata.keys())[:num_samples]):
                sequence = metadata[sequence_name]
                category = sequence['category']
                image_paths = sequence['image_paths']
                
                # fix paths
                image_paths = [p.replace('/export/group/datasets/co3d/', '/export/compvis-nfs/group/datasets/co3d/') for p in image_paths]
                source_image_path = os.path.join(self.dataset_directory, image_paths[0])
                target_image_paths = [os.path.join(self.dataset_directory, p) for p in image_paths[1:]]

                target_matrix = torch.tensor(annotations[sequence_name])

                for t_idx in range(len(target_image_paths)):
                    set_indices = (target_matrix[t_idx] != -1).nonzero(as_tuple=True)
                    set_values = target_matrix[t_idx][set_indices]
                    target_points, source_points = [], []
                    for i in range(set_indices[0].size(0)):
                        target_points.append(torch.tensor([set_indices[0][i], set_indices[1][i]]))
                        source_points.append(torch.tensor([set_values[i] // image_size, set_values[i] % image_size]))
                    
                    if len(target_points) == 0:
                        continue

                    # Calculate source and target bounding boxes
                    source_bbox = torch.tensor([source_points[0][0], source_points[0][1], source_points[0][0], source_points[0][1]])
                    target_bbox = torch.tensor([target_points[0][0], target_points[0][1], target_points[0][0], target_points[0][1]])

                    self.data.append({
                        'source_image_path': source_image_path,
                        'target_image_path': target_image_paths[t_idx],
                        'source_points': torch.stack(source_points, dim=0).float(),
                        'target_points': torch.stack(target_points, dim=0).float(),
                        'source_bbox': source_bbox,
                        'target_bbox': target_bbox,
                        'source_category': category,
                        'target_category': category
                    })

            # write data to h5py if it does not exist
            with h5py.File(os.path.join(self.dataset_directory, "cache.h5"), 'w') as h5file:
                for i, item in enumerate(data):
                    group = h5file.create_group(f'item_{i}')
                    group.create_dataset('source_image_path', data=item['source_image_path'])
                    group.create_dataset('target_image_path', data=item['target_image_path'])
                    group.create_dataset('source_points', data=item['source_points'].numpy())
                    group.create_dataset('target_points', data=item['target_points'].numpy())
                    group.create_dataset('source_bbox', data=item['source_bbox'])
                    group.create_dataset('target_bbox', data=item['target_bbox'])
                    group.create_dataset('source_category', data=item['source_category'])
                    group.create_dataset('target_category', data=item['target_category'])
        else:
            with h5py.File(os.path.join(self.dataset_directory, "cache.h5"), 'r') as h5file:
                for key in h5file.keys():
                    group = h5file[key]
                    data.append({
                        'source_image_path': group['source_image_path'][()],
                        'target_image_path': group['target_image_path'][()],
                        'source_points': torch.tensor(group['source_points'][:]).float(),
                        'target_points': torch.tensor(group['target_points'][:]).float(),
                        'source_bbox': torch.tensor(group['source_bbox'][:]),
                        'target_bbox': torch.tensor(group['target_bbox'][:]),
                        'source_category': group['source_category'][()],
                        'target_category': group['target_category'][()]
                    })

        return data
    
    def non_maximum_suppression(self, points, radius):
        """
        Perform non-maximum suppression on a set of points with a specified radius.
        
        Parameters:
        points (torch.Tensor): Tensor of shape (N, 2) containing the (x, y) coordinates of the points.
        radius (float): Radius for suppression.
        
        Returns:
        torch.Tensor: Tensor containing the points after non-maximum suppression.
        """
        suppressed = torch.zeros(points.shape[0], dtype=torch.bool)
        indices = []
        
        for i in range(points.shape[0]):
            if suppressed[i]:
                continue
            indices.append(i)
            distances = torch.norm(points - points[i], dim=1)
            suppressed = suppressed | (distances < radius)
        
        return points[indices], torch.tensor(indices)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # prevent memory leak

        # Load image
        sample['source_image'] = Image.open(sample['source_image_path'])
        sample['target_image'] = Image.open(sample['target_image_path'])

        # Save image size
        sample['source_size'] = sample['source_image'].size
        sample['target_size'] = sample['target_image'].size

        # Non-maximum suppression on points
        sample['source_points'], idx = self.non_maximum_suppression(sample['source_points'], 10)
        sample['target_points'] = sample['target_points'][idx]

        # image width to square image ratio
        if sample['source_size'][0] > sample['source_size'][1]: # W > H
            ratio = sample['source_size'][0] / sample['source_size'][1]
            sample['source_points'][:, 1] *= ratio
            sample['target_points'][:, 1] *= ratio
        else: # H > W
            ratio = sample['source_size'][1] / sample['source_size'][0]
            sample['source_points'][:, 0] *= ratio
            sample['target_points'][:, 0] *= ratio

        def rescale_points(points, old_size, new_size):
            """
            Rescale points to match new image size.

            Args:
                points (torch.Tensor): [N, 2] where each point is (x, y)
                old_size (tuple): (width, height)
                new_size (tuple): (width, height)

            Returns:
                torch.Tensor: [N, 2] where each point is (x, y)
            """
            x_scale = new_size[0] / old_size[0]
            y_scale = new_size[1] / old_size[1]
            scaled_points = torch.multiply(points, torch.tensor([x_scale, y_scale], device=points.device))
            return scaled_points

        # Resize points
        sample['source_points'] = rescale_points(sample['source_points'], (128, 128), sample['source_size'])
        sample['target_points'] = rescale_points(sample['target_points'], (128, 128), sample['target_size'])
        
        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample