import os
import json
import copy
import h5py
import torch
import scipy.io
import numpy as np
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO as COCOAPI

class ImageDataset(data.Dataset):
    """
    General dataset class for image datasets.
    """

    def __init__(self, config, preprocess=None):
        self.config = config
        self.dataset_directory = config['path']
        self.preprocess = preprocess
        self.split = config.get('split', 'test')
        self.image_pair = config.get('image_pair', False)
        self.num_samples = config.get('num_samples', None)
        self.image_sampling = config.get('image_sampling', 'random_category')
        self.top_k = self.config.get('top_k', 10)
        self.category_to_id = {}
        self.data = []

        self.hdf5_path = config.get('hdf5_path', None)
        self.hdf5_file = None
        if self.hdf5_path is not None:
            self.load_hdf5()
        else:
            self.load_data()

        if self.image_sampling == 'retrieval':
            embeddings = torch.load(self.config['embeddings_path'])
            embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min()) # min-max normalize
            self.weights = embeddings @ embeddings.transpose(0, 1)

    def load_hdf5(self):
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        keys_order = [key.decode('utf-8') for key in self.hdf5_file['keys_order']]
        for key in keys_order:
            group = self.hdf5_file[key]
            self.data.append({
                "image_path": key,
                **{key: group[key][()] for key in group.keys() if key != 'image'}
            })

    def load_data(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def create_category_to_id(self):
        self.category_to_id = {}
        for i, sample in enumerate(self.data):
            category = sample['category']
            if category not in self.category_to_id:
                self.category_to_id[category] = []
            self.category_to_id[category].append(i)

    def load_image(self, path):
        if self.hdf5_file is not None:
            group = self.hdf5_file[path]
            image = group['image'][()]
            size = group['size'][()]
        else:
            image = Image.open(path).convert('RGB')
            size = image.size
        return image, size

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # prevent memory leak

        if self.image_pair:
            if self.image_sampling == 'retrieval':
                weights = self.weights[idx]
                _, top_k_indices = torch.topk(weights, self.top_k)
                match_id = top_k_indices[np.random.choice(self.top_k)]
            elif self.image_sampling == 'random_category':
                match_id = np.random.choice(self.category_to_id[sample['category']])
            elif self.image_sampling == 'random':
                match_id = np.random.choice(len(self.data))
            elif self.image_sampling == 'same':
                match_id = idx

            matching_sample = self.data[match_id]
            sample['source_image'], sample['source_size'] = self.load_image(sample['image_path'])
            sample['target_image'], sample['target_size'] = self.load_image(matching_sample['image_path'])
            sample['source_category'] = sample['category']
            sample['target_category'] = matching_sample['category']
        else:
            sample['image'], sample['size'] = self.load_image(sample['image_path'])

        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample

class ImageNet(ImageDataset):
    """
    Dataset class for the ImageNet dataset.
    """

    def load_data(self):
        syn_to_class = {}

        with open(os.path.join(self.dataset_directory, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                syn_to_class[v[0]] = int(class_id)
        
        with open(os.path.join(self.dataset_directory, "ILSVRC2012_val_labels.json"), "rb") as f:
            val_to_syn = json.load(f)

        samples_dir = os.path.join(self.dataset_directory, "ILSVRC2012_" + self.split, "data")
        for entry in os.listdir(samples_dir):
            if self.split == "train":
                syn_id = entry
                target = syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.data.append({
                        "image_path": sample_path,
                        "category": target
                    })
            elif self.split == "val":
                syn_id = val_to_syn[entry]
                target = syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.data.append({
                    "image_path": sample_path,
                    "category": target
                })

class COCO(ImageDataset):
    """
    Dataset class for the COCO dataset.
    """

    def load_data(self):

        ann_file = os.path.join(self.dataset_directory, "annotations", f"instances_{self.split}2017.json")
        coco = COCOAPI(ann_file)

        cat_ids = coco.getCatIds()
        img_ids = coco.getImgIds()

        for img_id in img_ids:
            img = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            anns = [ann for ann in anns if ann['iscrowd'] == 0]

            if len(anns) == 0:
                continue

            image_path = os.path.join(self.dataset_directory, self.split + "2017", img['file_name'])
            categories = [coco.loadCats(ann['category_id'])[0]['name'] for ann in anns]

            self.data.append({
                "image_path": image_path,
                "category": categories[0],
            })

class PASCALPart(ImageDataset):
    """
    Dataset class for the PASCAL-Part dataset.
    """

    def __init__(self, config, preprocess=None):
        self.single_object = config.get('single_object', False)
        super().__init__(config, preprocess)

    def load_annotations(self, path):
        annotations = scipy.io.loadmat(path)["anno"]
        objects = annotations[0, 0]["objects"]
        objects_list = []

        for obj_idx in range(objects.shape[1]):
            obj = objects[0, obj_idx]

            classname = obj["class"][0]
            mask = obj["mask"]

            parts_list = []
            parts = obj["parts"]
            for part_idx in range(parts.shape[1]):
                part = parts[0, part_idx]
                part_name = part["part_name"][0]
                part_mask = part["mask"]
                parts_list.append({"part_name": part_name, "mask": part_mask})

            objects_list.append({"class": classname, "mask": mask, "parts": parts_list})

        return objects_list

    def load_data(self):
        annotation_directory = os.path.join(self.dataset_directory, "Annotations_Part")
        
        for annotation_filename in os.listdir(annotation_directory):
            annotations = self.load_annotations(os.path.join(annotation_directory, annotation_filename))
            if self.single_object and len(annotations) > 1:
                continue
            image_filename = annotation_filename.replace(".mat", ".jpg")
            image_path = os.path.join(self.dataset_directory, "VOC2010", "JPEGImages", image_filename)

            # get points from part mass centers
            # get bounding box from object mask in (x, y, w, h)
            points = []
            bbox = []
            for obj in annotations:
                for part in obj["parts"]:
                    mask = part["mask"]
                    y, x = np.where(mask)
                    points.append([np.mean(x), np.mean(y)])

                mask = obj["mask"]
                y, x = np.where(mask)
                x1, x2 = min(x), max(x)
                y1, y2 = min(y), max(y)
                bbox.append([x1, y1, x2 - x1, y2 - y1])

            self.data.append({
                "image_path": image_path,
                "categories": [obj["class"] for obj in annotations], 
                "annotations": annotations,
                "points": torch.tensor(points),
                "bbox": torch.tensor(bbox[0]), # single object
                "category": annotations[0]["class"], # single object
                "mask": annotations[0]["mask"] # single object
            })
    
    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # prevent memory leak

        if self.image_pair:
            # make it deterministic
            np.random.seed(23)
            random_category = np.random.choice(sample['categories']) # sample a random category from the image
            match_id = np.random.choice(self.category_to_id[random_category]) # sample a random image from the same category
            matching_sample = self.data[match_id]
            sample['source_image'], sample['source_size'] = self.load_image(sample['image_path'])
            sample['target_image'], sample['target_size'] = self.load_image(matching_sample['image_path'])
            sample['source_category'] = random_category
            sample['target_category'] = random_category
            sample['source_annotations'] = sample['annotations']
            sample['target_annotations'] = matching_sample['annotations']
            sample['source_points'] = sample['points']
            sample['target_points'] = matching_sample['points']
            sample['source_bbox'] = sample['bbox']
            sample['target_bbox'] = matching_sample['bbox']
            sample['source_mask'] = sample['mask']
            sample['target_mask'] = matching_sample['mask']
        else:
            sample['image'], sample['size'] = self.load_image(sample['image_path'])

        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample