import json
import os
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
from glob import glob
from PIL import Image
from torch.nn.functional import pad as F_pad
from tqdm import tqdm
import itertools
# General

def preprocess_kps_pad(kps, img_width, img_height, size, padding=True):
    """
    Adjusts the key points for an image resized to the given size.
    If padding is True, adjusts key points for padded resizing to preserve aspect ratio.
    If padding is False, allows image distortion without padding.
    
    Args:
        kps (torch.Tensor): Key points with shape (N, 3), where the last column indicates visibility.
        img_width (int): Original width of the image.
        img_height (int): Original height of the image.
        size (int): Desired size (new width and height) of the resized image.
        padding (bool): Whether to apply padding to preserve the aspect ratio.
        
    Returns:
        torch.Tensor: Adjusted key points.
        float: Scale factor used for width adjustment.
        float: Scale factor used for height adjustment.
        int: X offset (0 if padding=False).
        int: Y offset (0 if padding=False).
    """
    kps = kps.clone()

    if padding:
        # Logic for padded resizing to preserve aspect ratio
        scale = size / max(img_width, img_height)
        kps[:, [0, 1]] *= scale
        if img_height < img_width:
            new_h = int(np.around(size * img_height / img_width))
            offset_y = int((size - new_h) / 2)
            offset_x = 0
            kps[:, 1] += offset_y
        elif img_width < img_height:
            new_w = int(np.around(size * img_width / img_height))
            offset_x = int((size - new_w) / 2)
            offset_y = 0
            kps[:, 0] += offset_x
        else:
            offset_x = 0
            offset_y = 0
        scale_x = scale_y = scale
    else:
        # Logic for resizing without padding (image distortion)
        scale_x = size / img_width
        scale_y = size / img_height
        kps[:, 0] *= scale_x
        kps[:, 1] *= scale_y
        offset_x = 0
        offset_y = 0

    # Zero out any non-visible key points
    kps *= kps[:, 2:3].clone()

    return kps, scale_x, scale_y, offset_x, offset_y

def load_and_prepare_data(args):
    """
    Load and prepare dataset for training.

    Parameters:
    - PASCAL_TRAIN: Flag to indicate if training on PASCAL dataset.
    - BBOX_THRE: Flag to indicate if bounding box thresholds are used.
    - ANNO_SIZE: Annotation size.
    - SAMPLE: Sampling rate for the dataset.

    Returns:
    - files: List of file paths.
    - kps: Keypoints tensor.
    - cats: Categories tensor.
    - used_points_set: Used points set.
    - all_thresholds (optional): All thresholds.
    """

    # Determining the data directory and categories based on the training dataset
    if args.TRAIN_DATASET=='pascal':
        data_dir = 'data/PF-dataset-PASCAL'
        categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    else:
        data_dir = 'data/SPair-71k'
        categories = sorted(os.listdir(os.path.join(data_dir, 'ImageAnnotation')))

    files, kps, cats, used_points_set, all_thresholds = ([] for _ in range(5))

    # Loading data based on the dataset and preprocessing it
    for cat_idx, cat in tqdm(enumerate(categories), total=len(categories), desc="Processing Categories"):
        if args.TRAIN_DATASET=='pascal':
            single_files, single_kps, thresholds, used_points = load_pascal_data(data_dir, size=args.ANNO_SIZE, category=cat, split='train', subsample=args.SAMPLE, padding=not args.NO_PADDING)
        else:
            single_files, single_kps, thresholds, used_points = load_spair_data(data_dir, size=args.ANNO_SIZE, category=cat, split='trn', subsample=args.SAMPLE, padding=not args.NO_PADDING)
        
        files.extend(single_files)
        single_kps = F_pad(single_kps, (0, 0, 0, 30 - single_kps.shape[1], 0, 0), value=0)
        kps.append(single_kps)
        used_points_set.extend([used_points] * (len(single_files) // 2))
        cats.extend([cat_idx] * (len(single_files) // 2))
        if args.BBOX_THRE:
            all_thresholds.extend(thresholds)
    kps = torch.cat(kps, dim=0)

    # Shuffling the data
    shuffled_files, shuffled_kps, shuffled_cats, shuffled_used_points_set, shuffled_thresholds = shuffle_data(files, kps, cats, used_points_set, all_thresholds, args.BBOX_THRE)

    return shuffled_files, shuffled_kps, shuffled_cats, shuffled_used_points_set, shuffled_thresholds if args.BBOX_THRE else None

def shuffle_data(files, kps, cats, used_points_set, all_thresholds, BBOX_THRE):
    """
    Shuffle dataset pairs.

    Parameters are lists of files, keypoints, categories, used points, all thresholds, and a flag for bounding box thresholds.
    Returns shuffled lists.
    """
    pair_count = len(files) // 2
    pair_indices = torch.randperm(pair_count)
    actual_indices = pair_indices * 2

    shuffled_files = [files[idx] for i in actual_indices for idx in [i, i+1]]
    shuffled_kps = torch.cat([kps[idx:idx+2] for idx in actual_indices])
    shuffled_cats = [cats[i//2] for i in actual_indices]
    shuffled_used_points_set = [used_points_set[i//2] for i in actual_indices]
    shuffled_thresholds = [all_thresholds[idx] for i in actual_indices for idx in [i, i+1]] if BBOX_THRE else []

    return shuffled_files, shuffled_kps, shuffled_cats, shuffled_used_points_set, shuffled_thresholds

def load_eval_data(args, path, category, split):
    if args.EVAL_DATASET == 'pascal':
        files, kps, thresholds, used_kps = load_pascal_data(path, args.ANNO_SIZE, category, split, args.TEST_SAMPLE, not args.NO_PADDING)
    elif args.EVAL_DATASET == 'spair':
        files, kps, thresholds, used_kps = load_spair_data(path, args.ANNO_SIZE, category, split, args.TEST_SAMPLE, not args.NO_PADDING)
    elif args.EVAL_DATASET == 'cub':
        files, kps, thresholds, used_kps = load_cub_data(path, args.ANNO_SIZE, category, split, args.TEST_SAMPLE, not args.NO_PADDING)

    return files, kps, thresholds, used_kps

def get_dataset_info(args, split):
    if args.EVAL_DATASET == 'pascal':
        data_dir = 'data/PF-dataset-PASCAL'
        categories = sorted(os.listdir(os.path.join(data_dir, 'Annotations')))
    elif args.EVAL_DATASET == 'spair':
        data_dir = 'data/SPair-71k'
        categories = sorted(os.listdir(os.path.join(data_dir, 'ImageAnnotation')))
    elif args.EVAL_DATASET == 'cub':
        data_dir = 'data/CUB_200_2011'
        with open(os.path.join(data_dir, "classes.txt"), "r") as f:
            categories = [line.strip().split('.')[1].replace('_', '') for line in f.readlines()]

    return data_dir, categories, split

# SPair-71K

def load_spair_data(path="data/SPair-71k", size=256, category='cat', split='test', subsample=None, padding=True):
    np.random.seed(42)
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*:{category}.json'))
    if subsample is not None and subsample > 0:
        pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
    files = []
    thresholds = []
    kps = []
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    for pair in pairs:
        source_kps = torch.zeros(num_kps, 3)
        target_kps = torch.zeros(num_kps, 3)
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        source_json_name = source_fn.replace('JPEGImages','ImageAnnotation').replace('jpg','json')
        target_json_name = target_fn.replace('JPEGImages','ImageAnnotation').replace('jpg','json')
        source_bbox = np.asarray(data["src_bndbox"])    # (x1, y1, x2, y2)
        target_bbox = np.asarray(data["trg_bndbox"])
        with open(source_json_name) as f:
            file = json.load(f)
            kpts_src = file['kps']
        with open(target_json_name) as f:
            file = json.load(f)
            kpts_trg = file['kps']

        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        for i in range(30):
            point = kpts_src[str(i)]
            if point is None:
                source_kps[i, :3] = 0
            else:
                source_kps[i, :2] = torch.Tensor(point).float()  # set x and y
                source_kps[i, 2] = 1
        source_kps, src_scale_x, src_scale_y, src_x, src_y = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size, padding)
        
        for i in range(30):
            point = kpts_trg[str(i)]
            if point is None:
                target_kps[i, :3] = 0
            else:
                target_kps[i, :2] = torch.Tensor(point).float()
                target_kps[i, 2] = 1
        target_kps, trg_scale_x, trg_scale_y, trg_x, trg_y = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size, padding)
        
        if split == 'test' or split == 'val':
            # Use the y-scale factor for height-based thresholds
            thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]) * trg_scale_y)
        elif split == 'trn':
            # Use respective scales for source and target bounding boxes
            thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]) * src_scale_y)
            thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]) * trg_scale_y)

        kps.append(source_kps)
        kps.append(target_kps)
        files.append(source_fn)
        files.append(target_fn)
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]

    return files, kps, thresholds, used_kps

# Pascal

def read_mat(path, obj_name):
    r"""Reads specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj

def process_kps_pascal(kps):
    # Step 1: Reshape the array to (20, 2) by adding nan values
    num_pad_rows = 20 - kps.shape[0]
    if num_pad_rows > 0:
        pad_values = np.full((num_pad_rows, 2), np.nan)
        kps = np.vstack((kps, pad_values))
        
    # Step 2: Reshape the array to (20, 3) 
    # Add an extra column: set to 1 if the row does not contain nan, 0 otherwise
    last_col = np.isnan(kps).any(axis=1)
    last_col = np.where(last_col, 0, 1)
    kps = np.column_stack((kps, last_col))

    # Step 3: Replace rows with nan values to all 0's
    mask = np.isnan(kps).any(axis=1)
    kps[mask] = 0

    return torch.tensor(kps).float()

def load_pascal_data(path="data/PF-dataset-PASCAL", size=256, category='cat', split='test', subsample=None, padding=True):
    
    def get_points(point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
        Xpad = -np.ones(20)
        Xpad[: len(X)] = X
        Ypad = -np.ones(20)
        Ypad[: len(X)] = Y
        Zmask = np.zeros(20)
        Zmask[: len(X)] = 1
        point_coords = np.concatenate(
            (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1,20)), axis=0
        )
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
    
    np.random.seed(42)
    files = []
    kps = []
    test_data = pd.read_csv(f'{path}/{split}_pairs_pf_pascal.csv')
    cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    cls_ids = test_data.iloc[:,2].values.astype("int") - 1
    cat_id = cls.index(category)
    subset_id = np.where(cls_ids == cat_id)[0]
    # logger.info(f'Number of Pairs for {category} = {len(subset_id)}')
    subset_pairs = test_data.iloc[subset_id,:]
    src_img_names = np.array(subset_pairs.iloc[:,0])
    trg_img_names = np.array(subset_pairs.iloc[:,1])
    # print(src_img_names.shape, trg_img_names.shape)
    if not split.startswith('train'):
        point_A_coords = subset_pairs.iloc[:,3:5]
        point_B_coords = subset_pairs.iloc[:,5:]
    # print(point_A_coords.shape, point_B_coords.shape)
    for i in range(len(src_img_names)):
        src_fn= f'{path}/../{src_img_names[i]}'
        trg_fn= f'{path}/../{trg_img_names[i]}'
        src_size=Image.open(src_fn).size
        trg_size=Image.open(trg_fn).size

        if not split.startswith('train'):
            point_coords_src = get_points(point_A_coords, i).transpose(1,0)
            point_coords_trg = get_points(point_B_coords, i).transpose(1,0)
        else:
            src_anns = os.path.join(path, 'Annotations', category,
                                    os.path.basename(src_fn))[:-4] + '.mat'
            trg_anns = os.path.join(path, 'Annotations', category,
                                    os.path.basename(trg_fn))[:-4] + '.mat'
            point_coords_src = process_kps_pascal(read_mat(src_anns, 'kps'))
            point_coords_trg = process_kps_pascal(read_mat(trg_anns, 'kps'))

        # print(src_size)
        source_kps, src_scale_x, src_scale_y, src_x, src_y = preprocess_kps_pad(point_coords_src, src_size[0], src_size[1], size, padding)
        target_kps, trg_scale_x, trg_scale_y, trg_x, trg_y = preprocess_kps_pad(point_coords_trg, trg_size[0], trg_size[1], size, padding)
        kps.append(source_kps)
        kps.append(target_kps)
        files.append(src_fn)
        files.append(trg_fn)
    
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    # logger.info(f'Final number of used key points: {kps.size(1)}')
    return files, kps, None, used_kps

# PF-WILLOW
def load_pfwillow_data(path="data/PF-Willow", size=256, category=None, subsample=None, padding=True):
    """
    Load the PF-Willow dataset in a structured format.
    """
    np.random.seed(42)
    csv_file = os.path.join(path, 'test_pairs_pf.csv')
    pairs = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        pairs = list(reader)
    
    if subsample is not None and subsample > 0:
        pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
    
    files = []
    thresholds = []
    kps = []
    
    for row in pairs:
        source_image_path = os.path.join(path, row[0].replace('PF-dataset/', ''))
        target_image_path = os.path.join(path, row[1].replace('PF-dataset/', ''))
        
        row[2:] = list(map(float, row[2:]))
        source_points = torch.tensor(list(zip(row[2:12], row[12:22])), dtype=torch.float32)  # X, Y
        target_points = torch.tensor(list(zip(row[22:32], row[32:])), dtype=torch.float32)  # X, Y

        # Use min and max to get the bounding box
        source_bbox = np.array([source_points[:, 0].min(), source_points[:, 1].min(),
                                source_points[:, 0].max(), source_points[:, 1].max()], dtype=np.float32)
        target_bbox = np.array([target_points[:, 0].min(), target_points[:, 1].min(),
                                target_points[:, 0].max(), target_points[:, 1].max()], dtype=np.float32)

        # Convert from (x, y, x+w, y+h) to (x, y, w, h)
        source_bbox[2:] -= source_bbox[:2]
        target_bbox[2:] -= target_bbox[:2]

        # Get category from image path if needed
        img_category = source_image_path.split('(')[0] if category is None else category

        # Preprocess keypoints with padding
        source_kps, src_scale_x, src_scale_y, src_x, src_y = preprocess_kps_pad(source_points, source_bbox[2], source_bbox[3], size, padding)
        target_kps, trg_scale_x, trg_scale_y, trg_x, trg_y = preprocess_kps_pad(target_points, target_bbox[2], target_bbox[3], size, padding)

        # Use the target bounding box for thresholds
        thresholds.append(max(target_bbox[3], target_bbox[2]) * trg_scale_y)

        kps.append(source_kps)
        kps.append(target_kps)
        files.append(source_image_path)
        files.append(target_image_path)
    
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]

    return files, kps, thresholds, used_kps

# CUB
def load_cub_data(path="data/CUB-200-2011", size=256, category=None, split="train", subsample=None, padding=True):
    """
    Load the CUB-200-2011 dataset in a structured format, assuming images are in a flat directory structure.
    
    Parameters:
    - path: Path to the CUB dataset.
    - size: Image size for resizing.
    - split: Either 'train' or 'test' split.
    - subsample: Number of samples to randomly select, or None for all.
    - padding: Whether to apply padding during preprocessing.
    - category: Specify a single category to filter, or None for all categories.
    """
    np.random.seed(42)
    images_dir = os.path.join(path, 'images')

    # Load necessary metadata files
    with open(os.path.join(path, "images.txt"), "r") as f:
        images = [line.strip().split() for line in f.readlines()]

    with open(os.path.join(path, "train_test_split.txt"), "r") as f:
        train_test_split = [line.strip().split() for line in f.readlines()]

    with open(os.path.join(path, "parts/part_locs.txt"), "r") as f:
        part_locs = {}
        for line in f.readlines():
            img_id, _, x, y, visible = line.strip().split()
            if img_id not in part_locs:
                part_locs[img_id] = []
            part_locs[img_id].append((float(x), float(y), visible == '1'))

    with open(os.path.join(path, "image_class_labels.txt"), "r") as f:
        image_class_labels = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}

    with open(os.path.join(path, "bounding_boxes.txt"), "r") as f:
        bounding_boxes = {line.split()[0]: list(map(float, line.strip().split()[1:])) for line in f.readlines()}

    with open(os.path.join(path, "classes.txt"), "r") as f:
        classes = [line.strip().split('.')[1].replace('_', '') for line in f.readlines()]

    # Handle filtering by category
    category_id = None
    if category:
        try:
            category_id = classes.index(category) + 1  # 1-based index for CUB class IDs
        except ValueError:
            raise ValueError(f"Category '{category}' not found in CUB dataset classes.")

    # Filter images based on train/test split and optionally by category
    filtered_images = [
        (img_id, f"{classes[int(image_class_labels[img_id]) - 1]}_{os.path.basename(img_name)}")
        for img_id, img_name in images
        if (split == 'train' and int(train_test_split[int(img_id) - 1][1])) or
           (split == 'test' and not int(train_test_split[int(img_id) - 1][1])) and
           (category_id is None or image_class_labels[img_id] == category_id)
    ]

    files = []
    thresholds = []
    kps = []

    # Generate all pairs for each class or within the specified category
    class_ids = [category_id] if category_id else range(1, 201)
    for class_id in class_ids:
        class_images = [img for img in filtered_images if image_class_labels[img[0]] == class_id]
        pairs = list(itertools.combinations(class_images, 2))

        if subsample is not None and subsample > 0:
            pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]

        for source, target in pairs:
            source_image_path = os.path.join(images_dir, source[1])
            target_image_path = os.path.join(images_dir, target[1])

            source_points = np.array(part_locs[source[0]], dtype=float)
            target_points = np.array(part_locs[target[0]], dtype=float)

            # Filter out points that are not visible in either of the images
            visible_points = np.logical_and(source_points[:, 2], target_points[:, 2])
            if not visible_points.any():
                # If no visible keypoints, skip this pair
                continue
            source_points = torch.tensor(source_points[visible_points][:, :2], dtype=torch.float32)
            target_points = torch.tensor(target_points[visible_points][:, :2], dtype=torch.float32)

            # Bounding box data
            source_bbox = np.array(bounding_boxes[source[0]], dtype=np.float32)
            target_bbox = np.array(bounding_boxes[target[0]], dtype=np.float32)

            # Preprocess keypoints and bounding boxes
            print(source_points, source_bbox[2], source_bbox[3], size, padding)
            source_kps, src_scale_x, src_scale_y, src_x, src_y = preprocess_kps_pad(source_points, source_bbox[2], source_bbox[3], size, padding)
            target_kps, trg_scale_x, trg_scale_y, trg_x, trg_y = preprocess_kps_pad(target_points, target_bbox[2], target_bbox[3], size, padding)

            thresholds.append(max(target_bbox[2], target_bbox[3]) * trg_scale_y)

            kps.append(source_kps)
            kps.append(target_kps)
            files.append(source_image_path)
            files.append(target_image_path)

    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    return files, kps, thresholds, used_kps
