import os
import sys
import torch
import h5py
import json
import argparse
from tqdm import tqdm

from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.three_d import get_frame_data, points_2d_to_3d, points_3d_to_2d, screen_to_world, world_to_screen

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default="../data/co3d", help="Path to CO3D dataset root")
parser.add_argument("--output_root", type=str, default="../data/co3d", help="Output HDF5 file")
parser.add_argument("--max_frames", type=int, default=4, help="Maximum number of frames per sequence")
parser.add_argument("--image_width", type=int, default=128, help="Image width")
parser.add_argument("--image_height", type=int, default=128, help="Image height")
parser.add_argument("--quality_threshold", type=float, default=0.7, help="Quality threshold")
parser.add_argument("--thickness_threshold", type=float, default=0.01, help="Thickness threshold")
#parser.add_argument("--mutual_filter", action="store_true", help="Apply mutual filter")
args = parser.parse_args()

# Setup dataset root and paths
dataset_root = args.dataset_root
categories = ['teddybear', 'carrot', 'toytrain', 'kite', 'wineglass', 'plant', 'couch', 'parkingmeter', 'mouse', 'broccoli', 'toytruck', 'toybus', 'bench', 'laptop', 'remote', 'hairdryer', 'donut', 'handbag', 'toaster', 'cake', 'orange', 'cup', 'toyplane', 'backpack', 'sandwich', 'vase', 'chair', 'bottle', 'car', 'hotdog', 'microwave', 'tv', 'stopsign', 'cellphone', 'hydrant', 'baseballglove', 'apple', 'motorcycle', 'keyboard', 'bowl', 'banana', 'suitcase', 'book', 'skateboard', 'pizza', 'ball', 'frisbee', 'toilet', 'bicycle', 'baseballbat', 'umbrella']

image_width = args.image_width
image_height = args.image_height
max_frames = args.max_frames
quality_threshold = args.quality_threshold
thickness_threshold = args.thickness_threshold
mutual_filter = True #args.mutual_filter
image_size = (image_width, image_height)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create hdf5 file
os.makedirs(args.output_root, exist_ok=True)
output_file = os.path.join(args.output_root, f"co3d.h5")
hdf5_file = h5py.File(output_file, 'w')
data = hdf5_file.create_group("data")
metadata = {}

# Standard source matrix
source_matrix = torch.arange(image_width * image_height, dtype=torch.int32).reshape(image_height, image_width)

for category in categories:
    print(f"Processing {category}...")
    frame_file = os.path.join(dataset_root, category, "frame_annotations.jgz")
    sequence_file = os.path.join(dataset_root, category, "sequence_annotations.jgz")

    # Initialize dataset
    expand_args_fields(JsonIndexDataset)
    dataset = JsonIndexDataset(
        frame_annotations_file=frame_file,
        sequence_annotations_file=sequence_file,
        dataset_root=dataset_root,
        image_height=image_height,
        image_width=image_width,
        load_point_clouds=False,
        load_images=False,
        load_masks=False,
        load_depths=True,
        load_depth_masks=True,
        box_crop=False,
    )

    for i, sequence_name in enumerate(tqdm(dataset.seq_annots.keys())):
        """if dataset.seq_annots[sequence_name].point_cloud is None:
            continue

        quality = dataset.seq_annots[sequence_name].point_cloud.quality_score
        if quality < quality_threshold:
            continue"""

        target_matrix = -1 * torch.ones((max_frames-1, image_height, image_width), dtype=torch.int32)
        
        frame_data = get_frame_data(dataset, sequence_name, max_frames)
        if frame_data is None:
            continue
        camera = frame_data.camera.to(device)
        depth_mask = frame_data.depth_mask.to(device)
        depth_map = frame_data.depth_map.to(device)

        # Get source points from depth mask
        source_idx = 0
        source_points = (depth_mask[source_idx].squeeze(0) > .5).nonzero()[:, [1, 0]] # X, Y in screen space

        # Re-project source points on source image
        source_points_screen = points_2d_to_3d(source_points, depth_map[source_idx].squeeze(0)) # X, Y, Z in screen space
        source_points_world = screen_to_world(source_points_screen, camera[source_idx], image_size) # X, Y, Z in world space
        source_points_screen = world_to_screen(source_points_world, camera[source_idx], image_size) # Y, X, Z in screen space
        source_points = points_3d_to_2d(source_points_screen, image_size) # X, Y in screen space
        
        # Project points to all target images
        for target_idx in range(1, len(camera)):
            target_points_screen = world_to_screen(source_points_world, camera[target_idx], image_size) # Y, X, Z in screen space
            target_points = points_3d_to_2d(target_points_screen, image_size) # X, Y in screen space

            # Remove points that are outside of the image
            valid_indices = (target_points[:, 0] >= 0) & (target_points[:, 0] < image_width) & (target_points[:, 1] >= 0) & (target_points[:, 1] < image_height)
            target_points = target_points[valid_indices]
            source_points_t = source_points[valid_indices]

            if mutual_filter:
                # Re-project target points on target image
                temp_points_screen = points_2d_to_3d(target_points, depth_map[target_idx].squeeze(0)) # X, Y, Z in screen space
                temp_points_world = screen_to_world(temp_points_screen, camera[target_idx], image_size) # X, Y, Z in world space
                temp_points_screen = world_to_screen(temp_points_world, camera[target_idx], image_size) # Y, X, Z in screen space
                target_indices = torch.abs(target_points_screen[:, 2] - temp_points_screen[:, 2]) < thickness_threshold
                target_points = target_points[target_indices]
                source_points_t = source_points_t[target_indices]

            # Filter out points outside of the depth mask
            target_indices = depth_mask[target_idx, 0, target_points[:, 0], target_points[:, 1]] > .5
            target_points = target_points[target_indices]
            source_points_t = source_points_t[target_indices]

            # Create target matrix
            for i in range(len(target_points)):
                tx, ty = target_points[i]
                sx, sy = source_points_t[i]
                target_matrix[target_idx-1, ty, tx] = source_matrix[sy, sx]

        # Save to hdf5
        data.create_dataset(sequence_name, data=target_matrix)

        # Save image paths to metadata
        metadata[sequence_name] = {
            "category": category,
            "image_paths": frame_data.image_path,
        }

# Write metadata to json file
metadata_file = os.path.join(args.output_root, "metadata.json")
with open(metadata_file, "w") as f:
    json.dump(metadata, f)

# Close hdf5 file
hdf5_file.close()
print("Done!")