import h5py
import tqdm
import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import read_dataset_config, load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='COCO', help='Name of dataset to train on')
    parser.add_argument('--dataset_config', type=str, default='../configs/dataset_config.yaml', help='Path to dataset config file')

    # Parse arguments
    args = parser.parse_args()
    dataset_config = args.dataset_config
    dataset_name = args.dataset_name

    # Load dataset config
    dataset_config = read_dataset_config(dataset_config)

    # Load dataset parameters
    config = dataset_config[dataset_name]
    config['split'] = 'train'

    dataset = load_dataset(dataset_name, config)

    # Create HDF5 file
    hdf5_file = h5py.File(dataset_name + '.h5', 'w')

    # Save dataset data to HDF5
    keys_order = []
    for sample in tqdm.tqdm(dataset.data):
        image_path = sample['image_path']
        category = sample['category']
        image, size = dataset.load_image(image_path)

        key = os.path.basename(image_path)
        keys_order.append(key)
        group = hdf5_file.create_group(key)
        group.create_dataset('image', data=image)
        group.create_dataset('size', data=size)
        group.create_dataset('category', data=category)

    hdf5_file.create_dataset('keys_order', data=keys_order)
    hdf5_file.close()