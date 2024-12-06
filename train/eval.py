import os
import torch
import argparse
from tqdm import tqdm

import torch.utils.data as data
from accelerate import Accelerator

from utils.dataset import read_dataset_config, load_dataset, cache_dataset, Preprocessor, CacheDataset
from utils.model import read_model_config, load_model
from utils.correspondence import compute_pck_img, compute_pck_bbox, rescale_points
from utils.visualization import plot_results

os.environ["NCCL_P2P_LEVEL"] = "NVL" # Configures NCCL to use NVLink for peer-to-peer (P2P) communication if available

def eval(model, dataloader, accelerator, pck_threshold, use_cache=False, save_histograms=False, save_predictions=False, window_softargmax=False, pose_align=False):
    model.eval()

    pbar = tqdm(total=len(dataloader), disable=(not accelerator.is_main_process))

    pck_img = torch.tensor(0, device=accelerator.device)
    pck_bbox = torch.tensor(0, device=accelerator.device)
    keypoints = torch.tensor(0, device=accelerator.device)

    histograms = []
    predictions = []
    corrects = []
    distances = []
    
    for batch in dataloader:
        if use_cache:
            output = getattr(model, 'module', model).compute_correspondence(batch, return_histograms=save_histograms, window_softargmax=window_softargmax)
            if save_histograms:
                predicted_points, hists = output
            else:
                predicted_points = output
        else:
            # Run through model
            predicted_points = getattr(model, 'module', model).get_features_and_compute_correspondence(batch, return_histograms=save_histograms, window_softargmax=window_softargmax, pose_align=pose_align)
            
        # Calculate PCK values
        batch_size = len(predicted_points)
        target_points = batch['target_points']
        target_bbox = batch['target_bbox']
        target_size = batch['target_size']
        for b in range(batch_size):
            pck_img += compute_pck_img(predicted_points[b], target_points[b], target_size[b], pck_threshold)
            pck_bbox += compute_pck_bbox(predicted_points[b], target_points[b], target_bbox[b], pck_threshold)
            keypoints += len(target_points[b])

            if save_histograms:
                histograms.append(hists[b])
            if save_predictions:
                distances.append(torch.linalg.norm(predicted_points[b] - target_points[b], axis=-1))
                predictions.append(predicted_points[b])
                y, x, h, w = target_bbox[b]
                corrects.append(distances[-1] <= pck_threshold * max(h, w))

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({'PCK_img': (accelerator.gather_for_metrics(pck_img).sum() / accelerator.gather_for_metrics(keypoints).sum()).item() * 100,
                          'PCK_bbox': (accelerator.gather_for_metrics(pck_bbox).sum() / accelerator.gather_for_metrics(keypoints).sum()).item() * 100})

    pbar.close()
    
    if save_histograms:
        # save histograms for later analysis
        histograms = torch.stack(histograms, dim=0)
        histograms = accelerator.gather_for_metrics(histograms).mean(0)
        if accelerator.is_main_process:
            torch.save(histograms, f"histograms.pt")

    if save_predictions:
        # save predictions for later analysis
        predictions = accelerator.gather_for_metrics(predictions)
        corrects = accelerator.gather_for_metrics(corrects)
        distances = accelerator.gather_for_metrics(distances)
        if accelerator.is_main_process:
            #print("Saving predictions...", predictions, corrects, distances)
            torch.save([predictions, corrects, distances], f"predictions.pt")
            print("Saved.")

    pck_img = accelerator.gather_for_metrics(pck_img).sum()
    pck_bbox = accelerator.gather_for_metrics(pck_bbox).sum()
    keypoints = accelerator.gather_for_metrics(keypoints).sum()
    if accelerator.is_main_process:
        print(f"PCK_img: {(pck_img / keypoints) * 100:.2f}, PCK_bbox: {(pck_bbox / keypoints) * 100:.2f}")
    return pck_img / keypoints, pck_bbox / keypoints

def evaluate(model, dataloader, accelerator, pck_threshold, layers=None, use_cache=False, save_histograms=False, save_predictions=False, window_softargmax=False, pose_align=False):
    if layers is None:
        return eval(model, dataloader, accelerator, pck_threshold, use_cache, save_histograms, save_predictions, window_softargmax, pose_align)

    pck_img = []
    pck_bbox = []
    for l in layers:
        print(f"Layer {l}:")
        if hasattr(dataloader.dataset, 'set_layer'):
            dataloader.dataset.set_layer(l)
        pck_img_l, pck_bbox_l = eval(model, dataloader, accelerator, pck_threshold, use_cache, save_histograms, save_predictions, window_softargmax, pose_align)
        pck_img.append(pck_img_l)
        pck_bbox.append(pck_bbox_l)
    return pck_img, pck_bbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of model to evaluate')
    parser.add_argument('--dataset_names', type=str, nargs='+', default=['SPair-71k', 'PF-WILLOW', 'CUB-200-2011'], help='Names of the datasets to evaluate on')
    parser.add_argument('--model_config', type=str, default='configs/eval_config.yaml', help='Path to model config file')
    parser.add_argument('--dataset_config', type=str, default='configs/dataset_config.yaml', help='Path to dataset config file')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--pck_threshold', type=float, default=0.1, help='PCK threshold')
    parser.add_argument('--use_cache', action=argparse.BooleanOptionalAction, default=False, help='Precalculate features and use them for faster evaluation')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Directory to store cached features')
    parser.add_argument('--reset_cache', action=argparse.BooleanOptionalAction, default=False, help='Reset cache')
    parser.add_argument('--num_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False, help='Plot results')
    parser.add_argument('--save_histograms', action=argparse.BooleanOptionalAction, default=False, help='Save histograms for later analysis')
    parser.add_argument('--save_predictions', action=argparse.BooleanOptionalAction, default=False, help='Save confusions for later analysis')

    # Parse arguments
    args = parser.parse_args()
    model_name = args.model_name
    dataset_names = args.dataset_names
    dataset_config = args.dataset_config
    model_config = args.model_config
    num_workers = args.num_workers
    pck_threshold = args.pck_threshold
    use_cache = args.use_cache
    cache_dir = args.cache_dir
    reset_cache = args.reset_cache
    num_samples = args.num_samples
    plot = args.plot
    save_histograms = args.save_histograms
    save_predictions = args.save_predictions

    # Load model config
    model_config = read_model_config(model_config)[model_name]
    model_config['device'] = "cuda" # some models need to know the device (hedlin, luo, tang)

    # Get model parameters
    image_size = model_config.get('image_size', (512, 512))
    batch_size = model_config.get('batch_size', 8)
    drop_last_batch = model_config.get('drop_last_batch', False)
    grad_enabled = model_config.get('grad_enabled', False)
    rescale_data = model_config.get('rescale_data', False)
    image_range = model_config.get('image_range', (-1, 1))
    layers = model_config.get('layers', None)
    half_precision = model_config.get('half_precision', False)
    window_softargmax = model_config.get('window_softargmax', False)
    pose_align = model_config.get('pose_align', False)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if half_precision else "no", # bf16 for A100
    )

    # Load model
    model = load_model(model_name, model_config)
    model = accelerator.prepare(model)

    # Print number of parameters
    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")

    # Load dataset config
    dataset_config = read_dataset_config(dataset_config)
    
    # Print seperator
    if accelerator.is_main_process:
        print(f"\n{'='*30} Evaluate {model_name} {'='*30}\n")

    # Evaluate
    for dataset_name in dataset_names:
        if accelerator.is_main_process:
            print(f"Dataset: {dataset_name}")

        # Load dataset parameters
        config = dataset_config[dataset_name]
        dataset_num_samples = config.get('num_samples', None)
        random_sampling = config.get('random_sampling', False)

        preprocess = Preprocessor(image_size, preprocess_image = not use_cache, image_range=image_range, rescale_data=rescale_data)
        dataset = load_dataset(dataset_name, config, preprocess)

        # Limit number of samples if specified
        min_num_samples = min(filter(None, [num_samples, dataset_num_samples, len(dataset)]))
        if random_sampling:
            torch.manual_seed(42)
            dataset.data = [dataset.data[i] for i in torch.randperm(len(dataset))]
        dataset.data = dataset.data[:min_num_samples]

        # Cache dataset
        with accelerator.main_process_first():
            if use_cache:
                if not os.path.exists(cache_dir):
                    os.mkdir(cache_dir)
                cache_path = os.path.join(cache_dir, f"{model_name}_{dataset_name}.h5")
                if accelerator.is_main_process:
                    # if model has attribute module, use it to cache the dataset
                    dataset = cache_dataset(model.module if hasattr(model, 'module') else model, dataset, cache_path, reset_cache, batch_size, num_workers, accelerator.device, half_precision)
                else:
                    dataset = CacheDataset(dataset, cache_path)

        # Create dataloader
        def collate_fn(batch):
            output = {}
            for key in batch[0].keys():
                output[key] = [sample[key] for sample in batch]
                if key in ['source_image', 'target_image', 'source_features', 'target_features']:
                    output[key] = torch.stack(output[key])
            return output
        
        dataloader = data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=num_workers,
                                     drop_last=drop_last_batch,
                                     collate_fn=collate_fn)
        
        dataloader = accelerator.prepare(dataloader)

        with torch.set_grad_enabled(grad_enabled):
            pck_img, pck_bbox = evaluate(model, dataloader, accelerator, pck_threshold, layers, use_cache, save_histograms, save_predictions, window_softargmax, pose_align)

        if accelerator.is_main_process:
            if plot and layers is not None:
                if not os.path.exists('plots'):
                    os.mkdir('plots')
                plot_results(pck_img, pck_bbox, layers, f"plots/{model_name}_{dataset_name}.png")

        #break
    
    if accelerator.is_main_process:
        print(f"\n{'='*30} Finished {'='*30}\n")