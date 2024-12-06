import os
import torch
import pickle
import wandb
import argparse
from PIL import Image
from tqdm import tqdm
from loguru import logger
from itertools import chain
torch.set_num_threads(16)
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from preprocess_map import set_seed
from model_utils.projection_network import AggregationNetwork, DummyAggregationNetwork
import utils.utils_visualization as utils_visualization
from utils.logger import get_logger, log_geo_stats, update_stats, update_geo_stats, log_weighted_pcks, load_config
from utils.utils_geoware import SPAIR_GEO_AWARE, SPAIR_FLIP, SPAIR_FLIP_TRN, permute_indices, renumber_indices, flip_keypoints, renumber_used_points, optimized_kps_1_to_2
from utils.utils_correspondence import kpts_to_patch_idx, load_img_and_kps, convert_to_binary_mask, calculate_keypoint_transformation, get_distance, get_distance_mutual_nn
from utils.utils_dataset import load_eval_data, load_and_prepare_data, get_dataset_info

device  = 'cuda' if torch.cuda.is_available() else 'cpu'

def normalize_feats(args, feats, epsilon=1e-10):
    if not args.ONLY_DINO:  # separate norm
        feat_sd = feats[..., :640+1280+1280] #sd feature
        feat_dino = feats[..., 640+1280+1280:] #dino feature
        norms_sd = torch.linalg.norm(feat_sd, dim=-1)[:, :, None]
        norm_feats_sd = feat_sd / (norms_sd + epsilon)
        norms_dino = torch.linalg.norm(feat_dino, dim=-1)[:, :, None]
        norm_feats_dino = feat_dino / (norms_dino + epsilon)
        feats = torch.cat([norm_feats_sd, norm_feats_dino], dim=-1)
    # (b, w*h, c)
    norms = torch.linalg.norm(feats, dim=-1)[:, :, None]
    norm_feats = feats / (norms + epsilon)
    # norm_feats = feats / norms
    
    return norm_feats

def prepare_feature_paths_and_load(aggre_net, img_path, flip, ensemble, num_patches, device):
    # Construct feature paths
    feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
    suffix_flip = '_flip' if flip else ''
    ensemble_folder = f'features_ensemble{ensemble}' if ensemble > 1 else 'features'
    mask_path = f"{feature_base}_mask{suffix_flip}.png"
    sd_path = f"{feature_base}_sd{suffix_flip}.pt".replace('features', ensemble_folder)
    dino_path = f"{feature_base}_dino{suffix_flip}.pt".replace('features', ensemble_folder)

    # Load DINO features
    desc_dino = torch.load(dino_path).to(device)

    # If ONLY_DINO is not set, load SD features
    if not args.ONLY_DINO:
        sd_path = f"{feature_base}_sd{suffix_flip}.pt".replace('features', ensemble_folder)
        features_sd = torch.load(sd_path)
        for k in features_sd:
            features_sd[k] = features_sd[k].to(device)
        
        # Prepare descriptors
        desc_gathered = torch.cat([
            features_sd['s3'],
            F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            desc_dino
        ], dim=1)
    else:
        if desc_dino.shape[-2:] != (num_patches, num_patches):
            desc_dino = F.interpolate(desc_dino, size=(num_patches, num_patches), mode='bilinear', align_corners=False)
        desc_gathered = desc_dino  # Only DINO features

    desc = aggre_net(desc_gathered).reshape(1, 1, -1, num_patches**2).permute(0, 1, 3, 2)

    # Load mask if it exists
    mask = None
    if os.path.exists(mask_path):
        mask = convert_to_binary_mask(mask_path)
    return desc, mask

def get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, flip=False, flip2=False, img1=None, img2=None, device='cuda'):
    img_path_1 = files[pair_idx * 2]
    img_path_2 = files[pair_idx * 2 + 1]
    # save the imgs for cases if the feature doesn't exist
    img1_desc, mask1 = prepare_feature_paths_and_load(aggre_net, img_path_1, flip, args.ENSEMBLE, num_patches, device)
    img2_desc, mask2 = prepare_feature_paths_and_load(aggre_net, img_path_2, flip2, args.ENSEMBLE, num_patches, device)
    # normalize the desc
    img1_desc = normalize_feats(args, img1_desc[0])
    img2_desc = normalize_feats(args, img2_desc[0])
    return img1_desc, img2_desc, mask1, mask2

def compute_pck(args, save_path, aggre_net, files, kps, category=None, used_points=None, thresholds=None):
    out_results = []
    num_patches = args.NUM_PATCHES
    current_save_results = 0
    gt_correspondences, pred_correspondences, img_acc_001, img_acc_005, img_acc_01, len_kpts = ([] for _ in range(6))
    if thresholds is not None:
        thresholds = torch.tensor(thresholds).to(device)
        bbox_size=[]
    N = len(files) // 2
    pbar = tqdm(total=N)

    if args.COMPUTE_GEOAWARE_METRICS:   # get the geo-aware idx list
        geo_aware_count = geo_aware_total_count = 0
        geo_idx_all, influ_list_geo_filtered = [], []
        influ_list_geo = SPAIR_GEO_AWARE[category] if category in SPAIR_GEO_AWARE else None
        for item in influ_list_geo:
            item = [item] if isinstance(item, int) else item
            temp_list = [idx for idx in item if idx in used_points]
            if len(temp_list) >= 1:
                influ_list_geo_filtered.append(temp_list)
        raw_geo_aware = renumber_indices(influ_list_geo_filtered, counter=[0])
    
    if args.ADAPT_FLIP: # get the permute list for flipping
        FLIP_ANNO = SPAIR_FLIP[category]
        if sum(len(i) if isinstance(i, list) else 1 for i in FLIP_ANNO) == kps[0].shape[0]:
            permute_list = FLIP_ANNO
        else:
            influ_list_filtered = []
            influ_list = FLIP_ANNO
            for item in influ_list:
                item = [item] if isinstance(item, int) else item
                temp_list = [idx for idx in item if idx in used_points]
                if len(temp_list) >= 1:
                    influ_list_filtered.append(temp_list)
            permute_list = renumber_indices(influ_list_filtered, counter=[0])

    for pair_idx in range(N):
        # Load images and keypoints
        img1, img1_kps = load_img_and_kps(idx=2*pair_idx, files=files, kps=kps, img_size=args.ANNO_SIZE, edge=False)
        img2, img2_kps = load_img_and_kps(idx=2*pair_idx+1, files=files, kps=kps, img_size=args.ANNO_SIZE, edge=False)
        # Get mutual visibility
        vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
        vis2 = img2_kps[:, 2]
        # Get patch descriptors
        with torch.no_grad():
            img1_desc, img2_desc, mask1, mask2 = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, img1=img1, img2=img2)
        # Get patch index for the keypoints
        img1_patch_idx = kpts_to_patch_idx(args, img1_kps, num_patches)
        # Get similarity matrix
        kps_1_to_2 = calculate_keypoint_transformation(args, img1_desc, img2_desc, img1_patch_idx, num_patches)

        if args.ADAPT_FLIP:
            img1_flip = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img1_desc_flip, _, mask1_flip, _ = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, flip=True, img1=img1.transpose(Image.FLIP_LEFT_RIGHT), img2=img2)
            img1_kps_flip = flip_keypoints(img1_kps, args.ANNO_SIZE, permute_indices(permute_list, vis))
            img1_patch_idx_flip = kpts_to_patch_idx(args, img1_kps_flip, num_patches)
            kps_1_to_2_flip = calculate_keypoint_transformation(args, img1_desc_flip, img2_desc, img1_patch_idx_flip, num_patches)
            
            # get the distance for the flip and original img
            if args.MUTUAL_NN:
                original_dist = get_distance_mutual_nn(img1_desc, img2_desc)
                flip_dist = get_distance_mutual_nn(img1_desc_flip, img2_desc)
            else:
                original_dist = get_distance(img1_desc, img2_desc, mask1, mask2, num_patches)
                flip_dist = get_distance(img1_desc_flip, img2_desc, mask1_flip, mask2, num_patches)

            kps_1_to_2 = optimized_kps_1_to_2(args, kps_1_to_2, kps_1_to_2_flip, img1_kps, img2_kps, flip_dist, original_dist, vis, permute_list)

        # collect the result for more complicated eval
        single_result = {
            "src_fn": files[2*pair_idx],  # must
            "trg_fn": files[2*pair_idx+1],  # must
            # "category": category,
            # "used_points": used_points.cpu().numpy(),
            # "src_kpts": renumber_used_points(img1_kps, used_points).cpu().numpy(),
            # "trg_kpts": renumber_used_points(img2_kps, used_points).cpu().numpy(),
            "src_kpts_pred": renumber_used_points(kps_1_to_2.cpu(), used_points).cpu().detach().numpy(),  # must
            # "threshold": thresholds[pair_idx].item() if thresholds is not None else 0,
            "resize_resolution": args.ANNO_SIZE,  # must
        }
        out_results.append(single_result)

        gt_kps = img2_kps[vis][:, [1,0]]
        prd_kps = kps_1_to_2[vis][:, [1,0]]
        gt_correspondences.append(gt_kps)
        pred_correspondences.append(prd_kps)
        len_kpts.append(vis.sum().item())

        # compute per image acc
        if not args.KPT_RESULT: # per img result
            single_gt_correspondences = img2_kps[vis][:, [1,0]]
            single_pred_correspondences = kps_1_to_2[vis][:, [1,0]]
            alpha = torch.tensor([0.1, 0.05, 0.01]) if args.EVAL_DATASET != 'pascal' else torch.tensor([0.1, 0.05, 0.15])
            correct = torch.zeros(3)
            err = (single_gt_correspondences - single_pred_correspondences.cpu()).norm(dim=-1)
            err = err.unsqueeze(0).repeat(3, 1)
            if thresholds is not None:
                single_bbox_size = thresholds[pair_idx].repeat(vis.sum()).cpu()
                correct += (err < alpha.unsqueeze(-1) * single_bbox_size.unsqueeze(0)).float().mean(dim=-1)
            else:
                correct += (err < alpha.unsqueeze(-1) * args.ANNO_SIZE).float().mean(dim=-1)
            img_acc_01.append(correct[0].item())
            img_acc_005.append(correct[1].item())
            img_acc_001.append(correct[2].item())

        if thresholds is not None:
            pckthres = thresholds[pair_idx].repeat(vis.sum())
            bbox_size.append(pckthres)

        if args.COMPUTE_GEOAWARE_METRICS:
            geo_aware_list, geo_aware_full_list = ([] for _ in range(2))
            for item in raw_geo_aware:
                # convert to list
                item = [item] if isinstance(item, int) else item
                # check if all items are visible
                temp_list = [idx for idx in item if vis[idx]]
                temp_list2 = [idx for idx in item if vis2[idx]]
                # if more than 2 items are visible, add to geo_aware_list
                if len(temp_list2) >= 2 and len(temp_list) >= 1:
                    for temp_idx in temp_list:
                        geo_aware_list.append([temp_idx])
                    geo_aware_full_list.append(temp_list)
            
            geo_aware_idx = [item for sublist in geo_aware_list for item in sublist]
            geo_idx_mask = torch.zeros(len(vis)).bool()
            geo_idx_mask[geo_aware_idx] = True
            geo_idx_mask = geo_idx_mask[vis]
            geo_idx_all.append(torch.tensor(geo_idx_mask))
            
            # count the number of geo-aware pairs
            if len(geo_aware_full_list) > 0: 
                geo_aware_total_count += len(geo_aware_idx)     # per keypoint
                geo_aware_count += 1                            # per img
            
        if current_save_results!=args.TOTAL_SAVE_RESULT:
            if args.ADAPT_FLIP and (flip_dist < original_dist): # save the flip result
                utils_visualization.save_visualization(thresholds, pair_idx, vis, save_path, category, 
                       img1_kps_flip, img1_flip, img2, kps_1_to_2, img2_kps, args.ANNO_SIZE, args.ADAPT_FLIP)
            else:
                utils_visualization.save_visualization(thresholds, pair_idx, vis, save_path, category, 
                       img1_kps, img1, img2, kps_1_to_2, img2_kps, args.ANNO_SIZE, args.ADAPT_FLIP)
            current_save_results += 1

        pbar.update(1)
    if not args.KPT_RESULT:
        img_correct = torch.tensor([img_acc_01, img_acc_005, img_acc_001])
        img_correct = img_correct.mean(dim=-1).tolist()
        img_correct.append(N)
    else:
        img_correct = None
    gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
    pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
    alpha = torch.tensor([0.1, 0.05, 0.01]) if args.EVAL_DATASET != 'pascal' else torch.tensor([0.1, 0.05, 0.15])
    correct = torch.zeros(len(alpha))
    err = (pred_correspondences - gt_correspondences).norm(dim=-1)
    err = err.unsqueeze(0).repeat(len(alpha), 1)
    if thresholds is not None:
        bbox_size = torch.cat(bbox_size, dim=0).cpu()
        threshold = alpha.unsqueeze(-1) * bbox_size.unsqueeze(0)
        correct_all = err < threshold
    else:
        threshold = alpha * args.ANNO_SIZE
        correct_all = err < threshold.unsqueeze(-1)

    correct = correct_all.sum(dim=-1) / len(gt_correspondences)
    correct = correct.tolist()
    correct.append(len(gt_correspondences))
    alpha2pck = zip(alpha.tolist(), correct[:3]) if args.KPT_RESULT else zip(alpha.tolist(), img_correct[:3])
    logger.info(f'{category}...'+' | '.join([f'PCK-Transfer@{alpha:.2f}: {pck_alpha * 100:.2f}%'
        for alpha, pck_alpha in alpha2pck]))
    
    geo_score = []
    if args.COMPUTE_GEOAWARE_METRICS:
        geo_idx_all = torch.cat(geo_idx_all, dim=0).cpu()
        correct_geo = correct_all[:,geo_idx_all].sum(dim=-1) / geo_idx_all.sum().item()
        correct_geo = correct_geo.tolist()
        geo_score.append(geo_aware_count / N)
        geo_score.append(geo_aware_total_count / len(gt_correspondences))
        geo_score.extend(correct_geo)
        geo_score.append(geo_idx_all.sum().item())
        alpha2pck_geo = zip(alpha.tolist(), correct_geo[:3])
        logger.info(' | '.join([f'PCK-Transfer_geo-aware@{alpha:.2f}: {pck_alpha * 100:.2f}%'
                        for alpha, pck_alpha in alpha2pck_geo]))
        logger.info(f'Geo-aware occurance count: {geo_aware_count}, with ratio {geo_aware_count / N * 100:.2f}%; total count ratio {geo_aware_total_count / len(gt_correspondences) * 100:.2f}%')

    return correct, geo_score, out_results, img_correct

def eval(args, aggre_net, save_path, split='val'):
    aggre_net.eval()  # Set the network to evaluation mode
    # Configure data directory and categories based on the dataset type
    data_dir, categories, split = get_dataset_info(args, split)

    # Initialize lists for results and statistics
    total_out_results, pcks, pcks_05, pcks_01, weights, kpt_weights = ([] for _ in range(6))
    if args.COMPUTE_GEOAWARE_METRICS: geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo = ([] for _ in range(6))

    # Process each category
    for cat in categories:
        # Load data based on the dataset
        files, kps, thresholds, used_points = load_eval_data(args, data_dir, cat, split)
        # Compute PCK with or without bbox threshold
        compute_args = (save_path, aggre_net, files, kps, cat, used_points)
        pck, correct_geo, out_results, img_correct = compute_pck(args, *compute_args, thresholds=thresholds) if args.BBOX_THRE else compute_pck(args, *compute_args)
        total_out_results.extend(out_results)
        update_stats(args, pcks, pcks_05, pcks_01, weights, kpt_weights, pck, img_correct)
        if args.COMPUTE_GEOAWARE_METRICS: update_geo_stats(geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo, correct_geo)

    # Calculate and log weighted PCKs
    pck_010, pck_005, pck_001 = log_weighted_pcks(args, logger, pcks, pcks_05, pcks_01, weights)
    if args.COMPUTE_GEOAWARE_METRICS: log_geo_stats(args, geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo, kpt_weights, total_out_results)

    return pck_010, pck_005, pck_001, total_out_results

def main(args):
    set_seed(args.SEED)
    args.BBOX_THRE = not (args.IMG_THRESHOLD or args.EVAL_DATASET == 'pascal')
    if args.SAMPLE == 0: args.SAMPLE = None # use all the data
    feature_dims = [768] if args.ONLY_DINO else [640, 1280, 1280, 768] # dimensions for three layers of SD and one layer of DINOv2 features

    # Determine the evaluation type and project name based on args
    save_path = f'./results_{args.EVAL_DATASET}/eval_{args.NOTE}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not args.NOT_WANDB:
        wandb.init(project=args.EVAL_DATASET, name=f'eval_{args.NOTE}', config=args)
    
    logger = get_logger(save_path+'/result.log')
    logger.info(args)
    aggre_net = DummyAggregationNetwork()
    aggre_net.to(device)

    with torch.no_grad():
        _,_,_,result = eval(args, aggre_net, save_path, split='test')
        with open(save_path+'/result.pkl', 'wb') as f:
            pickle.dump(result, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # load config
    parser.add_argument('--config', type=str, default=None)                         # path to the config file

    # evaluation setup
    parser.add_argument('--NOTE', type=str, default='')                             # note for the experiment
    parser.add_argument('--SEED', type=int, default=0)                              # random seed
    parser.add_argument('--PAIR_AUGMENT', action='store_true', default=False)       # set true to use the pair augmentation
    parser.add_argument('--SAMPLE', type=int, default=0)                            # set the number of samples to use, 0 to use all
    parser.add_argument('--IMG_THRESHOLD', action='store_true', default=False)      # set true to use the image threshold
    parser.add_argument('--NOT_WANDB', action='store_true', default=False)          # set true to not use wandb
    parser.add_argument('--ENSEMBLE', type=int, default=1)                          # set the number of ensemble to use
    parser.add_argument('--TOTAL_SAVE_RESULT', type=int, default=0)                 # set the number of results to save
    parser.add_argument('--ANNO_SIZE', type=int, default=840)                       # set the size of the annotation image
    parser.add_argument('--NUM_PATCHES', type=int, default=60)                      # set the number of patches
    parser.add_argument('--TEST_SAMPLE', type=int, default=0)

    parser.add_argument('--EVAL_DATASET', type=str, default='spair')                # set the evaluation dataset, 'spair' for SPair-71k, 'pascal' for PF-Pascal, 'cub' for CUB_200_2011
    parser.add_argument('--COMPUTE_GEOAWARE_METRICS', action='store_true', default=False)   # set true to use the geo-aware count
    parser.add_argument('--KPT_RESULT', action='store_true', default=False)         # set true to evaluate per kpt result, in the paper, this is used for comparing unsupervised methods, following ASIC
    parser.add_argument('--ADAPT_FLIP', action='store_true', default=False)         # set true to use the flipped images, adaptive flip
    parser.add_argument('--MUTUAL_NN', action='store_true', default=False)          # set true to use the flipped images, adaptive flip, mutual nn as metric
    parser.add_argument('--SOFT_EVAL', action='store_true', default=False)          # set true to use the soft argmax eval
    parser.add_argument('--SOFT_EVAL_WINDOW', type=int, default=7)                  # set true to use the window soft argmax eval, window size is 2*SOFT_EVAL_WINDOW+1, 0 to be standard soft argmax
    parser.add_argument('--ONLY_DINO', action='store_true', default=False)          # set true to use only the DINO features
    parser.add_argument('--NO_PADDING', action='store_true', default=False)         # set true to not pad the image

    args = parser.parse_args()
    if args.config is not None: # load config file and update the args
        args_dict = vars(args)
        args_dict.update(load_config(args.config))
        args = argparse.Namespace(**args_dict)
    main(args)