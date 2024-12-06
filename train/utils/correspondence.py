import torch
from torchvision.transforms import ToTensor
from torch.nn.functional import interpolate
from torchvision.transforms import Normalize
from PIL import Image
import einops

def flip_points(points):
    """
    Flip x and y axis.

    Args:
        points (torch.Tensor): [N, 2] where each point is (x, y)

    Returns:
        torch.Tensor: [N, 2] where each point is (y, x)
    """
    return points[:, [1, 0]]

def flip_bbox(bbox):
    """
    Flip x, y and w, h axis.

    Args:
        bbox (torch.Tensor): [4] with (x, y, w, h)

    Returns:
        torch.Tensor: [4] with (y, x, h, w)
    """
    return bbox[[1, 0, 3, 2]]

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

def rescale_bbox(bbox, old_size, new_size):
    """
    Rescale bounding box to match new image size.

    Args:
        bbox (torch.Tensor): [4] with (x, y, w, h)
        old_size (tuple): (width, height)
        new_size (tuple): (width, height)

    Returns:
        torch.Tensor: [4] with (x, y, w, h)
    """
    x_scale = new_size[0] / old_size[0]
    y_scale = new_size[1] / old_size[1]
    scaled_bbox = torch.multiply(bbox, torch.tensor([x_scale, y_scale, x_scale, y_scale], device=bbox.device))
    return scaled_bbox

def preprocess_image(image_pil, size, range=[-1, 1], norm=False):
    """
    Convert PIL image to tensor and normalize to [-1, 1].

    Args:
        image_pil (PIL.Image): Image to preprocess
        size (tuple): (width, height)
        range (tuple): (min, max)
        norm (bool): Whether to normalize image

    Returns:
        torch.Tensor: [C, H, W]
    """
    image = ToTensor()(image_pil) # [C, H, W] and range [0, 1]
    image = interpolate(image.unsqueeze(0), size, mode="bilinear", antialias=True).squeeze(0)
        
    if norm:
        image = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image) # DINOv2 mean and std
    image = image * (range[1] - range[0]) + range[0] # range [min, max]
    return image

def preprocess_points(points, old_size, new_size):
    """
    Rescale points to match new image size and flip x and y axis.

    Args:
        points (torch.Tensor): [N, 2] where each point is (x, y)
        old_size (tuple): (width, height)
        new_size (tuple): (width, height)

    Returns:
        torch.Tensor: [N, 2] where each point is (y, x)
    """
    points = rescale_points(points, old_size, new_size) # [N, 2]
    points = flip_points(points) # flip x and y axis to match image
    return points

def postprocess_points(points, old_size, new_size):
    """
    Flip x and y axis and rescale points to match new image size.
    
    Args:
        points (torch.Tensor): [N, 2] where each point is (x, y)
        old_size (tuple): (width, height)
        new_size (tuple): (width, height)

    Returns:
        torch.Tensor: [N, 2] where each point is (x, y)
    """
    points = flip_points(points) # flip x and y axis to match image
    points = rescale_points(points, old_size, new_size) # [N, 2]
    return points

def preprocess_bbox(bbox, old_size, new_size):
    """
    Rescale bounding box to match new image size and flip x, y and w, h axis.

    Args:
        bbox (torch.Tensor): [4] with (x, y, w, h)
        old_size (tuple): (width, height)
        new_size (tuple): (width, height)

    Returns:
        torch.Tensor: [4] with (y, x, h, w)
    """
    bbox = rescale_bbox(bbox, old_size, new_size) # [4]
    bbox = flip_bbox(bbox) # flip x and y axis to match image
    return bbox

def postprocess_bbox(bbox, old_size, new_size):
    """
    Flip x, y and w, h axis and rescale bounding box to match new image size.

    Args:
        bbox (torch.Tensor): [4] with (x, y, w, h)
        old_size (tuple): (width, height)
        new_size (tuple): (width, height)

    Returns:
        torch.Tensor: [4] with (x, y, w, h)
    """
    bbox = flip_bbox(bbox) # flip x and y axis to match image
    bbox = rescale_bbox(bbox, old_size, new_size) # [4]
    return bbox

def points_to_idxs(points, size):
    """
    Convert points to indices (clamped to image size).
    
    Args:
        points (torch.Tensor): [B, N, 2] where each point is (y, x)
        size (tuple): (height, width)
    
    Returns:    
        torch.Tensor: [B, N] where each point is an index
    """
    h, w = size
    points_y = points[:, :, 0].float().clamp(0, h - 1)
    points_x = points[:, :, 1].float().clamp(0, w - 1)
    idxs = w * torch.round(points_y).long() + torch.round(points_x).long()
    return idxs

def idxs_to_points(idxs, size):
    """
    Convert indices to points.
    
    Args:
        idxs (torch.Tensor): [B, N] where each point is an index
        size (tuple): (height, width)
    
    Returns:    
        torch.Tensor: [B, N, 2] where each point is (y, x)
    """
    h, w = size
    points_y = idxs // w
    points_x = idxs % w
    points = torch.stack([points_y, points_x], dim=2)
    return points

def flatten_features(features):
    """
    Flatten features.

    Args:
        features (torch.Tensor): [B, C, H, W]
    
    Returns:
        torch.Tensor: [B, HxW, C]
    """
    return features.flatten(2).permute(0, 2, 1)

def normalize_features(features):
    """
    Normalize features (L2 norm).

    Args:
        features (torch.Tensor): [B, HxW, C]

    Returns:
        torch.Tensor: [B, HxW, C]
    """
    return features / torch.linalg.norm(features, dim=-1).unsqueeze(-1)

def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)

import torch.nn.functional as F

def softargmax2d(input, beta=100):
    """
    Apply the soft-argmax function on the input tensor.

    Parameters:
        input (torch.Tensor): The input tensor for which the soft-argmax is to be computed.
        beta (float, optional): The beta parameter to adjust the smoothness of the output. Default is 100.

    Returns:
        torch.Tensor: The soft-argmax output. [B, 2]
    """

    *_, h, w = input.shape

    input = input.reshape(*_, h * w)
    input = F.softmax(beta * input, dim=-1)

    indices_c, indices_r = torch.meshgrid(
        torch.linspace(0, 1, w),
        torch.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = indices_r.reshape(-1, h * w).to(input.device)
    indices_c = indices_c.reshape(-1, h * w).to(input.device)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    return torch.stack([result_r, result_c], dim=-1)

def window_softargmax2d(similarity_map, predicted_points, window_size=125, beta=50):
    """
    Apply the window soft-argmax function on the input tensor.

    Parameters:
        similarity_map (torch.Tensor): The similarity map tensor. [B, N, HxW]
        predicted_points (torch.Tensor): The predicted points tensor. [B, N, 2]
        window_size (int, optional): The size of the window. Default is 15.
        beta (float, optional): The beta parameter to adjust the smoothness of the output. Default is 100.

    Returns:
        torch.Tensor: The window soft-argmax output. [B, 2]
    """

    h, w = similarity_map.shape[-2:]
    windows = []
    for n, p in enumerate(predicted_points.permute(1, 0, 2)):  # [N, B, 2]
        window_masks = []
        for b in range(p.shape[0]):
            mask = torch.zeros_like(similarity_map[:, n])  # [B, H, W]
            x_center, y_center = p[b].long()
            x_min = max(x_center - window_size // 2, 0)
            x_max = min(x_center + window_size // 2, h)
            y_min = max(y_center - window_size // 2, 0)
            y_max = min(y_center + window_size // 2, w)
            mask[:, x_min:x_max+1, y_min:y_max+1] = 1
            window_masks.append(mask)
        window_masks = torch.stack(window_masks, dim=0)  # [B, H, W]
        masked_similarity = similarity_map[:, n] * window_masks # [B, H, W]
        windows.append(masked_similarity)
    windows = torch.stack(windows, dim=1)  # [B, N, H, W]
    predicted_points = softargmax2d(windows, beta).squeeze(2)  # [B, N, 2]
    return predicted_points

def compute_correspondence(source_features, target_features, source_points, source_size, target_size=None, return_histograms=False, batch_mode=True, window_softargmax=False):
    """
    Compute correspondence between source and target features using points from the source image.

    Args:
        source_features (torch.Tensor): [B, C, H, W]
        target_features (torch.Tensor): [B, C, H, W]
        source_points (torch.Tensor): [B, N, 2] where each point is (y, x)
        source_size (tuple): (width, height)
        target_size (tuple): (width, height)

    Returns:
        torch.Tensor: [B, N, 2] where each point is (y, x)
    """
    
    if target_size is None:
        target_size = source_size

    if not batch_mode:
        source_features = source_features.unsqueeze(0)
        target_features = target_features.unsqueeze(0)
        source_points = source_points.unsqueeze(0)

    # Get image sizes
    sw, sh = source_size
    tw, th = target_size

    # Resize features to match scale of points
    source_features = interpolate(source_features, (sh, sw), mode="bilinear")
    target_features = interpolate(target_features, (th, tw), mode="bilinear")

    # Use source points to get features
    source_idxs = points_to_idxs(source_points, (sh, sw)) # [B, N]
    source_features = flatten_features(source_features) # [B, HxW, C]
    source_features = normalize_features(source_features) # [B, HxW, C]
    source_features = source_features[torch.arange(source_features.shape[0])[:, None], source_idxs] # [B, N, C]
    
    # Calculate similarity map
    target_features = flatten_features(target_features) # [B, HxW, C]
    target_features = normalize_features(target_features) # [B, HxW, C]
    similarity_map = source_features @ target_features.transpose(1, 2) # [B, N, HxW]

    # Get max similarity for each point and convert to coordinates (y, x)
    predicted_idx = torch.argmax(similarity_map, dim=-1) # [B, N]
    predicted_points = idxs_to_points(predicted_idx, (th, tw)) # [B, N, 2]

    # Apply window soft-argmax (optional)
    if window_softargmax:
        predicted_points = window_softargmax2d(similarity_map.reshape(*similarity_map.shape[:2], th, tw), predicted_points)

    if not batch_mode:
        predicted_points = predicted_points.squeeze(0)

    if return_histograms:
        # Calculate histogram of similarity map relative to image size
        hist = similarity_map.reshape(-1, th * tw) # [BxN, HxW]
        hist = per_sample_min_max_normalization(hist) # [BxN, HxW]
        hist = torch.histc(hist, bins=100, min=0, max=1) # [BxN, 100]
        hist = hist / (th * tw)
        return predicted_points, hist
    
    return predicted_points

def compute_pck_img(predicted_points, target_points, image_size, threshold=0.1):
    """
    Compute PCK (Percent of Correct Keypoints) value with respect to image size.

    Args:
        predicted_points (torch.Tensor): [N, 2] where each point is (y, x)
        target_points (torch.Tensor): [N, 2] where each point is (y, x)
        image_size (tuple): (width, height)
        threshold (float): Threshold for PCK

    Returns:
        int: Number of correct keypoints
    """
    distances = torch.linalg.norm(predicted_points - target_points, axis=-1)
    pck = distances <= threshold * max(image_size)
    return pck.sum().item()


def compute_pck_bbox(predicted_points, target_points, target_bbox, threshold=0.1):
    """
    Compute PCK (Percent of Correct Keypoints) value with respect to bounding box size.

    Args:
        predicted_points (torch.Tensor): [N, 2] where each point is (y, x)
        target_points (torch.Tensor): [N, 2] where each point is (y, x)
        target_bbox (torch.Tensor): [4] with (x, y, w, h)
        threshold (float): Threshold for PCK

    Returns:
        int: Number of correct keypoints
    """
    y, x, h, w = target_bbox
    distances = torch.linalg.norm(predicted_points - target_points, axis=-1)
    pck = distances <= threshold * max(h, w)
    return pck.sum().item()