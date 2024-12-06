import torch
import torch.nn.functional as F

from utils.correspondence import points_to_idxs, idxs_to_points, rescale_points

def should_save(epoch, i, dataloader_length, checkpoint_percent):
    if checkpoint_percent < 1:  # Save at a fraction of an epoch
        steps_per_checkpoint = dataloader_length * checkpoint_percent
        return (i + 1) % int(steps_per_checkpoint) == 0
    else:  # Save every `checkpoint_percent` epochs
        return ((epoch + 1) % int(checkpoint_percent) == 0) and (i + 1 == dataloader_length)

def softmax_with_temperature(input, temperature=1.0):
    """
    Apply the softmax function with temperature on the input tensor.

    Parameters:
        input (torch.Tensor): The input tensor for which the softmax is to be computed.
        temperature (float, optional): The temperature parameter to adjust the smoothness of the output. Default is 1.0.
    
    Returns:
        torch.Tensor: The softmax probabilities with applied temperature.
    """
    scaled_input = input / temperature # Adjust the input based on the temperature
    return F.softmax(scaled_input, dim=-1) # Compute softmax on the scaled input

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

def softargmax1d(input, beta=100):
    """
    Apply the soft-argmax function on the input tensor.

    Parameters:
        input (torch.Tensor): The input tensor for which the soft-argmax is to be computed.
        beta (float, optional): The beta parameter to adjust the smoothness of the output. Default is 100.
    """
    *_, n = input.shape
    input = F.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n, device=input.device)
    return torch.sum((n - 1) * input * indices, dim=-1)

def separate_foreground(features):
    """
    Separate the foreground from the background in the input features.

    Parameters:
        features (torch.Tensor): The input features to be separated. [B, HW, C]

    Returns:
        torch.Tensor: The foreground masks for each image in the input features.
    """
    B, HW, C = features.shape
    foreground_masks = torch.zeros_like(features[:, :, 0], dtype=torch.bool, device=features.device)
    
    for b in range(B):
        flattened_features = features[b].view(-1, C)  # Shape: [HW, C]
        flattened_features = flattened_features - flattened_features.mean(dim=0, keepdim=True) # Center the data
        U, S, V = torch.pca_lowrank(flattened_features, q=1)
        scores = U[:, 0]  # First principal component scores for batch b
        threshold = scores.mean().item()  # Calculate mean for current batch's scores
        foreground_masks[b] = scores > threshold  # Update mask for batch b
    
    return ~foreground_masks

def separate_foreground_copca(source_features, target_features):
    """
    Separate the foreground from the background in the input features using Co-PCA.

    Parameters:
        source_features (torch.Tensor): The source features to be separated. [B, HW, C]
        target_features (torch.Tensor): The target features to be separated. [B, HW, C]

    Returns:
        torch.Tensor: The foreground masks for the source features.
        torch.Tensor: The foreground masks for the target features.
    """
    B, HW, C = source_features.shape
    source_masks = torch.zeros_like(source_features[:, :, 0], dtype=torch.bool, device=source_features.device)
    target_masks = torch.zeros_like(target_features[:, :, 0], dtype=torch.bool, device=target_features.device)

    for b in range(B):
        flattened_source_features = source_features[b].view(-1, C)  # Shape: [B*HW, C]
        flattened_target_features = target_features[b].view(-1, C)  # Shape: [B*HW, C]
        concatenated_features = torch.cat((flattened_source_features, flattened_target_features), dim=0)  # Shape: [2*HW, C]
        concatenated_features = concatenated_features - concatenated_features.mean(dim=0, keepdim=True) # Center the data
        U, S, V = torch.pca_lowrank(concatenated_features, q=1)
        scores = U[:, 0]  # First principal component scores for batch b
        threshold = scores.mean().item()  # Calculate mean for current batch's scores
        source_masks[b] = scores[:HW] > threshold  # Update mask for batch b
        target_masks[b] = scores[HW:] > threshold  # Update mask for batch b

    return ~source_masks, ~target_masks

def sample_points(features, feature_size, sampling_method=None, ground_truth_points=None, image_size=None, N=10):
    """
    Sample points from the features.

    Parameters:
        features (torch.Tensor): The source features of the student model.
        sampling_method (str, optional): The method to be used for sampling. Default is None.
        ground_truth_points (torch.Tensor, optional): The ground truth points to be used for sampling. Default is None.
        image_size (tuple): The size of the image, used for rescaling the ground truth points. Default is None.
        N (int, optional): The number of points to be sampled. Default is 10.

    Returns:
        torch.Tensor: The sampled points from the source features of the student model.
    """
    B, C, H, W = feature_size
    
    if sampling_method == 'ground_truth': # only for B=1
        points = rescale_points(ground_truth_points, image_size, (H, W)) # [B, N, 2]
        idxs = points_to_idxs(points, (H, W)) # [B, N]
    elif sampling_method == 'foreground': # only for B=1
        mask = separate_foreground(features).long() # [B, HxW]
        idxs = mask.nonzero(as_tuple=False)[:, 1].unsqueeze(0) # [1, N]
        points = idxs_to_points(idxs, (H, W)) # [B, N, 2]
    elif sampling_method == 'random_foreground':
        # Select N random non-zero indices from mask
        mask = separate_foreground(features).long() # [B, HxW]
        mask = mask.reshape(B, H, W) # [B, H, W]
        points = torch.zeros(B, N, 2, dtype=torch.long) # [B, N, 2]
        for b in range(B):
            points = 0
            while points < N:
                y, x = torch.randint(0, H, (1,)), torch.randint(0, W, (1,))
                if mask[b, y, x] == 1:
                    points[b, points, 0] = y
                    points[b, points, 1] = x
                    points += 1
        idxs = points_to_idxs(points, (H, W)) # [B, N]
    elif sampling_method == 'grid':
        # Select N points on a regular grid
        y = torch.linspace(0, H - 1, N, dtype=torch.long)
        x = torch.linspace(0, W - 1, N, dtype=torch.long)
        points = torch.stack(torch.meshgrid(y, x), dim=-1)
        idxs = points_to_idxs(points, (H, W))
    elif sampling_method == 'random':
        # Select N random indices
        idxs = torch.randint(0, H * W, (B, N), dtype=torch.long)
        points = idxs_to_points(idxs, (H, W))

    return idxs, points

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits, targets):
        # Source Image -> Target Image
        ce = self.cross_entropy(logits, targets)

        # Target Image -> Source Image
        rce = self.cross_entropy(logits.transpose(-1, -2), targets.transpose(-1, -2))

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss
    
class CLIPLoss(torch.nn.Module):
    def __init__(self, logit_scale=1.0):
        super(CLIPLoss, self).__init__()
        self.logit_scale = logit_scale

    def forward(self, source_features, target_features):
        source_logits = self.logit_scale * source_features @ target_features.transpose(-1, -2) # [B, N, N]
        target_logits = self.logit_scale * target_features @ source_features.transpose(-1, -2) # [B, N, N]
        labels = torch.arange(source_logits.shape[1], device=source_features.device, dtype=torch.long).unsqueeze(0) # [1, N]
        loss = F.cross_entropy(source_logits, labels) + F.cross_entropy(target_logits, labels)
        return loss / 2

# https://github.com/Junyi42/GeoAware-SC
def get_distance_mutual_nn(feature1, feature2):
    """
    Calculate the distance between the mutual nearest neighbors of two features.

    Parameters:
        feature1 (torch.Tensor): The first feature tensor.
        feature2 (torch.Tensor): The second feature tensor.

    Returns:
        float: The average distance between the mutual nearest neighbors.
    """

    distances_1to2 = torch.cdist(feature1, feature2)

    nearest_patch_indices_1to2 = torch.argmin(distances_1to2, dim=1)
    nearest_patch_indices_2to1 = torch.argmin(distances_1to2, dim=0)

    # get the mutual nearest neighbors
    mutual_nn_1to2 = torch.zeros_like(nearest_patch_indices_1to2)
    mutual_nn_2to1 = torch.zeros_like(nearest_patch_indices_2to1)
    for i in range(len(nearest_patch_indices_1to2)):
        if nearest_patch_indices_2to1[nearest_patch_indices_1to2[i]] == i:
            mutual_nn_1to2[i] = 1
            mutual_nn_2to1[nearest_patch_indices_1to2[i]] = 1

    # get the average distance of the mutual nearest neighbors
    avg_distance_1to2 = torch.min(distances_1to2, dim=1)[0][mutual_nn_1to2==1].mean()
    return avg_distance_1to2

SPAIR_FLIP = {
    'aeroplane': [0,1,2,3,[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19],[20,21],22,23,24],
    'bicycle': [0,1,[2,3],4,5,[6,7],8,[9,10],11,12,13], 
    'bird': [0,[1,2],3,[4,5],6,[7,8],9,[10,11],[12,13],[14,15],16],
    'boat': [0,[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],13],
    'bottle': [[0,1],[2,3],[4,5],[6,7],[8,9]],
    'bus': [[0,1],[2,3],[5,6],4,7,[8,18],[11,21],[9,19],[12,22],[10,20],[13,23],[14,15],[24,25],[16,17],[26,27], 28, 29, 30], 
    'car':[[0,1],[2,3],4,5,[6,7],8,9,[10,20],[13,23],[11,21],[14,24],[12,22],[15,25],[16,17],[26,27],[18,19],[28,29]],
    'cat':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14],
    'chair':[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]],
    'cow':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14,[15,16],[17,18],[19,20]],
    'dog':[[0,1],[2,3],[4,5],6,7,8,[9,10],[11,12],13,14,15],
    'horse':[[0,1],[2,3],[4,5],[6,7],8,9,[10,11],[12,13],14,15,[16,17],[18,19]],
    'motorbike':[[0,1],[2,3],4,5,6,7,8,9,10,11,12],
    'person':[[0,1],[2,3],4,5,6,7,[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]],
    'pottedplant':[[0,2],1,3,[4,5],[6,8],7],
    'sheep':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14,[15,16],[17,18],[19,20]],
    'train':[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17]],
    'tvmonitor':[[0,2],[4,6],1,5,[3,7],[8,10],[12,14],9,13,[11,15]]
}

def permute_indices(flip_list, vis=None):
    # Flatten the list to find the max index
    flat_list = [item for sublist in flip_list for item in (sublist if isinstance(sublist, list) else [sublist])]
    max_idx = max(flat_list)

    # Create a list of indices from 0 to max index
    indices = list(range(max_idx + 1))

    # Permute the indices where necessary
    for item in flip_list:
        if isinstance(item, list):
            # if all elem in item is True in vis, then flip
            if vis is None or all(vis[i] for i in item):
                for i in range(len(item)):
                    indices[item[i]] = item[(i + 1) % len(item)]

    return indices
