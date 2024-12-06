import torch
from torch import nn

from utils.correspondence import compute_correspondence, flatten_features, normalize_features
from utils.distillation import get_distance_mutual_nn, permute_indices, SPAIR_FLIP

class BaseModel(nn.Module):
    """
    Base model class.
    """

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    def forward(self, batch):
        raise NotImplementedError


class CacheModel(BaseModel):
    """
    Model that can be cached. Feature extraction and correspondence computation need to be seperable.
    """

    # I don't know why I need to define this again, but it doesn't work otherwise
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for module in self.children():
            module.to(*args, **kwargs)
        return self
    
    def get_features(self, image, category):
        raise NotImplementedError

    def compute_correspondence(self, batch, return_histograms=False, window_softargmax=False):
        if isinstance(batch['source_points'], list):
            predicted_points = []
            batch_size = len(batch['source_features'])
            for b in range(batch_size):
                predicted_points.append(compute_correspondence(batch['source_features'][b],
                                                            batch['target_features'][b],
                                                            batch['source_points'][b],
                                                            batch['source_size'][b],
                                                            batch['target_size'][b],
                                                            return_histograms,
                                                            False,
                                                            window_softargmax))
        else: # points are tensors
            predicted_points = compute_correspondence(batch['source_features'],
                                                    batch['target_features'],
                                                    batch['source_points'],
                                                    batch['source_size'],
                                                    batch['target_size'])
        return predicted_points

    def get_features_and_compute_correspondence(self, batch, return_histograms=False, window_softargmax=False, pose_align=False):
        """
        Get features and compute correspondence between source and target images.

        Args:
            batch (dict): Batch of data.
            return_histograms (bool): Return histograms.
            window_softargmax (bool): Use window softargmax for correspondence computation.
            pose_align (bool): Use pose alignment (Only for SPair-71k).

        Returns:
            predicted_points (list): Predicted points.
        """

        source_image = batch['source_image']
        target_image = batch['target_image']

        if not pose_align:
            images = torch.cat([source_image, target_image])
            categories = batch['source_category'] + batch['target_category']

            features = self.get_features(images, categories)
            if isinstance(features, list):
                features = features[0]
            batch['source_features'] = features[:len(batch['source_image'])]
            batch['target_features'] = features[len(batch['target_image']):]
            
            return self.compute_correspondence(batch, return_histograms, window_softargmax)

        source_images = []
        source_images.append(source_image)
        source_images.append(source_image.flip(3))
        #source_images.append(source_image.flip(2))
        #source_images.append(source_image.flip(3).flip(2))
        #source_images.append(source_image.transpose(2, 3))
        #source_images.append(source_image.transpose(2, 3).flip(3))
        #source_images.append(source_image.transpose(2, 3).flip(2))
        #source_images.append(source_image.transpose(2, 3).flip(3).flip(2))
        source_image = torch.cat(source_images, dim=0)

        images = torch.cat([source_image, target_image])
        categories = batch['source_category'] * len(source_images) + batch['target_category']

        features = self.get_features(images, categories)
        if isinstance(features, list):
            features = features[0]
        source_features = features[:len(source_image)]
        target_features = features[len(source_image):]

        # calculate distance between source and target features
        source_features_f = normalize_features(flatten_features(source_features)) # [B, HxW, C]
        target_features_f = normalize_features(flatten_features(target_features)) # [B, HxW, C]
        min_dist = float('inf')
        min_dist_idx = 0
        for i in range(len(source_features_f)):
            dist = get_distance_mutual_nn(source_features_f[i], target_features_f[0])
            if dist < min_dist:
                min_dist = dist
                min_dist_idx = i

        batch['source_features'] = source_features[min_dist_idx].unsqueeze(0)
        W, H = batch['source_size'][0]
        source_points = []
        source_points.append([torch.tensor([p[0], p[1]]) for p in batch['source_points'][0]]) # no switch
        source_points.append([torch.tensor([p[0], W-p[1]-1]) for p in batch['source_points'][0]]) # switch
        #source_points.append([torch.tensor([H-p[0]-1, p[1]]) for p in batch['source_points'][0]]) # switch
        #source_points.append([torch.tensor([H-p[0]-1, W-p[1]-1]) for p in batch['source_points'][0]]) # no switch
        #source_points.append([torch.tensor([p[1], p[0]]) for p in batch['source_points'][0]]) # switch
        #source_points.append([torch.tensor([p[1], H-p[0]-1]) for p in batch['source_points'][0]]) # no switch
        #source_points.append([torch.tensor([W-p[1]-1, p[0]]) for p in batch['source_points'][0]]) # no switch
        #source_points.append([torch.tensor([W-p[1]-1, H-p[0]-1]) for p in batch['source_points'][0]]) # switch

        # Permute the indices of the source points of the flipped image
        permute_list = SPAIR_FLIP[batch['source_category'][0]]
        vis = range(0, max([max(x) if isinstance(x, list) else x for x in permute_list]) + 1) # indices of possible points for category
        vis = [i in batch['mutual_visible_idx'][0] for i in vis] # list of mutual visibilities (True/False)
        permutation = permute_indices(permute_list, vis=vis) # permutation of indices
        source_points[1] = [source_points[1][batch['mutual_visible_idx'][0].index(i)] for i in permutation if i in batch['mutual_visible_idx'][0]]
        #source_points[2] = [source_points[2][batch['mutual_visible_idx'][0].index(i)] for i in permutation if i in batch['mutual_visible_idx'][0]]
        #source_points[4] = [source_points[4][batch['mutual_visible_idx'][0].index(i)] for i in permutation if i in batch['mutual_visible_idx'][0]] 
        #source_points[7] = [source_points[7][batch['mutual_visible_idx'][0].index(i)] for i in permutation if i in batch['mutual_visible_idx'][0]]

        batch['source_points'] = [torch.stack(source_points[min_dist_idx]).to(images.device)]
        batch['target_features'] = target_features

        if min_dist_idx >= 4:
            batch['source_size'] = [(H, W)]

        return self.compute_correspondence(batch, return_histograms, window_softargmax)

    def forward(self, image, category=None):
        return self.get_features(image, category)