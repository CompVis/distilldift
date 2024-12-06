import os
import torch
import numpy as np
import torchvision
import webdataset as wds
from einops import rearrange

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


class WebDataModule():
    def __init__(
        self,
        tar_base,
        batch_size,
        train=None,
        validation=None,
        test=None,
        num_workers=4,
        multinode=True,
        min_size=None,
        max_pwatermark=1.0,
        channel_last=False,
        val_batch_size=None,
        val_num_workers=None,
        **kwargs,
    ):
        super().__init__()
        print(f"Setting tar base to {tar_base}")
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode
        self.min_size = min_size  # filter out very small images
        self.max_pwatermark = max_pwatermark  # filter out watermarked images
        self.channel_last = channel_last
        self.val_batch_size = (
            val_batch_size if val_batch_size is not None else batch_size
        )
        self.val_num_workers = (
            val_num_workers if val_num_workers is not None else num_workers
        )

    def make_loader(self, dataset_config, train=True):
        # change range from [0,1] to [-1,1] and put channel last or first
        image_transforms = []
        if self.channel_last:
            lambda_fn = lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")
        else:
            lambda_fn = lambda x: x * 2.0 - 1.0
        image_transforms.extend(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float32),
                torchvision.transforms.Lambda(lambda_fn),
            ]
        )
        if "image_transforms" in dataset_config:
            image_transforms.extend(dataset_config["image_transforms"])
        image_transforms = torchvision.transforms.Compose(image_transforms)

        # this is crucial to set correct image key to get the transofrms applied correctly
        img_key = dataset_config.get("image_key", "image.png")
        transform_dict = {}
        transform_dict.update({img_key: image_transforms})

        if "postprocess" in dataset_config:
            postprocess = dataset_config["postprocess"]
        else:
            postprocess = None

        # some illustration how shuffle works in webdataset
        # https://github.com/webdataset/webdataset/issues/71
        # TL;DR: set the shuffle as big as you can afford ->len(files)
        shuffle = dataset_config.get("shuffle", 0)
        shardshuffle = shuffle > 0

        nodesplitter = (
            wds.shardlists.split_by_node
            if self.multinode
            else wds.shardlists.single_node_only
        )

        tars = os.path.join(self.tar_base, dataset_config.get("shards", ""))

        dset = (
            wds.WebDataset(
                tars,
                nodesplitter=nodesplitter,
                shardshuffle=shardshuffle,
                handler=wds.warn_and_continue,
            )
            .repeat()
            .shuffle(shuffle)
        )
        print(f"Loading webdataset with {len(dset.pipeline[0].urls)} shards.")

        dset = (
            dset
            # .select(self.filter_keys)
            .decode("rgb", handler=wds.warn_and_continue).map_dict(
                **transform_dict, handler=wds.warn_and_continue
            )
        )

        # change name of image key to be consistent with other datasets
        renaming = dataset_config.get("rename", None)
        if renaming is not None:
            dset = dset.rename(**renaming)

        if postprocess is not None:
            dset = dset.map(postprocess)

        bs = self.batch_size if train else self.val_batch_size
        nw = self.num_workers if train else self.val_num_workers
        dset = dset.batched(bs, partial=False, collation_fn=dict_collation_fn)
        loader = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=nw)

        return loader

    def filter_size(self, x):
        try:
            valid = True
            if self.min_size is not None and self.min_size > 1:
                try:
                    valid = (
                        valid
                        and x["json"]["original_width"] >= self.min_size
                        and x["json"]["original_height"] >= self.min_size
                    )
                except Exception:
                    valid = False
            if self.max_pwatermark is not None and self.max_pwatermark < 1.0:
                try:
                    valid = valid and x["json"]["pwatermark"] <= self.max_pwatermark
                except Exception:
                    valid = False
            return valid
        except Exception:
            return False

    def filter_keys(self, x):
        try:
            return ("jpg" in x) and ("txt" in x)
        except Exception:
            return False

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)