import math
import re
import os
from os.path import join
import torch
from torch.utils.data import Dataset
import numpy as np
from dipy.io.image import load_nifti_data
from utils.data.data_manipulation import rescale, crop


class MICCAIDataset(Dataset):
    def __init__(self,
                 data_path,
                 normalization=None,
                 filter_list=None,
                 unet_params: dict = None,
                 simplify=False,
                 scaling: dict = None,
                 crop_dist=None,
                 precomputed: dict = None):
        """
        MICCAI Dataset class that can be used with a normal torch.utils.data.Dataloader in experiments.
        Data loading is probably quite inefficient, could be improved in future work

        :param data_path: Path to MICCAI folder,
        i.e. '/storage/group/dataset_mirrors/medimg_public/miccai-brats-20/MICCAI_BraTS2020_TrainingData'
        :param normalization: How to normalize the data:
            - None: Apply no normalization, not recommended
            - range: Scale to range [0,1]
            - instance: Apply usual instance normalization
            - instance_nonzero: Apply instance normalization only to nonzero values
            - full: Normalize by subtracting global mean over all scans in dataset and dividing through global
            mean of variance
            - full_nonzero: Same as 'full' but only for nonzero values
        :param filter_list: Restrict dataset to specific sample ids, useful for train/test split
        :param unet_params: Parameters of the U-Net for cropping distance computation, see get_dimensions method
        :param simplify: Whether to summarize all tumor classes as one class
        :param scaling: Whether to apply scaling to specific samples
        :param crop_dist: Whether to apply a manual cropping distance instead of automatic computation (deprecated)
        :param precomputed: Dictionary with precomputed mean and var for 'full' normalization
        """
        super(MICCAIDataset, self).__init__()
        img_list = os.listdir(data_path)
        r = re.compile("BraTS20(.)*")
        img_list = list(filter(r.match, img_list))
        if filter_list is not None:
            img_list = list(
                filter(
                    (lambda x: int(x[17] + x[18] + x[19]) in filter_list and int(x[17] + x[18] + x[19]) not in [176]),
                    img_list))
        assert len(img_list) > 0, "Something went wrong with dataset path or filenames"
        self.data_dirs = [os.path.join(data_path, img_id) for img_id in img_list]
        self.data_path = data_path
        self.normalization = normalization
        self.unet_params = unet_params
        self.simplify = simplify
        self.scaling = scaling
        self.crop_dist = crop_dist
        if normalization == "instance":
            self.norm = torch.nn.InstanceNorm3d(4, affine=False, track_running_stats=False)
        elif normalization == "full":
            if precomputed is not None:
                self.mean = precomputed.get("mean")
                self.var = precomputed.get("var")
            else:
                self.mean, self.var = compute_overall_mean_and_variance(self.data_dirs)
        elif normalization == "full_nonzero":
            if precomputed is not None:
                self.mean = precomputed.get("mean")
                self.var = precomputed.get("var")
            else:
                self.mean, self.var = compute_overall_mean_and_variance(self.data_dirs, nonzero=True)

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.tolist()
        if isinstance(idx, int):
            sample_data, sample_seg, sample_id = get_data_point(self.data_dirs[idx], self.unet_params, self.simplify,
                                                                self.scaling, self.crop_dist)
            sample_data = torch.tensor(sample_data).float()
            sample_seg = torch.tensor(sample_seg.astype(int))
        else:
            assert self.scaling is None, "Samples with different spatial size in one batch is not a good idea"
            sample_data = []
            sample_seg = []
            sample_id = []
            for i in idx:
                _data, _seg, _id = get_data_point(self.data_dirs[i], self.unet_params, self.simplify, self.scaling,
                                                  self.crop_dist)
                _data = torch.tensor(_data)
                _seg = torch.tensor(_seg)
                sample_data.append(_data)
                sample_seg.append(_seg)
                sample_id.append(_id)
            sample_data = torch.stack(sample_data).float()
            sample_seg = torch.stack(sample_seg).float()

        if self.normalization == "instance":
            """
            Currently uses instance normalization
            """
            single_sample = sample_data.ndim == 4
            if single_sample:
                sample_data = sample_data[None]
            with torch.no_grad():
                sample_data = self.norm(sample_data)
            if single_sample:
                sample_data = sample_data[0]

        elif self.normalization == "range":
            sample_data = sample_data / sample_data.max()

        elif self.normalization == "instance_nonzero":
            single_sample = sample_data.ndim == 4
            if single_sample:
                sample_data = sample_data[None]
            sample_data = sample_data.numpy()
            mask = sample_data > 0
            mean = np.mean(sample_data, axis=(2, 3, 4), where=mask)
            var = np.var(sample_data, axis=(2, 3, 4), where=mask)
            sample_data = np.divide(np.subtract(sample_data, mean[:, :, None, None, None], where=mask),
                                    var[:, :, None, None, None], where=mask)
            if single_sample:
                sample_data = sample_data[0]
            sample_data = torch.tensor(sample_data)

        elif self.normalization == "full":
            single_sample = sample_data.ndim == 4
            if single_sample:
                sample_data = sample_data[None]
            sample_data = sample_data.numpy()
            mean = self.mean[None]
            var = self.var[None]
            sample_data = np.divide(np.subtract(sample_data, mean[:, :, None, None, None]),
                                    var[:, :, None, None, None])
            if single_sample:
                sample_data = sample_data[0]
            sample_data = torch.tensor(sample_data).float()

        elif self.normalization == "full_nonzero":
            single_sample = sample_data.ndim == 4
            if single_sample:
                sample_data = sample_data[None]
            sample_data = sample_data.numpy()
            mask = sample_data > 0
            mean = self.mean[None]
            var = self.var[None]
            sample_data = np.divide(np.subtract(sample_data, mean[:, :, None, None, None], where=mask),
                                    var[:, :, None, None, None], where=mask)
            if single_sample:
                sample_data = sample_data[0]
            sample_data = torch.tensor(sample_data).float()

        return {"data": sample_data, "target": sample_seg, "id": sample_id}

    def get_parameters(self):
        return {
            "data_path": self.data_path,
            "normalization": self.normalization,
            "unet_params": self.unet_params,
            "simplify": self.simplify,
            "scaling": self.scaling,
            "size": len(self),
            "crop_dist": self.crop_dist
        }


def get_data_point(
        path,
        unet_params: dict = None,
        simplify=False,
        scaling: dict = None,
        crop_dist=None
):
    """
    Load sample from path, possibly apply scaling and possibly crop to be usable as input for U-Net

    :param path: Path to sample
    :param unet_params: Parameters of U-Net if needed for cropping
    :param simplify: Whether to simplify segmentation
    :param scaling: Scaling parameters if wanted
    :param crop_dist: Manual crop-distance if wanted
    :return: data, segmentation, title
    """
    file_list = os.listdir(path)
    assert len(file_list) > 0
    stem = file_list[0][:20]
    number = int(stem[-3:])
    t1 = join(path, stem) + '_t1.nii.gz'
    t1_data = load_nifti_data(t1)
    t1ce = join(path, stem) + '_t1ce.nii.gz'
    t1ce_data = load_nifti_data(t1ce)
    t2 = join(path, stem) + '_t2.nii.gz'
    t2_data = load_nifti_data(t2)
    flair = join(path, stem) + '_flair.nii.gz'
    flair_data = load_nifti_data(flair)
    seg = join(path, stem) + '_seg.nii.gz'
    seg = load_nifti_data(seg).astype(float)
    data = np.array([t1_data, t1ce_data, t2_data, flair_data]).astype(float)
    if simplify:
        seg = (seg >= 1).astype(float)[None]
    else:
        seg_full = (seg >= 1).astype(float)
        seg_core = (seg >= 2).astype(float)
        seg_enhancing = (seg == 4).astype(float)
        seg = np.array([seg_full, seg_core, seg_enhancing])
    if scaling is not None:
        scaling_n_th_sample = scaling.get("n", 3)
        #if number % scaling_n_th_sample == 0:
        if np.random.rand() < 1 / scaling_n_th_sample:
            random_scaling = scaling.get("random", False)
            if random_scaling:
                scaling_range = scaling.get("range", (0.5, 0.9))
                scaling_scale = np.random.uniform(*scaling_range)
            else:
                scaling_scale = scaling.get("scale", 0.8)
            data = torch.tensor(data)
            data = rescale(data, scaling_scale).numpy()
            seg = torch.tensor(seg)
            seg = rescale(seg, scaling_scale, "nearest").numpy().astype(int)
    if unet_params is not None:
        sizes = []
        for d in np.shape(seg)[1:]:
            sizes.append(d - get_dimensions(d, **unet_params))
        crop_dist = []
        for s in sizes:
            crop_dist.append(math.floor(s / 2))
            crop_dist.append(math.ceil(s / 2))
    else:
        crop_dist = crop_dist
    if crop_dist is not None:
        return crop(data, crop_dist=crop_dist), crop(seg, crop_dist=crop_dist), f"scan{number}"
    else:
        return data, seg, f"scan{number}"


def get_dimensions(
        x,
        kernel_size,
        strides,
        padding=None,
        debug=False,
        monai=False
):
    """
    Compute largest size of sample that can be used with the given U-Net configuration
    """
    if padding is None:
        padding = [0] * len(strides)
    elif isinstance(padding, int):
        padding = [padding] * len(strides)
    else:
        assert len(padding) == len(strides)
    if not monai:
        orig_x = x
        while orig_x >= kernel_size:
            x = orig_x
            x_ = orig_x
            debug_string = "Feature Map Sizes: "
            for i in range(len(strides)):
                debug_string += f"{x}->"
                first = (x - kernel_size + 2 * padding[i])
                if first % strides[i] != 0:
                    orig_x -= 1
                    break
                x = (x - kernel_size) // strides[i] + 1
            if x_ == orig_x:
                if debug:
                    print(debug_string + f"{x}")
                break
        return orig_x
    else:
        if len(strides) == 1:
            return x
        else:
            if x % 2 == 1:
                x -= 1
            while x >= kernel_size:
                if math.floor((x + strides[0] - 1) / strides[0]) % math.prod(strides[1:]) == 0:
                    return x
                else:
                    x -= 2


def compute_overall_mean_and_variance(path_list, nonzero=False):
    """
    Compute overall mean and variance of a data set at given path

    :param path_list: Paths to files
    :param nonzero: Whether to only look at nonzero values in samples
    :return: Mean values of mean and variance
    """
    mean = 0.0
    var = 0.0
    num_samples = 0
    for path in path_list:
        data, _, _ = get_data_point(path)
        num_samples += 1
        if nonzero:
            mask = data > 0
            mean += np.mean(data, axis=(1, 2, 3), where=mask)
            var += np.var(data, axis=(1, 2, 3), where=mask)
        else:
            mean += np.mean(data, axis=(1, 2, 3))
            var += np.var(data, axis=(1, 2, 3))
    return mean / num_samples, var / num_samples
