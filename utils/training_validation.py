"""
Script for evaluation and visualization of trained models
"""
import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from utils.loss import get_tp_fp_fn_tn
import yaml


def plot_results(scans, targets, predictions_equi, predictions_non_equi):
    """
    Plot the results of the segmentations: Target, Baseline and U-Net

    :param scans: Data
    :param targets: Target Segmentations
    :param predictions_equi: Segmentations of scale-equivariant U-Net
    :param predictions_non_equi: Segmentations of baseline method
    :return: Figure with plotted segmentations
    """
    rows = ['Ground truth', 'Scale-Equivariant', 'Baseline']

    ncols = len(scans)
    fig_size_x = ncols * 3
    figsize = (fig_size_x, 3 * 4)
    fig, axes = plt.subplots(nrows=3, ncols=ncols, figsize=figsize)

    """
    pad = 5  # in points
    
    if ncols == 1:
        for ax, row in zip(axes, rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')
    else:
        for ax, row in zip(axes[:, 0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')
    """
    if ncols == 1:
        scan, target, pred_equi, pred_non = scans[1], targets[0], predictions_equi[0], predictions_non_equi[0]
        index = slice_index_with_most_difference(target, pred_equi[0, 0], pred_non[0, 0])

        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')

        plot_slice_img(axes[0], scan[index].numpy(), target[index].numpy())
        plot_slice_img(axes[1], scan[index].numpy(), pred_equi[0, 0, index].numpy())
        plot_slice_img(axes[2], scan[index].numpy(), pred_non[0, 0, index].numpy())
    else:
        for i, (scan, target, pred_equi, pred_non) in enumerate(
                zip(scans, targets, predictions_equi, predictions_non_equi)):
            index = slice_index_with_most_difference(target, pred_equi[0, 0], pred_non[0, 0])

            axes[0, i].axis('off')
            axes[1, i].axis('off')
            axes[2, i].axis('off')

            plot_slice_img(axes[0, i], scan[..., index].numpy(), target[..., index].numpy(),
                           file_name=f"brain_scans/gt_{i}.png")
            plot_slice_img_true_false(axes[1, i], scan[..., index].numpy(), pred_equi[0, 0, ..., index].numpy(),
                                      target[..., index].numpy(), file_name=f"brain_scans/equi_{i}.png")
            plot_slice_img_true_false(axes[2, i], scan[..., index].numpy(), pred_non[0, 0, ..., index].numpy(),
                                      target[..., index].numpy(), file_name=f"brain_scans/base_{i}.png")

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. We'll need
    # to make some room. These numbers are manually tweaked.
    # You could automatically calculate them, but it's a pain.
    # fig.subplots_adjust(left=0.15, top=0.95)
    return fig


def plot_slice_img(ax, scan: np.ndarray, segmentation: np.ndarray, file_name=None):
    # scans (and segmentations) are defined in RAS orientation,
    # where R -> x-axis, A -> y-axis, S -> z-axis and the axis are given in the orders zyx).
    # The z-axis is already selected by the slice, so yx are left
    # we want left to be shown right and front on the top so we need to flip both axes
    scan = np.rot90(np.flip(scan, (0, 1)), 1)
    segmentation = np.rot90(np.flip(segmentation, (0, 1)), 1)

    ax.imshow(scan, cmap='gray', alpha=1.0, interpolation='none')

    cmap = plt.cm.spring
    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:128, -1] = np.linspace(0, 1, 128)
    alpha_cmap[0, -1] = 0.
    alpha_cmap = ListedColormap(alpha_cmap)
    ax.imshow(segmentation, cmap=alpha_cmap, alpha=.7, interpolation='none')

    if file_name is not None:
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(scan, cmap='gray', alpha=1.0, interpolation='none')
        plt.imshow(segmentation, cmap=alpha_cmap, alpha=.7, interpolation='none')
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=300)


def plot_slice_img_true_false(ax, scan, segmentation, target, file_name=None):
    # show true positives in green, false positives and false negatives in red
    scan = np.rot90(np.flip(scan, (0, 1)), 1)
    segmentation = np.rot90(np.flip(segmentation, (0, 1)), 1)
    target = np.rot90(np.flip(target, (0, 1)), 1)

    ax.imshow(scan, cmap='gray', alpha=1.0, interpolation='none')

    def get_mask(pred, target):
        tp = np.zeros_like(pred)
        fp = np.zeros_like(pred)
        fn = np.zeros_like(pred)

        tp[pred > 0.5] = target[pred > 0.5]
        fp[pred > 0.5] = 1 - target[pred > 0.5]
        fn[pred < 0.5] = target[pred < 0.5]
        return tp, fp, fn

    tp, fp, fn = get_mask(segmentation, target)

    cmap = plt.cm.get_cmap("viridis")
    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:128, -1] = np.linspace(0, 1, 128)
    alpha_cmap[0, -1] = 0.
    alpha_cmap = ListedColormap(alpha_cmap)
    ax.imshow(tp, cmap=alpha_cmap, alpha=.7, interpolation='none')

    # color map with pure red for all values with 1.0 and
    fn_cmap = ListedColormap(np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 1]
    ]))
    ax.imshow(fn, cmap=fn_cmap, alpha=.7, interpolation='none')

    # color map with pure blue for all values with 1.0 and 0.0 alpha for all values with 0.0
    fp_cmap = ListedColormap(np.array([
        [0, 0, 1, 0],
        [0, 0, 1, 1]
    ]))
    ax.imshow(fp, cmap=fp_cmap, alpha=.7, interpolation='none')

    if file_name is not None:
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(scan, cmap='gray', alpha=1.0, interpolation='none')
        plt.imshow(tp, cmap=alpha_cmap, alpha=.7, interpolation='none')
        plt.imshow(fn, cmap=fn_cmap, alpha=.7, interpolation='none')
        plt.imshow(fp, cmap=fp_cmap, alpha=.7, interpolation='none')
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=300)


def slice_index_with_most_lesions(target_lesions: torch.Tensor):
    lesion_count = target_lesions.sum(dim=(1, 2))  # Dim (Z)
    return int(torch.argmax(lesion_count))


def slice_index_with_most_difference(target_lestions, ours_pred, baseline_pred):
    # Select slices where our method performs better than the baseline (i.e. more lesions are detected)
    # First compute the number of lesions detected by both methods
    ours_detected = target_lestions * ours_pred
    baseline_detected = target_lestions * baseline_pred
    ours_lesion_count = ours_detected.sum(dim=(0, 1))  # Dim (Z)
    baseline_lesion_count = baseline_detected.sum(dim=(0, 1))  # Dim (Z)
    # Then compute the difference
    difference = ours_lesion_count - baseline_lesion_count
    return int(torch.argmax(difference))


def compute_scores(dataset, method, file_title, methodeval=False):
    """
    Compute validation scores of a method

    :param dataset: Dataset that should be used
    :param method: Method that should be tested
    :param file_title: Title of results file where results are stored
    :param methodeval: Whether to call method.eval() before tests or not
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    method.to(device)

    dice_list = []
    precision_list = []
    recall_list = []
    balanced_acc_list = []
    if methodeval:
        method.eval()
    count = 0
    for sample in dataset:
        data = sample["data"][None].to(device=device)
        target = sample["target"][None].to(device=device)
        with torch.no_grad():
            prediction = method(data)
            prediction = torch.sigmoid(prediction)
            tp, fp, fn, tn = get_tp_fp_fn_tn(prediction, target)
        dice_list.append(((2 * tp) / (2 * tp + fp + fn + 1e-8)).mean().item())
        precision_list.append((tp / (tp + fp)).mean().item())
        recall_list.append((tp / (tp + fn)).mean().item())
        balanced_acc_list.append(((tp / (tp + fn) + tn / (tn + fp)) / 2).mean().item())
        print(count)
        count += 1

    data = {
        "dice": dice_list,
        "precision": precision_list,
        "recall": recall_list,
        "bal_accuracy": balanced_acc_list
    }
    with open(file_title + ".yml", "w") as outfile:
        yaml.dump(data, outfile)
    print(file_title + ".yml")


from utils.data.dataset import MICCAIDataset
from models.unet.unet import UNet
from models.baseline.baseline import BaseLineModel
import torch

if __name__ == "__main__":
    print("Starting evaluation")
    data_path = "{DATA_PATH/MICCAI_BraTS2020_TrainingData}"
    unet_params = {
        "kernel_size": 5,
        "strides": [2, 2, 2]
    }

    b_unet_params = {
        "kernel_size": 5,
        "strides": [2, 2, 2],
        "monai": True
    }

    scaling = {
        "n": 2,
        "random": False,
        "scale": 0.9,
        "range": (0.7, 0.9)
    }

    # np.random.seed(420)
    use_scaling = False

    sample_list = list(range(251, 370, 1))
    u_test_dataset = MICCAIDataset(data_path, filter_list=sample_list, unet_params=unet_params,
                                   simplify=True, normalization="instance", scaling=scaling if use_scaling else None)
    b_test_dataset = MICCAIDataset(data_path, filter_list=sample_list, unet_params=b_unet_params,
                                   simplify=True, normalization="instance", scaling=scaling if use_scaling else None)

    print("Datasets loaded")

    u_checkpoint_path = "{Checkpoint path of the best scale-equivariant model}", "{Title}"
    u_model = UNet.load_from_checkpoint(u_checkpoint_path[0])
    compute_scores(u_test_dataset, u_model, u_checkpoint_path[1])

    b_checkpoint_path = "{Checkpoint path of the best baseline model}", "{Title}"
    b_model = BaseLineModel.load_from_checkpoint(b_checkpoint_path[0])
    compute_scores(b_test_dataset, b_model, b_checkpoint_path[1])

    # load a few samples from the test dataset, pass them through the models and plot the results
    scans = []
    targets = []
    predictions_equi = []
    predictions_non_equi = []

    u_model = UNet.load_from_checkpoint(u_checkpoint_path[0])
    b_model = BaseLineModel.load_from_checkpoint(b_checkpoint_path[0])

    import torch.nn.functional as F

    print("Starting inference...")

    for i in range(0, 20):
        scans.append(u_test_dataset[i]["data"][0])
        targets.append(u_test_dataset[i]["target"][0])
        with torch.no_grad():
            predictions_equi.append(torch.sigmoid(u_model(u_test_dataset[i]["data"][None])))
            prediction_non_equi = torch.sigmoid(b_model(b_test_dataset[i]["data"][None]))
        predictions_non_equi.append(
            F.interpolate(prediction_non_equi, size=predictions_equi[-1].shape[2:], mode="trilinear"))

    print("Done with inference, plotting results...")

    fig = plot_results(scans=scans, targets=targets, predictions_equi=predictions_equi,
                       predictions_non_equi=predictions_non_equi)

    fig.savefig(f"results/test.png", dpi=300)
