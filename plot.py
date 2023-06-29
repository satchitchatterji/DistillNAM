from typing import Optional, Tuple

import matplotlib
import numpy as np

# NOTE comment this line if it raises errors
matplotlib.use("TkAgg")

from matplotlib import pyplot as plt


def plot_dataset_sample(
    torchviz_dataset, rng: Optional[np.random.Generator] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Simply plot 25 samples from a torchvision dataset split in a 5x5 grid.

    NOTE No regards for class representation here ;)
    """
    fig, axes = plt.subplots(5, 5, figsize=(10, 8))
    if rng is None:
        rng = np.random.default_rng()
    ids = rng.integers(0, torchviz_dataset.data.shape[0], 25)
    # some targets in torchvision.datasets are lists, e.g. CIFAR10
    imgs, labels = torchviz_dataset.data[ids], np.array(torchviz_dataset.targets)[ids]
    human_class_map = {v: k for k, v in torchviz_dataset.class_to_idx.items()}
    show_args, imgs_plot = {}, imgs
    if len(imgs.shape) < 4 or imgs.shape[3] == 1:
        # either no RGB channel, or squeeze out (possible) placeholder channel
        imgs_plot = imgs.squeeze()
        show_args = {"cmap": "gray"}
    for img, label, ax in zip(imgs_plot, labels, axes.flat):
        ax.imshow(img, **show_args)
        ax.set_title(f"{label.item()}: {human_class_map[label.item()]}")
        ax.set_axis_off()
    datapoint_shape = "x".join(map(str, imgs[0].shape))
    fig.suptitle(f"Normalized {type(torchviz_dataset).__name__} ({datapoint_shape})")
    return fig, axes
