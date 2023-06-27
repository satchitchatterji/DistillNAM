from typing import Optional, Tuple

import matplotlib
import numpy as np
from torchvision.datasets import MNIST

# NOTE comment this line if it raises errors
matplotlib.use("TkAgg")

from matplotlib import pyplot as plt


def plot_MNIST_sample(
    mnist: MNIST, rng: Optional[np.random.Generator] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Simply plot 25 samples from a MNIST dataset split in a 5x5 grid."""
    fig, axes = plt.subplots(5, 5, figsize=(10, 8))
    if rng is None:
        rng = np.random.default_rng()
    ids = rng.integers(0, mnist.data.size(0), 25)
    imgs, labels = mnist.data[ids], mnist.targets[ids]
    for img, label, ax in zip(imgs, labels, axes.flat):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(label.item())
        ax.set_axis_off()
    return fig, axes
