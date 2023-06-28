import multiprocessing as mp
from typing import Optional

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


def get_MNIST(
    train: bool = True,
    download_path: str = "./",
    batch_size: int = 64,
    n_workers: Optional[int] = None,
) -> DataLoader:
    mnist_dataset = MNIST(
        download_path,
        train=train,
        download=True,
        transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
    )
    return DataLoader(
        mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers or mp.cpu_count(),
    )
