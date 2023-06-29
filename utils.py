import multiprocessing as mp
from typing import Optional

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


def get_torchviz_loader(
    dataset_cls,
    download_path: str = "./",
    train: bool = True,
    batch_size: int = 64,
    n_workers: Optional[int] = None,
) -> DataLoader:
    dataset = dataset_cls(
        download_path,
        train=train,
        download=True,
        transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
    )
    n_workers = n_workers or mp.cpu_count()
    print(f"Created DataLoader for {dataset_cls.__name__} with {n_workers} workers")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )


def get_MNIST(**kwargs) -> DataLoader:
    return get_torchviz_loader(MNIST, **kwargs)


def get_CIFAR10(**kwargs) -> DataLoader:
    return get_torchviz_loader(CIFAR10, **kwargs)
