import os
from typing import Optional

import torch
from pl_bolts.datamodules import MNISTDataModule


def get_MNIST(
    download_dir: Optional[str] = "./",
    normalize: bool = True,
    num_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    num_workers = num_workers or int(os.cpu_count() / 2)
    batch_size = 256 if torch.cuda.is_available() else 64
    print(
        f"Datasets root: {download_dir} batch size: {batch_size} n_workers: {num_workers}"
    )
    return MNISTDataModule(
        data_dir=download_dir,
        normalize=normalize,
        num_workers=num_workers,
        batch_size=batch_size,
    )


# from torchvision.datasets import CIFAR10, MNIST
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, Normalize, ToTensor

# def get_torchviz_loader(
#     dataset_cls,
#     download_path: str = "./",
#     train: bool = True,
#     batch_size: int = 64,
#     n_workers: Optional[int] = None,
# ) -> DataLoader:
#     dataset = dataset_cls(
#         download_path,
#         train=train,
#         download=True,
#         transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
#     )
#     n_workers = n_workers or int(os.cpu_count() / 2)
#     print(f"Created DataLoader for {dataset_cls.__name__} with {n_workers} workers")
#     return DataLoader(
#         dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers
#     )


# def get_MNIST(**kwargs) -> DataLoader:
#     return get_torchviz_loader(MNIST, **kwargs)


# def get_CIFAR10(**kwargs) -> DataLoader:
#     return get_torchviz_loader(CIFAR10, **kwargs)
