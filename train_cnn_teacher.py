import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import models
import utils as u


class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# TODO some early termination callback to get a good teacher
# TODO save trained weights
def train(
    args,
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model: nn.Module, device: torch.device, test_loader: DataLoader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

if __name__ == "__main__":
    # simulate argparse
    args = DummyArgs(log_interval=10, epochs=15, data_root="~/datasets")

    train_loader = u.get_MNIST(download_path=args.data_root)
    test_loader = u.get_MNIST(download_path=args.data_root, train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.MNISTCnn().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


    # import numpy as np
    # import plot
    # from matplotlib import pyplot as plt
    # rng = np.random.default_rng(42)
    # fig, axes = plot.plot_MNIST_sample(train_loader.dataset, rng)
    # plt.tight_layout()
    # plt.show()
