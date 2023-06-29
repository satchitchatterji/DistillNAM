import pytorch_lightning as L
import torch
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy


# architecture reference:
# https://medium.com/@jaredmcmullen1/developing-a-simple-cnn-for-mnist-f98c38f0d38d
class MNISTCnn(nn.Module):
    def __init__(self):
        super().__init__()
        conv_block = lambda in_chan, out_chan: nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # MNIST input: 28x28x1
        self.conv1 = conv_block(1, 32)  # 26x26x32 conv -> 13x13x32 maxpool
        self.conv2 = conv_block(32, 64)  # 24x24x64 -> 5x5x64 maxpool
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(5 * 5 * 64, out_features=10)

    def forward(self, x: T) -> T:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.classifier(out.view(out.size(0), -1))
        return out


# structure taken from
# https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
# FIXME only train and validation loss logged?!
class LitMNIST(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = MNISTCnn()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
