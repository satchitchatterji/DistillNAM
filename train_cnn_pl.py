from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn, optim
from torch.utils.data import DataLoader

import models
import utils


class MNISTCnnModule(pl.LightningModule):
    def __init__(self, model_hparams, optimizer_name, optimizer_hparams):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = models.MNISTCnn()
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 1, 28, 28), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # AdamW is Adam with a correct implementation of weight decay
        # (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
        optimizer = optim.AdamW(self.parameters())
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    # def validation_step(self, batch, batch_idx):
    #     imgs, labels = batch
    #     preds = self.model(imgs).argmax(dim=-1)
    #     acc = (labels == preds).float().mean()
    #     # By default logs it per epoch (weighted average over batches)
    #     self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and
        # returns it afterwards
        self.log("test_acc", acc)


def train_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    ckpt_path: str,
    max_epochs: int = 180,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=ckpt_path,
        accelerator="gpu" if device.type.startswith("cuda") else "cpu",
        max_epochs=max_epochs,  # How many epochs to train for if no patience is set
        # Save the best checkpoint based on the maximum val_acc recorded. Saves
        # only weights and not optimizer
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")
        ],
        enable_progress_bar=True,
    )
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = ckpt_path + ".ckpt"
    if Path(pretrained_filename).is_file():
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = MNISTCnnModule.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = MNISTCnnModule(None, None, None)
        trainer.fit(model, train_loader, None)
        # Load best checkpoint after training
        model = MNISTCnnModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Test best model on test set
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"]}

    return model, result


if __name__ == "__main__":
    DATA_ROOT = "~/datasets"

    train_loader = utils.get_MNIST(download_path=DATA_ROOT)
    test_loader = utils.get_MNIST(download_path=DATA_ROOT, train=False)

    train_model(train_loader, test_loader, "./MNIST_CNN")
