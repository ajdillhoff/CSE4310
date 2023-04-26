import torch
import torch.nn as nn
import pytorch_lightning as pl

from model import ViTHPE


class ViTHPELightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = ViTHPE(hparams)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, y_pixel, center, scale_factor = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        # TODO: Convert output to pixel space and compare with y_pixel
        #       Use center and scale_factor to convert
        #       Make sure to log the loss

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        # TODO: Convert output to pixel space and compare with y_pixel
        #       Use center and scale_factor to convert
        #       Make sure to log the loss

        self.log("test_loss", loss)
        return loss