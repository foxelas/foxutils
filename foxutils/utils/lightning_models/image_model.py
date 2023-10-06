# PyTorch
import warnings

# Torchvision
import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class ImageModel(pl.LightningModule):
    def __init__(self, num_input_channels, width, height, model_class, **model_hyperparameters):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_class(**model_hyperparameters, width=width, height=height)
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    @staticmethod
    def _get_reconstruction_loss(batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, labels = batch
        loss = F.mse_loss(x, labels, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.2,
                                                               patience=20,
                                                               min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
