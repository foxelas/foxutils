from .torch_models import Encoder, Decoder, get_reconstruction_loss
from .core_utils import SEED, models_dir

from os.path import join as pathjoin
from os.path import isfile, splitext

# PyTorch
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics

# Torchvision
import torchvision
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

# Setting the seed
pl.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# You are using a CUDA device ('NVIDIA GeForce RTX 3060 Laptop GPU') that has Tensor Cores. To properly utilize them,
# you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for
# performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision
# .html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('medium')


#################################################################################

def get_lightning_log_dir(model_name):
    lightning_log_dir = pathjoin(models_dir, model_name)
    return lightning_log_dir


def get_lightning_checkpoint_path(lightning_log_dir, checkpoint_version):
    checkpoint_path = lightning_log_dir.replace('\\', '/') + "/lightning_logs/version_" + str(checkpoint_version) \
                      + "/checkpoints/"
    return checkpoint_path


def get_lightning_checkpoint_file(lightning_log_dir, checkpoint_version, checkpoint_file):
    checkpoint_path = get_lightning_checkpoint_path(lightning_log_dir, checkpoint_version)
    pretrained_filename = pathjoin(checkpoint_path, f"{checkpoint_file}.ckpt")
    return pretrained_filename


def pl_load_trained_model(target_model_class, save_name):
    save_name, save_ext = splitext(save_name)
    if save_ext is None:
        save_ext = 'pts'
    filename = pathjoin(models_dir, save_name + '.' + save_ext)
    print(f'Model is loaded from location: {filename}')
    target_model = PredictionModel(target_model_class)
    target_model.load_state_dict(torch.load(filename))
    target_model.eval()
    return target_model


def pl_load_trained_model_from_checkpoint(target_model_class, checkpoint_path):
    print(f'Model is loaded from checkpoint: {checkpoint_path}')
    target_model = PredictionModel.load_from_checkpoint(checkpoint_path=checkpoint_path, model_class=target_model_class)
    target_model.eval()
    return target_model


#################################################################################
# LightningModule classes

def get_binary_label(model, x):
    outputs = torch.squeeze(model(x))
    preds = torch.round(outputs)
    return outputs, preds


def get_label_probs(model, x):
    outputs = model(x)
    return outputs, outputs


def get_argmax_label(model, x):
    outputs = model(x)
    _, preds = torch.max(outputs, 1)
    return outputs, preds


def get_sgd_optimizer(self):
    optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
    return optimizer


class PredictionModel(pl.LightningModule):
    def __init__(self, model_class, task="binary", num_classes=2, num_labels=2, forward_function=get_binary_label,
                 loss_fun=nn.BCELoss, configure_optimizers_fun=None, **model_hyperparameters):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fun', 'model_class'])
        self.forward_function = forward_function
        self.loss_fun = loss_fun  # F.binary_cross_entropy()
        if len(model_hyperparameters) == 0:
            self.model = model_class
        else:
            self.model = model_class(**model_hyperparameters)

        if configure_optimizers_fun is None:
            self.automatic_optimization = True
        else:
            self.automatic_optimization = False
            self.configure_optimizers = configure_optimizers_fun

        self.train_precision = torchmetrics.Precision(task=task, num_classes=num_classes, num_labels=num_labels)
        self.valid_precision = torchmetrics.Precision(task=task, num_classes=num_classes, num_labels=num_labels)
        self.train_recall = torchmetrics.Recall(task=task, num_classes=num_classes, num_labels=num_labels)
        self.valid_recall = torchmetrics.Recall(task=task, num_classes=num_classes, num_labels=num_labels)
        self.train_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, num_labels=num_labels)
        self.valid_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, num_labels=num_labels)

    def forward(self, x):
        _, preds = self.forward_function(self.model, x)
        return preds

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        feature_vectors, labels = batch
        outputs, preds = self.forward_function(self.model, feature_vectors)

        loss = self.loss_fun(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        self.train_acc(preds, labels)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.train_recall(preds, labels)
        self.log('train_recall', self.train_recall, prog_bar=True, on_step=False, on_epoch=True)

        self.train_precision(preds, labels)
        self.log('train_precision', self.train_precision, prog_bar=True, on_step=False, on_epoch=True)

        optimizer = self.optimizers()
        optimizer.zero_grad()
        #self.manual_backward(loss)
        optimizer.step()

        sch = self.lr_schedulers()
        sch.step()
        return loss


    def validation_step(self, batch, batch_idx):
        feature_vectors, labels = batch
        outputs, preds = self.forward_function(self.model, feature_vectors)

        loss = self.loss_fun(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        self.valid_acc(preds, labels)
        self.log('val_acc', self.valid_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.valid_recall(preds, labels)
        self.log('val_recall', self.valid_recall, prog_bar=True, on_step=False, on_epoch=True)

        self.valid_precision(preds, labels)
        self.log('val_precision', self.valid_precision, prog_bar=True, on_step=False, on_epoch=True)
        return loss


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


class Autoencoder(pl.LightningModule):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32):
        super().__init__()
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim, width, height)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim, width, height)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = get_reconstruction_loss(x, x_hat)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_mse_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_mse_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_mse_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_mse_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


#################################################################################
# Lightning Training

def train_predictive_model(target_model, lightning_log_dir, data_generators, epochs=2):
    lr_logger = LearningRateMonitor("epoch")
    logger = TensorBoardLogger(lightning_log_dir)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")

    trainer = pl.Trainer(
        log_every_n_steps=4,
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        enable_model_summary=True,
        callbacks=[lr_logger,
                   early_stop_callback,
                   PrintLossCallback(every_n_epochs=5),
                   ModelCheckpoint(monitor='val_loss', save_weights_only=False)
                   ],
        logger=logger)

    trainer.fit(
        target_model,
        train_dataloaders=data_generators["train"],
        val_dataloaders=data_generators["valid"])

    return target_model


class PrintLossCallback(pl.Callback):
    def __init__(self, every_n_epochs=20):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print(f"Epoch: {trainer.current_epoch}, "
              f"Train Loss: {metrics['train_loss']:.4f}"
              f"Validation Loss: {metrics['val_loss']:.4f}")


class GenerateCallbackForImageReconstruction(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=20):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


def train_image_reconstruction_model(target_model, lightning_log_dir, data_generators, epochs=2, has_checkpoint=False,
                                     checkpoint_path=None, pretrained_filename=None, callback_data=None):
    lr_logger = LearningRateMonitor("epoch")
    logger = TensorBoardLogger(lightning_log_dir)
    early_stop_callback = EarlyStopping(monitor="val_mse_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger,
                   early_stop_callback,
                   ModelCheckpoint(save_weights_only=True),
                   GenerateCallbackForImageReconstruction(callback_data, every_n_epochs=1)],
        logger=logger)

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    if has_checkpoint and isfile(pretrained_filename):
        print("Found pretrained model, loading...")

        target_model = target_model.load_from_checkpoint(pretrained_filename)
        # target_model = load_and_train.pl_load_trained_model_from_checkpoint(model_class, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        hyperparams = checkpoint["hyper_parameters"]
        print(f'Loaded hyperparameters: {hyperparams}')

    else:
        trainer.fit(target_model, data_generators["train"], data_generators["valid"])

    # Test best model on validation and test set
    val_result = trainer.test(target_model, data_generators["valid"], verbose=False)
    test_result = trainer.test(target_model, data_generators["test"], verbose=False)
    result = {"test": test_result, "val": val_result}
    return target_model, result
