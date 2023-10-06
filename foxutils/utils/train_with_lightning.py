import warnings
from os.path import isfile, splitext
from os.path import join as pathjoin

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# PyTorch
import torch
# Torchvision
import torchvision
# Kornia
from kornia import augmentation
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torch import nn

from .core_utils import SEED, models_dir


#################################################################################
def set_run_params():
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


set_run_params()


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
    if '.' not in checkpoint_file:
        checkpoint_file = checkpoint_file + '.ckpt'
    pretrained_filename = pathjoin(checkpoint_path, checkpoint_file)
    return pretrained_filename


def pl_load_trained_model(target_model_class, weight_path, **model_params):
    weight_path_file, weight_path_ext = splitext(weight_path)
    assert weight_path_ext == '.pts', True
    print(f'Model is loaded from location: {weight_path}')
    target_model = target_model_class(**model_params)
    target_model.load_state_dict(torch.load(weight_path))
    target_model.eval()
    return target_model


# PyTorch Lightning >= 2.0
def pl_load_trained_model_from_checkpoint(target_model_class, checkpoint_path, **model_params):
    checkpoint_path_file, checkpoint_path_ext = splitext(checkpoint_path)
    assert checkpoint_path_ext == '.ckpt', True
    print(f'Model is loaded from checkpoint: {checkpoint_path}')
    target_model = target_model_class.load_from_checkpoint(checkpoint_path=checkpoint_path, **model_params)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    hyperparams = checkpoint["hyper_parameters"]
    print(f'Loaded hyperparameters: {hyperparams}')
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


def get_conf_matrix_fig(cm, class_mapping=None):
    df_cm = pd.DataFrame(cm)
    if class_mapping:
        inv_map = {v: k for k, v in class_mapping.items()}
        df_cm.rename(columns=inv_map, index=inv_map, inplace=True)

    plt.figure(figsize=(10, 7))
    fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt='d').get_figure()
    plt.close(fig_)
    return fig_


# Define a custom weight initialization function
def custom_weights_init(layer):
    if isinstance(layer, nn.Linear):
        layer.reset_parameters()
        # nn.init.xavier_uniform_(layer.weight)
        # layer.bias.data.fill_(0.01)

    if isinstance(layer, nn.Conv2d):
        layer.reset_parameters()


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def get_default_transforms(self, p):
        default_transforms = nn.Sequential(
            augmentation.RandomHorizontalFlip(p=p),
            augmentation.RandomRotation(degrees=10.0, p=p, keepdim=False),
            # augmentation.RandomBrightness(p=p),
            augmentation.RandomPerspective(0.1, p=p, keepdim=False),
            # augmentation.RandomThinPlateSpline(scale=0.1, p=p),
        )
        return default_transforms

    def get_color_transforms(self):
        transform = augmentation.ColorJitter(0.5, 0.5, 0.5, 0.5)
        return transform

    def keep_original_dim(self, x: Tensor, orig_dim: tuple) -> Tensor:
        new_dim = x.shape[-2:]
        if self.keep_orig_dim:
            transform = nn.Sequential(
                augmentation.CenterCrop((new_dim[0] - 30, new_dim[1] - 30), p=1, keepdim=True),
                augmentation.Resize(orig_dim, p=1, keepdim=True),
            )
            return transform(x)
        else:
            return x

    def __init__(self, transforms=None, p=0.5, keep_orig_dim=False) -> None:
        super().__init__()
        self.keep_orig_dim = keep_orig_dim
        if transforms is None:
            self.transforms = self.get_default_transforms(p)
        else:
            self.transforms = transforms

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        orig_dim = x.shape[-2:]
        x_out = self.transforms(x)  # BxCxHxW
        x_out = self.keep_original_dim(x_out, orig_dim)
        return x_out


#################################################################################
# Lightning Training

def train_predictive_model(target_model_class, lightning_log_dir, data_generators, epochs=2, early_stopping_patience=5,
                           **model_params):
    target_model = target_model_class(**model_params)

    lr_logger = LearningRateMonitor("epoch")
    logger = TensorBoardLogger(lightning_log_dir)

    callbacks = [lr_logger,
                 PrintLossCallback(every_n_epochs=5),
                 ModelCheckpoint(monitor='val_acc', save_top_k=1, save_weights_only=False, mode='max')
                 ]

    if early_stopping_patience is not None:
        print(f'Early stop callback with patience {early_stopping_patience} enabled')
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=1e-6, patience=early_stopping_patience,
                                            verbose=True, mode="max")
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        log_every_n_steps=4,
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        enable_model_summary=True,
        callbacks=callbacks,
        logger=logger)

    trainer.fit(
        target_model,
        train_dataloaders=data_generators["train"],
        val_dataloaders=data_generators["valid"])

    checkpoint_path = trainer.checkpoint_callback.best_model_path
    target_model = pl_load_trained_model_from_checkpoint(target_model_class, checkpoint_path, **model_params)

    return target_model, trainer


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


def train_image_reconstruction_model(target_model_class, lightning_log_dir, data_generators, epochs=2,
                                     has_checkpoint=False, checkpoint_path=None, pretrained_filename=None,
                                     callback_data=None, **model_params):
    target_model = target_model_class(**model_params)

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
                   ModelCheckpoint(monitor='val_acc', save_top_k=1, save_weights_only=False, mode='max'),
                   GenerateCallbackForImageReconstruction(callback_data, every_n_epochs=1)],
        logger=logger)

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    if has_checkpoint and isfile(pretrained_filename):
        print("Found pretrained model, loading...")

        target_model = target_model.load_from_checkpoint(pretrained_filename)
        # target_model = pl_load_trained_model_from_checkpoint(target_model_class, checkpoint_path, **model_params)

    else:
        trainer.fit(target_model, data_generators["train"], data_generators["valid"])

        checkpoint_path = trainer.checkpoint_callback.best_model_path
        print('Loading model from the best path: ', checkpoint_path)
        target_model = pl_load_trained_model_from_checkpoint(target_model_class, checkpoint_path, **model_params)

    # Test best model on validation and test set
    val_result = trainer.test(target_model, data_generators["valid"], verbose=False)
    test_result = trainer.test(target_model, data_generators["test"], verbose=False)
    result = {"test": test_result, "val": val_result}
    return target_model, result, trainer
