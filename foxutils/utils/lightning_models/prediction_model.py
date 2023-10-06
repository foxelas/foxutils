# PyTorch
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torchmetrics
# Torchvision
import torchvision
# Kornia
from kornia import tensor_to_image
from torch import nn

from ..train_with_lightning import DataAugmentation, get_binary_label, get_conf_matrix_fig


class PredictionModel(pl.LightningModule):
    def __init__(self, model_class, task="binary", num_classes=2, num_labels=2, forward_function=get_binary_label,
                 loss_fun=nn.BCELoss, configure_optimizers_fun=None, average='micro', has_augmentation=False,
                 aug_transforms=None, aug_p=0.5, class_mapping=None, **model_hyperparameters):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fun', 'model_class'])
        self.class_mapping = class_mapping
        self.has_augmentation = has_augmentation
        self.transform = DataAugmentation(transforms=aug_transforms, p=aug_p, keep_orig_dim=True)
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

        self.average = average
        self.train_precision = torchmetrics.Precision(task=task, num_classes=num_classes, top_k=1,
                                                      num_labels=num_labels, average=average)
        self.valid_precision = torchmetrics.Precision(task=task, num_classes=num_classes, top_k=1,
                                                      num_labels=num_labels, average=average)
        self.train_recall = torchmetrics.Recall(task=task, num_classes=num_classes, top_k=1,
                                                num_labels=num_labels, average=average)
        self.valid_recall = torchmetrics.Recall(task=task, num_classes=num_classes, top_k=1,
                                                num_labels=num_labels, average=average)
        self.train_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, top_k=1,
                                               num_labels=num_labels, average=average)
        self.valid_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, top_k=1,
                                               num_labels=num_labels, average=average)
        self.train_cm = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.valid_cm = torchmetrics.ConfusionMatrix(num_classes=num_classes)

        self.training_step_preds = []
        self.training_step_labels = []

        self.validation_step_preds = []
        self.validation_step_labels = []

    def forward(self, x):
        _, preds = self.forward_function(self.model, x)
        return preds

    def show_batch(self, dataloader, win_size=(10, 10), transform=None):
        def _to_vis(data):
            if transform is not None:
                data = [transform(x) for x in data]
            return tensor_to_image(torchvision.utils.make_grid(data, nrow=8))

        imgs, labels = next(iter(dataloader))
        if self.has_augmentation:
            imgs_aug = self.transform(imgs)  # apply transforms
        else:
            imgs_aug = imgs

        # use matplotlib to visualize
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs))
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs_aug))

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training and self.has_augmentation:
            x = self.transform(x)  # => we perform GPU/Batched data augmentation
        return x, y

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        feature_vectors, labels = batch

        # zero the parameter gradients
        optimizer = self.optimizers()
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs, preds = self.forward_function(self.model, feature_vectors)

            loss = self.loss_fun(outputs, labels)
            self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
            self.logger.experiment.add_scalars('loss', {'train': loss}, self.global_step)

            # backward + optimize only if in training phase
            self.manual_backward(loss)
            optimizer.step()

            self.training_step_preds.append(preds)
            self.training_step_labels.append(labels)

        return loss

    def on_train_epoch_end(self):
        preds = torch.cat(self.training_step_preds)
        labels = torch.cat(self.training_step_labels)

        acc = self.train_acc(preds, labels)
        self.logger.experiment.add_scalars('acc', {'train': acc}, self.global_step)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        rc = self.train_recall(preds, labels)
        self.logger.experiment.add_scalars('recall', {'train': rc}, self.global_step)
        self.log('train_recall', rc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        pc = self.train_precision(preds, labels)
        self.logger.experiment.add_scalars('precision', {'train': pc}, self.global_step)
        self.log('train_precision', pc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        cm = self.train_cm(preds, labels)
        computed_confusion = cm.detach().cpu().numpy().astype(int)
        fig_ = get_conf_matrix_fig(computed_confusion, self.class_mapping)
        self.logger.experiment.add_figure("Train Confusion Matrix", fig_, self.current_epoch)

        # Free memory
        self.training_step_preds.clear()
        self.training_step_labels.clear()

        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch, batch_idx):
        feature_vectors, labels = batch
        outputs, preds = self.forward_function(self.model, feature_vectors)

        loss = self.loss_fun(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.logger.experiment.add_scalars('loss', {'valid': loss}, self.global_step)

        self.validation_step_preds.append(preds)
        self.validation_step_labels.append(labels)

        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.validation_step_preds)
        labels = torch.cat(self.validation_step_labels)

        acc = self.valid_acc(preds, labels)
        self.logger.experiment.add_scalars('acc', {'valid': acc}, self.global_step)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        rc = self.valid_recall(preds, labels)
        self.logger.experiment.add_scalars('recall', {'valid': rc}, self.global_step)
        self.log('val_recall', rc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        pc = self.valid_precision(preds, labels)
        self.logger.experiment.add_scalars('precision', {'valid': pc}, self.global_step)
        self.log('val_precision', pc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        cm = self.valid_cm(preds, labels)
        computed_confusion = cm.detach().cpu().numpy().astype(int)
        fig_ = get_conf_matrix_fig(computed_confusion, self.class_mapping)
        self.logger.experiment.add_figure("Valid Confusion Matrix", fig_, self.current_epoch)

        # Free memory
        self.validation_step_preds.clear()
        self.validation_step_labels.clear()

    def configure_optimizers(self):
        return self.configure_optimizers_fun()

