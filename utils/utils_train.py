import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
# import sys
# sys.path.insert(0, '../../foxutils/')
# from utils import utils_display
from utils import utils, utils_display
from .utils import SEED, BATCH_SIZE, MAX_EPOCHS

import IPython
import IPython.display

import joblib
import torchmetrics
from torch import nn
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import torch.nn.functional as F

from os.path import join as pathjoin
from os.path import splitext
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import EarlyStopping as tfEarlyStopping
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

set_random_seed(SEED)

import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

torch.manual_seed(SEED)
output_threshold = 0.5

settings = utils.settings
datasets_dir = utils.datasets_dir
models_dir = utils.models_dir
token_dir = utils.token_dir
preprocessed_folder = utils.preprocessed_folder
extracted_folder = utils.extracted_folder
device = utils.device


def validate_model(data_generator, target_model, criterion):
    valid_loss_val = 0.0
    target_model.eval()  # Optional when not using Model Specific layer

    correct = 0
    counts = 0
    cm = np.zeros((2, 2))
    for feature_vectors, labels in data_generator:
        outputs = target_model(feature_vectors)
        loss = criterion(torch.squeeze(outputs), labels)
        predicted = torch.squeeze(outputs).round().detach().numpy()
        expected = labels.detach().numpy()

        cm_ = confusion_matrix(expected, predicted)
        cm = [[cm[i][j] + cm_[i][j] for j in range(len(cm[0]))] for i in range(len(cm))]

        correct += (predicted == expected).sum()
        counts += len(expected)
        valid_loss_val += loss.item()

    accuracy = 100 * (correct.item()) / counts

    return accuracy, valid_loss_val / len(data_generator), cm


def train_and_validate_model(data_generator_train, data_generator_validate, target_model, epochs, optimizer, criterion):
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []

    for epoch in range(epochs):

        correct = 0
        counts = 0
        cm = np.zeros((2, 2))

        train_loss_val = 0.0
        target_model.train()  # Optional when not using Model Specific layer
        for i, (feature_vectors, labels) in enumerate(data_generator_train):
            try:
                optimizer.zero_grad()
                outputs = target_model(feature_vectors)
                loss = criterion(torch.squeeze(outputs), labels)
                # Loss.append(loss.item())
                predicted = torch.squeeze(outputs).round().detach().numpy()
                expected = labels.detach().numpy()

                cm_ = confusion_matrix(expected, predicted)
                cm = [[cm[i][j] + cm_[i][j] for j in range(len(cm[0]))] for i in range(len(cm))]

                correct += (predicted == expected).sum()
                counts += len(expected)

                loss.backward()
                optimizer.step()

                train_loss_val += loss.item()

            except ValueError:
                print(f'Error in applying for batch {i}')
                print(f'Outputs {outputs}')
                print(f'Labels {labels}')

        train_accuracy_val = 100 * (correct.item()) / counts
        train_loss_val = train_loss_val / len(data_generator_train)
        train_accuracy.append(train_accuracy_val)
        train_loss.append(train_loss_val)

        if data_generator_validate:
            accuracy, loss, cm = validate_model(data_generator_validate, target_model, criterion)
            valid_accuracy.append(accuracy)
            valid_loss.append(loss)
            print('Epoch: {}. Train Loss: {:.2f}. Train Accuracy: {:.2f}. Valid Loss: {:.2f}. Valid Accuracy: {:.2f}'
                  .format(epoch, train_loss_val, train_accuracy_val, loss, accuracy))
            print('Valid Confusion matrix {}'.format(cm))

        utils_display.plot_accuracy_per_epoch(train_accuracy, valid_accuracy)
        utils_display.plot_loss_per_epoch(train_loss, valid_loss)

    return target_model


###########################################################
# Performance metrics

def get_performance_metrics(true_labels, predicted_labels, model_name=None, show_cm=True):
    cm = confusion_matrix(true_labels, predicted_labels)
    if show_cm:
        utils_display.plot_confusion_matrix(cm, model_name)

    acc, precision, recall = get_metrics_from_confusion_matrix(cm, show_cm)

    return acc, cm


def get_binary_confusion_matrix_values(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]

    return tp, fp, fn, tn


def get_metrics_from_confusion_matrix(cm, print_results=True):
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]

    acc = (tp + tn) / (tp + fp + tn + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)

    if print_results:
        print('Accuracy: ', "{:.0%}".format(acc), ' of predictions for asset price direction (up/down) are correct')
        print('Precision: ', "{:.0%}".format(precision), ' of predicted upward trend cases are correct')
        print('Recall: ', "{:.0%}".format(recall), ' of all actual upward trend cases are predicted as such')

    return acc, precision, recall


def store_performance(model_name, df_performance, true_labels, predicted_labels, train_true_labels,
                      train_predicted_labels):
    acc, cm = get_performance_metrics(true_labels, predicted_labels, model_name)
    acc_train, cm_train = get_performance_metrics(train_true_labels, train_predicted_labels, model_name, False)

    df_performance.loc[model_name] = {'name': model_name, 'val_accuracy': acc, 'val_confusion_matrix': cm,
                                      'train_accuracy': acc_train, 'train_confusion_matrix': cm_train}
    return df_performance


def get_error_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    rmse = np.sqrt(mse)

    df_results = pd.DataFrame({'MSE': [mse], 'RMSE': [rmse], 'MAE': [mae], 'MAPE': [mape]})
    return df_results


###############################################################
# Train with lightning
class LitTargetModel(pl.LightningModule):
    def __init__(self, model_class, **model_hyperparameters):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_class(**model_hyperparameters)
        self.loss_fun = nn.BCELoss()  # F.binary_cross_entropy()
        self.train_precision = torchmetrics.Precision(task="binary", num_classes=2)
        self.valid_precision = torchmetrics.Precision(task="binary", num_classes=2)
        self.train_recall = torchmetrics.Recall(task="binary", num_classes=2)
        self.valid_recall = torchmetrics.Recall(task="binary", num_classes=2)
        self.train_acc = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.valid_acc = torchmetrics.Accuracy(task="binary", num_classes=2)
        print('Loss function: BCE')

    def forward(self, x):
        outputs = torch.squeeze(self.model(x))
        preds = torch.round(outputs)
        return preds

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        feature_vectors, labels = batch
        outputs = torch.squeeze(self.model(feature_vectors))
        preds = torch.round(outputs)

        loss = self.loss_fun(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        self.train_acc(preds, labels)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.train_recall(preds, labels)
        self.log('train_recall', self.train_recall, prog_bar=True, on_step=False, on_epoch=True)

        self.train_precision(preds, labels)
        self.log('train_precision', self.train_precision, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    @staticmethod
    def train_epoch_end(result):
        loss = sum(r['loss'] for r in result) / len(result)
        print(f'Loss at training epoch end {loss}')

    def validation_step(self, batch, batch_idx):
        feature_vectors, labels = batch
        outputs = torch.squeeze(self.model(feature_vectors))
        preds = torch.round(outputs)

        loss = self.loss_fun(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        self.valid_acc(preds, labels)
        self.log('val_acc', self.valid_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.valid_recall(preds, labels)
        self.log('val_recall', self.valid_recall, prog_bar=True, on_step=False, on_epoch=True)

        self.valid_precision(preds, labels)
        self.log('val_precision', self.valid_precision, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        print('Optimizer SGD with lr=0.001')
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer


class ImageModel(pl.LightningModule):
    def __init__(self, num_input_channels, width, height, model_class, **model_hyperparameters):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_class(**model_hyperparameters, width=width, height=height)
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
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
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)


def train_and_validate_with_lightning(data_generators, target_model, epochs):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger)

    trainer.fit(
        target_model,
        train_dataloaders=data_generators["train"],
        val_dataloaders=data_generators["valid"])


def save_trained_model(target_model, save_name):
    save_name, save_ext = splitext(save_name)
    filename = utils.mkdir_if_not_exist(pathjoin(models_dir, save_name + '.pts'))
    torch.save(target_model.state_dict(), filename)
    # torch.save(target_model, filename.replace('.model', '.pt'))
    print(f'Model is saved at location: {filename}')


def load_trained_model(target_model_class, save_name):
    save_name, save_ext = splitext(save_name)
    filename = pathjoin(models_dir, save_name + '.pts')
    print(f'Model is loaded from location: {filename}')
    target_model = target_model_class
    target_model.load_state_dict(torch.load(filename))
    # target_model = torch.load(filename.replace('.model', '.pt'))
    target_model.eval()
    return target_model


def pl_load_trained_model(target_model_class, save_name):
    filename = pathjoin(models_dir, save_name)
    print(f'Model is loaded from location: {filename}')
    target_model = LitTargetModel(target_model_class)
    target_model.load_state_dict(torch.load(filename.replace('.model', '.pts')))
    target_model.eval()
    return target_model


def pl_load_trained_model_from_checkpoint(target_model_class, checkpoint_path):
    print(f'Model is loaded from checkpoint: {checkpoint_path}')
    target_model = LitTargetModel.load_from_checkpoint(checkpoint_path=checkpoint_path, model_class=target_model_class)
    target_model.eval()
    return target_model


##################################################################################################


def apply_scaling(df, scaler, has_fit=True):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    other_cols = [x for x in df.columns if x not in numeric_cols]

    df_ = df.select_dtypes(include=np.number)
    if has_fit:
        df_ = scaler.fit_transform(df_)
    else:
        df_ = scaler.transform(df_)

    df_ = pd.DataFrame(df_, columns=numeric_cols)
    for x in other_cols:
        df_[x] = df[x]

    return scaler, df_


def make_train_val_test(data_df, val_size=0.3, test_size=0.05):
    train_df, val_df = train_test_split(data_df, test_size=val_size, random_state=SEED, shuffle=False)
    train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=SEED, shuffle=False)

    print(f'Train length: {len(train_df)}')
    print(f'Val length: {len(val_df)}')
    print(f'Test length: {len(test_df)}')

    scaler = MinMaxScaler()
    scaler, train_df = apply_scaling(train_df, scaler, has_fit=True)
    _, val_df = apply_scaling(val_df, scaler, has_fit=False)
    _, test_df = apply_scaling(test_df, scaler, has_fit=False)

    return train_df, val_df, test_df, scaler


def compile_and_fit(model, window, patience=2, max_epochs=MAX_EPOCHS):
    early_stopping = tfEarlyStopping(monitor='val_loss',
                                     patience=patience,
                                     mode='min')

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,verbose=0, validation_data=window.val, callbacks=[early_stopping])

    #IPython.display.clear_output()

    return history
