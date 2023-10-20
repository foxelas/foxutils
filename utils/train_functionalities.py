import os
import pickle
from os.path import join as pathjoin
from os.path import splitext

import kornia
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision.io import read_image

from . import core_utils, display_and_plot, train_with_lightning
from .core_utils import SEED, logger

###########################################################
MAX_EPOCHS = 20
NUM_WORKERS = 0
BATCH_SIZE = 16
ImageFile.LOAD_TRUNCATED_IMAGES = True

###########################################################

torch.manual_seed(SEED)
output_threshold = 0.5

settings = core_utils.settings
datasets_dir = core_utils.datasets_dir
default_models_dir = core_utils.models_dir
token_dir = core_utils.token_dir
device = core_utils.device


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
                logger.info(f'Error in applying for batch {i}')
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
            logger.info('Epoch: {}. Train Loss: {:.2f}. Train Accuracy: {:.2f}. Valid Loss: {:.2f}. Valid Accuracy: {:.2f}'
                  .format(epoch, train_loss_val, train_accuracy_val, loss, accuracy))
            print('Valid Confusion matrix {}'.format(cm))

        display_and_plot.plot_accuracy_per_epoch(train_accuracy, valid_accuracy)
        display_and_plot.plot_loss_per_epoch(train_loss, valid_loss)

    return target_model


###########################################################
# Performance metrics

def get_performance_metrics(true_labels, predicted_labels, model_name=None, show_cm=True):
    cm = confusion_matrix(true_labels, predicted_labels)
    if show_cm:
        display_and_plot.plot_confusion_matrix(cm, model_name)

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


def save_trained_model(target_model, save_name, models_dir=default_models_dir):
    save_name, save_ext = splitext(save_name)
    filename = pathjoin(models_dir, save_name + '.pts')
    core_utils.mkdir_if_not_exist(filename)

    torch.save(target_model.state_dict(), filename)
    logger.info(f'Model is saved at location: {filename}')


def load_trained_model(target_model_class, save_name, models_dir=default_models_dir):
    save_name, save_ext = splitext(save_name)
    filename = pathjoin(models_dir, save_name + '.pts')
    logger.info(f'Model is loaded from location: {filename}')
    target_model = target_model_class
    target_model.load_state_dict(torch.load(filename))
    # target_model = torch.load(filename.replace('.model', '.pt'))
    target_model.eval()
    return target_model


def pickle_model(target_model, save_name, models_dir=default_models_dir):
    save_name, save_ext = splitext(save_name)
    filename = pathjoin(models_dir, save_name + '.pkl')
    f = open(filename, "wb")
    f.write(pickle.dumps(target_model))
    f.close()
    logger.info(f'Model is saved at location: {filename}')


def unpickle_model(save_name, models_dir=default_models_dir):
    save_name, save_ext = splitext(save_name)
    filename = pathjoin(models_dir, save_name + '.pkl')
    target_model = pickle.loads(open(filename, "rb").read())
    logger.info(f'Model is loaded from location: {filename}')
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


def inverse_scaling(df, scaler):
    numeric_cols = scaler.get_feature_names_out()
    missing_cols = [x for x in numeric_cols if x not in df.columns]
    for x in missing_cols:
        df[x] = 0

    df = df.reindex(numeric_cols, axis=1)
    df_ = scaler.inverse_transform(df)
    df_ = pd.DataFrame(df_, columns=numeric_cols)
    return df_


def get_train_val_test_size(data_size, val_percentage=0.3, test_percentage=0.05):
    indices = np.arange(data_size)
    train_df, test_df = train_test_split(indices, test_size=test_percentage, random_state=SEED, shuffle=False)
    train_df, val_df = train_test_split(train_df, test_size=val_percentage, random_state=SEED, shuffle=False)
    train_size = len(train_df)
    val_size = len(val_df)
    test_size = len(test_df)

    return train_size, val_size, test_size


def scale_data(data_df, train_size=None):
    if train_size is None:
        train_size = len(data_df)

    scaler = MinMaxScaler()
    scaler, train_df = apply_scaling(data_df.iloc[0:train_size], scaler, has_fit=True)
    _, data_df = apply_scaling(data_df, scaler, has_fit=False)
    return data_df, scaler


def make_scale_train_val_test(data_df, val_percentage=0.3, test_percentage=0.05):
    train_df, test_df = train_test_split(data_df, test_size=test_percentage, random_state=SEED, shuffle=False)
    train_df, val_df = train_test_split(train_df, test_size=val_percentage, random_state=SEED, shuffle=False)

    logger.info(f'Train length: {len(train_df)}')
    logger.info(f'Val length: {len(val_df)}')
    logger.info(f'Test length: {len(test_df)}')

    scaler = MinMaxScaler()
    scaler, train_df = apply_scaling(train_df, scaler, has_fit=True)
    _, val_df = apply_scaling(val_df, scaler, has_fit=False)
    _, test_df = apply_scaling(test_df, scaler, has_fit=False)

    return train_df, val_df, test_df, scaler


def make_data_loader_with_torch(dataset, batch_size=BATCH_SIZE, shuffle=False, show_size=False):
    data_generator = TorchDataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    if show_size:
        logger.info(f'Number of images in the data loader: {len(dataset)}')
    return data_generator


def make_train_val_test_generators_with_torch(dataset, val_percentage, test_percentage, batch_size=BATCH_SIZE,
                                              show_size=False):
    permutation_generator = torch.Generator().manual_seed(SEED)
    splits = [1 - (val_percentage + test_percentage), val_percentage, test_percentage]
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, splits,
                                                                             generator=permutation_generator)
    if show_size:
        logger.info(f'Number of train images: {len(train_dataset)}')
        logger.info(f'Number of valid images: {len(val_dataset)}')
        logger.info(f'Number of test images: {len(test_dataset)}')

    data_generators = {"train": TorchDataLoader(train_dataset, shuffle=False, batch_size=batch_size),
                       "valid": TorchDataLoader(val_dataset, shuffle=False, batch_size=batch_size),
                       "test": TorchDataLoader(test_dataset, shuffle=False, batch_size=batch_size)}

    return data_generators


################################################################################

# import matplotlib.pyplot as plt

def augment_image_dataset_by_class(df, target_column, target_classes, num_per_class, image_dataset_dir,
                                   target_classes_dict):
    for c in target_classes:
        to_be_augmented = df[df[target_column] == c].copy()
        orig_num = len(to_be_augmented)
        expected = num_per_class - orig_num
        while len(to_be_augmented) < expected:
            to_be_augmented = pd.concat([to_be_augmented, to_be_augmented.iloc[-(num_per_class - orig_num):]])
        to_be_augmented = to_be_augmented.iloc[:expected + 1]
        logger.info(f'To be augmented {len(to_be_augmented)} for {target_classes_dict[c]}.')

        aug_tfm = train_with_lightning.DataAugmentation(p=0.5, keep_orig_dim=True)

        for folder in to_be_augmented["folder"].unique():
            core_utils.mkdir_if_not_exist(pathjoin(image_dataset_dir, folder + "aug", ""))

        for (folder, file) in zip(to_be_augmented["folder"].values, to_be_augmented["file"].values):
            i = 0
            new_file = "_".join([folder + "aug" + "%s", file.split("_")[-1]]) % i
            savedir = pathjoin(image_dataset_dir, folder + "aug", "")
            while os.path.exists(pathjoin(savedir, new_file)):
                i += 1
                new_file = "_".join([folder + "aug" + "%s", file.split("_")[-1]]) % i

            try:
                img = read_image(pathjoin(image_dataset_dir, folder, file)).squeeze()
                # pil_img = T.ToPILImage()(img)
                # plt.imshow(np.asarray(pil_img))
                # plt.show()

                imgs_aug = aug_tfm(img.float()).squeeze()
                imgs_aug = kornia.tensor_to_image(imgs_aug.byte())
                pil_img = T.ToPILImage()(imgs_aug)
                # plt.imshow(np.asarray(pil_img))
                # plt.show()
                pil_img.save(pathjoin(savedir, new_file))
                # print(new_file)
                # break
            except RuntimeError:
                logger.info(f'Error in augmenting {pathjoin(image_dataset_dir, folder, file)}.')

        logger.info(f'Finished augmentation for {target_classes_dict[c]}.')


def get_label_and_prob_string(label, prob):
    return f'{label} ({prob:.2f})'