from os.path import join as pathjoin
from os.path import splitext

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from foxutils.utils import core_utils, display_and_plot
from .core_utils import SEED

import pickle

###########################################################
MAX_EPOCHS = 20
NUM_WORKERS = 0
BATCH_SIZE = 16

###########################################################

torch.manual_seed(SEED)
output_threshold = 0.5

settings = core_utils.settings
datasets_dir = core_utils.datasets_dir
models_dir = core_utils.models_dir
token_dir = core_utils.token_dir
preprocessed_folder = core_utils.preprocessed_folder
extracted_folder = core_utils.extracted_folder
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


def save_trained_model(target_model, save_name):
    save_name, save_ext = splitext(save_name)

    filename = pathjoin(models_dir, save_name + '.pts')
    core_utils.mkdir_if_not_exist(filename)

    torch.save(target_model.state_dict(), filename)
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

def pickle_model(target_model, save_name):
    save_name, save_ext = splitext(save_name)
    filename = pathjoin(models_dir, save_name + '.pkl')
    f = open(filename, "wb")
    f.write(pickle.dumps(target_model))
    f.close()
    print(f'Model is saved at location: {filename}')

def unpickle_model(save_name):
    save_name, save_ext = splitext(save_name)
    filename = pathjoin(models_dir, save_name + '.pkl')
    target_model = pickle.loads(open(filename, "rb").read())
    print(f'Model is loaded from location: {filename}')
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


def make_data_loader_with_torch(dataset, batch_size=BATCH_SIZE, shuffle=False, show_size=False):
    data_generator = TorchDataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    if show_size:
        print(f'Number of images in the data loader: {len(dataset)}')
    return data_generator


def make_train_val_test_generators_with_torch(dataset, val_percentage, test_percentage, batch_size=BATCH_SIZE,
                                              show_size=False):
    permutation_generator = torch.Generator().manual_seed(SEED)
    splits = [1-(val_percentage+test_percentage), val_percentage, test_percentage]
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, splits,
                                                                             generator=permutation_generator)
    if show_size:
        print(f'Number of train images: {len(train_dataset)}')
        print(f'Number of valid images: {len(val_dataset)}')
        print(f'Number of test images: {len(test_dataset)}')

    data_generators = {"train": TorchDataLoader(train_dataset, shuffle=False, batch_size=batch_size),
                        "valid": TorchDataLoader(val_dataset, shuffle=False, batch_size=batch_size),
                       "test": TorchDataLoader(test_dataset, shuffle=False, batch_size=batch_size)}

    return data_generators

