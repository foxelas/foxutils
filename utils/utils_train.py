
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import sys
sys.path.insert(0, '../../foxutils/')
from utils import utils_display


torch.manual_seed(42)
output_threshold = 0.5


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

