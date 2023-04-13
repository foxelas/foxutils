from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop

import joblib

import utils_train

models_dir = utils_train.models_dir

model_svm_file = pathjoin(models_dir, "trained_model_svm")
model_lr_file = pathjoin(models_dir, "trained_model_lr")
model_nn_file = pathjoin(models_dir, "trained_model_simple_nn")
model_cnn_file = pathjoin(models_dir, "trained_model_simple_cnn")
scaler_file = pathjoin(models_dir, "trained_scaler")