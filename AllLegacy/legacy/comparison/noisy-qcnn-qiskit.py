# QCNN structure following
# https://pennylane.ai/qml/demos/tutorial_learning_few_data.html
# but running on a noisy device

import os.path
import time
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from qiskit_aer.noise import NoiseModel
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from qiskit_aer import AerSimulator
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA, COBYLA
import json
import time
import shutup
import pickle
from qiskit_ibm_provider import IBMProvider
PROVIDER = IBMProvider()
noisy_backend = PROVIDER.get_backend('ibm_perth')
noise_model = NoiseModel.from_backend(noisy_backend)

shutup.please()

DATA_PATH = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/tiny-handwritten.pkl"

seed = 0
rng = np.random.default_rng(seed=seed)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def load_tiny_digits(path = DATA_PATH, kind="train"):
    with open(path,'rb') as f:
            mnist = pickle.load(f)
    if kind == 'train':
        return mnist["training_images"], mnist["training_labels"]
    else:
        return mnist["test_images"], mnist["test_labels"]

def load_data(num_train, num_test, rng, one_hot=False):
    data_path = DATA_PATH
    features, labels = load_tiny_digits(data_path)
    features = np.array(features)
    labels = np.array(labels)
    # only use first four classes
    features = features[np.where((labels == 0) | (labels == 1) | (labels == 2) | (labels == 3))]
    labels = labels[np.where((labels == 0) | (labels == 1) | (labels == 2) | (labels == 3))]

    # subsample train and test split
    train_indices = rng.choice(len(labels), num_train, replace=False)
    test_indices = rng.choice(
        np.setdiff1d(range(len(labels)), train_indices), num_test, replace=False
    )

    x_train, y_train = features[train_indices], labels[train_indices]
    x_train = [x_train[i]/np.linalg.norm(x_train[i], axis=1).reshape((-1, 1)) for i in range(num_train)]
    x_test, y_test = features[test_indices], labels[test_indices]
    x_test = [x_test[i]/np.linalg.norm(x_test[i], axis=1).reshape((-1, 1)) for i in range(num_test)]
    if one_hot:
        train_labels = np.zeros((len(y_train), 4))
        test_labels = np.zeros((len(y_test), 4))
        train_labels[np.arange(len(y_train)), y_train] = 1
        test_labels[np.arange(len(y_test)), y_test] = 1

        y_train = train_labels
        y_test = test_labels

    return (
        x_train,
        y_train,
        x_test,
        y_test,
    )

def data_encoding_circ(image_array):
    circ = QuantumCircuit(6)
