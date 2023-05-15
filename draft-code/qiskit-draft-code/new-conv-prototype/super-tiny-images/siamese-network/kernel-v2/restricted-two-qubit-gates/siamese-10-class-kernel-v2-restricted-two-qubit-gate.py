import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
from typing import List, Tuple, Union
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import os.path

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from qiskit_aer import AerSimulator
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.circuit import ParameterVector
import json
import time
import shutup
import pickle
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from SPSAGradOptimiser.qiskit_opts.SPSA_Adam import ADAMSPSA

shutup.please()

DATA_PATH = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/tiny-handwritten.pkl"

def nowtime():
    return str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))

def add_padding(matrix: np.ndarray,
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix.
    from https://github.com/detkov/Convolution-From-Scratch/blob/main/convolution.py
    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we
        addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix
    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding

    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r: n + r, c: m + c] = matrix

    return padded_matrix

def _check_params(matrix, kernel, stride, dilation, padding):
    """
    from https://github.com/detkov/Convolution-From-Scratch/blob/main/convolution.py
    :param matrix:
    :param kernel:
    :param stride:
    :param dilation:
    :param padding:
    :return:
    """
    params_are_correct = (isinstance(stride[0], int) and isinstance(stride[1], int) and
                          isinstance(dilation[0], int) and isinstance(dilation[1], int) and
                          isinstance(padding[0], int) and isinstance(padding[1], int) and
                          stride[0] >= 1 and stride[1] >= 1 and
                          dilation[0] >= 1 and dilation[1] >= 1 and
                          padding[0] >= 0 and padding[1] >= 0)
    assert params_are_correct, 'Parameters should be integers equal or greater than default values.'
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    n, m = matrix.shape
    matrix = matrix if list(padding) == [0, 0] else add_padding(matrix, padding)
    n_p, m_p = matrix.shape

    if not isinstance(kernel, np.ndarray):
        kernel = np.array(kernel)
    k = kernel.shape

    kernel_is_correct = k[0] % 2 == 1 and k[1] % 2 == 1
    assert kernel_is_correct, 'Kernel shape should be odd.'
    matrix_to_kernel_is_correct = n_p >= k[0] and m_p >= k[1]
    assert matrix_to_kernel_is_correct, 'Kernel can\'t be bigger than matrix in terms of shape.'

    h_out = np.floor((n + 2 * padding[0] - k[0] - (k[0] - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - k[1] - (k[1] - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    out_dimensions_are_correct = h_out > 0 and w_out > 0
    assert out_dimensions_are_correct, 'Can\'t apply input parameters, one of resulting output dimension is ' \
                                       'non-positive.'

    return matrix, kernel, k, h_out, w_out

def extract_convolution_data(matrix: Union[List[List[float]], List[List[List[float]]], np.ndarray],
                             kernel_size:Tuple[int, int]=(3, 3),
                             stride:Tuple[int, int] = (3, 3),
                             dilation:Tuple[int, int]=(1, 1),
                             padding: Tuple[int, int]=(0,0),
                             encoding_gate_parameter_size:int=3,
                             ) -> List[List[List[float]]]:
    kernel_placeholder = np.ones(kernel_size)
    matrix, kernel, k, h_out, w_out = _check_params(matrix, kernel_placeholder, stride, dilation, padding)
    b = k[0] // 2, k[1] // 2
    center_x_0 = b[0] * dilation[0]
    center_y_0 = b[1] * dilation[1]
    output = []
    for i in range(h_out):
        center_x = center_x_0 + i * stride[0]
        indices_x = [center_x + l * dilation[0] for l in range(-b[0], b[0] + 1)]
        row = []
        for j in range(w_out):
            center_y = center_y_0 + j * stride[1]
            indices_y = [center_y + l * dilation[1] for l in range(-b[1], b[1] + 1)]
            submatrix = matrix[indices_x, :][:, indices_y]
            unpadded_data = submatrix.flatten().tolist()
            num_data_gates = len(unpadded_data)//encoding_gate_parameter_size + 1
            data_pad_size = encoding_gate_parameter_size * num_data_gates - len(unpadded_data)
            padded_data = submatrix.flatten().tolist()
            for _ in range(data_pad_size):
                padded_data.append(0)
            row.append(padded_data)
        output.append(row)
    return output

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

def load_tiny_digits(path, kind="train"):
    if os.path.isfile(path):
        with open(path,'rb') as f:
            mnist = pickle.load(f)
    else:
        import tinyhandwrittendigits
        tinyhandwrittendigits.init()
        with open(DATA_PATH, 'rb') as f:
            mnist = pickle.load(f)
    if kind == 'train':
        return mnist["training_images"], mnist["training_labels"]
    else:
        return mnist["test_images"], mnist["test_labels"]

def select_data(labels=[0,1,2,3,4,5,6,7,8,9], num_data_per_label_train = 3, num_test_per_label=1, rng=np.random.default_rng(seed=42)):
    data_path = DATA_PATH
    features, all_labels = load_tiny_digits(data_path)
    features = np.array(features) * (2*np.pi) # in the dataset, we converted the range of values to [0,1]
    all_labels = np.array(all_labels)
    selected_train_images = []
    test_images= {}
    test_images['data'] = []
    test_images['labels'] = []
    for label in labels:
        data_for_label = features[np.where(all_labels == label)]
        # sample some data
        train_indices = rng.choice(len(data_for_label), num_data_per_label_train, replace=False)
        test_indices = rng.choice(
        np.setdiff1d(range(len(data_for_label)), train_indices), num_test_per_label, replace=False)
        extracted_data = [extract_convolution_data(data_for_label[train_indices][i], kernel_size=(5, 5), stride=(3, 3), dilation=(1, 1),
                                        padding=(0, 0), encoding_gate_parameter_size=15) for i in range(num_data_per_label_train)]
        for i in range(num_data_per_label_train):
            selected_train_images.append((extracted_data[i], label))
        test_images['data'].append(extract_convolution_data(data_for_label[test_indices][0], kernel_size=(5, 5), stride=(3, 3), dilation=(1, 1),
                                        padding=(0, 0), encoding_gate_parameter_size=15))
        test_images['labels'].append(label)

    # pair the training data
    paired_extracted_data = []
    for i in range(len(selected_train_images)):
        for j in range(len(selected_train_images)):
            if i!=j:
                paired_extracted_data.append((selected_train_images[i], selected_train_images[j]))

    return paired_extracted_data, test_images

def su4_circuit(params):
    su4 = QuantumCircuit(2, name='su4')
    su4.u(params[0], params[1], params[2], qubit=0)
    su4.u(params[3], params[4], params[5], qubit=1)
    su4.cx(0,1)
    su4.ry(params[6], 0)
    su4.rz(params[7], 1)
    su4.cx(1, 0)
    su4.ry(params[8], 0)
    su4.cx(0, 1)
    su4.u(params[9], params[10], params[11], 0)
    su4.u(params[12], params[13], params[14], 1)
    su4_inst = su4.to_instruction()
    su4 = QuantumCircuit(2)
    su4.append(su4_inst, list(range(2)))
    return su4

def conv_circuit(params):
    target = QuantumCircuit(2, name='conv')
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    conv_circ = QuantumCircuit(2)
    conv_circ.append(target, list(range(2)))
    return conv_circ

