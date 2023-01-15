import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit import Aer
from dask.distributed import LocalCluster, Client
from concurrent.futures import ThreadPoolExecutor
from noisyopt import minimizeSPSA
from qiskit.algorithms.optimizers import COBYLA, SPSA, GradientDescent
import json
import time
import shutup


shutup.please()

from qiskit_ibm_provider import IBMProvider
PROVIDER = IBMProvider()

# get IBM's simulator backend
IBMQ_QASM_SIMULATOR = PROVIDER.get_backend('ibmq_qasm_simulator')

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


def load_data(num_train, num_test, rng, stride=(3,3), kernel_size=(3,3),encoding_gate_parameter_size:int=3, one_hot=True):
    """Return training and testing data of digits dataset."""
    data_folder = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/fashion"
    features, labels = load_fashion_mnist(data_folder)
    features = [features[i].reshape(28, 28) for i in range(len(features))]
    features = np.array(features)

    # only use first four classes
    features = features[np.where((labels == 0) | (labels == 1) | (labels == 2) | (labels == 3))]
    labels = labels[np.where((labels == 0) | (labels == 1) | (labels == 2) | (labels == 3))]

    # normalize data
    features = features / 255

    # subsample train and test split
    train_indices = rng.choice(len(labels), num_train, replace=False)
    test_indices = rng.choice(
        np.setdiff1d(range(len(labels)), train_indices), num_test, replace=False
    )

    x_train, y_train = features[train_indices], labels[train_indices]
    x_train = [extract_convolution_data(x_train[i], stride=stride, kernel_size=kernel_size, encoding_gate_parameter_size=encoding_gate_parameter_size) for i in range(num_train)]
    x_test, y_test = features[test_indices], labels[test_indices]
    x_test = [extract_convolution_data(x_test[i], stride=stride, kernel_size=kernel_size, encoding_gate_parameter_size=encoding_gate_parameter_size) for i in range(num_test)]
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
                             encoding_gate_parameter_size:int=3) -> List[List[List[float]]]:
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

def load_fashion_mnist(path, kind='train'):
    # from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

