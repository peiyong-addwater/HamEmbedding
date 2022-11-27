import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
from typing import List, Tuple, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import jax

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import optax  # optimization using jax

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
                             stride:Tuple[int, int] = (1, 1),
                             dilation:Tuple[int, int]=(1, 1),
                             padding: Tuple[int, int]=(0,0)) -> np.ndarray:
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
            row.append(submatrix.flatten().tolist())
        output.append(row)
    return np.array(output)

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

def SU4(params, wires):
    """
    A 15-parameter SU4 gate, from FIG. 6 of PHYSICAL REVIEW RESEARCH 4, 013117 (2022)
    :param params:
    :param wires:
    :return:
    """
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=wires)
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])

def su4_single_conv_encoding(theta,w, data, wires):
    """
    A two-qubit encoding gate using the compact data re-upload approach, with the SU4 layer.
    Each SU4 layer has 15 parameters. With the compact approach, it can encode 15 data points for a sinle layer.
    parameters passed to the quantum gates = theta + w.x, where . is the elementwise multiplication.
    :param theta:
    :param w:
    :param data:
    :param wires:
    :return:
    """
    num_layers = len(data)//15+1
    data_pad_size = len(data)-15*num_layers
    padded_data = jnp.array(data)
    for _ in range(data_pad_size):
        padded_data = jnp.append(padded_data, 0)
    gate_params = jnp.array(theta) + jnp.multiply(jnp.array(w), padded_data)
    for i in range(num_layers):
        SU4(gate_params[15*i:15*(i+1)], wires=wires)

def conv_repuload_encoding(theta, w, data):
    num_row, num_columns = data.shape[0], data.shape[1]
    for i in range(0, num_row, 2):
        encode_data = data[i]
        for j in range(num_columns):
            su4_single_conv_encoding(theta, w, encode_data[j], wires=[i, i+1])
    for i in range(1, num_row-2, 2):
        encode_data = data[i]
        for j in range(num_columns):
            su4_single_conv_encoding(theta, w, encode_data[j], wires=[i, i+1])

def pooling_layer(weights, wires):
    """ from https://pennylane.ai/qml/demos/tutorial_learning_few_data.html
    Adds a pooling layer to a circuit.
    Args:
        weights (np.array): Array with the weights of the conditional U3 gate.
        wires (list[int]): List of wires to apply the pooling layer on.
    """
    n_wires = len(wires)
    assert len(wires) >= 2, "this circuit is too small!"

    for indx, w in enumerate(wires):
        if indx % 2 == 1 and indx < n_wires:
            m_outcome = qml.measure(w)
            qml.cond(m_outcome, qml.U3)(*weights, wires=wires[indx - 1])

def convolutional_layer(weights, wires, skip_first_layer=True):
    """from https://pennylane.ai/qml/demos/tutorial_learning_few_data.html
    Adds a convolutional layer to a circuit.
    Args:
        weights (np.array): 1D array with 15 weights of the parametrized gates.
        wires (list[int]): Wires where the convolutional layer acts on.
        skip_first_layer (bool): Skips the first two U3 gates of a layer.
    """
    n_wires = len(wires)
    assert n_wires >= 3, "this circuit is too small!"

    for p in [0, 1]:
        for indx, w in enumerate(wires):
            if indx % 2 == p and indx < n_wires - 1:
                if indx % 2 == 0 and not skip_first_layer:
                    qml.U3(*weights[:3], wires=[w])
                    qml.U3(*weights[3:6], wires=[wires[indx + 1]])
                qml.IsingXX(weights[6], wires=[w, wires[indx + 1]])
                qml.IsingYY(weights[7], wires=[w, wires[indx + 1]])
                qml.IsingZZ(weights[8], wires=[w, wires[indx + 1]])
                qml.U3(*weights[9:12], wires=[w])
                qml.U3(*weights[12:], wires=[wires[indx + 1]])

def conv_and_pooling(kernel_weights, n_wires, skip_first_layer=True):
    """from https://pennylane.ai/qml/demos/tutorial_learning_few_data.html
    Apply both the convolutional and pooling layer.
    requires 15+3 = 18 parameters"""
    convolutional_layer(kernel_weights[:15], n_wires, skip_first_layer=skip_first_layer)
    pooling_layer(kernel_weights[15:], n_wires)
def dense_layer(weights, wires):
    """from https://pennylane.ai/qml/demos/tutorial_learning_few_data.html
    Apply an arbitrary unitary gate to a specified set of wires."""
    qml.ArbitraryUnitary(weights, wires)

if __name__ == '__main__':
    import seaborn as sns

    sns.set()

    seed = 42
    rng = np.random.default_rng(seed=seed)

    KERNEL_SIZE = (3, 3)
    STRIDE = (3, 3)
    NUM_CONV_POOL_LAYERS = 2
    FINAL_LAYER_QUBITS = 3

    n_test = 1000
    n_epochs = 100
    n_reps = 10

    _, _, _, num_conv_rows, _ = _check_params(np.random.rand(28 * 28).reshape(28, 28), kernel=np.random.random(KERNEL_SIZE),
                                          stride=STRIDE, dilation=(1, 1), padding=(0, 0))
    num_wires = num_conv_rows + 1
    print(num_wires)
    device = qml.device("default.qubit", wires=num_wires)
