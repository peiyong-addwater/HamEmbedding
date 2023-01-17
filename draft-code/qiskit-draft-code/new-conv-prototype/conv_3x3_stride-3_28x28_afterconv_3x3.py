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
    x_train = [prepare_data_28x28_9x9_3x3_kernel_3x3(x_train[i]) for i in range(num_train)]
    x_test, y_test = features[test_indices], labels[test_indices]
    x_test = [prepare_data_28x28_9x9_3x3_kernel_3x3(x_test[i]) for i in range(num_test)]
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

def prepare_data_28x28_9x9_3x3_kernel_3x3(img_matrix):
    extracted_conv_data = extract_convolution_data(img_matrix, kernel_size=(3,3), stride=(3,3),dilation=(1,1), padding=(0,0), encoding_gate_parameter_size=3)
    rearranged_data = []
    for i in [0, 1, 2]:
        for j in [0,1,2]:
            rearranged_data.append(extracted_conv_data[i][j])
        for j in [3,4,5]:
            rearranged_data.append(extracted_conv_data[i][j])
        for j in [6,7,8]:
            rearranged_data.append(extracted_conv_data[i][j])
    for i in [3,4,5]:
        for j in [0,1,2]:
            rearranged_data.append(extracted_conv_data[i][j])
        for j in [3,4,5]:
            rearranged_data.append(extracted_conv_data[i][j])
        for j in [6,7,8]:
            rearranged_data.append(extracted_conv_data[i][j])
    for i in [6,7,8]:
        for j in [0,1,2]:
            rearranged_data.append(extracted_conv_data[i][j])
        for j in [3,4,5]:
            rearranged_data.append(extracted_conv_data[i][j])
        for j in [6,7,8]:
            rearranged_data.append(extracted_conv_data[i][j])
    return rearranged_data




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

def kernel_3x3(data_in_kernel_view, conv_params, pooling_params):
    """
    A three-qubit circuit that takes 9 data pixels, with 9 convolution parameters, and 12 parameters for pooling.
    :param data_in_kernel_view:
    :param conv_params:
    :param pooling_params:
    :return:
    """
    qreg = QuantumRegister(3, name='conv-pooling')
    creg = ClassicalRegister(2, name="pooling-meas")
    circ = QuantumCircuit(qreg, creg, name='conv-encode-3x3')

    circ.h(qreg)
    # encode the pixel data
    circ.u(data_in_kernel_view[0], data_in_kernel_view[1], data_in_kernel_view[2], qubit=qreg[0])
    circ.u(data_in_kernel_view[3], data_in_kernel_view[4], data_in_kernel_view[5], qubit=qreg[1])
    circ.u(data_in_kernel_view[6], data_in_kernel_view[7], data_in_kernel_view[8], qubit=qreg[2])
    # conv parameters
    circ.u(conv_params[0], conv_params[1], conv_params[2], qubit=qreg[0])
    circ.u(conv_params[3], conv_params[4], conv_params[5], qubit=qreg[1])
    circ.u(conv_params[6], conv_params[7], conv_params[8], qubit=qreg[2])

    circ.barrier(qreg)

    # measurement and pooling
    circ.measure(qreg[1], creg[0])
    circ.measure(qreg[2], creg[1])
    circ.u(pooling_params[0], pooling_params[1], pooling_params[2], qubit=qreg[0]).c_if(creg[0], 1)
    circ.u(pooling_params[3], pooling_params[4], pooling_params[5], qubit=qreg[0]).c_if(creg[0], 0)
    circ.u(pooling_params[6], pooling_params[7], pooling_params[8], qubit=qreg[0]).c_if(creg[1], 1)
    circ.u(pooling_params[9], pooling_params[10], pooling_params[11], qubit=qreg[0]).c_if(creg[1], 0)
    # reset the last two qubits
    circ.barrier(qreg)
    circ.reset(qreg[1])
    circ.reset(qreg[2])

    return circ

# draw the kernel circuit
# data_in_kernel = ParameterVector("x", length=9)
# kernel_param = ParameterVector("θ", length=9)
# pooling_param = ParameterVector("p", length=12)
# kernel_circ = kernel_3x3(data_in_kernel, kernel_param, pooling_param)
# kernel_circ.draw(output='mpl', style='bw', filename="kernel.png", fold=-1)

def conv_layer_1(data_in_kernel_on_first_feature_map, params):
    """

    :param data_in_kernel_on_first_feature_map: should be a list of 9 lists which contains the data involved to produce
    a 3 by 3 section of the 9 by 9 feature map
    :param params: conv and pooling parameters, 9+12=21
    :return:
    """
    qreg = QuantumRegister(11, name='conv')
    creg = ClassicalRegister(2, name = "pooling-meas")
    circ = QuantumCircuit(qreg, creg, name='conv-layer-1')
    conv_kernel_param = params[:9]
    pooling_param = params[9:]
    for i in range(9):
        conv_op = kernel_3x3(data_in_kernel_on_first_feature_map[i], conv_kernel_param, pooling_param)
        circ.compose(conv_op, qubits=qreg[i:i+3], clbits=creg, inplace=True)
        circ.barrier(qreg)
    return circ

# # draw the conv 1 layer
# data = []
# for i in range(9):
#     single_qubit_data = []
#     for j in range(9):
#         single_qubit_data.append(ParameterVector(f"x_{i}{j}", length=9))
#     data.append(single_qubit_data)
# parameter_conv_1 = ParameterVector("θ", length=9+12)
# # data in view (for the second feature map)
# data_in_view = []
# for i in [0,1,2]:
#     for j in [0,1,2]:
#         data_in_view.append(data[i][j])
#
# first_conv_layer = conv_layer_1(data_in_view, parameter_conv_1)
# first_conv_layer.draw(output='mpl', filename='conv_1.png', style='bw', fold=-1)

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

def ising_u3(params):
    circ = QuantumCircuit(2, name='u3-ising-u3')
    circ.u(params[0], params[1], params[2], 0)
    circ.u(params[3], params[4], params[5], 1)
    circ.rxx(params[6], 0, 1)
    circ.ryy(params[7], 0, 1)
    circ.rzz(params[8], 0, 1)
    circ.u(params[9], params[10], params[11], 0)
    circ.u(params[12], params[13], params[14], 1)
    return circ

def three_qubit_conv(params):
    """
    a convolution-pooling layer with 2 two-qubit blocks operating on three qubits.
    first two of the three qubits will be measured and u3 gates controlled by the measurement results
    this can be viewed as the third conv layer
    :param params:
    :return:
    """
def conv_layer_2(params):
    """
    each two-qubit block requires 15 parameters.
    8 blocks in total, 8*15 = 120 parameters
    :param params:
    :return:
    """
    qreg = QuantumRegister(11, name='conv')
    circ = QuantumCircuit(qreg, name='conv-layer-2')
    for i in range(8):
        circ.compose(ising_u3(params[i*15:15*(i+1)]), qubits=[qreg[i], qreg[i+1]], inplace=True)
        circ.barrier(qreg)
    # swap the data to the last qubit
    circ.swap(qreg[8], qreg[9])
    circ.swap(qreg[9], qreg[10])
    circ.barrier(qreg)
    # reset all qubits except for the last one
    for i in range(10):
        circ.reset(qreg[i])
    return circ

def conv_1_and_2(data_in_second_kernel_view, params):
    """

    :param data_in_second_kernel_view:
    :param params:
    :return:
    """
    qreg = QuantumRegister(11, name='conv')
    creg = ClassicalRegister(2, name="pooling-meas")
    circ = QuantumCircuit(qreg, creg, name='conv-layer-1-2')
    conv_1_params = params[:21]
    conv_2_params = params[21:]
    conv_1 = conv_layer_1(data_in_second_kernel_view, conv_1_params)#.to_instruction()
    circ.compose(conv_1, qubits=qreg, clbits=creg, inplace=True)
    conv_2 = conv_layer_2(conv_2_params)
    circ.compose(conv_2, qubits=qreg, inplace=True)
    circ.barrier(qreg)
    return circ

# # draw the conv 1 and 2 layer
# data = []
# for i in range(9):
#     single_qubit_data = []
#     for j in range(9):
#         single_qubit_data.append(ParameterVector(f"x_{i}{j}", length=9))
#     data.append(single_qubit_data)
# parameter_conv_1_2 = ParameterVector("θ", length=9+12+120)
# # data in view (for the second feature map)
# rng = np.random.default_rng(seed=42)
# data = load_data(10,10,rng)[0][0]
# #
# conv_layer = conv_1_and_2(data[0:9], parameter_conv_1_2)
# conv_layer.draw(output='mpl', filename='conv_1_and_2_with_data.png', style='bw', fold=-1)

def full_circ(prepared_data, params):
    """
    conv 1 & 2 need 9+12+120 = 141 parameters in total
    to reduce the number of qubits, we also adopt an asynchronized structure to process the 3x3 feature map.
    this part of the circuit requires 3 convolution-like operations
    :param prepared_data:
    :param params:
    :return:
    """

