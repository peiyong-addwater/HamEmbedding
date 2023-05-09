import os.path

import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA
import json
import time
import shutup
import pickle

from qiskit.circuit import ParameterVector

shutup.please()

from SPSAGradOptimiser.qiskit_opts.SPSA_Adam import ADAMSPSA
from qiskit_ibm_provider import IBMProvider
PROVIDER = IBMProvider()

DATA_PATH = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/tiny-handwritten.pkl"

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
        with open("/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/tiny-handwritten.pkl", 'rb') as f:
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
        extracted_data = [extract_convolution_data(data_for_label[train_indices][i], kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1),
                                        padding=(0, 0), encoding_gate_parameter_size=9) for i in range(num_data_per_label_train)]
        for i in range(num_data_per_label_train):
            selected_train_images.append((extracted_data[i], label))
        test_images['data'].append(extract_convolution_data(data_for_label[test_indices][0], kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1),
                                        padding=(0, 0), encoding_gate_parameter_size=9))
        test_images['labels'].append(label)

    # pair the training data
    """paired_extracted_data = []
    for i in range(len(selected_train_images)):
        for j in range(len(selected_train_images)):
            if i!=j:
                paired_extracted_data.append((selected_train_images[i], selected_train_images[j]))
    """
    return selected_train_images #, test_images

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

def su4_9_params(params):
    su4 = QuantumCircuit(2, name='su4-9-params')
    su4.cx(0, 1)
    su4.ry(params[0], 0)
    su4.rz(params[1], 1)
    su4.cx(1, 0)
    su4.ry(params[2], 0)
    su4.cx(0, 1)
    su4.u(params[3], params[4], params[6], 0)
    su4.u(params[6], params[7], params[8], 1)
    su4_inst = su4.to_instruction()
    su4 = QuantumCircuit(2)
    su4.append(su4_inst, list(range(2)))
    return su4

# draw the 9-param version of su4
param_9 = ParameterVector('θ', 9)
su4_9p = su4_9_params(param_9)
su4_9p.decompose().draw(output='mpl', filename='su4_9p.png', style='bw')

def three_q_interaction(params):
    """
    The three-qubit interaction circuit, bottom-to-top
    :param params: 15+15-3 = 27 parameters
    :return:
    """
    circ = QuantumCircuit(3, name="three-qubit-interaction")
    circ.compose(su4_circuit(params[0:15]), [1, 2], inplace=True)
    circ.u(params[15], params[16], params[17], 0)
    circ.cx(0, 1)
    circ.ry(params[18], 0)
    circ.rz(params[19], 1)
    circ.cx(1, 0)
    circ.ry(params[20], 0)
    circ.cx(0, 1)
    circ.u(params[21], params[22], params[23], 0)
    circ.u(params[24], params[25], params[26], 1)
    circ_inst = circ.to_instruction()
    circ = QuantumCircuit(3)
    circ.append(circ_inst, list(range(3)))
    return circ

# draw the three-qubit interaction circuit
param_27 = ParameterVector('θ', 27)
three_q_circ = three_q_interaction(param_27)
three_q_circ.decompose().draw(output='mpl', filename='three_q_circ.png', style='bw', fold=-1)

def memory_cell(params):
    """
    a four-qubit memory cell, with the 4th qubit reset and awaiting new data
    :param params: 15+4+27 = 46 parameters
    :return:
    """
    circ = QuantumCircuit(4, name="memory-cell")
    circ.compose(su4_circuit(params[0:15]), [2, 3], inplace=True)
    circ.cu(params[15], params[16], params[17], params[18], 3, 2)
    circ.reset(3)
    circ.compose(three_q_interaction(params[19:46]), [0, 1, 2], inplace=True)
    circ_inst = circ.to_instruction()
    circ = QuantumCircuit(4)
    circ.append(circ_inst, list(range(4)))
    return circ

# draw the memory cell
param_46 = ParameterVector('θ', 46)
mem_circ = memory_cell(param_46)
mem_circ.decompose().draw(output='mpl', filename='mem_circ.png', style='bw', fold=-1)

def kernel_3x3x1(padded_data_in_kernel, conv_pooling_params):
    """

    :param padded_data_in_kernel: 3x3 = 9
    :param conv_pooling_params: 9+9+4+9+4 = 35
    :return:
    """
    qreg = QuantumRegister(3, name="conv-pooling")
    circ = QuantumCircuit(qreg, name="conv-encode-3x3")
    circ.h(qreg)
    circ.barrier()
    # encode the pixel data into the rotation parameter of U3 gates
    circ.u(padded_data_in_kernel[0], padded_data_in_kernel[1], padded_data_in_kernel[2], qreg[0])
    circ.u(padded_data_in_kernel[3], padded_data_in_kernel[4], padded_data_in_kernel[5], qreg[1])
    circ.u(padded_data_in_kernel[6], padded_data_in_kernel[7], padded_data_in_kernel[8], qreg[2])
    circ.barrier()
    # convolution kernel with two 9-param su4 gates
    circ.compose(su4_9_params(conv_pooling_params[0:9]), [qreg[0], qreg[1]], inplace=True)
    circ.compose(su4_9_params(conv_pooling_params[9:18]), [qreg[1], qreg[2]], inplace=True)
    circ.barrier()
    # pooling with cu gates
    circ.cu(conv_pooling_params[18], conv_pooling_params[19], conv_pooling_params[20], conv_pooling_params[21], qreg[2], qreg[1])
    circ.barrier()
    # 9-param su4 on the first two qubits
    circ.compose(su4_9_params(conv_pooling_params[22:31]), [qreg[0], qreg[1]], inplace=True)
    circ.barrier()
    # pooling with cu gates
    circ.cu(conv_pooling_params[31], conv_pooling_params[32], conv_pooling_params[33], conv_pooling_params[34], qreg[1], qreg[0])
    circ.barrier()
    # reset qubits 1 and 2
    circ.reset(qreg[1])
    circ.reset(qreg[2])

    circ_inst = circ.to_instruction()
    circ = QuantumCircuit(3)
    circ.append(circ_inst, list(range(3)))

    return circ

# draw the 3x3x1 kernel circuit
data_single_kernel = ParameterVector('x', 9)
conv_pooling_params = ParameterVector('θ', 35)
kernel_3x3x1(data_single_kernel, conv_pooling_params).decompose().draw(output='mpl', filename='kernel_3x3x1.png', style='bw', fold=-1)

def encode(data_for_complete_feature_map, params):
    """
    An RNN-like encoding circuit to encode an 8x8 image, with 3x3 kernel, stride=1
    :param data_for_complete_feature_map: length 9 list of length 9 lists
    :param params: 15+35+27+46=123 parameters
    :return:
    """
    circ = QuantumCircuit(6, name="encode")
    circ.h(0)
    circ.h(1)
    circ.barrier()
    circ.compose(su4_circuit(params[0:15]), [0, 1], inplace=True)
    circ.compose(kernel_3x3x1(data_for_complete_feature_map[0], params[15:50]), [2, 3, 4], inplace=True)
    circ.compose(three_q_interaction(params[50:77]), [0, 1, 2], inplace=True)
    circ.barrier()
    for index in range(1, 6 * 6):
        circ.compose(kernel_3x3x1(data_for_complete_feature_map[index], params[15:50]), [3,4,5], inplace=True)
        circ.compose(memory_cell(params[77:123]), [0, 1, 2, 3], inplace=True)
        circ.barrier()
    circ_inst = circ.to_instruction()
    circ = QuantumCircuit(6)
    circ.append(circ_inst, list(range(6)))
    return circ

# draw the encoding circuit
data = []
for i in range(6):
    for j in range(6):
        data.append(ParameterVector(f"x_{i}{j}", length=9))

parameter_encode = ParameterVector('θ', 123)
encode(data, parameter_encode).decompose().draw(output='mpl', filename='encode.png', style='bw')

def backbone_qnn(data_for_complete_feature_map, params):
    """
    The backbone QNN to generate the feature map.
    First section of the circuit is the encoding circuit, which is an RNN-like circuit with 123 parameters
    After the encoding circuit, the first three qubits will contain information for the image, the rest qubits (last three) will be fresh.
    :param data_for_complete_feature_map:
    :param params: 123+3*15=168 parameters
    :return:
    """
    circ = QuantumCircuit(6, 6, name="backbone")
    circ.compose(encode(data_for_complete_feature_map, params[0:123]), list(range(6)), inplace=True)
    circ.barrier()
    for i in range(3):
        circ.compose(su4_circuit(params[123 + i * 15:123 + (i + 1) * 15]), [i, 3 + i], inplace=True)
    circ.barrier()
    circ.measure([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    return circ

# draw the backbone QNN
parameter_qnn = ParameterVector('θ', 168)
backbone_qnn(data, parameter_qnn).draw(output='mpl', filename='backbone_qnn.png', style='bw')




