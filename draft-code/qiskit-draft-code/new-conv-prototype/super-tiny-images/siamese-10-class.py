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

shutup.please()

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
        with open("../../../../data/mini-digits/tiny-handwritten.pkl", 'rb') as f:
            mnist = pickle.load(f)
    if kind == 'train':
        return mnist["training_images"], mnist["training_labels"]
    else:
        return mnist["test_images"], mnist["test_labels"]

def load_data(num_train, num_test, rng, one_hot=True):
    data_path = "../../../../data/mini-digits/tiny-handwritten.pkl"
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
    x_train = [extract_convolution_data(x_train[i], kernel_size=(5, 5), stride=(3, 3), dilation=(1, 1),
                                        padding=(0, 0), encoding_gate_parameter_size=15) for i in range(num_train)]
    x_test, y_test = features[test_indices], labels[test_indices]
    x_test = [extract_convolution_data(x_test[i], kernel_size=(5, 5), stride=(3, 3), dilation=(1, 1),
                                       padding=(0, 0), encoding_gate_parameter_size=15) for i in range(num_test)]
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

def select_data(labels=[0,1,2,3,4,5,6,7,8,9], num_data_per_label_train = 3, num_test_per_label=1, rng=np.random.default_rng(seed=42)):
    data_path = "../../../../data/mini-digits/tiny-handwritten.pkl"
    features, all_labels = load_tiny_digits(data_path)
    features = np.array(features)
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

def kernel_5x5(padded_data_in_kernel_view, conv_params, pooling_params):
    """
    45 + 18 parameters
    :param padded_data_in_kernel_view:
    :param conv_params:
    :param pooling_params:
    :return:
    """
    qreg = QuantumRegister(4, name="conv-pooling")
    creg = ClassicalRegister(3, name='pooling-meas')
    circ = QuantumCircuit(qreg, creg, name = "conv-encode-5x5")

    circ.h(qreg)
    # encode the pixel data
    circ.compose(su4_circuit(padded_data_in_kernel_view[:15]), qubits=qreg[:2], inplace=True)
    circ.compose(su4_circuit(padded_data_in_kernel_view[15:]), qubits=qreg[2:], inplace=True)
    # convolution parameters
    circ.compose(su4_circuit(conv_params[:15]), qubits=[qreg[1], qreg[2]], inplace=True)
    circ.compose(su4_circuit(conv_params[15:30]), qubits=[qreg[0], qreg[1]], inplace=True)
    circ.compose(su4_circuit(conv_params[30:45]), qubits=[qreg[2], qreg[3]], inplace=True)
    # measurement and pooling
    circ.measure(qreg[1:], creg)
    circ.u(pooling_params[0], pooling_params[1], pooling_params[2], qubit=qreg[0]).c_if(creg[0], 1)
    circ.u(pooling_params[3], pooling_params[4], pooling_params[5], qubit=qreg[0]).c_if(creg[0], 0)
    circ.u(pooling_params[6], pooling_params[7], pooling_params[8], qubit=qreg[0]).c_if(creg[1], 1)
    circ.u(pooling_params[9], pooling_params[10], pooling_params[11], qubit=qreg[0]).c_if(creg[1], 0)
    circ.u(pooling_params[12], pooling_params[13], pooling_params[14], qubit=qreg[0]).c_if(creg[2], 1)
    circ.u(pooling_params[15], pooling_params[16], pooling_params[17], qubit=qreg[0]).c_if(creg[2], 0)
    # reset the last three qubits
    circ.barrier(qreg)
    circ.reset(qreg[1])
    circ.reset(qreg[2])
    circ.reset(qreg[3])



    return circ

def conv_layer_1(data_for_entire_2x2_feature_map, params):
    """
    conv-1 requires 45 + 18 parameters
    8 qubits
    :param data_for_first_row_of_5x5_feature_map:
    :param params:
    :return:
    """
    qreg = QuantumRegister(7, name='conv')
    creg = ClassicalRegister(3, name="pooling-meas")
    circ = QuantumCircuit(qreg, creg, name='conv-layer-1')
    conv_kernel_param = params[:45]
    pooling_param = params[45:]
    qubit_counter = 0
    for i in range(2):
        for j in range(2):
            conv_op = kernel_5x5(data_for_entire_2x2_feature_map[i][j], conv_kernel_param, pooling_param)#.to_instruction(label="Conv5x5")
            circ.compose(conv_op, qubits=qreg[qubit_counter:qubit_counter + 4], clbits=creg, inplace=True)
            circ.barrier(qreg)
            qubit_counter+=1
    return circ

def full_circ(prepared_data_twin, params):
    """
    45 + 18 parameters
    :param prepared_data_twin:
    :param params:
    :return:
    """
    network_qreg = QuantumRegister(4+7, name="network-qreg")
    pooling_measure = ClassicalRegister(3, name='pooling-measure')
    swap_test_creg = ClassicalRegister(1, name="swap-test-meas")
    circ = QuantumCircuit(network_qreg, pooling_measure, swap_test_creg)

    img0, label0 = prepared_data_twin[0]
    img1, label1 = prepared_data_twin[1]

    # network for image 0
    circ.compose(conv_layer_1(img0, params), qubits=network_qreg[:7], clbits=pooling_measure, inplace=True)
    circ.barrier(network_qreg)
    # network for image 1
    circ.compose(conv_layer_1(img1, params), qubits=network_qreg[4:], clbits=pooling_measure,inplace=True)
    circ.barrier(network_qreg)
    # swap test
    circ.h(network_qreg[-1])
    for i in range(4):
        circ.cswap(control_qubit=network_qreg[-1], target_qubit1=network_qreg[i], target_qubit2=network_qreg[i+4])
    circ.h(network_qreg[-1])
    circ.measure(network_qreg[-1], swap_test_creg)

    return circ

# draw the conv 1 layer
# data0 = []
# for i in range(2):
#     row = []
#     for j in range(2):
#         row.append(ParameterVector(f"x0_{i}{j}", length=30))
#     data0.append(row)
#
# data1 = []
# for i in range(2):
#     row = []
#     for j in range(2):
#         row.append(ParameterVector(f"x1_{i}{j}", length=30))
#     data1.append(row)
# parameter_conv_1 = ParameterVector("θ", length=45 + 18)
# data = ((data0,0), (data1,1))
# siamese_circ = full_circ(data, parameter_conv_1)
# siamese_circ.draw(output='mpl', filename='siamese-conv-5x5.png', style='bw', fold=-1, scale=0.5)
# params = np.random.random(45+18)
# extracted_data = select_data()
# one_data = extracted_data[0][0]
# siamese_full = full_circ(one_data, params)
# backend_sim = Aer.get_backend('aer_simulator')
# convnet = transpile(siamese_full, backend_sim)
# job = backend_sim.run(convnet, shots=2048)
# results = job.result()
# counts = results.get_counts()
# print(counts)
# swap_test_counts = {"0":0, "1":0}
# for key in counts.keys():
#     swap_test_meas = key.split(' ')[0]
#     swap_test_counts[swap_test_meas] += counts[key]
# print(swap_test_counts)
def get_state_overlap_from_counts(counts:dict):
    swap_test_counts = {"0": 0, "1": 0}
    for key in counts.keys():
        swap_test_meas = key.split(' ')[0]
        swap_test_counts[swap_test_meas] += counts[key]
    prob_0 = swap_test_counts['0']/sum(swap_test_counts.values())
    return 2*prob_0-1

def single_data_pair_overlap_sim(params, data, shots = 2048):
    backend_sim = Aer.get_backend('aer_simulator')
    convnet = transpile(full_circ(data, params), backend_sim)
    job = backend_sim.run(convnet, shots=shots)
    results = job.result()
    counts = results.get_counts()
    return get_state_overlap_from_counts(counts)

if __name__ == '__main__':
    import matplotlib as mpl
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import json

    NUM_SHOTS = 2048
    N_WORKERS = 8
    MAX_JOB_SIZE = 10

    BACKEND_SIM = Aer.get_backend('aer_simulator')
    EXC = ThreadPoolExecutor(max_workers=N_WORKERS)
    BACKEND_SIM.set_options(executor=EXC)
    BACKEND_SIM.set_options(max_job_size=MAX_JOB_SIZE)
    BACKEND_SIM.set_options(max_parallel_experiments=0)
    seed = 42
    rng = np.random.default_rng(seed=seed)
    KERNEL_SIZE = (5, 5)
    STRIDE = (3, 3)
    n_epochs = 500




