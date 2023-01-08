import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit import Aer
from dask.distributed import LocalCluster, Client
from concurrent.futures import ThreadPoolExecutor
from noisyopt import minimizeSPSA
from qiskit.algorithms.optimizers import COBYLA, SPSA
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

# draw the su4 circuit
# params_su4_draw = ParameterVector("θ", length=15)
# circuit_su4_draw = su4_circuit(params_su4_draw).decompose()
# circuit_su4_draw.draw(output='mpl', filename="su4_circuit.png",style='bw')

def single_kernel_encoding(kernel_params, data_in_kernel_view):
    """
    Size of the data_params should be the same as the size of the kernel_params
    Encoding with U3 gates
    :param kernel_params: Should be an integer times of number of parameter in a single encoding unitary
    :param data_in_kernel_view: needs to be padded
    :return:
    """
    num_combo_gates = len(kernel_params)//3
    encoding_circ = QuantumCircuit(1, name="Single Kernel Encoding")
    for i in range(num_combo_gates):
        encoding_circ.u(data_in_kernel_view[3 * i], data_in_kernel_view[3 * i + 1], data_in_kernel_view[3 * i + 2], 0)
        encoding_circ.u(kernel_params[3*i], kernel_params[3*i+1], kernel_params[3*i+2], 0)
    circ_inst = encoding_circ.to_instruction()
    encoding_circ = QuantumCircuit(1)
    encoding_circ.append(circ_inst, [0])
    return encoding_circ

# draw the single kernel encoding circuit.
# kernel_params_draw = ParameterVector("θ", length=9)
# data_draw = ParameterVector("x", length=9)
# ske_circuit = single_kernel_encoding(kernel_params_draw, data_draw).decompose()
# ske_circuit.draw(output='mpl', filename='single-kernel-encoding-circuit.png', style='bw')

def convolution_reupload_encoding(kernel_params, data):
    num_qubits, num_conv_per_qubit = len(data), len(data[0])
    encoding_circ = QuantumCircuit(num_qubits, name="Encoding Layer")
    for j in range(num_conv_per_qubit):
        for i in range(num_qubits):
            single_qubit_data = data[i]
            encoding_circ = encoding_circ.compose(single_kernel_encoding(kernel_params, single_qubit_data[j]), qubits=[i])
    inst = encoding_circ.to_instruction()
    encoding_circ = QuantumCircuit(num_qubits)
    encoding_circ.append(inst, list(range(num_qubits)))

    return encoding_circ

# draw the full encoding circuit corresponding to a 9 by 9 feature map
# kernel_params_draw = ParameterVector("θ", length=9)
# data
# data = []
# for i in range(9):
#     single_qubit_data = []
#     for j in range(9):
#         single_qubit_data.append(ParameterVector(f"x_{i}{j}", length=9))
#     data.append(single_qubit_data)
# conv_encode_circ = convolution_reupload_encoding(kernel_params_draw, data).decompose()
# conv_encode_circ.draw(output='mpl', style='bw', filename="conv_encoding_9x9_feature_map.png", fold=-1)

def entangling_after_encoding(params):
    """
    Entangling layer with su4 gates
    :param params:
    :return:
    """
    num_qubits = len(params) // 15 + 1
    circ = QuantumCircuit(num_qubits, name="Entangling Layer")
    for i in range(num_qubits-1):
        circ.compose(su4_circuit(params[15*i:15*(i+1)]), [i, i+1], inplace=True)
    circ_inst = circ.to_instruction()
    circ = QuantumCircuit(num_qubits)
    circ.append(circ_inst, list(range(num_qubits)))
    return circ

# draw the entangling layer
# entangling_params = ParameterVector("θ", length=15*8) # 9 qubits
# entangling_circ = entangling_after_encoding(entangling_params).decompose()
# entangling_circ.draw(output='mpl', style='bw', filename='entangling_after_encoding.png', fold=-1)

def convolution_layer(params):
    """
    same structure as entangling_after_encoding
    :param params:
    :return:
    """
    num_qubits = len(params) // 15 + 1
    circ = QuantumCircuit(num_qubits, name="Convolution Layer")
    for i in range(num_qubits - 1):
        circ.compose(su4_circuit(params[15 * i:15 * (i + 1)]), [i, i + 1], inplace=True)
    circ_inst = circ.to_instruction()
    circ = QuantumCircuit(num_qubits)
    circ.append(circ_inst,list(range(num_qubits)))
    return circ

def simple_conv_net_9x9_encoding_4_class(params, single_image_data):
    """

    encoding has 9 parameters;

    entangling has 15*8 = 120 parameters;

    first pooling layer utilizes the information on the bottom 4 qubits. The bottom 4 qubits will act as control qubits,
    and the target will be the first four qubits (we'll leave the fifth qubit out). Each gate on the target qubit will
    be a U3 gate. Total parameters: 4*3 = 12

    after the first pooling layer, there will be 5 qubits left (first 5 qubits). Then there will be 4 SU4 layers between
    qubits (0,1), (1,2), (2,3), (3,4) acting as a convolutional layer. Total number of parameters: 15*4 = 60

    second pooling layer utilizes the information on qubits 2 and 3, which acts as control qubits for the U3 gates on
    qubits 0 and 1. Total number of parameters: 6

    after the second pooling layer, a SU4 layer will be acting on the first two qubits. Total number of parameters: 15

    total number of parameters is 9+120+12+60+6+15 = 222
    :param params:
    :param single_image_data:
    :return:
    """
    qreg = QuantumRegister(9)
    prob_meas = ClassicalRegister(2, name='classification')

    circ = QuantumCircuit(qreg, prob_meas)

    # data re-uploading layer
    circ.compose(convolution_reupload_encoding(params[:9], single_image_data), qubits=qreg, inplace=True)
    # entangling after encoding layer
    circ.compose(entangling_after_encoding(params[9:9+15*8]), qubits=qreg, inplace=True)
    # first pooling layer
    first_pooling_params = params[9+15*8:9+15*8+12]
    for i in range(4):
        circ.cu3(first_pooling_params[3*i], first_pooling_params[3*i+1], first_pooling_params[3*i+2], control_qubit=qreg[i+5], target_qubit=qreg[i])
    # convolution layer
    first_convolution_params = params[9+15*8+12:9+15*8+12+15*4]
    for i in range(4):
        circ.compose(su4_circuit(first_convolution_params[15*i:15*i+15]), qubits=[qreg[i], qreg[i+1]], inplace=True)
    # second pooling layer
    second_pooling_params = params[9+15*8+12+15*4:9+15*8+12+15*4+6]
    for i in range(2):
        circ.cu3(second_pooling_params[3*i], second_pooling_params[3*i+1], second_pooling_params[3*i+2], control_qubit=qreg[i+2], target_qubit=qreg[i])
    # second conv layer
    second_conv_params = params[9+15*8+12+15*4+6:]
    circ.compose(su4_circuit(second_conv_params), qubits=[qreg[0], qreg[1]], inplace=True)


    # output classification probabilities
    circ.measure(qreg[:2], prob_meas)

    return circ

# draw the conv net
# data = []
# for i in range(9):
#     single_qubit_data = []
#     for j in range(9):
#         single_qubit_data.append(ParameterVector(f"x_{i}{j}", length=9))
#     data.append(single_qubit_data)
# parameter_convnet = ParameterVector("θ", length=222)
# convnet_draw = simple_conv_net_9x9_encoding_4_class(parameter_convnet, data)
# convnet_draw.draw(output='mpl', filename='simple_conv_net_9x9_encoding_4_class.png', style='bw', fold=-1)
#
# exit(0)

# run the circuit with random data, see what kind of measurements will appear in the output
# backend_sim = Aer.get_backend('aer_simulator')
# seed = 42
# rng = np.random.default_rng(seed=seed)
# data = load_data(10,10,rng)[0][0]
# print(len(data), len(data[0]))
# labels = load_data(10,10,rng)[1]
# print(labels)
# parameter_convnet = np.random.random(222)
# sample_run_convnet = transpile(simple_conv_net_9x9_encoding_4_class(parameter_convnet, data), backend_sim)
# job = backend_sim.run(sample_run_convnet, shots = 4096)
# results = job.result()
# counts = results.get_counts()
# print(counts)
# print(list(counts.keys())[0].split(' '))
# probs = [0]*4
# for key in counts.keys():
#     classification = int(key.split(' ')[0], 2)
#     probs[classification] = probs[classification]+counts[key]
# probs = [c/sum(probs) for c in probs]
# print(probs, sum(probs))
# exit(0)

def get_probs_from_counts(counts, num_classes=4):
    """
    for count keys like '00 011 0010', where the first two digits are the class
    :param counts:
    :param num_classes:
    :return:
    """
    probs = [0]*num_classes
    for key in counts.keys():
        classification = int(key.split(' ')[0], 2)
        probs[classification] = probs[classification] + counts[key]
    probs = [c / sum(probs) for c in probs]
    return probs

def avg_softmax_cross_entropy_loss_with_one_hot_labels(y_target, y_prob):
    """
    average cross entropy loss after softmax.
    :param y:
    :param y_pred:
    :return:
    """
    # print(y_prob)
    # print(np.sum(np.exp(y_prob), axis=1), 1)
    y_prob = np.divide(np.exp(y_prob), np.sum(np.exp(y_prob), axis=1).reshape((-1,1)))
    # print(y_prob)
    # print("|||")
    return -np.sum(y_target*np.log(y_prob))/len(y_target)

def single_data_probs_sim(params, data, shots = 2048):
    backend_sim = Aer.get_backend('aer_simulator')
    convnet = transpile(simple_conv_net_9x9_encoding_4_class(params, data), backend_sim)
    job = backend_sim.run(convnet, shots=shots)
    results = job.result()
    counts = results.get_counts()
    probs = get_probs_from_counts(counts, num_classes=4)
    return probs


def batch_avg_accuracy(probs, labels):
    """
    average accuracy with one-hot labels
    :param probs:
    :param labels:
    :return:
    """
    preds = np.argmax(probs, axis=1)
    targets = np.argmax(labels, axis=1)
    return np.mean(np.array(preds == targets).astype(int))


if __name__ == '__main__':
    import matplotlib as mpl
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    NUM_SHOTS = 512
    N_WORKERS = 8
    MAX_JOB_SIZE = 10

    BACKEND_SIM = Aer.get_backend('aer_simulator')
    EXC = ThreadPoolExecutor(max_workers=N_WORKERS) # 125 secs/iteration for 20 train 20 test
    #EXC = Client(address=LocalCluster(n_workers=N_WORKERS, processes=True)) # 150 secs/iteration for 20 train 20 test
    BACKEND_SIM.set_options(executor=EXC)
    BACKEND_SIM.set_options(max_job_size=MAX_JOB_SIZE)
    BACKEND_SIM.set_options(max_parallel_experiments=0)

    seed = 42
    rng = np.random.default_rng(seed=seed)
    KERNEL_SIZE = (3, 3)
    STRIDE = (3, 3)
    n_test = 20
    n_epochs = 100
    n_reps = 3
    train_sizes = [20, 200, 500]


    def batch_data_probs_sim(params, data_list):
        """
        no ThreadPoolExecutor, 1024 shots,  40 train, 100 test, SPSA, single epoch time around 370 seconds;
        with ThreadPoolExecutor, n_workers=12, max_job_size =1, 1024 shots, 40 train, 100 test, SPSA, single epoch
        time around 367 seconds
        with dask Client, n_workers=12, max_job_size =1, 1024 shots, 40 train, 100 test, SPSA, single epoch time
        around 446 seconds, but will encounter OSError: [Errno 24] Too many open files
        :param params:
        :param data_list:
        :param shots:
        :param n_workers:
        :param max_job_size:
        :return:
        """
        circs = [transpile(simple_conv_net_9x9_encoding_4_class(params, data), BACKEND_SIM) for data in data_list]
        results = BACKEND_SIM.run(circs, shots=NUM_SHOTS).result()
        counts = results.get_counts()
        probs = [get_probs_from_counts(count, num_classes=4) for count in counts]
        return np.array(probs)


    def batch_data_loss_avg(params, data_list, labels):
        probs = batch_data_probs_sim(params, data_list)
        return avg_softmax_cross_entropy_loss_with_one_hot_labels(labels, probs)


    def train_model(n_train, n_test, n_epochs, rep, rng):
        """

        :param n_train:
        :param n_test:
        :param n_epochs:
        :param rep:
        :return:
        """
        x_train, y_train, x_test, y_test = load_data(n_train, n_test, rng)
        params = np.random.random(1209)
        train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], [], []
        print(f"Training with {n_train} data, testing with {n_test} data, for {n_epochs} epochs...")
        cost = lambda xk: batch_data_loss_avg(xk, x_train, y_train)
        start = time.time()

        # For the callback function for SPSA in qiskit
        # 5 arguments needed: number of function evaluations, parameters, loss, stepsize, accepted
        def callback_fn(xk):
            train_prob = batch_data_probs_sim(xk, x_train)
            train_cost = avg_softmax_cross_entropy_loss_with_one_hot_labels(y_train, train_prob)
            train_cost_epochs.append(train_cost)
            test_prob = batch_data_probs_sim(xk, x_test)
            test_cost = avg_softmax_cross_entropy_loss_with_one_hot_labels(y_test, test_prob)
            test_cost_epochs.append(test_cost)
            train_acc = batch_avg_accuracy(train_prob, y_train)
            test_acc = batch_avg_accuracy(test_prob, y_test)
            train_acc_epochs.append(train_acc)
            test_acc_epochs.append(test_acc)
            iteration_num = len(train_cost_epochs)
            time_till_now = time.time() - start
            avg_epoch_time = time_till_now / iteration_num
            if iteration_num % 1 == 0:
                print(
                    f"Rep {rep}, Training with {n_train} data, Training at Epoch {iteration_num}, train acc "
                    f"{np.round(train_acc, 4)}, "
                    f"train cost {np.round(train_cost, 4)}, test acc {np.round(test_acc, 4)}, test cost "
                    f"{np.round(test_cost, 4)}, avg epoch time "
                    f"{round(avg_epoch_time, 4)}, total time {round(time_till_now, 4)}")

        def callback_fn_qiskit_spsa(n_func_eval, xk, next_loss, stepsize, accepted):
            train_prob = batch_data_probs_sim(xk, x_train)
            #train_cost = avg_softmax_cross_entropy_loss_with_one_hot_labels(y_train, train_prob)
            train_cost_epochs.append(next_loss)
            test_prob = batch_data_probs_sim(xk, x_test)
            test_cost = avg_softmax_cross_entropy_loss_with_one_hot_labels(y_test, test_prob)
            test_cost_epochs.append(test_cost)
            train_acc = batch_avg_accuracy(train_prob, y_train)
            test_acc = batch_avg_accuracy(test_prob, y_test)
            train_acc_epochs.append(train_acc)
            test_acc_epochs.append(test_acc)
            iteration_num = len(train_cost_epochs)
            time_till_now = time.time() - start
            avg_epoch_time = time_till_now / iteration_num
            if iteration_num % 1 == 0:
                print(
                    f"Rep {rep}, Training with {n_train} data, Training at Epoch {iteration_num}, train acc "
                    f"{np.round(train_acc, 4)}, "
                    f"train cost {np.round(next_loss, 4)}, test acc {np.round(test_acc, 4)}, test cost "
                    f"{np.round(test_cost, 4)}, avg epoch time "
                    f"{round(avg_epoch_time, 4)}, total time {round(time_till_now, 4)}")

        bounds = [(0, 2 * np.pi)] * 1209
        # COBYLA single iteration around 97 seconds, SPSA (noisyopt) is 150 seconds at the same condition.
        # opt = SPSA(maxiter=n_epochs, callback=callback_fn_qiskit_spsa)
        opt = COBYLA(maxiter=n_epochs, callback=callback_fn)
        res = opt.minimize(
            cost,
            x0 = params,
            bounds=bounds
        )
        # according to Spall, IEEE, 1998, 34, 817-823,
        # one typically finds that in a high-noise setting (Le., poor quality measurements of L(theta))
        # it is necessary to pick a smaller a and larger c than in a low-noise setting.
        # res = minimizeSPSA(
        #     cost,
        #     x0=params,
        #     niter=n_epochs,
        #     paired=False,
        #     bounds=bounds,
        #     c=1,
        #     a=0.05,
        #     callback=callback_fn
        # )
        optimized_params = res.x
        return dict(
            n_train=[n_train] * (n_epochs+1),
            step=np.arange(1, n_epochs + 1 +1, dtype=int),
            train_cost=train_cost_epochs,
            train_acc=train_acc_epochs,
            test_cost=test_cost_epochs,
            test_acc=test_acc_epochs,
        ), optimized_params




    def run_iterations(n_train, rng):
        results_df = pd.DataFrame(
            columns=["train_acc", "train_cost", "test_acc", "test_cost", "step", "n_train"]
        )
        for rep in range(n_reps):
            results, _ = train_model(n_train=n_train, n_test=n_test, n_epochs=n_epochs, rep=rep, rng=rng)
            results_df = pd.concat(
                [results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True
            )
        return results_df

    results_df = run_iterations(n_train=train_sizes[0], rng =rng)
    for n_train in train_sizes[1:]:
        results_df = pd.concat([results_df, run_iterations(n_train=n_train, rng=rng)])
    # aggregate dataframe
    df_agg = results_df.groupby(["n_train", "step"]).agg(["mean", "std"])
    df_agg = df_agg.reset_index()

    sns.set_style('whitegrid')
    colors = sns.color_palette()
    fig, axes = plt.subplots(ncols=2, figsize=(16.5, 5))

    # plot losses and accuracies
    for i, n_train in enumerate(train_sizes):
        df = df_agg[df_agg.n_train == n_train]

        dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
        lines = ["o-", "x--", "o-", "x--"]
        labels = [fr"$N={n_train}$", None, fr"$N={n_train}$", None]
        axs = [0, 0, 1,1]

        for k in range(4):
            ax = axes[axs[k]]
            ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=10, color=colors[i], alpha=0.8)

    # format loss plot
    ax = axes[0]
    ax.set_title('Train and Test Losses', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # format loss plot
    ax = axes[1]
    ax.set_title('Train and Test Accuracies', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0., 1.05)

    legend_elements = [
                          mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes)
                      ] + [
                          mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
                          mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
                      ]

    axes[0].legend(handles=legend_elements, ncol=3)
    axes[1].legend(handles=legend_elements, ncol=3)
    plt.savefig(f"qiskit-fashion-mnist-simple-conv-multiclass-results-{n_test}-test-{n_reps}-reps.pdf")

