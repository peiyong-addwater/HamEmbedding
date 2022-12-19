import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit import Aer
from dask.distributed import LocalCluster, Client
from concurrent.futures import ThreadPoolExecutor
from noisyopt import minimizeSPSA
from qiskit.algorithms.optimizers import COBYLA, SPSA, BOBYQA
import pybobyqa
from skquant.opt import minimize
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

def conv_net_9x9_encoding_4_class(params, single_image_data):
    """

    encoding has 9 parameters;

    entangling has 15*8 = 120 parameters;

    first pooling layer measures bottom 4 qubits, there are 2^4 = 16 different measurement results
    which leads to 32 different conv operations on the remaining 5 qubits based on the measurement results
    then the first pooling layer has 16*(4*15) = 960 parameters;

    second pooling layer measures 3 of the remaining 5 qubits, there are 2^3 = 8 different measurement results
    which leads to 4 different conv operations on the remaining 2 qubits based on the measurement results
    then the second pooling layer has 8*(15) = 120 parameters;

    total number of parameters is 9+120+960+120 = 1209 > 2^9 = 512, over-parameterization achieved.

    This structure has some flavor of decision trees.
    :param params:
    :param single_image_data:
    :return:
    """
    qreg = QuantumRegister(9)
    pooling_layer_1_meas = ClassicalRegister(4, name='pooling1meas')
    pooling_layer_2_meas = ClassicalRegister(3, name='pooling2meas')
    prob_meas = ClassicalRegister(2, name='classification')

    circ = QuantumCircuit(qreg, pooling_layer_1_meas, pooling_layer_2_meas, prob_meas)

    # data re-uploading layer
    circ.compose(convolution_reupload_encoding(params[:9], single_image_data), qubits=qreg, inplace=True)
    # entangling after encoding layer
    circ.compose(entangling_after_encoding(params[9:9+15*8]), qubits=qreg, inplace=True)
    # first pooling layer
    circ.measure(qreg[5:], pooling_layer_1_meas)
    first_pooling_params = params[9+15*8:9+15*8+16*(4*15)]
    for i in range(16):
        with circ.if_test((pooling_layer_1_meas, i)):
            circ.append(convolution_layer(first_pooling_params[15*4*i:15*4*(i+1)]).to_instruction(), qreg[:5])
    # second pooling layer
    circ.measure(qreg[2:5], pooling_layer_2_meas)
    second_pooling_params = params[9+15*8+16*(4*15):9+15*8+16*(4*15)+8*15]
    for i in range(8):
        with circ.if_test((pooling_layer_2_meas, i)):
            circ.append(convolution_layer(second_pooling_params[15*i:15*(i+1)]).to_instruction(), qreg[:2])
    # output classification probabilities
    circ.measure(qreg[:2], prob_meas)

    return circ

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
    convnet = transpile(conv_net_9x9_encoding_4_class(params, data), backend_sim)
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
    import json

    NUM_SHOTS = 512
    N_WORKERS = 8
    MAX_JOB_SIZE = 10
    BUDGET = 1000
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
    train_sizes = [20, 200]


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
        circs = [conv_net_9x9_encoding_4_class(params, data) for data in data_list]
        results = BACKEND_SIM.run(circs, shots=NUM_SHOTS).result()
        counts = results.get_counts()
        probs = [get_probs_from_counts(count, num_classes=4) for count in counts]
        return np.array(probs)


    def batch_data_loss_avg(params, data_list, labels):
        probs = batch_data_probs_sim(params, data_list)
        return avg_softmax_cross_entropy_loss_with_one_hot_labels(labels, probs)


    def train_model(n_train, n_test, rep, rng):
        """

        :param n_train:
        :param n_test:
        :param n_epochs:
        :param rep:
        :return:
        """
        x_train, y_train, x_test, y_test = load_data(n_train, n_test, rng)
        params = np.random.random(1209)
        print(f"Rep {rep}, Training with {n_train} data, testing with {n_test} data, for max {BUDGET} function evaluations...")
        cost = lambda xk: batch_data_loss_avg(xk, x_train, y_train)
        start = time.time()
        bounds = np.array([[0, 2 * np.pi]] * 1209)


        result, history = \
            minimize(cost, params, bounds, BUDGET, method='bobyqa',do_logging=True, print_progress=True)

        optval = result.optval
        optparams = result.optpar


        return optval, optparams, history

    def run_iterations(n_train, rng):
        results = {}
        for rep in range(n_reps):
            optval, optparams, history = train_model(n_train=n_train, n_test=n_test, rep=rep, rng=rng)
            results[rep] = {}
            results[rep]['optval'] = optval
            results[rep]['optparams'] = optparams
            results[rep]['history'] = history
        return results

    for n_train in train_sizes:
        filename = f"qiskit-fashion-mnist-multiclass-results{n_train}-train-{n_test}-test.json"
        res = run_iterations(n_train, rng)
        with open(filename, 'w') as f:
            json.dump(res, f, indent=4, cls=NpEncoder)
