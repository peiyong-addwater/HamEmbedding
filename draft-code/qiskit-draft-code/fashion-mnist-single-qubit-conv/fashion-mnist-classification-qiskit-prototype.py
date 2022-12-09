import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit import Aer
from dask.distributed import LocalCluster, Client
from concurrent.futures import ThreadPoolExecutor
from noisyopt import minimizeSPSA
import json
import time
import shutup
shutup.please()

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

# draw the conv net
# data = []
# for i in range(9):
#     single_qubit_data = []
#     for j in range(9):
#         single_qubit_data.append(ParameterVector(f"x_{i}{j}", length=9))
#     data.append(single_qubit_data)
# parameter_convnet = ParameterVector("θ", length=1629)
# convnet_draw = conv_net_9x9_encoding_4_class(parameter_convnet, data)
# convnet_draw.draw(output='mpl', filename='conv_net_9x9_encoding_4_class.png', style='bw', fold=-1)


# run the circuit with random data, see what kind of measurements will appear in the output
# backend_sim = Aer.get_backend('aer_simulator')
# seed = 42
# rng = np.random.default_rng(seed=seed)
# data = load_data(10,10,rng)[0][0]
# print(len(data), len(data[0]))
# labels = load_data(10,10,rng)[1]
# print(labels)
# parameter_convnet = np.random.random(1629)
# sample_run_convnet = transpile(conv_net_9x9_encoding_4_class(parameter_convnet, data), backend_sim)
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

def avg_softmax_cross_entropy_loss_with_one_hot_labels(y, y_prob):
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
    return -np.sum(y*np.log(y_prob))/len(y)

def single_data_probs_sim(params, data, shots = 2048):
    backend_sim = Aer.get_backend('aer_simulator')
    convnet = transpile(conv_net_9x9_encoding_4_class(params, data), backend_sim)
    job = backend_sim.run(convnet, shots=shots)
    results = job.result()
    counts = results.get_counts()
    probs = get_probs_from_counts(counts, num_classes=4)
    return probs

def batch_data_probs_sim(params, data_list, shots=2048, n_workers = 8, max_job_size =1):
    """
    no ThreadPoolExecutor, 1024 shots,  40 train, 100 test, SPSA, single epoch time around 370 seconds;
    with ThreadPoolExecutor, n_workers=12, max_job_size =1, 1024 shots, 40 train, 100 test, SPSA, single epoch time around 367 seconds
    with dask Client, n_workers=12, max_job_size =1, 1024 shots, 40 train, 100 test, SPSA, single epoch time around 446 seconds, but will encounter OSError: [Errno 24] Too many open files
    :param params:
    :param data_list:
    :param shots:
    :param n_workers:
    :param max_job_size:
    :return:
    """
    backend_sim = Aer.get_backend('aer_simulator')
    circs = [conv_net_9x9_encoding_4_class(params, data) for data in data_list]
    # exc = Client(address=LocalCluster(n_workers=n_workers, processes=True))
    exc = ThreadPoolExecutor(max_workers=n_workers)
    backend_sim.set_options(executor=exc)
    backend_sim.set_options(max_job_size=max_job_size)
    results = backend_sim.run(circs, shots=shots).result()
    # if using dask, close the Client
    # exc.close()
    counts = results.get_counts()
    probs = [get_probs_from_counts(count, num_classes=4) for count in counts]
    return np.array(probs)

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

def batch_data_loss_avg(params, data_list, labels, shots = 2048, n_workers=8, max_job_size =1):
    probs = batch_data_probs_sim(params, data_list, shots, n_workers, max_job_size)
    return avg_softmax_cross_entropy_loss_with_one_hot_labels(labels, probs)

def train_model(n_train, n_test, n_epochs, rep, rng, shots = 2048, n_workers=8, max_job_size =1):
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
    cost = lambda xk:batch_data_loss_avg(xk, x_train, y_train, shots, n_workers, max_job_size)
    start = time.time()

    def callback_fn(xk):
        train_prob = batch_data_probs_sim(xk, x_train, shots, n_workers, max_job_size)
        train_cost = avg_softmax_cross_entropy_loss_with_one_hot_labels(train_prob, y_train)
        train_cost_epochs.append(train_cost)
        test_prob = batch_data_probs_sim(xk, x_test, shots, n_workers, max_job_size)
        test_cost = avg_softmax_cross_entropy_loss_with_one_hot_labels(test_prob, y_test)
        test_cost_epochs.append(test_cost)
        train_acc = batch_avg_accuracy(train_prob, y_train)
        test_acc = batch_avg_accuracy(test_prob, y_test)
        train_acc_epochs.append(train_acc)
        test_acc_epochs.append(test_acc)
        iteration_num = len(train_cost_epochs)
        time_till_now = time.time()-start
        avg_epoch_time = time_till_now/iteration_num
        if iteration_num % 1 == 0:
            print(
                f"Rep {rep}, Training with {n_train} data, Training at Epoch {iteration_num}, train acc {np.round(train_acc, 4)}, "
                f"train cost {np.round(train_cost, 4)}, test acc {np.round(test_acc, 4)}, test cost {np.round(test_cost, 4)}, avg epoch time "
                f"{round(avg_epoch_time, 4)}, total time {round(time_till_now, 4)}")
    res = minimizeSPSA(
        cost,
        x0 = params,
        niter=n_epochs,
        paired=False,
        c=0.15,
        a=0.2,
        callback=callback_fn
    )
    return res


if __name__ == '__main__':
    # import qiskit
    # from qiskit_aer import AerSimulator
    # from dask.distributed import LocalCluster, Client
    # from math import pi
    #
    #
    # def q_exec():
    #      # Generate circuits
    #      circ = qiskit.QuantumCircuit(15, 15)
    #      circ.h(0)
    #      circ.cx(0, 1)
    #      circ.cx(1, 2)
    #      circ.p(pi / 2, 2)
    #      circ.measure([0, 1, 2], [0, 1, 2])
    #
    #      circ2 = qiskit.QuantumCircuit(7, 7)
    #      circ2.h(0)
    #      circ2.cx(0, 1)
    #      circ2.cx(1, 2)
    #      circ2.p(pi / 2, 2)
    #      circ2.measure([0, 1, 2], [0, 1, 2])
    #
    #      circ3 = qiskit.QuantumCircuit(3, 3)
    #      circ3.h(0)
    #      circ3.cx(0, 1)
    #      circ3.cx(1, 2)
    #      circ3.p(pi / 2, 2)
    #      circ3.measure([0, 1, 2], [0, 1, 2])
    #
    #      circ4 = qiskit.QuantumCircuit(2, 2)
    #      circ4.h(0)
    #      circ4.cx(0, 1)
    #      circ4.p(pi / 2, 1)
    #      circ4.measure([0, 1], [0, 1])
    #
    #      circ_list = [circ, circ2, circ3, circ4]
    #
    #      #exc = Client(address=LocalCluster(n_workers=4, processes=True))
    #      exc = ThreadPoolExecutor(max_workers=12)
    #
    #      # Set executor and max_job_size
    #      qbackend = AerSimulator()
    #      qbackend.set_options(executor=exc)
    #      qbackend.set_options(max_job_size=12)
    #      result = qbackend.run(circ_list).result()
    #      return result
    # res = q_exec()
    # print(res)
    # print(res.get_counts())
    # exit(0)


    seed = 42
    rng = np.random.default_rng(seed=seed)
    KERNEL_SIZE = (3, 3)
    STRIDE = (3, 3)
    n_test = 100
    n_epochs = 5
    n_reps = 5
    train_sizes = [10, 200, 500, 1000]
    res = train_model(train_sizes[0], n_test=n_test, n_epochs=n_epochs, rep=0, rng=rng, shots = 1024, n_workers=10, max_job_size =10)
    print(res)
    print(res.keys())
    print(res["x"])
    print(res["fun"])

