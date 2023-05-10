import os.path
import math
import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA, COBYLA
import json
import time
import shutup
import pickle
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from SPSAGradOptimiser.qiskit_opts.SPSA_Adam import ADAMSPSA
from qiskit.circuit import ParameterVector

shutup.please()

from qiskit_ibm_provider import IBMProvider
PROVIDER = IBMProvider()

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
            num_data_gates = math.ceil(len(unpadded_data)//encoding_gate_parameter_size)
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
"""
# draw the 9-param version of su4
param_9 = ParameterVector('θ', 9)
su4_9p = su4_9_params(param_9)
su4_9p.decompose().draw(output='mpl', filename='su4_9p.png', style='bw')
"""
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

"""
# draw the three-qubit interaction circuit
param_27 = ParameterVector('θ', 27)
three_q_circ = three_q_interaction(param_27)
three_q_circ.decompose().draw(output='mpl', filename='three_q_circ.png', style='bw', fold=-1)
"""

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

"""
# draw the memory cell
param_46 = ParameterVector('θ', 46)
mem_circ = memory_cell(param_46)
mem_circ.decompose().draw(output='mpl', filename='mem_circ.png', style='bw', fold=-1)
"""

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

"""
# draw the 3x3x1 kernel circuit
data_single_kernel = ParameterVector('x', 9)
conv_pooling_params = ParameterVector('θ', 35)
kernel_3x3x1(data_single_kernel, conv_pooling_params).decompose().draw(output='mpl', filename='kernel_3x3x1.png', style='bw', fold=-1)
"""

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

"""
# draw the encoding circuit
data = []
for i in range(6):
    for j in range(6):
        data.append(ParameterVector(f"x_{i}{j}", length=9))

parameter_encode = ParameterVector('θ', 123)
encode(data, parameter_encode).decompose().draw(output='mpl', filename='encode.png', style='bw')
"""

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

"""
# draw the backbone QNN
parameter_qnn = ParameterVector('θ', 168)
backbone_qnn(data, parameter_qnn).draw(output='mpl', filename='backbone_qnn.png', style='bw')
"""

"""
# sample some data
sample_data = select_data()
print([data[1] for data in sample_data])
print(len(sample_data[0][0]))
print(len(sample_data[0][0][0]))
flatten_data_list = [item for sublist in sample_data[0][0] for item in sublist]
print(len(flatten_data_list))
"""
def get_prob_vec_from_count_dict(counts:dict, n_qubits:int=6):
    """

    :param counts:
    :param n_qubits:
    :return:
    """
    keys = list(counts.keys())
    shots = sum(counts.values())
    prob_vec = np.zeros(2**n_qubits)
    for key in keys:
        index = int(key, 2)
        prob_vec[index] = counts[key] / shots

    return prob_vec

if __name__ == '__main__':
    import matplotlib as mpl
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import json
    import os

    print(os.getcwd())

    NUM_SHOTS = 1024
    N_WORKERS = 11
    MAX_JOB_SIZE = 1
    N_PARAMS = 168
    BACKEND_SIM = Aer.get_backend('aer_simulator')
    EXC = ThreadPoolExecutor(max_workers=N_WORKERS)
    BACKEND_SIM.set_options(executor=EXC)
    BACKEND_SIM.set_options(max_job_size=MAX_JOB_SIZE)
    BACKEND_SIM.set_options(max_parallel_experiments=0)
    # BACKEND_SIM.set_options(device='GPU') # GPU is probably more suitable for a few very large circuits instead of a large number of small-to-medium sized circuits
    seed = 42
    rng = np.random.default_rng(seed=seed)
    KERNEL_SIZE = (3, 3)
    STRIDE = (1, 1)
    n_epochs = 500
    n_img_per_label = 2
    curr_t = nowtime()
    save_filename = curr_t + "_" + f"siamese-10-class-qiskit-mnist-3x3-conv-classical-features-tiny-image-results-{n_img_per_label}-img_per_class-COBYLA.json"
    checkpointfile = None
    if checkpointfile is not None:
        with open(checkpointfile, 'r') as f:
            checkpoint = json.load(f)
            print("Loaded checkpoint file: " + checkpointfile)
        params = checkpoint['params']
    else:
        params = np.random.uniform(low=-np.pi, high=np.pi, size= N_PARAMS)

    def get_batch_prob_vectors(params, dataset:List[Tuple[List[List[float]],int]]):
        """

        :param params:
        :param dataset: full dataset. Each entry is a tuple containing a 6 by 6 list and a label
        :return:
        """
        all_circs = []
        circ_name_label_dict = {}
        circ_count = 0
        for data in dataset:
            flatten_data_list = [item for sublist in data[0] for item in sublist]
            #circ = transpile(backbone_qnn(flatten_data_list, params), BACKEND_SIM)
            circ = backbone_qnn(flatten_data_list, params).decompose(reps=4) # decompose everything
            circ.name = f"circ_{circ_count}"
            # print(circ.name)
            circ_name_label_dict[f"circ_{circ_count}"] = data[1]
            all_circs.append(circ)
            circ_count+=1

        # run the circuits in parallel
        job = BACKEND_SIM.run(all_circs, shots=NUM_SHOTS)
        result_dict = job.result().to_dict()["results"]
        result_counts = job.result().get_counts()
        prob_vec_list = [] # list of tuples (prob_vec, label)
        for i in range(len(all_circs)):
            name = result_dict[i]["header"]["name"]
            counts = result_counts[i]
            label = circ_name_label_dict[name]
            prob_vec = get_prob_vec_from_count_dict(counts, n_qubits=6)
            prob_vec_list.append((prob_vec, label))

        return prob_vec_list

    """
    test_params = np.random.randn(N_PARAMS)
    data_sample = select_data()
    start_test = time.time()
    test_prob_vecs = get_batch_prob_vectors(test_params, data_sample)
    end_test = time.time()
    print("Size of the data: " + str(len(data_sample)))
    print("Time for single run: " + str(end_test - start_test)) # 102 seconds; 58 seconds with just decompose 4 reps instead of transpile
    print(len(test_prob_vecs))
    print([item[1] for item in test_prob_vecs])
    """

    def contrastive_loss(params, dataset:List[Tuple[List[List[float]],int]], margin = 1):
        """

        :param params:
        :param dataset:
        :return:
        """
        prob_vecs = get_batch_prob_vectors(params, dataset)
        loss = 0
        count = 0
        for i in range(len(prob_vecs)):
            for j in range(len(prob_vecs)):
                if i != j:
                    count= count + 1
                    if prob_vecs[i][1] == prob_vecs[j][1]:
                        loss += (np.linalg.norm(prob_vecs[i][0] - prob_vecs[j][0]))**2
                    else:
                        loss += max(0, margin - np.linalg.norm(prob_vecs[i][0] - prob_vecs[j][0]))**2

        return loss / count

    def train_model(n_img_per_label, n_epochs, starting_point,rng):
        train_cost_epochs = []
        data_list = select_data(num_data_per_label_train=n_img_per_label, rng=rng)
        cost = lambda xk: contrastive_loss(xk, data_list, margin=1)
        start = time.time()
        def callback_fn(xk):
            train_cost = cost(xk)
            train_cost_epochs.append(train_cost)
            iteration_num = len(train_cost_epochs)
            time_till_now = time.time() - start
            avg_epoch_time = time_till_now / iteration_num
            if iteration_num % 1 == 0:
                print(
                    f"Training with {len(data_list)} images with {n_img_per_label} image(s) per class, at Epoch {iteration_num}, "
                    f"train cost {np.round(train_cost, 4)}, "
                    f"avg epoch time "
                    f"{round(avg_epoch_time, 4)}, total time {round(time_till_now, 4)}")

        def callback_fn_qiskit_spsa(n_func_eval, xk, next_loss, stepsize, accepted):
            train_cost = next_loss
            train_cost_epochs.append(train_cost)
            iteration_num = len(train_cost_epochs)
            time_till_now = time.time() - start
            avg_epoch_time = time_till_now / iteration_num
            if iteration_num % 1 == 0:
                print(
                    f"Training with {len(data_list)} images with {n_img_per_label} image(s) per class, at Epoch {iteration_num}, "
                    f"train cost {np.round(train_cost, 4)}, "
                    f"avg epoch time "
                    f"{round(avg_epoch_time, 4)}, total time {round(time_till_now, 4)}")

        bounds = [(0, 2 * np.pi)] * (N_PARAMS)
        """
        opt = COBYLA(maxiter=n_epochs, callback=callback_fn)
        res = opt.minimize(
            cost,
            x0=starting_point,
            bounds=bounds
        )
        optimized_params = res.x
        return dict(
            losses=train_cost_epochs,
            params=optimized_params
        )"""
        opt = ADAMSPSA(maxiter=n_epochs, amsgrad=True)
        """
        res = opt.minimize(
            cost,
            x0=starting_point,
            bounds=bounds
        )
        """
        # optimized_params = res.x
        res = opt.optimize(
            num_vars=N_PARAMS,
            objective_function=cost,
            gradient_function=None,
            variable_bounds=bounds,
            initial_point=starting_point,
            verbose=True
        )
        optimized_params = res[0]
        train_cost_epochs = res[3]
        return dict(
            losses=train_cost_epochs,
            params=optimized_params
        )


    res = train_model(n_img_per_label=n_img_per_label, n_epochs=n_epochs, starting_point=params, rng=rng)

    with open(save_filename, 'w') as f:
        json.dump(res, f, indent=4, cls=NpEncoder)











