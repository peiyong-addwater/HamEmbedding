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
        """
        # pair the training data
        paired_extracted_data = []
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

def kernel_5x5_v2(padded_data_in_kernel_view, conv_params, pooling_params):
    """

    :param padded_data_in_kernel_view:
    :param conv_params: 21
    :param pooling_params: 18
    :return:
    """
    qreg = QuantumRegister(4, name="conv-pooling")
    creg = ClassicalRegister(5, name='pooling-meas')
    circ = QuantumCircuit(qreg, creg, name="conv-encode-5x5")
    circ.h(qreg)
    # encode the pixel data
    circ.compose(su4_circuit(padded_data_in_kernel_view[:15]), qubits=qreg[:2], inplace=True)
    circ.compose(su4_circuit(padded_data_in_kernel_view[15:]), qubits=qreg[2:], inplace=True)
    # convolution
    circ.compose(conv_circuit(conv_params[0:3]), qubits=[qreg[1], qreg[2]], inplace=True)
    circ.compose(conv_circuit(conv_params[3:6]), qubits=[qreg[0], qreg[1]], inplace=True)
    circ.compose(conv_circuit(conv_params[6:9]), qubits=[qreg[2], qreg[3]], inplace=True)
    # collapse two of the four qubits
    # the measured qubits will remain in the measured state (sometimes with a phase)
    circ.measure(qreg[0], creg[3])
    circ.measure(qreg[2], creg[4])
    circ.barrier()
    circ.compose(conv_circuit(conv_params[9:12]), qubits=[qreg[0], qreg[1]], inplace=True)
    circ.compose(conv_circuit(conv_params[12:15]), qubits=[qreg[2], qreg[3]], inplace=True)
    circ.barrier()
    circ.measure(qreg[1], creg[3])
    circ.measure(qreg[3], creg[4])
    circ.barrier()
    circ.compose(conv_circuit(conv_params[15:18]), qubits=[qreg[0], qreg[1]], inplace=True)
    circ.compose(conv_circuit(conv_params[18:21]), qubits=[qreg[2], qreg[3]], inplace=True)
    # measurement and pooling
    circ.measure(qreg[1:], creg[:3])
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

# draw the kernel
data_single_kernel = ParameterVector('x', 30)
conv_params = ParameterVector('θ', 21)
pooling_params = ParameterVector('φ', 18)
circ = kernel_5x5_v2(data_single_kernel, conv_params, pooling_params)
circ.draw(output='mpl', filename="kernel_5x5_v2_restricted_two_qubit_gate.png", style='bw', fold=-1)

def backbone_net(data_for_entire_2x2_feature_map, params):
    """
    ingle kernel requires 21+18 parameters
    three consective linear transformation requires 3*3=9 parameters
    two consective non-linear transformation requires 3*2*2=12 parameters
    total number of parameters: 21+18+9+12 = 60
    :param data_for_entire_2x2_feature_map:
    :param params:
    :return:
    """
    qreg = QuantumRegister(7, name='conv')
    creg = ClassicalRegister(5, name="pooling-meas")
    res_creg = ClassicalRegister(4, name="res-meas")
    circ = QuantumCircuit(qreg, creg, res_creg, name='backbone')
    conv_kernel_param = params[:21]
    pooling_param = params[21:21 + 18]
    linear_params = params[21 + 18:21 + 18 + 9]
    non_linear_transformation_param = params[21 + 18 + 9:21 + 18 + 9 + 12]
    qubit_counter = 0
    for i in range(2):
        for j in range(2):
            conv_op = kernel_5x5_v2(data_for_entire_2x2_feature_map[i][j], conv_kernel_param,
                                 pooling_param)  # .to_instruction(label="Conv5x5")
            circ.compose(conv_op, qubits=qreg[qubit_counter:qubit_counter + 4], clbits=creg, inplace=True)
            circ.barrier(qreg)
            qubit_counter += 1
    # linear layer on the feature produced by conv kernels transformation
    circ.compose(conv_circuit(linear_params[:3]), qubits=[qreg[0], qreg[1]], inplace=True)
    circ.compose(conv_circuit(linear_params[3:6]), qubits=[qreg[1], qreg[2]], inplace=True)
    circ.compose(conv_circuit(linear_params[6:9]), qubits=[qreg[2], qreg[3]], inplace=True)
    circ.barrier()
    # non-linear transformation, just like those in the kernel
    circ.measure(qreg[0], creg[3])
    circ.measure(qreg[2], creg[4])
    circ.barrier()
    circ.compose(conv_circuit(non_linear_transformation_param[:3]), qubits=[qreg[0], qreg[1]], inplace=True)
    circ.compose(conv_circuit(non_linear_transformation_param[3:6]), qubits=[qreg[2], qreg[3]], inplace=True)
    circ.measure(qreg[1], creg[3])
    circ.measure(qreg[3], creg[4])
    circ.barrier()
    circ.compose(conv_circuit(non_linear_transformation_param[6:9]), qubits=[qreg[0], qreg[1]], inplace=True)
    circ.compose(conv_circuit(non_linear_transformation_param[9:12]), qubits=[qreg[2], qreg[3]], inplace=True)

    circ.measure([qreg[0], qreg[1], qreg[2], qreg[3]], res_creg)

    return circ

# draw the backbone network
data = []
for i in range(2):
    row = []
    for j in range(2):
        row.append(ParameterVector(f"x_{i}{j}", length=30))
    data.append(row)

parameter_backbone = ParameterVector("θ", length=60)
backbone = backbone_net(data, parameter_backbone)
backbone.draw(output='mpl', filename='backbone-5x5-input-8x8_restricted_two_qubit_gate.png', style='bw')
print("backbone circuit picture saved...")
"""
test_params = np.random.randn(60)
data_sample = select_data()[0][0]
test_circ = backbone_net(data_sample, test_params).decompose(reps=4)
BACKEND_SIM = Aer.get_backend('aer_simulator')
job = BACKEND_SIM.run(test_circ)
result_counts = job.result().get_counts()
print(result_counts)
"""
def get_prob_vec_from_count_dict(counts:dict, n_qubits:int=4):
    """
    get the probability vector from the counts dictionary
    :param counts:
    :param n_qubits:
    :return:
    """
    prob_vec = np.zeros(2**n_qubits)
    shots = sum(counts.values())
    for key, value in counts.items():
        feature_measure = key.split(" ")[0]
        prob_vec[int(feature_measure, 2)] += value
    return prob_vec/shots


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
    N_PARAMS = 60
    BACKEND_SIM = Aer.get_backend('aer_simulator')
    EXC = ThreadPoolExecutor(max_workers=N_WORKERS)
    BACKEND_SIM.set_options(executor=EXC)
    BACKEND_SIM.set_options(max_job_size=MAX_JOB_SIZE)
    BACKEND_SIM.set_options(max_parallel_experiments=0)
    # BACKEND_SIM.set_options(device='GPU') # GPU is probably more suitable for a few very large circuits instead of a large number of small-to-medium sized circuits
    seed = 1701
    rng = np.random.default_rng(seed=seed)
    KERNEL_SIZE = (5, 5)
    STRIDE = (3, 3)
    n_epochs = 500*2
    n_img_per_label = 8
    curr_t = nowtime()
    save_filename = curr_t + "_" + f"siamese-10-class-qiskit-mnist-5x5-conv-restricted-2q-gate-classical-features-tiny-image-results-{n_img_per_label}-img_per_class-ADAM-SPSA.json"
    checkpointfile = "20230523-110519_siamese-10-class-qiskit-mnist-5x5-conv-restricted-2q-gate-classical-features-tiny-image-results-4-img_per_class-ADAM-SPSA.json"
    if checkpointfile is not None:
        with open(checkpointfile, 'r') as f:
            checkpoint = json.load(f)
            print("Loaded checkpoint file: " + checkpointfile)
        params = np.array(checkpoint['params'])
    else:
        params = np.random.uniform(low=-np.pi, high=np.pi, size= N_PARAMS)

    def get_batch_prob_vectors(params, dataset:List[Tuple[List[List[float]], int]]):
        """

        :param params:
        :param dataset:
        :return:
        """
        all_circs = []
        circ_name_label_dict = {}
        circ_count = 0
        for data, label in dataset:
            circ = backbone_net(data, params).decompose(reps=4) # decompose everything
            circ.name = f"circ_{circ_count}"
            all_circs.append(circ)
            circ_name_label_dict[circ.name] = label
            circ_count += 1

        # run the circuits in parallel
        job = BACKEND_SIM.run(all_circs, shots=NUM_SHOTS)
        result_dict = job.result().to_dict()["results"]
        result_counts = job.result().get_counts()
        prob_vec_list = []  # list of tuples (prob_vec, label)
        for i in range(len(all_circs)):
            name = result_dict[i]["header"]["name"]
            counts = result_counts[i]
            label = circ_name_label_dict[name]
            prob_vec = get_prob_vec_from_count_dict(counts, n_qubits=4)
            prob_vec_list.append((prob_vec, label))

        return prob_vec_list

    """
    test_params = np.random.randn(N_PARAMS)
    data_sample = select_data()
    start_test = time.time()
    test_prob_vecs = get_batch_prob_vectors(test_params, data_sample)
    end_test = time.time()
    print("Size of the data: " + str(len(data_sample)))
    print("Time for single run: " + str(
        end_test - start_test))  #  16.312373399734497 for 3 images per label
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
        bounds = [(0, 2 * np.pi)] * (N_PARAMS)
        opt = ADAMSPSA(maxiter=n_epochs, amsgrad=True)
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

