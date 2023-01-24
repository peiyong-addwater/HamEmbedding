import os.path

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
import pickle


shutup.please()

from qiskit_ibm_provider import IBMProvider
PROVIDER = IBMProvider()

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
        with open("tiny-handwritten.pkl", 'rb') as f:
            mnist = pickle.load(f)
    if kind == 'train':
        return mnist["training_images"], mnist["training_labels"]
    else:
        return mnist["test_images"], mnist["test_labels"]

def load_data(num_train, num_test, rng, one_hot=True):
    data_path = "tiny-handwritten.pkl"
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

# img_matrix = np.random.randn(8,8)
# extracted_conv_data = extract_convolution_data(img_matrix, kernel_size=(5, 5), stride=(3, 3), dilation=(1, 1),
#                                                  padding=(0, 0), encoding_gate_parameter_size=15)
# print(np.array(extracted_conv_data).shape) # (2,2,30)

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
            conv_op = kernel_5x5(data_for_entire_2x2_feature_map[i][j], conv_kernel_param, pooling_param)
            circ.compose(conv_op, qubits=qreg[qubit_counter:qubit_counter + 4], clbits=creg, inplace=True)
            circ.barrier(qreg)
            qubit_counter+=1
    return circ

# draw the conv 1 layer
data = []
for i in range(2):
    row = []
    for j in range(2):
        row.append(ParameterVector(f"x_{i}{j}", length=30))
    data.append(row)
parameter_conv_1 = ParameterVector("θ", length=45 + 18)
first_conv_layer = conv_layer_1(data, parameter_conv_1)
first_conv_layer.draw(output='mpl', filename='conv-5x5-1.png', style='bw', fold=-1)

def full_circ(prepared_data, params):
    """
    conv-1 requires 45 + 18 parameters

    after conv-1, we'll perform three su4 gates between neighbouring 2 qubits on the first 4 qubits (0, 1, 2, 3):
    (2,3), (1,2), (0,1), then measure qubits 0 and 1 for classification results. This step requires 15*3=45 parameters

    total number of parameters 45+18+45 = 108
    :param prepared_data:
    :param params:
    :return:
    """
    qreg = QuantumRegister(7)
    pooling_measure = ClassicalRegister(3, name='pooling-measure')
    classification_reg = ClassicalRegister(2, name='classification')
    conv_1_parameters = params[:45+18]
    final_layer_parameters = params[45+18:]
    circ = QuantumCircuit(qreg, pooling_measure, classification_reg)
    # conv-1
    circ.compose(conv_layer_1(prepared_data, conv_1_parameters), qubits=qreg, clbits=pooling_measure, inplace=True)
    circ.barrier(qreg)
    # final layer
    circ.compose(su4_circuit(final_layer_parameters[:15]), qubits=[qreg[2], qreg[3]], inplace=True)
    circ.compose(su4_circuit(final_layer_parameters[15:30]), qubits=[qreg[1], qreg[2]], inplace=True)
    circ.compose(su4_circuit(final_layer_parameters[30:45]), qubits=[qreg[0], qreg[1]], inplace=True)
    circ.barrier(qreg)
    # classification measurement
    circ.measure(qreg[:2], classification_reg)
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
    convnet = transpile(full_circ(data, params), backend_sim)
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

# rng = np.random.default_rng(seed=42)
# data = load_data(10,10,rng)[0][0]
# params = ParameterVector('w', length=108)
# full_conv_net = full_circ(data, params)
# full_conv_net.draw(output='mpl', filename='full-circ-5x5.png', style='bw', fold=-1)
# params = np.random.random(108)
# full_conv_net = full_circ(data, params)
# backend_sim = Aer.get_backend('aer_simulator')
# start = time.time()
# job = backend_sim.run(transpile(full_conv_net, backend_sim), shots = 2048)
# results = job.result()
# counts = results.get_counts()
# prob = single_data_probs_sim(params, data)
# end =  time.time()
# print(counts)
# print(prob)
# print(sum(prob))
# print(end-start) # 0.730126142501831 seconds for 2048 shots
if __name__ == '__main__':
    import matplotlib as mpl
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    NUM_SHOTS = 512
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
    n_test = 20
    n_epochs = 200
    n_reps = 3
    train_sizes = [20, 200, 500]

    def batch_data_probs_sim(params, data_list):
        """

        :param params:
        :param data_list:
        :param shots:
        :param n_workers:
        :param max_job_size:
        :return:
        """
        circs = [full_circ(data, params) for data in data_list]
        results = BACKEND_SIM.run(transpile(circs, BACKEND_SIM), shots=NUM_SHOTS).result()
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
        params = np.random.random(108)
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

        bounds = [(0, 2 * np.pi)] * (108)

        opt = SPSA(maxiter=n_epochs, callback=callback_fn_qiskit_spsa)
        # opt = COBYLA(maxiter=n_epochs, callback=callback_fn)
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
            n_train=[n_train] * (n_epochs),
            step=np.arange(1, n_epochs + 1, dtype=int),
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
        axs = [0, 0, 2, 2]

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
    plt.savefig(f"qiskit-fashion-mnist-5x5-conv-multiclass-tiny-image-results-{n_test}-test-{n_reps}-reps.pdf")