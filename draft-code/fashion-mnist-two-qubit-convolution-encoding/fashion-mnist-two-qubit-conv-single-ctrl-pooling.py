import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
from typing import List, Tuple, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import jax

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
    data_pad_size = 15*num_layers-len(data)
    padded_data = jnp.array(data)
    theta = jnp.array(theta)
    w = jnp.array(w)
    for _ in range(data_pad_size):
        padded_data = jnp.append(padded_data, 0)
    gate_params = jnp.add(theta, jnp.multiply(w, padded_data))#jnp.array(theta) + jnp.multiply(jnp.array(w), padded_data)
    for i in range(num_layers):
        SU4(gate_params[15*i:15*(i+1)], wires=wires)

def conv_repuload_encoding(theta, w, data, wires):
    num_row, num_columns = data.shape[0], data.shape[1]
    for j in range(num_columns):
        for i in range(0, num_row+1, 2):
            row_data = data[i]
            su4_single_conv_encoding(theta, w, row_data[j], wires=[wires[i], wires[i + 1]])
        qml.Barrier(wires=wires, only_visual=True)
        for i in range(1, num_row+1 - 2, 2):
            row_data = data[i]
            su4_single_conv_encoding(theta, w, row_data[j], wires=[wires[i], wires[i + 1]])
        qml.Barrier(wires=wires, only_visual=True)


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
    import time

    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

    import seaborn as sns

    sns.set()

    seed = 42
    rng = np.random.default_rng(seed=seed)

    KERNEL_SIZE = (5, 5)
    STRIDE = (5, 5)
    NUM_CONV_POOL_LAYERS = 2
    FINAL_LAYER_QUBITS = 2

    n_test = 100
    n_epochs = 100
    n_reps = 10

    train_sizes = [2, 10, 100, 1000]

    _, _, _, num_conv_rows, _ = _check_params(np.random.rand(28 * 28).reshape(28, 28), kernel=np.random.random(KERNEL_SIZE),
                                          stride=STRIDE, dilation=(1, 1), padding=(0, 0))
    num_wires = num_conv_rows + 1
    print(num_wires)
    device = qml.device("default.qubit", wires=num_wires)

    num_su4_each_conv = KERNEL_SIZE[0]*KERNEL_SIZE[1]//15 + 1
    theta_size = num_su4_each_conv*15
    w_size = num_su4_each_conv*15

    device = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(device, interface="jax")
    def conv_net(theta, w, conv_weights, last_layer_params, image_conv_extract):
        num_conv_layers = conv_weights.shape[1]
        wires = list(range(num_wires))
        for wire in wires:
            qml.Hadamard(wires=wire)
        qml.Barrier(wires=wires, only_visual=True)
        conv_repuload_encoding(theta, w, image_conv_extract, wires)
        qml.Barrier(wires=wires, only_visual=True)
        for j in range(num_conv_layers):
            conv_and_pooling(conv_weights[:, j], wires, skip_first_layer=(not j == 0))
            wires = wires[::2]
            qml.Barrier(wires=wires, only_visual=True)

        dense_layer(last_layer_params, wires)
        return qml.probs(wires=(0))

    def init_weights():
        """Initializes random weights for the QCNN model."""
        theta = pnp.random.normal(loc=0, scale=1, size=theta_size, requires_grad=True)
        w = pnp.random.normal(loc=0, scale=1, size=w_size, requires_grad=True)
        conv_weights = pnp.random.normal(loc=0, scale=1, size=(18, NUM_CONV_POOL_LAYERS), requires_grad=True)
        weights_last = pnp.random.normal(loc=0, scale=1, size=4 ** FINAL_LAYER_QUBITS - 1, requires_grad=True)
        return jnp.array(theta),jnp.array(w), jnp.array(conv_weights), jnp.array(weights_last)


    fig, ax = qml.draw_mpl(conv_net, style='black_white')(pnp.random.normal(loc=0, scale=1, size=theta_size, requires_grad=True),
                                                          pnp.random.normal(loc=0, scale=1, size=w_size, requires_grad=True),
                                                          pnp.random.normal(loc=0, scale=1, size=(18, NUM_CONV_POOL_LAYERS), requires_grad=True),
                                                          pnp.random.normal(loc=0, scale=1, size=4 ** FINAL_LAYER_QUBITS - 1, requires_grad=True),
                                                          extract_convolution_data(np.random.rand(28*28).reshape((28,28)),stride=STRIDE, kernel_size=KERNEL_SIZE))
    plt.savefig("circuit-su4-encoding-multiclass.pdf")

    def load_data(num_train, num_test, rng, stride = STRIDE, kernel_size = KERNEL_SIZE):
        """Return training and testing data of digits dataset."""
        data_folder = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/fashion"
        features, labels = load_fashion_mnist(data_folder)
        features = [features[i].reshape(28,28) for i in range(len(features))]
        features = np.array(features)

        # only use first four classes
        # features = features[np.where((labels == 0) | (labels == 1)|(labels == 2) | (labels == 3))]
        # labels = labels[np.where((labels == 0) | (labels == 1)|(labels == 2) | (labels == 3))]
        features = features[np.where((labels == 0) | (labels == 1))]
        labels = labels[np.where((labels == 0) | (labels == 1))]

        # normalize data
        features = features / 255


        # subsample train and test split
        train_indices = rng.choice(len(labels), num_train, replace=False)
        test_indices = rng.choice(
            np.setdiff1d(range(len(labels)), train_indices), num_test, replace=False
        )

        x_train, y_train = features[train_indices], labels[train_indices]
        x_train = np.array([extract_convolution_data(x_train[i],stride=stride, kernel_size=kernel_size) for i in range(num_train)])
        x_test, y_test = features[test_indices], labels[test_indices]
        x_test = np.array([extract_convolution_data(x_test[i],stride=stride, kernel_size=kernel_size) for i in range(num_test)])
        return (
            jnp.asarray(x_train),
            jnp.asarray(y_train),
            jnp.asarray(x_test),
            jnp.asarray(y_test),
        )

    @jax.jit
    def compute_out(theta, w, conv_weights, weights_last, features, labels):
        cost = lambda theta, w, conv_weights, weights_last, feature, label:conv_net(theta, w, conv_weights, weights_last, feature)[
            label
        ]
        return jax.vmap(cost, in_axes=(None, None, None, None, 0, 0), out_axes=0)(
            theta, w, conv_weights, weights_last, features, labels
        )

    def compute_accuracy(theta, w, conv_weights, weights_last, features, labels):
        out = compute_out(theta, w, conv_weights,weights_last, features, labels)
        return jnp.sum(out>0.5)/len(out)

    def compute_cost(theta, w, conv_weights, weights_last, features, labels):
        out = compute_out(theta, w, conv_weights,weights_last, features, labels)
        return 1.0 - jnp.sum(out) / len(labels)

    value_and_grad = jax.jit(jax.value_and_grad(compute_cost, argnums=[0, 1, 2, 3]))

    def train_qcnn(n_train, n_test, n_epochs):
        """
        Args:
            n_train  (int): number of training examples
            n_test   (int): number of test examples
            n_epochs (int): number of training epochs
            desc  (string): displayed string during optimization

        Returns:
            dict: n_train,
            steps,
            train_cost_epochs,
            train_acc_epochs,
            test_cost_epochs,
            test_acc_epochs

        """
        # load data
        x_train, y_train, x_test, y_test = load_data(n_train, n_test, rng)
        theta, w, conv_weights, weights_last = init_weights()
        cosine_decay_scheduler = optax.cosine_decay_schedule(0.5, decay_steps=n_epochs, alpha=0.95)
        optimizer = optax.sgd(learning_rate=cosine_decay_scheduler, nesterov=True, momentum=0.5)
        opt_state = optimizer.init((theta, w, conv_weights, weights_last))
        train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], [], []

        print(f"Training with {n_train} data, testing with {n_test} data, for {n_epochs} epochs...")
        for step in range(n_epochs):
            epoch_start = time.time()
            # Training step with (adam) optimizer
            train_cost, grad_circuit = value_and_grad(theta, w, conv_weights, weights_last, x_train, y_train)
            updates, opt_state = optimizer.update(grad_circuit, opt_state)
            #print(updates)
            theta, w, conv_weights, weights_last = optax.apply_updates((theta, w, conv_weights, weights_last), updates)
            train_cost_epochs.append(train_cost)
            # compute accuracy on training data
            train_acc = compute_accuracy(theta, w, conv_weights, weights_last, x_train,
                                         y_train)
            train_acc_epochs.append(train_acc)
            # compute accuracy and cost on testing data
            test_out = compute_out(theta, w, conv_weights, weights_last, x_test,
                                   y_test)
            test_acc = jnp.sum(test_out > 0.5) / len(test_out)
            test_acc_epochs.append(test_acc)
            test_cost = 1.0 - jnp.sum(test_out) / len(test_out)
            test_cost_epochs.append(test_cost)

            epoch_end = time.time()
            print(
                f"Training with {n_train} data, Training at Epoch {step}, train acc {train_acc}, test acc {test_acc}, time {round(epoch_end-epoch_start,4)} seconds")

        return dict(
            n_train=[n_train] * n_epochs,
            step=np.arange(1, n_epochs + 1, dtype=int),
            train_cost=train_cost_epochs,
            train_acc=train_acc_epochs,
            test_cost=test_cost_epochs,
            test_acc=test_acc_epochs,
        )

    def run_iterations(n_train):
        results_df = pd.DataFrame(
            columns=["train_acc", "train_cost", "test_acc", "test_cost", "step", "n_train"]
        )

        for _ in range(n_reps):
            results = train_qcnn(n_train=n_train, n_test=n_test, n_epochs=n_epochs)
            results_df = pd.concat(
                [results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True
            )

        return results_df


    # run training for multiple sizes
    results_df = run_iterations(n_train=train_sizes[0])
    for n_train in train_sizes[1:]:
        results_df = pd.concat([results_df, run_iterations(n_train=n_train)])

    # aggregate dataframe
    df_agg = results_df.groupby(["n_train", "step"]).agg(["mean", "std"])
    df_agg = df_agg.reset_index()

    sns.set_style('whitegrid')
    colors = sns.color_palette()
    fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))

    generalization_errors = []

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

        # plot final loss difference
        dif = df[df.step == 100].test_cost["mean"] - df[df.step == 100].train_cost["mean"]
        generalization_errors.append(dif)

    # format loss plot
    ax = axes[0]
    ax.set_title('Train and Test Losses', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # format generalization error plot
    ax = axes[1]
    ax.plot(train_sizes, generalization_errors, "o-", label=r"$gen(\alpha)$")
    ax.set_xscale('log')
    ax.set_xticks(train_sizes)
    ax.set_xticklabels(train_sizes)
    ax.set_title(r'Generalization Error $gen(\alpha) = R(\alpha) - \hat{R}_N(\alpha)$', fontsize=14)
    ax.set_xlabel('Training Set Size')

    # format loss plot
    ax = axes[2]
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
    axes[2].legend(handles=legend_elements, ncol=3)

    axes[1].set_yscale('log', base=2)
    plt.savefig(f"fashion-mnist-twoclass-su4-encoding-results-{n_test}-test-{n_reps}-reps.pdf")
