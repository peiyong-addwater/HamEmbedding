import pennylane as qml

qml.disable_return()  # Turn of the experimental return feature,
# see https://docs.pennylane.ai/en/stable/code/api/pennylane.enable_return.html#pennylane.enable_return

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.data import DataLoader
from with_pennylane_torch.dataset import TinyHandwrittenDigitsDataset
import time

from with_pennylane_torch.torch_module_prob import RecurentQNNNoPosCodeV1
from with_pennylane_torch.qnns import RecurrentCircV1
from torch.utils.tensorboard import SummaryWriter
import json
import os
from sklearn.svm import SVC

latest_checkpoint_file = "epoch-00045-checkpoint.pth"
checkpoint_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/tiny-handwritten-digits/contrastive_learning/checkpoint/checkpoints-20230816-170130"
hyper_params = json.load(open(os.path.join(checkpoint_dir, "training_hyperparams.json"), 'r'))['model_hyperparams']
latest_checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint_file))

L1 = hyper_params['L1']
L2 = hyper_params['L2']
L_MC = hyper_params['L_MC']
n_mem_qubits = hyper_params['n_mem_qubits']
n_patch_qubits = hyper_params['n_patch_qubits']
forget_gate = hyper_params['forget_gate']

trained_params = {
    "four_pixel_encode_params": latest_checkpoint['model']['net.qlayer.four_pixel_encode_params'],
    "sixteen_pixel_encode_params": latest_checkpoint['model']['net.qlayer.sixteen_pixel_encode_params'],
    "mem_init_params": latest_checkpoint['model']['net.qlayer.mem_init_params'],
    "mem_patch_interact_params": latest_checkpoint['model']['net.qlayer.mem_patch_interact_params'],
    "mem_computation_params": latest_checkpoint['model']['net.qlayer.mem_computation_params']
}

n_ssl_qubits = 2*n_mem_qubits+ n_patch_qubits
ssl_wires = list(range(n_ssl_qubits))
ssl_dev = qml.device("default.mixed", wires=ssl_wires)
@qml.qnode(ssl_dev, interface="torch")
def ssl_swap_test(x1, x2):
    """
    The input image data is assumed to be of shape (...,  64).
    :param x1:
    :param x2:
    :return:
    """
    x1_wires = ssl_wires[:n_mem_qubits+n_patch_qubits]
    x1_mem_wires = x1_wires[:n_mem_qubits]
    x2_wires = ssl_wires[n_mem_qubits:]
    x2_mem_wires = x2_wires[:n_mem_qubits]
    swap_test_qubit = ssl_wires[2*n_mem_qubits]
    print(x1_wires)
    print(x2_wires)
    print(swap_test_qubit)
    print(x2_mem_wires)
    print(x1_mem_wires)
    # encode x1
    RecurrentCircV1(
        x1,
        trained_params['four_pixel_encode_params'],
        trained_params['sixteen_pixel_encode_params'],
        trained_params['mem_init_params'],
        trained_params['mem_patch_interact_params'],
        trained_params['mem_computation_params'],
        x1_wires,
        L1,
        L2,
        L_MC,
        forget_gate
    )
    # encode x2
    RecurrentCircV1(
        x2,
        trained_params['four_pixel_encode_params'],
        trained_params['sixteen_pixel_encode_params'],
        trained_params['mem_init_params'],
        trained_params['mem_patch_interact_params'],
        trained_params['mem_computation_params'],
        x1_wires,
        L1,
        L2,
        L_MC,
        forget_gate
    )
    # swap test
    qml.Hadamard(wires=swap_test_qubit)
    for i in range(n_mem_qubits):
        qml.CSWAP(wires=[swap_test_qubit, x1_mem_wires[i], x2_mem_wires[i]])
    qml.Hadamard(wires=swap_test_qubit)
    return qml.probs(wires=swap_test_qubit)



if __name__ == "__main__":

    img_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/images"
    csv_file = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/annotated_labels.csv"

    x1 = torch.randn(64)
    x2 = torch.randn(64)
    print(ssl_swap_test(x1, x2))

