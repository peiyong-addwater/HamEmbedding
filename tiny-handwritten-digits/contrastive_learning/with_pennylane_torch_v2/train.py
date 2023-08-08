import pennylane as qml

qml.disable_return()  # Turn of the experimental return feature,
# see https://docs.pennylane.ai/en/stable/code/api/pennylane.enable_return.html#pennylane.enable_return

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import TinyHandwrittenDigitsDataset
from utils import cut8x8to4x4PatchesNoPos
from recurrent_structure_no_pos_encoding import RecurrentCircV1

# data paths
img_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/images"
csv_file="/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/annotated_labels.csv"

# structural parameters
#TODO: incorporate this into the construction of the PyTorch model, then only import the Pytorch model
N_MEM_QUBITS = 4 # don't change this unless change the model structure
N_PATCH_QUBITS = 4
N_QUBITS = N_MEM_QUBITS + N_PATCH_QUBITS
WIRES = list(range(N_QUBITS))
L1 = 2
L2 = 2
L_MC = 1
RESET_FIRST_MEM_QUBIT = False

# Training parameters
BATCH_SIZE = 10
N_EPOCHS = 100
LEARNING_RATE = 0.01
DEVICE = 'gpu'

# define qnode
dev = qml.device('default.mixed', wires=N_QUBITS)
@qml.qnode(dev, interface='torch')
def qnode(
        inputs,
        patch_encode_params,
        patch_rot_crot_params,
        mem_init_params,
        mem_patch_interact_params,
        mem_computation_params,
):
    RecurrentCircV1(
        inputs,
        patch_encode_params,
        patch_rot_crot_params,
        mem_init_params,
        mem_patch_interact_params,
        mem_computation_params,
        wires=WIRES,
        L1=L1,
        L2=L2,
        L_MC=L_MC,
        Optional_reset_first_mem_qubit=RESET_FIRST_MEM_QUBIT
    )
    return qml.probs(wires=range(N_MEM_QUBITS))

WEIGHT_SHAPES = {
    'patch_encode_params': (L2,L1*6*4),
    'patch_rot_crot_params': (L2,21),
    'mem_init_params': (12 * N_MEM_QUBITS - 9),
    'mem_patch_interact_params': (36),
    'mem_computation_params': (L_MC, 12 * N_MEM_QUBITS - 9),
}

QLAYER = qml.qnn.TorchLayer(qnode, WEIGHT_SHAPES)

mock_model = nn.Sequential(QLAYER)
mock_data = torch.randn(3,64)
#print(mock_data.shape)
#print(mock_model(mock_data))