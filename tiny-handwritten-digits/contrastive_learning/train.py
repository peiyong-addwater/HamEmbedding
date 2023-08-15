import pennylane as qml

qml.disable_return()  # Turn of the experimental return feature,
# see https://docs.pennylane.ai/en/stable/code/api/pennylane.enable_return.html#pennylane.enable_return

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.data import DataLoader
from with_pennylane_torch.dataset import TinyHandwrittenDigitsDataset
from with_pennylane_torch.byol import BYOL
from with_pennylane_torch.torch_module_prob import RecurentQNNNoPosCodeV1
from with_pennylane_torch.image_transform import DEFAULT_TRANSFORM
import time

# data paths
img_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/images"
csv_file="/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/annotated_labels.csv"

# structural parameters
#TODO: incorporate this into the construction of the PyTorch model, then only import the Pytorch model
N_MEM_QUBITS = 4 # don't change this unless change the model structure
N_PATCH_QUBITS = 4

L1 = 2
L2 = 2
L_MC = 1
RESET_FIRST_MEM_QUBIT = False

model_hyperparams = {
    "L1": L1,
    "L2": L2,
    "L_MC": L_MC,
    "n_mem_qubits": N_MEM_QUBITS,
    "n_patch_qubits": N_PATCH_QUBITS,
    "forget_gate": RESET_FIRST_MEM_QUBIT
}

# Training parameters
BATCH_SIZE = 10
N_EPOCHS = 100
LEARNING_RATE = 0.01
DEVICE = 'gpu'

model = RecurentQNNNoPosCodeV1(L1, L2, L_MC, N_MEM_QUBITS, N_PATCH_QUBITS)
dataset = TinyHandwrittenDigitsDataset(csv_file, img_dir)

#print(model)

learner = BYOL(
    model,
    net_class=RecurentQNNNoPosCodeV1,
    net_hyperparam_dict=model_hyperparams,
    image_size=8,
    projection_size = 16,
    projection_hidden_size=128,
    augment_fn=DEFAULT_TRANSFORM,
    hidden_layer='qlayer'
)
dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

for batch, (X, y) in enumerate(dataloader):
    start = time.time()
    loss = learner(X)
    end = time.time()
    print(loss, end - start)

    break