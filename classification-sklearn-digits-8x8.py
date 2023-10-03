import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageTask')

import torch
from torch.utils.data import DataLoader
import time
from torch.utils.tensorboard import SummaryWriter
import json
import os

from Models.data import PatchedDigitsDataset
from Models.qiskit_models import ClassificationSamplerQNN8x8Image

def nowtime():
    return str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=100)
    parser.add_argument('--train_batches', type=int, required=False, default=10)
    parser.add_argument('--epochs', type=int, required=False, default=50)
    parser.add_argument('--n_mem_qubits', type=int, required=False, default=2)
    parser.add_argument('--n_mem_interact_qubits', type=int, required=False, default=1)
    parser.add_argument('--n_patch_interact_qubits', type=int, required=False, default=1)
    parser.add_argument('--n_mem_comp_layers', type=int, required=False, default=1)
    parser.add_argument('--n_classification_layers', type=int, required=False, default=1)
    parser.add_argument('--spsa_batchsize', type=int, required=False, default=1)
    parser.add_argument('--working_dir', type=str, required=False, default='/home/peiyongw/Desktop/Research/QML-ImageTask')
    parser.add_argument('--prev_checkpoint', type=str, required=False, default=None)
    parser.add_argument('--n_single_patch_reupload', type=int, required=False, default=2)

    args = parser.parse_args()
    wd = args.working_dir
    os.chdir(wd)

    # old checkpoint
    prev_checkpoint = args.prev_checkpoint

    # hyperparameters
    BATCH_SIZE = args.batch_size
    TRAIN_BATCHES = args.train_batches
    EPOCHS = args.epochs
    N_MEM_QUBITS = args.n_mem_qubits
    N_MEM_INTERACT_QUBITS = args.n_mem_interact_qubits
    N_PATCH_INTERACT_QUBITS = args.n_patch_interact_qubits
    N_MEM_COMP_LAYERS = args.n_mem_comp_layers
    N_CLASSIFICATION_LAYERS = args.n_classification_layers
    SPSA_BATCHSIZE = args.spsa_batchsize
    N_SINGLE_PATCH_REUPLOAD = args.n_single_patch_reupload

    nt = nowtime()
    log_dir = f"logs-{nt}"
    checkpoint_dir = os.path.join('checkpoint', f'checkpoints-{nt}')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    writer = SummaryWriter(os.path.join('runs', log_dir))

    model_hyperparams = {
        "n_mem_qubits": N_MEM_QUBITS,
        "n_mem_interact_qubits": N_MEM_INTERACT_QUBITS,
        "n_patch_interact_qubits": N_PATCH_INTERACT_QUBITS,
        "n_mem_comp_layers": N_MEM_COMP_LAYERS,
        "n_classification_layers": N_CLASSIFICATION_LAYERS,
        "spsa_batchsize": SPSA_BATCHSIZE,
        "n_single_patch_reupload": N_SINGLE_PATCH_REUPLOAD
    }

    training_hyperparams = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "train_batches": TRAIN_BATCHES,
        "model_hyperparams": model_hyperparams
    }

    with open(os.path.join(checkpoint_dir, 'training_hyperparams.json'), 'w') as f:
        json.dump(training_hyperparams, f, indent=4)

    model = ClassificationSamplerQNN8x8Image(

    )