import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageTask')

import torch
from torch import nn
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
    """
    For remote tensorboard:
    if tensorboard haven't been started:
    ssh -L 16006:127.0.0.1:6006 peiyongw@10.100.238.77
    tensorboard --logdir=<log dir> --port=6006
    Then in local browser: 
    127.0.0.1:16006
    or
    localhost:16006
    """
    import warnings
    warnings.filterwarnings("ignore")
    import argparse
    import torchmetrics
    import numpy as np
    import random
    from qiskit_algorithms.utils import algorithm_globals

    task_name = 'classification-sklearn-digits-8x8-samplerQNN-4x4-patch'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=100)
    parser.add_argument('--train_batches', type=int, required=False, default=10)
    parser.add_argument('--epochs', type=int, required=False, default=500*2)
    parser.add_argument('--n_mem_qubits', type=int, required=False, default=3)
    parser.add_argument('--n_mem_interact_qubits', type=int, required=False, default=2)
    parser.add_argument('--n_patch_interact_qubits', type=int, required=False, default=2)
    parser.add_argument('--n_mem_comp_layers', type=int, required=False, default=1)
    parser.add_argument('--n_classification_layers', type=int, required=False, default=1)
    parser.add_argument('--spsa_batchsize', type=int, required=False, default=1)
    parser.add_argument('--working_dir', type=str, required=False, default='/home/peiyongw/Desktop/Research/QML-ImageTask')
    parser.add_argument('--prev_checkpoint', type=str, required=False, default=None)
    parser.add_argument('--load_optimizer', type=bool, required=False, default=False)
    parser.add_argument('--n_single_patch_reupload', type=int, required=False, default=1)
    parser.add_argument('--lr', type=float, required=False, default=0.1)
    parser.add_argument('--spsa_epsilon', type=float, required=False, default=0.2)
    parser.add_argument('--seed', type=int, required=False, default=1701)

    args = parser.parse_args()
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    algorithm_globals.random_seed = seed
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
    LR = args.lr
    SPSA_EPSILON = args.spsa_epsilon

    nt = nowtime()
    log_dir = f"logs-{task_name}-{nt}"
    checkpoint_dir = os.path.join('checkpoint', f'checkpoints-{task_name}-{nt}')
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
        "n_single_patch_reupload": N_SINGLE_PATCH_REUPLOAD,
        "spsa_epsilon": SPSA_EPSILON,
    }

    training_hyperparams = {
        "task_name": task_name,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "train_batches": TRAIN_BATCHES,
        "model_hyperparams": model_hyperparams,
        "lr": LR,
        "seed": seed,
    }

    with open(os.path.join(checkpoint_dir, 'training_hyperparams.json'), 'w') as f:
        json.dump(training_hyperparams, f, indent=4)

    device = 'cpu'

    model = ClassificationSamplerQNN8x8Image(
        num_single_patch_reuploading=N_SINGLE_PATCH_REUPLOAD,
        num_mem_qubits=N_MEM_QUBITS,
        num_mem_interact_qubits=N_MEM_INTERACT_QUBITS,
        num_patch_interact_qubits=N_PATCH_INTERACT_QUBITS,
        num_mem_comp_layers=N_MEM_COMP_LAYERS,
        num_classification_layers=N_CLASSIFICATION_LAYERS,
        spsa_batchsize=SPSA_BATCHSIZE,
        spsa_epsilon=SPSA_EPSILON
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    criterion = nn.CrossEntropyLoss()

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10, top_k=1)

    # load old checkpoint
    if prev_checkpoint is not None:
        checkpoint = torch.load(prev_checkpoint)
        model.load_state_dict(checkpoint['model'])
        if args.load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Load from previous checkpoint {prev_checkpoint}")

    # data
    dataset = PatchedDigitsDataset()
    train_size = int(TRAIN_BATCHES * BATCH_SIZE) # reduce the train size
    val_size = int(0.2 * train_size)  # reduce the val size
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                       num_workers=10), \
        DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=10), \
        DataLoader(test_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=10)

    batch_iters = 0
    all_start = time.time()

    model = model.to(device)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0
        total_acc = 0
        n_train_batches = len(train_loader)
        for i, (x, y) in enumerate(train_loader):
            batch_start = time.time()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = accuracy(y_pred, y)
            writer.add_scalar('Loss/train_batch', loss.item(), batch_iters)
            writer.add_scalar('Accuracy/train_batch', acc.item(), batch_iters)
            total_loss = total_loss + loss.item()
            total_acc = total_acc + acc.item()
            loss.backward()
            optimizer.step()
            batch_end = time.time()
            print(f"Epoch {epoch} batch {i + 1}/{n_train_batches} loss: {loss.item()} acc: {acc.item()} time: {batch_end - batch_start}")
            batch_iters += 1
        writer.add_scalar('Loss/train_epoch', total_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train_epoch', total_acc / len(train_loader), epoch)
        print(f"Epoch {epoch} train loss: {total_loss / len(train_loader)}, train acc: {total_acc / len(train_loader)}, train time: {time.time() - epoch_start}")
        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch)
            if weight.grad is not None:
                writer.add_histogram(f'{name}.grad', weight.grad, epoch)

        if (epoch) % 20 == 0 or epoch == EPOCHS - 1:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'epoch-{str(epoch).zfill(5)}-checkpoint.pth'))
            print(f"Epoch {epoch} checkpoint saved")

        if (epoch) % 1 == 0:
            total_loss = 0
            total_acc = 0
            model.eval()
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y= y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                acc = accuracy(y_pred, y)
                total_loss = total_loss + loss.item()
                total_acc = total_acc + acc.item()
            writer.add_scalar('Loss/val_epoch', total_loss / len(val_loader), epoch)
            writer.add_scalar('Accuracy/val_epoch', total_acc / len(val_loader), epoch)
            model.train()
            print(
                f"Epoch {epoch} val loss: {total_loss / len(val_loader)}, val acc: {total_acc / len(val_loader)}, train + val time: {time.time() - epoch_start}")

        print("=====================================================")

    final_chpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
    torch.save(final_chpt, os.path.join(checkpoint_dir, f'final-checkpoint.pth'))


