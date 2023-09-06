import pennylane as qml

qml.disable_return()  # Turn of the experimental return feature,
# see https://docs.pennylane.ai/en/stable/code/api/pennylane.enable_return.html#pennylane.enable_return

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.data import DataLoader
from with_pennylane_torch.dataset import TinyHandwrittenDigitsDataset
from with_pennylane_torch.image_transform import DEFAULT_TRANSFORM
import time





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


    from with_pennylane_torch.torch_module_prob_experimental import RecurentQNNNoPosCodeV1
    from with_pennylane_torch.byol import BYOL
    from torch.utils.tensorboard import SummaryWriter
    import json
    import os


    # data paths
    img_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/images"
    csv_file = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/annotated_labels.csv"

    BATCH_SIZE = 4
    TRAIN_BATCHES = 100
    EPOCHS = 50

    # structural parameters
    N_MEM_QUBITS = 4  # don't change this unless change the model structure
    N_PATCH_QUBITS = 4

    L1 = 2
    L2 = 2
    L_MC = 1
    RESET_FIRST_MEM_QUBIT = True

    model_hyperparams = {
        "L1": L1,
        "L2": L2,
        "L_MC": L_MC,
        "n_mem_qubits": N_MEM_QUBITS,
        "n_patch_qubits": N_PATCH_QUBITS,
        "forget_gate": RESET_FIRST_MEM_QUBIT
    }

    training_hyperparams = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "model_hyperparams": model_hyperparams
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    base_model = RecurentQNNNoPosCodeV1(L1, L2, L_MC, N_MEM_QUBITS, N_PATCH_QUBITS, RESET_FIRST_MEM_QUBIT)

    base_model = base_model.to(device)

    ssl_model = BYOL(
        base_model,
        RecurentQNNNoPosCodeV1,
        model_hyperparams,
        image_size=8,
        hidden_layer=-1,
        projection_size=256,
        projection_hidden_size=4096,
        augment_fn=DEFAULT_TRANSFORM,
        augment_fn2=DEFAULT_TRANSFORM,
        moving_average_decay=0.99,
        use_momentum=True
    )

    ssl_model = ssl_model.to(device)

    optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.01, amsgrad=True)


    dataset = TinyHandwrittenDigitsDataset(csv_file, img_dir)

    train_size = int(TRAIN_BATCHES*BATCH_SIZE) # reduce the train size
    val_size = int(0.2 * train_size) # reduce the val size
    test_size = len(dataset) - train_size- val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10), \
                                            DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=10), \
                                            DataLoader(test_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=10)

    batch_iters = 0
    all_start = time.time()
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0
        n_train_batches = len(train_loader)
        for i, (x, _) in enumerate(train_loader):
            batch_start = time.time()
            x = x.to(device)
            loss = ssl_model(x)
            total_loss = total_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_end = time.time()
            print(f"Epoch {epoch} batch {i+1}/{n_train_batches} loss: {loss.item()} time: {batch_end - batch_start}")
            batch_iters += 1
        print(f"Epoch {epoch} train loss: {total_loss / len(train_loader)}, train time: {time.time() - epoch_start}")

        if (epoch) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model': ssl_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, f'epoch-{str(epoch).zfill(5)}-checkpoint.pth')
            print(f"Epoch {epoch} checkpoint saved")
        if (epoch) % 10 == 0:
            total_loss = 0
            ssl_model.eval()
            for i, (x, _) in enumerate(val_loader):
                x = x.to(device)
                loss = ssl_model(x)
                total_loss = total_loss + loss.item()
            ssl_model.train()
            print(f"Epoch {epoch} val loss: {total_loss / len(val_loader)}, train + val time: {time.time() - epoch_start}")




