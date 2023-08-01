import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
from backbone_circ import SSLCircFourQubitZ
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')
import time
import shutup
import pickle
import json
DATA_FILE = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/tiny-handwritten-with-augmented-as-rotation-angles-patches.pkl"
def nowtime():
    return str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))

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

def prob_z_sim(a, b):
    """
    take two 16-dim prob vectors, return the similarity
    :param a:
    :param b:
    :return:
    """
    return torch.dot(a,b)

def createBatches(data, batchSize, seed = 0, type = "train", n_batches = None):
    """
    Create batches of data.

    :param data: Loaded data file "tiny-handwritten-with-augmented-as-rotation-angles-patches.pkl"
    :param batchSize: The size of each batch.
    :param seed: The random seed.
    :return: A list of batches.
    """
    train_data = data[type]
    batches = []
    rng = np.random.default_rng(seed)
    np.random.shuffle(train_data)
    for i in range(0, len(train_data), batchSize):
        batch_data_dict = train_data[i:i+batchSize]
        aug_data = []
        for j in range(len(batch_data_dict)):
            random_chosen_two = rng.choice(batch_data_dict[j]['augmentations'], 2, replace=False)
            # each batch of batches has 2N augmented images for batch size N
            aug_data.append(torch.tensor(random_chosen_two[0] , requires_grad=False))
            aug_data.append(torch.tensor(random_chosen_two[1] , requires_grad=False))
        single_batch = torch.stack(aug_data)
        batches.append(single_batch)
        if n_batches is not None:
            if len(batches) == n_batches:
                break
    return batches

WIRES = [0, 1, 2, 3, 4, 5, 6, 7]
dev = qml.device("default.mixed", wires=WIRES)

@qml.qnode(dev, interface="torch")
@qml.transforms.merge_rotations(atol=1e-8, include_gates=None)
@qml.transforms.cancel_inverses
@qml.transforms.commute_controlled(direction="right")
def get_z(
        patched_img: torch.Tensor,
        single_patch_encoding_parameter: torch.Tensor,
        single_patch_d_and_r_phase_parameter: torch.Tensor,
        phase_parameter: torch.Tensor,
        two_patch_2_q_pqc_parameter: torch.Tensor,
        projection_head_parameter: torch.Tensor
):
    """
    This function returns the z representation of the input image.
    :param patched_img: a 4 by 4 by 4 tensor of image patches, where the last index is the original pixels
    :param single_patch_encoding_parameter:"θ" 12N element array, where N is the number of data re-uploading repetitions
    :param single_patch_d_and_r_phase_parameter: "ϕ" 2 element array
    :param phase_parameter: "ψ" 2 element array
    :param two_patch_2_q_pqc_parameter: "ξ" 15-parameter array, since we are using su4 gate
    :param projection_head_parameter: "η" using the FourQubitGenericParameterisedLayer, 39-parameter array
    :return: prob vector on the first 4 qubits
    """
    SSLCircFourQubitZ(
        patched_img,
        single_patch_encoding_parameter,
        single_patch_d_and_r_phase_parameter,
        phase_parameter,
        two_patch_2_q_pqc_parameter,
        projection_head_parameter,
        wires=WIRES
    )
    return qml.probs(wires=WIRES[0:4])

def get_batch_z_serial(
        batch_patches: torch.Tensor,
        single_patch_encoding_parameter: torch.Tensor,
        single_patch_d_and_r_phase_parameter: torch.Tensor,
        phase_parameter: torch.Tensor,
        two_patch_2_q_pqc_parameter: torch.Tensor,
        projection_head_parameter: torch.Tensor
):
    """
    This function returns the z representation of the input image.
    :param batch_patches: (2*batchsize, 4, 4, 4)
    :param single_patch_encoding_parameter:
    :param single_patch_d_and_r_phase_parameter:
    :param phase_parameter:
    :param two_patch_2_q_pqc_parameter:
    :param projection_head_parameter:
    :return:
    """
    batch_z = []
    for i in range(batch_patches.shape[0]):
        batch_z.append(get_z(
            batch_patches[i],
            single_patch_encoding_parameter,
            single_patch_d_and_r_phase_parameter,
            phase_parameter,
            two_patch_2_q_pqc_parameter,
            projection_head_parameter
        ))
    return torch.stack(batch_z)


if __name__ == "__main__":
    import pickle
    import math
    curr_t = nowtime()
    save_filename = curr_t + "_" + "10q_circ_4q_rep_SimCLR_probs_z_training_result.json"
    batch_size = 10 # for 32G total memory, batch size 10 is the maximum, 20 sec per batch
    val_ratio = 0.8
    n_batches = 300
    n_data_reuploading_layers = 1
    n_theta = 6 * n_data_reuploading_layers*2
    n_phi = 2
    n_psai = 2
    n_xi = 15
    n_eta = 39
    n_params = n_theta + n_phi + n_psai + n_xi + n_eta
    print("n_params: ", n_params)
    theta = torch.randn(n_theta, requires_grad=True)
    phi = torch.randn(n_phi, requires_grad=True)
    psai = torch.randn(n_psai, requires_grad=True)
    xi = torch.randn(n_xi, requires_grad=True)
    eta = torch.randn(n_eta, requires_grad=True)
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    train_val_batches = createBatches(data, batch_size, seed=1701, n_batches=n_batches)
    print(train_val_batches[0].shape)
    train_batches = train_val_batches[:math.floor(len(train_val_batches)*(1-val_ratio))]
    val_batches = train_val_batches[math.floor(len(train_val_batches)*(1-val_ratio)):]
    test_batches = createBatches(data, batch_size, seed=1701, type="test", n_batches=math.floor(n_batches*val_ratio))

    start_time = time.time()
    batch_z = get_batch_z_serial(
            train_val_batches[0],
            theta,
            phi,
            psai,
            xi,
            eta
        )
    end = time.time()
    print(batch_z)
    print(batch_z.shape)
    print("time elapsed: ", end - start_time)
