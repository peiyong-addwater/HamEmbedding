import pennylane as qml
from pennylane import numpy as pnp
import json
import torch
import numpy as np
import time
from typing import List, Tuple, Union, Optional
from pennylane.wires import Wires
from utils import Reset0
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')
from backbone_hierarchical_encoding_QFT_Mixing import backboneQFTMixing

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
            aug_data.append(torch.Tensor(random_chosen_two[0] ))
            aug_data.append(torch.Tensor(random_chosen_two[1] ))
        batches.append(torch.stack(aug_data))
        if n_batches is not None:
            if len(batches) == n_batches:
                break
    return batches

WIRES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
dev = qml.device("default.mixed", wires=WIRES)
@qml.qnode(dev, interface="torch")
def probs_z(
        patched_img: Union[np.ndarray, torch.Tensor, pnp.ndarray],
        encode_parameters: Union[np.ndarray, pnp.ndarray, torch.Tensor],
        single_patch_phase_parameters: Union[np.ndarray, pnp.ndarray, torch.Tensor],
        local_patches_phase_parameters: Union[np.ndarray, pnp.ndarray, torch.Tensor],
        final_layer_parameters:Optional[Union[np.ndarray, torch.Tensor, pnp.ndarray]],
):
    """
    The circuit for generating "z" as in the SimCLR paper with the probs on the first four qubits
    :param patched_img: All 16 image patches, a 4 by 4 by 4 array
    :param encode_parameters: "θ", 6N*2*2 element array, where N is the number of "layers" of data-reuploading
    :param single_patch_phase_parameters: "φ", 2*2 element array
    :param local_patches_phase_parameters: "ψ", 2 element array
    :param final_layer_parameters: "η" 12L parameters if using FourQubitParameterisedLayer, 3L parameters if using PermutationInvariantFourQLayer
    :param final_layer_type: "generic" or "permutation-invariant"
    :param observables:
    :return:
    """
    backboneQFTMixing(
        patched_img=patched_img,
        encode_parameters=encode_parameters,
        single_patch_phase_parameters=single_patch_phase_parameters,
        local_patches_phase_parameters=local_patches_phase_parameters,
        final_layer_parameters=final_layer_parameters,
        final_layer_type='generic',
        wires=WIRES
    )
    return qml.probs(wires=[0,1,2,3])

def get_batch_z(
        batch: torch.Tensor,
        encode_parameters: torch.Tensor,
        single_patch_phase_parameters: torch.Tensor,
        local_patches_phase_parameters: torch.Tensor,
        final_layer_parameters:torch.Tensor,
):
    """

    :param batch: shape (batchsize*2, 4, 4, 4), each image has two different augmentations
    :param encode_parameters: "θ", 6N*2*2 element array, where N is the number of "layers" of data-reuploading
    :param single_patch_phase_parameters: "φ", 2*2 element array
    :param local_patches_phase_parameters: "ψ", 2 element array
    :param final_layer_parameters: "η" 12L parameters if using FourQubitParameterisedLayer, 3L parameters if using PermutationInvariantFourQLayer
    :return: shape (batchsize*2, 16)
    """
    return probs_z(
        patched_img=batch,
        encode_parameters=encode_parameters,
        single_patch_phase_parameters=single_patch_phase_parameters,
        local_patches_phase_parameters=local_patches_phase_parameters,
        final_layer_parameters=final_layer_parameters
    )


if __name__ == "__main__":
    import pickle
    import math
    curr_t = nowtime()
    save_filename = curr_t + "_" + "10q_circ_4q_rep_SimCLR_probs_z_training_result.json"
    batch_size = 2
    val_ratio = 0.
    n_batches = 10
    n_data_reuploading_layers = 2
    n_theta = 6 * n_data_reuploading_layers*2*2
    n_phi = 2*2
    n_psai = 2
    n_eta = 12

    theta = torch.randn(n_theta, requires_grad=True)
    phi = torch.randn(n_phi, requires_grad=True)
    psai = torch.randn(n_psai, requires_grad=True)
    eta = torch.randn(n_eta, requires_grad=True)

    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    train_val_batches = createBatches(data, batch_size, seed=1701, n_batches=n_batches)
    train_batches = train_val_batches[:math.floor(len(train_val_batches)*(1-val_ratio))]
    val_batches = train_val_batches[math.floor(len(train_val_batches)*(1-val_ratio)):]
    test_batches = createBatches(data, batch_size, seed=1701, type="test", n_batches=math.floor(n_batches*val_ratio))

    print(get_batch_z(train_val_batches[0], theta, phi, psai, eta).shape)