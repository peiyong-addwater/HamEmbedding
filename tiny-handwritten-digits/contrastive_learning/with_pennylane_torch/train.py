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

def lij(batch_z, i, j, tau = 1):
    """
    :param batch_z: (2*batch_size, 16)
    :param i: index, starting from 1, according to the SimCLR paper
    :param j: index, starting from 1, according to the SimCLR paper
    :param tau: temperature
    :return:
    """
    zi = batch_z[i-1]
    zj = batch_z[j-1]
    numerator = torch.exp(prob_z_sim(zi, zj)/tau)
    denominator = 0
    for k in range(1, batch_z.shape[0]+1):
        if k != i:
            zk = batch_z[k-1]
            denominator += torch.exp(prob_z_sim(zi, zk)/tau)
    return -torch.log(numerator/denominator)

def L(batch_z, tau=1):
    loss = 0
    batch_size = batch_z.shape[0]//2
    for k in range(1, batch_size+1):
        loss += lij(batch_z, 2*k-1, 2*k, tau)
        loss += lij(batch_z, 2*k, 2*k-1, tau)
    return loss/(2*batch_size)


if __name__ == "__main__":
    import pickle
    import math
    curr_t = nowtime()
    save_filename = curr_t + "_" + "8q_circ_4q_rep_SimCLR_probs_z_training_result.json"
    checkpointfile = None
    # hyperparameters
    batch_size = 5 # Memory occupied: ~14 GB when just finished the first batch, then rise up to ~21 GB and stay there
    val_ratio = 0.2
    n_batches = 100
    init_lr = 1e-1
    maxiter = 100
    n_data_reuploading_layers = 1
    n_theta = 6 * n_data_reuploading_layers*2
    n_phi = 2
    n_psai = 2
    n_xi = 15
    n_eta = 39
    n_params = n_theta + n_phi + n_psai + n_xi + n_eta
    if checkpointfile is not None:
        with open(checkpointfile, 'r') as f:
            checkpoint = json.load(f)
            print("Loaded checkpoint file: " + checkpointfile)
        params = torch.tensor(checkpoint['params'], requires_grad=True)
        n_data_reuploading_layers = checkpoint['n_data_reuploading_layers']
        n_theta = 6 * n_data_reuploading_layers*2
        n_params = n_theta + n_phi + n_psai + n_xi + n_eta
    else:
        params = torch.randn(n_params, requires_grad=True)
    print("n_params: ", n_params)

    def batch_cost(params, batch):
        output = get_batch_z_serial(
            batch,
            params[0:n_theta],
            params[n_theta:n_theta+n_phi],
            params[n_theta+n_phi:n_theta+n_phi+n_psai],
            params[n_theta+n_phi+n_psai:n_theta+n_phi+n_psai+n_xi],
            params[n_theta+n_phi+n_psai+n_xi:n_theta+n_phi+n_psai+n_xi+n_eta]
        )
        return L(output, tau=1)

    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    train_val_batches = createBatches(data, batch_size, seed=1701, n_batches=n_batches)
    train_batches = train_val_batches[:math.floor(len(train_val_batches)*(1-val_ratio))]
    val_batches = train_val_batches[math.floor(len(train_val_batches)*(1-val_ratio)):]
    test_batches = createBatches(data, batch_size, seed=1701, type="test", n_batches=math.floor(n_batches*val_ratio))

    def train_model_autodiff(
            train_batches: List[torch.Tensor],
            val_batches: List[torch.Tensor],
            test_batches: List[torch.Tensor],
            starting_point: torch.Tensor=params,
            n_epochs: int=maxiter,
            learning_rate: float=init_lr
    ):
        params = starting_point
        train_start = time.time()
        train_loss_list = []
        val_loss_list = []
        test_loss_list = []
        all_optimisation_iterations_loss_list = []
        opt = torch.optim.Adam([params], lr=learning_rate)
        for epoch in range(n_epochs):
            epoch_start = time.time()
            batch_loss_list = []
            for batchid in range(len(train_batches)):
                batch_start_time = time.time()
                batch = train_batches[batchid]
                opt.zero_grad()
                loss = batch_cost(params, batch)
                loss.backward()
                opt.step()
                batch_loss_list.append(loss.item())
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                print(
                    f"----Training at Epoch {epoch + 1}, Batch {batchid + 1}/{len(train_batches)}, Objective = {np.round(batch_loss_list[-1], 4)}, Batch Time = {np.round(batch_time, 4)}")
            epoch_end_1 = time.time()
            epoch_time_1 = epoch_end_1 - epoch_start
            all_optimisation_iterations_loss_list.extend(batch_loss_list)
            batch_avg_loss = np.mean(batch_loss_list)
            train_loss_list.append(batch_avg_loss)
            print(
                f"Training at Epoch {epoch + 1}, Objective = {np.round(batch_avg_loss, 4)}, Train Epoch Time = {np.round(epoch_time_1, 4)}")
            if val_batches is not None:
                with torch.no_grad():
                    batch_loss_list = []
                    for batchid in range(len(val_batches)):
                        batch = val_batches[batchid]
                        loss = batch_cost(params, batch)
                        batch_loss_list.append(loss.item())
                    batch_avg_loss = np.mean(batch_loss_list)
                    val_loss_list.append(batch_avg_loss)
                    print(f"Validation at Epoch {epoch + 1}, Objective = {np.round(batch_avg_loss, 4)}")
            if test_batches is not None:
                with torch.no_grad():
                    batch_loss_list = []
                    for batchid in range(len(test_batches)):
                        batch = test_batches[batchid]
                        loss = batch_cost(params, batch)
                        batch_loss_list.append(loss.item())
                    batch_avg_loss = np.mean(batch_loss_list)
                    test_loss_list.append(batch_avg_loss)
                    print(f"Testing at Epoch {epoch + 1}, Objective = {np.round(batch_avg_loss, 4)}")
            epoch_end_2 = time.time()
            epoch_time_2 = epoch_end_2 - epoch_end_1
            print(f"Epoch {epoch + 1} Time = {np.round(epoch_time_2, 4)}")
        train_end = time.time()
        train_time = train_end - train_start
        print(f"Training Time = {np.round(train_time, 4)}")
        params = params.detach().cpu().numpy()
        return params, train_loss_list, val_loss_list, test_loss_list, all_optimisation_iterations_loss_list, train_time

    params, train_loss_list, val_loss_list, test_loss_list, all_optimisation_iterations_loss_list, train_time = train_model_autodiff(
        train_batches,
        val_batches,
        test_batches,
        starting_point=params,
        n_epochs=maxiter,
        learning_rate=init_lr
    )
    res_dict = {
        "params": params,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "test_loss_list": test_loss_list,
        "all_optimisation_iterations_loss_list": all_optimisation_iterations_loss_list,
        "train_time": train_time,
        "n_data_reuploading_layers": n_data_reuploading_layers,
        "n_theta": n_theta,
        "n_phi": n_phi,
        "n_psai": n_psai,
        "n_xi": n_xi,
        "n_eta": n_eta,
        "n_params": n_params,
        "init_lr": init_lr,
        "maxiter": maxiter,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "val_ratio": val_ratio
    }
    with open(save_filename, 'w') as f:
        json.dump(res_dict, f, cls=NpEncoder, indent=4)




