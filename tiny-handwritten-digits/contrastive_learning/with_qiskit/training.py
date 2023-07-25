import multiprocessing

import math
import numpy as np
import dill
from multiprocessing import Process, Queue
from typing import List, Tuple, Union, Callable, Optional
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer
from qiskit.compiler import transpile
from concurrent.futures import ThreadPoolExecutor
from qiskit.quantum_info import random_clifford
from qiskit.tools import parallel_map
from multiprocessing import Pool
import dask
import qiskit
import json
import time
import shutup
import pickle
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from SPSAGradOptimiser.qiskit_opts.SPSA_Adam import ADAMSPSA
from qiskit.circuit import ParameterVector
import os
from shadow_representation import getzSingleArg, getzFromImagePatches

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

def operator_2_norm(R):
    """
    Calculate the operator 2-norm.

    Args:
        R (array): The operator whose norm we want to calculate.

    Returns:
        Scalar corresponding to the norm.
    """
    return np.sqrt(np.trace(R.conjugate().transpose() @ R))

def complexMatrixSim(A, B):
    norm = np.real_if_close(operator_2_norm(A - B))
    distance = 1-np.exp(-norm) # normalise to [0, 1]
    return 1 - distance

def SPSAGradient(
        x_center:np.ndarray,
        k:int,
        f:Callable[[np.ndarray], float],
        c=0.2,
        alpha=0.602,
        gamma=0.101,
        A=None,
        a=None,
        maxiter=None
):
    """

    :param x_center:
    :param k:
    :param f: the cost function. Must only have one argument, which is the parameters, and the output should be a scalar
    :param c:
    :param alpha:
    :param gamma:
    :param A:
    :param a:
    :param maxiter:
    :return:
    """
    if not maxiter and not A:
        raise TypeError("One of the parameters maxiter or A must be provided.")
    if not A:
        A = maxiter * 0.1
    if not a:
        a = 0.05 * (A + 1) ** alpha
    ck = c / k ** gamma
    delta = np.random.choice([-1, 1], size=x_center.shape)
    multiplier = ck * delta
    thetaplus = np.array(x_center) + multiplier
    thetaminus = np.array(x_center) - multiplier
    # the output of the cost function should be a scalar
    yplus = f(thetaplus)
    yminus = f(thetaminus)
    grad = [(yplus - yminus) / (2 * ck * di) for di in delta]
    # Clip gradient
    grad = np.clip(grad, -np.pi, np.pi)
    return grad

def AdamUpdateWithSPSAGrad(
        x_center:np.ndarray,
        t:int,
        f:Callable[[np.ndarray], float],
        lr:float,
        beta_1:float,
        beta_2:float,
        noise_factor:float,
        eps:float,
        maxiter:Union[int, None] = None,
        amsgrad=True,
        c=0.2,
        alpha=0.602,
        gamma=0.101,
        A=None,
        a=None,
):
    m = np.zeros_like(x_center)
    v = np.zeros_like(x_center)
    if amsgrad:
        v_eff = np.zeros_like(x_center)
    params = params_new = x_center
    derivative = SPSAGradient(params, t, f, c, alpha, gamma, A, a, maxiter)
    m = beta_1 * m + (1 - beta_1) * derivative
    v = beta_2 * v + (1 - beta_2) * (derivative ** 2)
    lr_eff = lr * np.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
    if not amsgrad:
        params_new = params-lr_eff*m.flatten()/(
            np.sqrt(v.flatten())+noise_factor
        )
    else:
        v_eff = np.maximum(v_eff, v)
        params_new = params-lr_eff*m.flatten()/(
            np.sqrt(v_eff.flatten())+noise_factor
        )
    current_obj = f(params)
    return params_new, current_obj

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
            aug_data.append(random_chosen_two[0])
            aug_data.append(random_chosen_two[1])
        batches.append(aug_data)
        if n_batches is not None:
            if len(batches) == n_batches:
                break
    return batches

def getBatchz(
        batch,
        parameters: Union[ParameterVector, np.ndarray],
        num_single_patch_data_reuploading_layers: int,
        num_single_patch_d_and_r_repetitions: int,
        num_four_patch_d_and_r_repetitions: int,
        num_two_patch_2_q_pqc_layers: int,
        num_finishing_4q_layers: int,
        device_backend=Aer.get_backend('aer_simulator'),
        simulation:bool=True,
        shadow_type = "pauli",
        shots = 100,
        n_shadows = 50,
        parallel = True,
        seed = 1701

):
    datasize = len(batch)
    parallel_args_list = zip(
        batch,
        [parameters] * datasize,
        [num_single_patch_data_reuploading_layers] * datasize,
        [num_single_patch_d_and_r_repetitions] * datasize,
        [num_four_patch_d_and_r_repetitions] * datasize,
        [num_two_patch_2_q_pqc_layers] * datasize,
        [num_finishing_4q_layers] * datasize,
        list(range(datasize)),
        [device_backend] * datasize,
        [simulation] * datasize,
        [shadow_type] * datasize,
        [shots] * datasize,
        [n_shadows] * datasize,
        [parallel] * datasize,
        [seed] * datasize
    )
    z_and_indices = [getzSingleArg(args) for args in parallel_args_list]
    z_and_indices = dask.compute(z_and_indices)[0]
    res = dict()
    for c in z_and_indices:
        res[c[1]+1] = c[0] # c[1] is the index, needs to start from 1 for the convenience of loss calculation, c[0] is the z
    return res

def lij(res_dict, i, j, tau = 1):
    zi = res_dict[i]
    zj = res_dict[j]
    numerator = np.exp(complexMatrixSim(zi, zj)/tau)
    denominator = 0
    for k in res_dict.keys():
        if k!=i:
            zk = res_dict[k]
            denominator += np.exp(complexMatrixSim(zi, zk)/tau)
    return -np.log(numerator/denominator)

def L(res_dict, tau = 1):
    loss = 0
    batch_size = len(res_dict)//2
    assert batch_size == len(res_dict)/2
    for k in range(1, batch_size+1):
        loss += lij(res_dict, 2*k-1, 2*k, tau)
        loss += lij(res_dict, 2*k, 2*k-1, tau)
    return loss/(2*batch_size)


def batchCost(
        parameters: Union[ParameterVector, np.ndarray],
        batch,
        num_single_patch_data_reuploading_layers: int,
        num_single_patch_d_and_r_repetitions: int,
        num_four_patch_d_and_r_repetitions: int,
        num_two_patch_2_q_pqc_layers: int,
        num_finishing_4q_layers: int,
        device_backend=Aer.get_backend('aer_simulator'),
        simulation:bool=True,
        shadow_type = "pauli",
        shots = 100,
        n_shadows = 50,
        parallel = True,
        seed = 1701,
        tau = 1
):
    res_dict = getBatchz(
        batch,
        parameters,
        num_single_patch_data_reuploading_layers,
        num_single_patch_d_and_r_repetitions,
        num_four_patch_d_and_r_repetitions,
        num_two_patch_2_q_pqc_layers,
        num_finishing_4q_layers,
        device_backend,
        simulation,
        shadow_type,
        shots,
        n_shadows,
        parallel,
        seed
    )
    return L(res_dict, tau)

if __name__ == "__main__":
    curr_t = nowtime()
    save_filename = curr_t + "_" + "8q_circ_4q_rep_SimCLR_classical_shadow_training_result.json"
    checkpointfile = None
    # hyperparameters
    batch_size = 10
    val_ratio = 0.2
    n_batches = 10
    num_single_patch_data_reuploading_layers = 1
    num_single_patch_d_and_r_repetitions = 2
    num_four_patch_d_and_r_repetitions = 2
    num_two_patch_2_q_pqc_layers = 1
    num_finishing_4q_layers = 1
    shots = 10
    n_shadows = 20
    shadow_type = "clifford"
    init_lr = 1e-1
    beta_1 = 0.9
    beta_2 = 0.999
    noise_factor = 1e-8
    eps = 1e-8
    c=0.2
    alpha=0.602
    gamma=0.101
    maxiter=1
    simulation = True
    hyperparameters = {
        "num_single_patch_data_reuploading_layers": num_single_patch_data_reuploading_layers,
        "num_single_patch_d_and_r_repetitions": num_single_patch_d_and_r_repetitions,
        "num_four_patch_d_and_r_repetitions": num_four_patch_d_and_r_repetitions,
        "num_two_patch_2_q_pqc_layers": num_two_patch_2_q_pqc_layers,
        "num_finishing_4q_layers": num_finishing_4q_layers,
        "shots": shots,
        "n_shadows": n_shadows,
        "shadow_type": shadow_type,
        "simulation": simulation,
        "init_lr": init_lr,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "noise_factor": noise_factor,
        "eps": eps,
        "c": c,
        "alpha": alpha,
        "gamma": gamma,
        "n_epoches": maxiter,
        "previous_checkpoint": checkpointfile,
        "save_filename": save_filename,
        "batch_size": batch_size,
        "A": None,
        "a": None
    }
    train_kwargs = {
        "num_single_patch_data_reuploading_layers": num_single_patch_data_reuploading_layers,
        "num_single_patch_d_and_r_repetitions": num_single_patch_d_and_r_repetitions,
        "num_four_patch_d_and_r_repetitions": num_four_patch_d_and_r_repetitions,
        "num_two_patch_2_q_pqc_layers": num_two_patch_2_q_pqc_layers,
        "num_finishing_4q_layers": num_finishing_4q_layers,
        "shots": shots,
        "n_shadows": n_shadows,
        "shadow_type": shadow_type,
        "simulation": simulation,
        "init_lr": init_lr,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "noise_factor": noise_factor,
        "eps": eps,
        "c": c,
        "alpha": alpha,
        "gamma": gamma,
        "n_epoches": maxiter,
        "A": None,
        "a": None
    }

    single_patch_encoding_parameter_dim = 6 * num_single_patch_data_reuploading_layers * num_single_patch_d_and_r_repetitions
    single_patch_d_and_r_phase_parameter_dim = num_single_patch_d_and_r_repetitions
    four_patch_d_and_r_phase_parameter_dim = num_four_patch_d_and_r_repetitions
    two_patch_2_q_pqc_parameter_dim = 6 * num_two_patch_2_q_pqc_layers
    finishing_4q_layers_dim = 12 * num_finishing_4q_layers  # finishing_layer_parameter

    TOTAL_PARAM_DIM = single_patch_encoding_parameter_dim + single_patch_d_and_r_phase_parameter_dim + four_patch_d_and_r_phase_parameter_dim + two_patch_2_q_pqc_parameter_dim + finishing_4q_layers_dim

    print("Total number of parameters:", TOTAL_PARAM_DIM)
    if checkpointfile is not None:
        with open(checkpointfile, 'r') as f:
            checkpoint = json.load(f)
            print("Loaded checkpoint file: " + checkpointfile)
        params = np.array(checkpoint['params'])
    else:
        params = np.random.uniform(low=-np.pi, high=np.pi, size=TOTAL_PARAM_DIM)

    # Load data
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    train_val_batches = createBatches(data, batch_size, seed=1701, n_batches=n_batches)
    train_batches = train_val_batches[:math.floor(len(train_val_batches)*(1-val_ratio))]
    val_batches = train_val_batches[math.floor(len(train_val_batches)*(1-val_ratio)):]
    test_batches = createBatches(data, batch_size, seed=1701, type="test", n_batches=math.floor(n_batches*val_ratio))

    def train_model_adam_spsa(
            train_batches:List[List[np.ndarray]],
            val_batches:Optional[List[List[np.ndarray]]]=None,
            test_batches:Optional[List[List[np.ndarray]]]=None,
            n_epoches=100,
            starting_point=params,
            num_single_patch_data_reuploading_layers: int=1,
            num_single_patch_d_and_r_repetitions: int=2,
            num_four_patch_d_and_r_repetitions: int=2,
            num_two_patch_2_q_pqc_layers: int=1,
            num_finishing_4q_layers: int=1,
            init_lr: float=1e-3,
            beta_1: float=0.9,
            beta_2: float=0.999,
            noise_factor: float=1e-8,
            eps: float=1e-8,
            amsgrad: bool=True,
            c=0.2,
            alpha=0.602,
            gamma=0.101,
            A=None,
            a=None,
            tau=1,
            device_backend=Aer.get_backend('aer_simulator'),
            simulation: bool = True,
            shadow_type="pauli",
            shots=100,
            n_shadows=50,
            parallel=True,
            seed=1701
    ):
        maxiter = len(train_batches)*n_epoches
        optimisation_counter = 1 # k in spsa, t in adam
        train_loss_list=[]
        val_loss_list=[]
        test_loss_list=[]
        all_optimisation_iterations_loss_list = []
        train_start = time.time()
        lr = init_lr
        params = starting_point
        for epoch in range(n_epoches):
            epoch_start = time.time()
            batch_loss_list = []
            for batchid in range(len(train_batches)):
                batch_start_time = time.time()
                batch = train_batches[batchid]
                costfn = lambda x: batchCost(
                    x,
                    batch,
                    num_single_patch_data_reuploading_layers,
                    num_single_patch_d_and_r_repetitions,
                    num_four_patch_d_and_r_repetitions,
                    num_two_patch_2_q_pqc_layers,
                    num_finishing_4q_layers,
                    device_backend,
                    simulation,
                    shadow_type,
                    shots,
                    n_shadows,
                    parallel,
                    seed,
                    tau
                )
                params, current_obj = AdamUpdateWithSPSAGrad(
                    params,
                    optimisation_counter,
                    costfn,
                    lr,
                    beta_1,
                    beta_2,
                    noise_factor,
                    eps,
                    maxiter,
                    amsgrad,
                    c,
                    alpha,
                    gamma,
                    A,
                    a
                )
                batch_loss_list.append(current_obj)
                optimisation_counter += 1
                batch_end_time = time.time()
                batch_time = batch_end_time-batch_start_time
                print(f"----Training at Epoch {epoch+1}, Batch {batchid+1}/{len(train_batches)}, Objective = {np.round(current_obj, 4)}, Batch Time = {np.round(batch_time, 4)}")
            epoch_end_1 = time.time()
            epoch_time_1 = epoch_end_1-epoch_start
            all_optimisation_iterations_loss_list.extend(batch_loss_list)
            batch_avg_loss = np.mean(batch_loss_list)
            train_loss_list.append(batch_avg_loss)
            print(f"Training at Epoch {epoch+1}, Objective = {np.round(batch_avg_loss, 4)}, Train Epoch Time = {np.round(epoch_time_1, 4)}")
            if val_batches is not None:
                # concatenate all the batches together
                val_batch = []
                for batch in val_batches:
                    val_batch.extend(batch)
                val_costfn = lambda x: batchCost(
                    x,
                    val_batch,
                    num_single_patch_data_reuploading_layers,
                    num_single_patch_d_and_r_repetitions,
                    num_four_patch_d_and_r_repetitions,
                    num_two_patch_2_q_pqc_layers,
                    num_finishing_4q_layers,
                    device_backend,
                    simulation,
                    shadow_type,
                    shots,
                    n_shadows,
                    parallel,
                    seed,
                    tau
                )
                val_loss = val_costfn(params)
                val_loss_list.append(val_loss)
                epoch_time_2 = time.time()-epoch_end_1
                print(f"Validation at Epoch {epoch+1}, Objective = {np.round(val_loss, 4)}, Val Time = {np.round(epoch_time_2, 4)}")
            if test_batches is not None:
                # concatenate all the batches together
                test_batch = []
                for batch in test_batches:
                    test_batch.extend(batch)
                test_costfn = lambda x: batchCost(
                    x,
                    test_batch,
                    num_single_patch_data_reuploading_layers,
                    num_single_patch_d_and_r_repetitions,
                    num_four_patch_d_and_r_repetitions,
                    num_two_patch_2_q_pqc_layers,
                    num_finishing_4q_layers,
                    device_backend,
                    simulation,
                    shadow_type,
                    shots,
                    n_shadows,
                    parallel,
                    seed,
                    tau
                )
                test_loss = test_costfn(params)
                test_loss_list.append(test_loss)
                epoch_time_3 = time.time()-epoch_end_1
                print(f"Testing at Epoch {epoch+1}, Objective = {np.round(test_loss, 4)}, Test Time = {np.round(epoch_time_3, 4)}")
            epoch_end = time.time()
            epoch_time = epoch_end-epoch_start
            print(f"Epoch {epoch+1} Total Time = {np.round(epoch_time, 4)}")
        train_end = time.time()
        train_time = train_end-train_start
        print(f"Training Total Time = {np.round(train_time, 4)}")
        return params, train_loss_list, val_loss_list, test_loss_list, all_optimisation_iterations_loss_list

    params, train_loss_list, val_loss_list, test_loss_list, all_optimisation_iterations_loss_list = train_model_adam_spsa(
        train_batches=train_batches,
        val_batches=val_batches,
        test_batches=test_batches,
        device_backend=Aer.get_backend('aer_simulator'),
        starting_point=params,
        **train_kwargs
    )
    # save the results
    with open(save_filename, 'w') as f:
        json.dump({
            "hyperparameters": hyperparameters,
            "train_loss_list": train_loss_list,
            "val_loss_list": val_loss_list,
            "test_loss_list": test_loss_list,
            "all_optimisation_iterations_loss_list": all_optimisation_iterations_loss_list,
            "params": params
        }, f, cls=NpEncoder)



