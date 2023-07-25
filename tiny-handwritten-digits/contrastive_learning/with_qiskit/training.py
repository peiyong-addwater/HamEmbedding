import multiprocessing

import math
import numpy as np
import dill
from multiprocessing import Process, Queue
from typing import List, Tuple, Union, Callable
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

def createBatches(data, batchSize, seed = 0):
    """
    Create batches of data.

    :param data: Loaded data file "tiny-handwritten-with-augmented-as-rotation-angles-patches.pkl"
    :param batchSize: The size of each batch.
    :param seed: The random seed.
    :return: A list of batches.
    """
    train_data = data["train"]
    batches = []
    rng = np.random.default_rng(seed)
    for i in range(0, len(train_data), batchSize):
        batch_data_dict = train_data[i:i+batchSize]
        aug_data = []
        for j in range(len(batch_data_dict)):
            random_chosen_two = rng.choice(batch_data_dict[j]['augmentations'], 2, replace=False)
            aug_data.append(random_chosen_two[0])
            aug_data.append(random_chosen_two[1])
        batches.append(aug_data)
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
    datasize = len(batch) # 2*batch_size
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
    batch_size = 20
    num_single_patch_data_reuploading_layers = 1
    num_single_patch_d_and_r_repetitions = 2
    num_four_patch_d_and_r_repetitions = 2
    num_two_patch_2_q_pqc_layers = 1
    num_finishing_4q_layers = 1
    shots = 100
    n_shadows = 50
    shadow_type = "pauli"
    init_lr = 1e-3
    beta_1 = 0.9
    beta_2 = 0.999
    noise_factor = 1e-8
    eps = 1e-8
    c=0.2
    alpha=0.602
    gamma=0.101
    maxiter=100
    hyperparameters = {
        "num_single_patch_data_reuploading_layers": num_single_patch_data_reuploading_layers,
        "num_single_patch_d_and_r_repetitions": num_single_patch_d_and_r_repetitions,
        "num_four_patch_d_and_r_repetitions": num_four_patch_d_and_r_repetitions,
        "num_two_patch_2_q_pqc_layers": num_two_patch_2_q_pqc_layers,
        "num_finishing_4q_layers": num_finishing_4q_layers,
        "shots": shots,
        "n_shadows": n_shadows,
        "shadow_type": shadow_type,
        "init_lr": init_lr,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "noise_factor": noise_factor,
        "eps": eps,
        "c": c,
        "alpha": alpha,
        "gamma": gamma,
        "maxiter": maxiter,
        "previous_checkpoint": checkpointfile,
        "save_filename": save_filename,
        "batch_size": batch_size
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
    batches = createBatches(data, batch_size, seed=1701)

    def train_model_adam_spsa(
            data_batches,
            n_epoches,
            starting_point,
            num_single_patch_data_reuploading_layers: int,
            num_single_patch_d_and_r_repetitions: int,
            num_four_patch_d_and_r_repetitions: int,
            num_two_patch_2_q_pqc_layers: int,
            num_finishing_4q_layers: int,
            init_lr: float,
            beta_1: float,
            beta_2: float,
            noise_factor: float,
            eps: float,
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
        maxiter = len(data_batches)*n_epoches



