# Based on https://github.com/ryanlevy/shadow-tutorial/blob/main/Tutorial_Shadow_State_Tomography.ipynb
import multiprocessing

import math
import numpy as np
import dill
from multiprocessing import Process, Queue
from typing import List, Tuple, Union
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


class DillProcess(Process):
    # from https://stackoverflow.com/a/72776044

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = dill.dumps(self._target)  # Save the target function as bytes, using dill

    def run(self):
        if self._target:
            self._target = dill.loads(self._target)    # Unpickle the target function before executing
            self._target(*self._args, **self._kwargs)  # Execute the target function



pauli_list = [
    np.eye(2),
    np.array([[0.0, 1.0], [1.0, 0.0]]),
    np.array([[0, -1.0j], [1.0j, 0.0]]),
    np.array([[1.0, 0.0], [0.0, -1.0]]),
]
s_to_pauli = {
    "I": pauli_list[0],
    "X": pauli_list[1],
    "Y": pauli_list[2],
    "Z": pauli_list[3],
}

def bitGateMap(qc:QuantumCircuit, g:str, qi:int):
    '''Map X/Y/Z string to qiskit ops'''
    if g == "X":
        qc.h(qi)
    elif g == "Y":
        qc.sdg(qi)
        qc.h(qi)
    elif g == "Z":
        pass
    else:
        raise NotImplementedError(f"Unknown gate {g}")

def Minv(N, X):
    '''inverse shadow channel'''
    return ((2 ** N + 1.)) * X - np.eye(2 ** N)

# Note: Qiskit uses little-endian bit ordering
def rotGate(g):
    '''produces gate U such that U|psi> is in Pauli basis g'''
    if g=="X":
        return 1/np.sqrt(2)*np.array([[1.,1.],[1.,-1.]])
    elif g=="Y":
        return 1/np.sqrt(2)*np.array([[1.,-1.0j],[1.,1.j]])
    elif g=="Z":
        return np.eye(2)
    else:
        raise NotImplementedError(f"Unknown gate {g}")

def operator_2_norm(R):
    """
    Calculate the operator 2-norm.

    Args:
        R (array): The operator whose norm we want to calculate.

    Returns:
        Scalar corresponding to the norm.
    """
    return np.sqrt(np.trace(R.conjugate().transpose() @ R))

def complexMatrixDiff(A, B):
    return np.real_if_close(operator_2_norm(A - B))

def constructCliffordShadowSingleCirc(n_qubits:int,
                   base_circuit:QuantumCircuit,
                   clifford,
                   circ_name:str,
                   shadow_register:Union[QuantumRegister, List[QuantumRegister], List[int]],
                   device_backend=Aer.get_backend('aer_simulator'),
                   transpile_circ:bool=False):
    """

    :param n_qubits: number of qubits involved in the shadow
    :param base_circuit: the base circuit that produce the target state, should be only quantum regs
    :param clifford: the random Clifford that is going to be appended to the base circuit
    :param circ_name: name of the circuit
    :param shadow_register: the quantum register involved in the shadow
    :param device_backend: backend to run the circuit
    :param transpile_circ: whether to transpile the circuit
    :return:
    """
    shadow_meas = ClassicalRegister(n_qubits, name="shadow")
    qc = base_circuit.copy()
    qc.add_register(shadow_meas)
    qc.append(clifford.to_instruction(), shadow_register)
    qc.measure(shadow_register, shadow_meas)
    qc = qc.decompose(reps=4)
    if transpile_circ:
        qc = transpile(qc, device_backend)
    qc.name = circ_name
    return circ_name, qc, clifford

@dask.delayed
def constructCliffordShadowSingeCirSingleArg(args):
    n_qubits, base_circuit, clifford, circ_name, shadow_register, device_backend, transpile_circ = args
    return constructCliffordShadowSingleCirc(n_qubits=n_qubits, base_circuit=base_circuit, clifford=clifford,
                                             circ_name=circ_name, shadow_register=shadow_register,
                                             device_backend=device_backend, transpile_circ=transpile_circ)


def cliffordShadow(n_shadows:int,
                   n_qubits:int,
                   base_circuit:QuantumCircuit,
                   shadow_register:Union[QuantumRegister, List[QuantumRegister], List[int]],
                   device_backend=Aer.get_backend('aer_simulator'),
                   reps:int=1,
                   seed = 1701,
                   parallel:bool=True,
                   simulation:bool=True,
                   transpile_circ:bool=True
                   ):
    """
    Shadow state tomography with Clifford circuits
    :param n_shadows: number of classical shadows
    :param n_qubits: number of qubits in the target state/classical shadow
    :param base_circuit: the circuit producing the target state
    :param shadow_register: the registers involving the target state
    :param device_backend: backend to run the circuit
    :param reps: number of measurement repetitions/shots
    :param seed: random seed for generating the Clifford circuit
    :param simulation: whether running on a simulator
    :param parallel: whether running in parallel
    :param transpile_circ: whether to transpile the circuits
    :return:
    """
    rng = np.random.default_rng(seed)
    cliffords = [random_clifford(n_qubits, seed=rng) for _ in range(n_shadows)]
    if simulation and parallel:
        N_WORKERS = 11
        MAX_JOB_SIZE = 10
        EXC = ThreadPoolExecutor(max_workers=N_WORKERS)
        device_backend.set_options(executor=EXC)
        device_backend.set_options(max_job_size=MAX_JOB_SIZE)
        device_backend.set_options(max_parallel_experiments=0)

    # args for the parallelled construction of the shadow circuits
    n_qubits_list = [n_qubits] * n_shadows
    base_circuit_list = [base_circuit] * n_shadows
    circ_name_list = [f"Shadow_{i}" for i in range(n_shadows)]
    shadow_register_list = [shadow_register] * n_shadows
    dev_backend_list = [device_backend] * n_shadows
    transpile_circ_list = [transpile_circ] * n_shadows
    parallel_args_list = zip(n_qubits_list, base_circuit_list, cliffords, circ_name_list, shadow_register_list,dev_backend_list,transpile_circ_list)
    shadow_circ_and_names = [constructCliffordShadowSingeCirSingleArg(args) for args in parallel_args_list]
    shadow_circ_and_names=dask.compute(shadow_circ_and_names)[0]
    shadow_circs = []
    cliffords_dict = {}
    for (name, circ, clifford) in shadow_circ_and_names:
        shadow_circs.append(circ)
        cliffords_dict[name] = clifford
    job = device_backend.run(shadow_circs, shots=reps)
    result_dict = job.result().to_dict()["results"]
    result_counts = job.result().get_counts()
    shadows = []
    for i in range(n_shadows):
        name = result_dict[i]["header"]["name"]
        counts = result_counts[i]
        mat = cliffords_dict[name].adjoint().to_matrix()
        for bit, count in counts.items():
            Ub = mat[:, int(bit, 2)] # this is Udag|b>
            shadows.append(Minv(n_qubits, np.outer(Ub, Ub.conj())) * count)
    rho_shadow = np.sum(shadows, axis=0) / (n_shadows * reps)
    return rho_shadow

def constructPauliShadowSingleCirc(n_qubits:int,
                                   base_circuit:QuantumCircuit,
                                   pauli_string:List[str],
                                   circ_name:str,
                                   shadow_register:Union[QuantumRegister, List[QuantumRegister], List[int]],
                                   device_backend=Aer.get_backend('aer_simulator'),
                                   transpile_circ:bool=False
                                   ):
    """

    :param n_qubits: number of qubits involved in the classical shadow
    :param base_circuit: the base circuit that produces the state to be tomographed
    :param pauli_string: the random Pauli string to be measured
    :param circ_name: name of the circuit
    :param shadow_register: the quantum register involving the classical shadow
    :param device_backend: the backend to run the circuit
    :param transpile_circ: whether to transpile the circuit
    :return:
    """
    shadow_meas = ClassicalRegister(n_qubits, name="shadow")
    qc_m = base_circuit.copy()
    qc_bitstring = QuantumCircuit(n_qubits)
    for j, bit in enumerate(pauli_string):
        bitGateMap(qc_bitstring, bit, j)
    qc_m.append(qc_bitstring.to_instruction(), shadow_register)
    qc_m.add_register(shadow_meas)
    qc_m.measure(shadow_register, shadow_meas)
    qc_m = qc_m.decompose(reps=4)
    if transpile_circ:
        qc_m = transpile(qc_m, device_backend)
    qc_m.name = circ_name
    return circ_name, qc_m, pauli_string

@dask.delayed
def constructPauliShadowSingleCircSingleArg(args):
    return constructPauliShadowSingleCirc(*args)

def pauliShadow(
        n_shadows:int,
        n_qubits:int,
        base_circuit:QuantumCircuit,
        shadow_register:Union[QuantumRegister, List[QuantumRegister], List[int]],
        device_backend=Aer.get_backend('aer_simulator'),
        reps:int=1,
        seed = 1701,
        parallel:bool=True,
        simulation:bool=True,
        transpile_circ:bool=False
):
    """
    Shadow state tomography with random Pauli measurements
    :param n_shadows: number of classical shadows
    :param n_qubits: number of qubits involved for the classical shadows
    :param base_circuit: the base circuit producing the target state
    :param shadow_register: the quantum register involving the target state
    :param device_backend: device to run/simulate the circuit
    :param reps: number of repetitions/shots for the measurement
    :param seed: random seed
    :param parallel: whether to run in parallel
    :param simulation: whether to run on a simulator
    :param transpile_circ: whether to transpile the circuit
    :return:
    """
    if simulation and parallel:
        N_WORKERS = 11
        MAX_JOB_SIZE = 10
        EXC = ThreadPoolExecutor(max_workers=N_WORKERS)
        device_backend.set_options(executor=EXC)
        device_backend.set_options(max_job_size=MAX_JOB_SIZE)
        device_backend.set_options(max_parallel_experiments=0)
    rng = np.random.default_rng(seed)
    scheme = [rng.choice(['X', 'Y', 'Z'], size=n_qubits).tolist() for _ in range(n_shadows)]
    shadow_circs = []
    shadow_circ_name_pauli_dict = {}
    # args for the parallelled construction of the shadow circuits
    n_qubits_list = [n_qubits]*n_shadows
    base_circuit_list = [base_circuit] * n_shadows
    circ_name_list = [f"Shadow_{i}" for i in range(n_shadows)]
    shadow_register_list = [shadow_register] * n_shadows
    dev_backend_list = [device_backend] * n_shadows
    transpile_circ_list = [transpile_circ] * n_shadows
    parallel_arg_list = zip(n_qubits_list, base_circuit_list, scheme, circ_name_list, shadow_register_list, dev_backend_list, transpile_circ_list)
    shadow_circ_and_names = [constructPauliShadowSingleCircSingleArg(args) for args in parallel_arg_list]
    shadow_circ_and_names = dask.compute(shadow_circ_and_names)[0]
    for circ_name, circ, pauli_string in shadow_circ_and_names:
        shadow_circs.append(circ)
        shadow_circ_name_pauli_dict[circ_name] = pauli_string

    job = device_backend.run(shadow_circs, shots=reps)
    result_dict = job.result().to_dict()["results"]
    result_counts = job.result().get_counts()
    shadows = []
    for i in range(n_shadows):
        name = result_dict[i]["header"]["name"]
        counts = result_counts[i]
        pauli_string = shadow_circ_name_pauli_dict[name]
        for bit, count in counts.items():
            mat = 1.
            # reverse ordering since Qiskit is little-endian
            for j, bi in enumerate(bit[::-1]):
                b = rotGate(pauli_string[j])[int(bi), :]
                mat = np.kron(Minv(1, np.outer(b.conj(), b)), mat)
            shadows.append(mat * count)
    rho_shadow = np.sum(shadows, axis=0) / (n_shadows * reps)
    return rho_shadow



if __name__ == '__main__':
    """
    Benchmarking the "accuracy" of shadow state tomography
    """
    import matplotlib.pyplot as plt
    import json
    from qiskit.visualization.state_visualization import plot_state_city
    from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
    from backbone_circ_with_hierarchical_encoding import backboneCircFourQubitFeature
    import time


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

    # Times of sampling for calculating shadows
    SAMPLES = 5

    GLOBAL_RNG = np.random.default_rng(42)

    # Structural parameters of the backbone circuit
    N = 1 # number of data-reuploading layers for a single patch
    L = 1 # number of D&R repetitions in the single patch D&R layer
    M = 1 # number of 4-qubit parameterised layers after the 2x2 local patches in the LocalTokenMixing layer
    K = 1 # number of 4-qubit  parameterised layers at the end of the backbone circuit, right before producing the learned representation
    THETA_DIM = 6*N*L # single_patch_encoding_parameter
    PHI_DIM = L # single_patch_d_and_r_phase_parameter
    GAMMA_DIM = 12*M # four_q_param_layer_parameter_local_patches
    OMEGA_DIM = 1 # local_token_mixing_phase_parameter
    ETA_DIM = 12*K # finishing_layer_parameter
    TOTAL_PARAM_DIM = THETA_DIM + PHI_DIM + GAMMA_DIM + OMEGA_DIM + ETA_DIM
    print("Total number of parameters:", TOTAL_PARAM_DIM)

    # Each "image" in the data file is a 4x4x4 array, each element in the first 4x4 is a flattend patch
    DATA_FILE = "/home/peiyongw/Desktop/Research/QML-ImageClassification/tiny-handwritten-digits/contrastive_learning/with_qiskit/tiny-handwritten-as-rotation-angles-patches.pkl"
    with open(DATA_FILE, "rb") as f:
        patched_data = pickle.load(f)
    image_patches = patched_data["training_patches"] + patched_data["test_patches"]

    def calculate_shadows_single_image_single_parameter(image, parameters, shadow_type, n_shadows, seed):
        """
        Calculate the Clifford and Pauli shadows, as well as extracting the original density matrix, for a single image with a single set of parameter

        :param image: patches of a single image
        :param parameters: a single set of parameters, dim = THETA_DIM + PHI_DIM + GAMMA_DIM + OMEGA_DIM + ETA_DIM
        :return:
        """
        # Construct the backbone circuit
        single_patch_encoding_parameter = parameters[:THETA_DIM]
        single_patch_d_and_r_phase_parameter = parameters[THETA_DIM:THETA_DIM+PHI_DIM]
        four_q_param_layer_parameter_local_patches = parameters[THETA_DIM+PHI_DIM:THETA_DIM+PHI_DIM+GAMMA_DIM]
        local_token_mixing_phase_parameter = parameters[THETA_DIM+PHI_DIM+GAMMA_DIM:THETA_DIM+PHI_DIM+GAMMA_DIM+OMEGA_DIM]
        finishing_layer_parameter = parameters[THETA_DIM+PHI_DIM+GAMMA_DIM+OMEGA_DIM:THETA_DIM+PHI_DIM+GAMMA_DIM+OMEGA_DIM+ETA_DIM]
        backbone = backboneCircFourQubitFeature(
            image_patches=image,
            single_patch_encoding_parameter=single_patch_encoding_parameter,
            single_patch_d_and_r_phase_parameter=single_patch_d_and_r_phase_parameter,
            four_q_param_layer_parameter_local_patches=four_q_param_layer_parameter_local_patches,
            local_token_mixing_phase_parameter=local_token_mixing_phase_parameter,
            finishing_layer_parameter=finishing_layer_parameter
        )
        # The reduced density matrix of the first 4 qubits
        rho_actual = partial_trace(DensityMatrix(backbone), [4, 5, 6, 7, 8, 9])
        rho_actual = rho_actual.data
        # The Clifford shadow
        if shadow_type == "clifford":
            rho_shadow = cliffordShadow(
                n_shadows=n_shadows,
                n_qubits=4,
                base_circuit=backbone,
                shadow_register=[0,1, 2, 3],
                reps=1,
                transpile_circ=False,
                seed=seed
            )
        # The Pauli shadow
        elif shadow_type == "pauli":
            rho_shadow = pauliShadow(
                n_shadows=n_shadows,
                n_qubits=4,
                base_circuit=backbone,
                shadow_register=[0,1, 2, 3],
                reps=1,
                transpile_circ=False,
                seed=seed
            )
        else:
            raise ValueError("shadow_type must be either clifford or pauli")
        return rho_actual, rho_shadow

    pauli_shadow_sizes = [100, 500, 1000, 2000, 5000]
    clifford_shadow_sizes = [100, 500, 1000, 2000, 5000]
    # calculate pauli shadow accuracy
    pauli_shadow_accuracies = []
    pauli_shadow_time=[]
    for n in pauli_shadow_sizes:
        seeds = GLOBAL_RNG.integers(low=0, high=10000, size=SAMPLES)
        pauli_shadow_accuracies_single_size = []
        pauli_shadow_time_single_size=[]
        print("Calculating Pauli shadow accuracy for n_shadows =", n)
        for seed in seeds:
            start = time.time()
            print("Calculating Pauli shadow accuracy for n_shadows =", n, "and seed =", seed)
            parameters = GLOBAL_RNG.uniform(low=-np.pi, high=np.pi, size=TOTAL_PARAM_DIM)
            img = GLOBAL_RNG.choice(image_patches)
            rho_actual, rho_shadow = calculate_shadows_single_image_single_parameter(image=img, parameters=parameters, shadow_type="pauli", n_shadows=n, seed=seed)
            pauli_shadow_accuracies_single_size.append(complexMatrixDiff(rho_actual, rho_shadow))
            end = time.time()
            pauli_shadow_time_single_size.append(end-start)
            print("Time taken:", end-start)
        pauli_shadow_accuracies.append(pauli_shadow_accuracies_single_size)
        pauli_shadow_time.append(pauli_shadow_time_single_size)
    # calculate clifford shadow accuracy
    clifford_shadow_accuracies = []
    clifford_shadow_time = []
    for n in clifford_shadow_sizes:
        print("Calculating Clifford shadow accuracy for n_shadows =", n)
        seeds = GLOBAL_RNG.integers(low=0, high=10000, size=SAMPLES)
        clifford_shadow_accuracies_single_size = []
        clifford_shadow_time_single_size = []
        for seed in seeds:
            start = time.time()
            print("Calculating Clifford shadow accuracy for n_shadows =", n, "and seed =", seed)
            parameters = GLOBAL_RNG.uniform(low=-np.pi, high=np.pi, size=TOTAL_PARAM_DIM)
            img = GLOBAL_RNG.choice(image_patches)
            rho_actual, rho_shadow = calculate_shadows_single_image_single_parameter(image=img, parameters=parameters, shadow_type="clifford", n_shadows=n, seed=seed)
            clifford_shadow_accuracies_single_size.append(complexMatrixDiff(rho_actual, rho_shadow))
            end = time.time()
            clifford_shadow_time_single_size.append(end-start)
            print("Time taken:", end-start)
        clifford_shadow_accuracies.append(clifford_shadow_accuracies_single_size)
        clifford_shadow_time.append(clifford_shadow_time_single_size)
    # save the results
    res_dict = {
        "pauli_shadows": (pauli_shadow_sizes,pauli_shadow_accuracies),
        "clifford_shadows": (clifford_shadow_sizes,clifford_shadow_accuracies),
        "pauli_shadow_time": (pauli_shadow_sizes, pauli_shadow_time),
        "clifford_shadow_time": (clifford_shadow_sizes, clifford_shadow_time)
    }
    with open("shadow_accuracy_benchmark_with_reset.json", "w") as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)
    print("Done!")
    print("Results saved to shadow_accuracy_benchmark_with_reset.json")
    print("Plotting the results...")
    # plot the results with error bars
    distances_pauli = np.zeros((SAMPLES, len(pauli_shadow_sizes)))
    for j in range(len(pauli_shadow_sizes)):
        for i in range(SAMPLES):
            distances_pauli[i, j] = pauli_shadow_accuracies[j][i]
    distances_clifford = np.zeros((SAMPLES, len(clifford_shadow_sizes)))
    for j in range(len(clifford_shadow_sizes)):
        for i in range(SAMPLES):
            distances_clifford[i, j] = clifford_shadow_accuracies[j][i]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0].errorbar(pauli_shadow_sizes, np.mean(distances_pauli, axis=0), yerr=np.std(distances_pauli, axis=0), fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    ax[0].set_xlabel("Number of Pauli Shadows")
    ax[0].set_ylabel("Distance to Actual State")
    ax[0].set_title("Pauli Shadows")
    ax[1].errorbar(clifford_shadow_sizes, np.mean(distances_clifford, axis=0), yerr=np.std(distances_clifford, axis=0), fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    ax[1].set_xlabel("Number of Clifford Shadows")
    ax[1].set_ylabel("Distance to Actual State")
    ax[1].set_title("Clifford Shadows")
    plt.savefig("shadow_accuracy_benchmark_with_reset.png")
    plt.close()












