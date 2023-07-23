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
from backbone_circ_with_hierarchical_encoding import backboneCircFourQubitFeature
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from SPSAGradOptimiser.qiskit_opts.SPSA_Adam import ADAMSPSA
from qiskit.circuit import ParameterVector
import os

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

# configurations for single circuit simulation parallelisation
N_WORKERS = 11
MAX_JOB_SIZE = 10

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

def getzFromImagePatches(
        image_patches:Union[List[List[ParameterVector]], np.ndarray],
        parameters:Union[ParameterVector, np.ndarray],
        num_single_patch_data_reuploading_layers:int,
        num_single_patch_d_and_r_repetitions:int,
        num_four_patch_d_and_r_repetitions:int,
        num_two_patch_2_q_pqc_layers:int,
        num_finishing_4q_layers:int,
        device_backend=Aer.get_backend('aer_simulator'),
        simulation:bool=True,
        shadow_type = "pauli",
        shots = 100,
        n_shadows = 50,
        parallel = True,
        seed = 1701
):
    single_patch_encoding_parameter_dim = 6 * num_single_patch_data_reuploading_layers * num_single_patch_d_and_r_repetitions
    single_patch_d_and_r_phase_parameter_dim = num_single_patch_d_and_r_repetitions
    four_patch_d_and_r_phase_parameter_dim = num_four_patch_d_and_r_repetitions
    two_patch_2_q_pqc_parameter_dim = 6 * num_two_patch_2_q_pqc_layers
    finishing_4q_layers_dim = 12 * num_finishing_4q_layers  # finishing_layer_parameter

    single_patch_encoding = parameters[:single_patch_encoding_parameter_dim]
    single_patch_d_and_r_phase = parameters[
                                 single_patch_encoding_parameter_dim:single_patch_encoding_parameter_dim + single_patch_d_and_r_phase_parameter_dim]
    four_patch_d_and_r_phase = parameters[
                               single_patch_encoding_parameter_dim + single_patch_d_and_r_phase_parameter_dim:single_patch_encoding_parameter_dim + single_patch_d_and_r_phase_parameter_dim + four_patch_d_and_r_phase_parameter_dim]
    two_patch_2_q_pqc = parameters[
                        single_patch_encoding_parameter_dim + single_patch_d_and_r_phase_parameter_dim + four_patch_d_and_r_phase_parameter_dim:single_patch_encoding_parameter_dim + single_patch_d_and_r_phase_parameter_dim + four_patch_d_and_r_phase_parameter_dim + two_patch_2_q_pqc_parameter_dim]
    finishing_layer_parameter = parameters[
                                single_patch_encoding_parameter_dim + single_patch_d_and_r_phase_parameter_dim + four_patch_d_and_r_phase_parameter_dim + two_patch_2_q_pqc_parameter_dim:]

    backbone = backboneCircFourQubitFeature(
        image_patches=image_patches,
        single_patch_encoding_parameter=single_patch_encoding,
        single_patch_d_and_r_phase_parameter=single_patch_d_and_r_phase,
        four_patch_d_and_r_phase_parameter=four_patch_d_and_r_phase,
        two_patch_2_q_pqc_parameter=two_patch_2_q_pqc,
        finishing_layer_parameter=finishing_layer_parameter
    )
    # The Clifford shadow
    if shadow_type == "clifford":
        rho_shadow = cliffordShadow(
            n_shadows=n_shadows,
            n_qubits=4,
            base_circuit=backbone,
            shadow_register=[0, 1, 2, 3],
            reps=shots,
            transpile_circ=False if simulation else True,
            seed=seed,
            parallel=parallel,
            simulation=simulation,
            device_backend=device_backend
        )
    # The Pauli shadow
    elif shadow_type == "pauli":
        rho_shadow = pauliShadow(
            n_shadows=n_shadows,
            n_qubits=4,
            base_circuit=backbone,
            shadow_register=[0, 1, 2, 3],
            reps=shots,
            transpile_circ=False if simulation else True,
            seed=seed,
            parallel = parallel,
            simulation = simulation,
            device_backend = device_backend
        )
    else:
        raise ValueError("shadow_type must be either clifford or pauli")
    return rho_shadow

@dask.delayed
def getzSingleArg(args):
    return getzFromImagePatches(*args)