# Based on https://github.com/ryanlevy/shadow-tutorial/blob/main/Tutorial_Shadow_State_Tomography.ipynb
import math
import numpy as np
import dill
from multiprocessing import Process
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer
from qiskit.compiler import transpile
from concurrent.futures import ThreadPoolExecutor
from qiskit.quantum_info import random_clifford
from qiskit.tools import parallel_map
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
from backbone_circ_with_hierarchical_encoding import backboneCircFourQubitFeature

class DillProcess(Process):
    # from https://stackoverflow.com/a/72776044

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = dill.dumps(self._target)  # Save the target function as bytes, using dill

    def run(self):
        if self._target:
            self._target = dill.loads(self._target)    # Unpickle the target function before executing
            self._target(*self._args, **self._kwargs)  # Execute the target function


# Each "image" in the data file is a 4x4x4 array, each element in the first 4x4 is a flattend patch
DATA_FILE = "/home/peiyongw/Desktop/Research/QML-ImageClassification/tiny-handwritten-digits/contrastive_learning/with_qiskit/tiny-handwritten-as-rotation-angles-patches.pkl"

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
                   transpile_circ:bool=True):
    """

    :param n_qubits:
    :param base_circuit:
    :param clifford:
    :param circ_name:
    :param shadow_register:
    :param device_backend:
    :param transpile_circ:
    :return:
    """
    #TODO: parallel construction of clifford shadow circuits
    pass



def cliffordShadow(n_shadows:int,
                   n_qubits:int,
                   base_circuit:QuantumCircuit,
                   shadow_register:Union[QuantumRegister, List[QuantumRegister], List[int]],
                   device_backend=Aer.get_backend('aer_simulator'),
                   reps:int=1,
                   seed = 1701,
                   parallel:bool=True,
                   simulation:bool=True,
                   transpile:bool=True
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
    :param transpile: whether to transpile the circuits
    :return:
    """
    rng = np.random.default_rng(seed)
    cliffords = [random_clifford(n_qubits, seed=rng) for _ in range(n_shadows)]
    shadow_circs = []
    cliffords_dict = {}
    if simulation and parallel:
        N_WORKERS = 11
        MAX_JOB_SIZE = 10
        EXC = ThreadPoolExecutor(max_workers=N_WORKERS)
        device_backend.set_options(executor=EXC)
        device_backend.set_options(max_job_size=MAX_JOB_SIZE)
        device_backend.set_options(max_parallel_experiments=0)


    for i in range(len(cliffords)):
        #print(i)
        shadow_meas = ClassicalRegister(n_qubits, name="shadow")
        clifford = cliffords[i]
        qc = base_circuit.copy()
        qc.add_register(shadow_meas)
        qc.append(clifford.to_instruction(), shadow_register)
        qc.measure(shadow_register, shadow_meas)
        qc = qc.decompose(reps=4)
        qc.name = f"Shadow_{i}"
        cliffords_dict[f"Shadow_{i}"] = clifford
        shadow_circs.append(qc)
    name_list = [qc.name for qc in shadow_circs]
    #TODO: parallel transpilation of the circuits
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
                                   transpile_circ:bool=True
                                   ):
    """

    :param n_qubits:
    :param base_circuit:
    :param pauli_string:
    :param circ_name:
    :param shadow_register:
    :param device_backend:
    :param transpile_circ:
    :return:
    """
    # TODO: parallel construction of Pauli shadow circuits
    pass

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
        transpile:bool=False
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
    :param transpile: whether to transpile the circuit
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
    for i in range(n_shadows):
        print(i)
        name = f"Shadow_{i}"
        shadow_meas = ClassicalRegister(n_qubits, name="shadow")
        qc_m = base_circuit.copy()
        bit_string = scheme[i]
        qc_bitstring = QuantumCircuit(n_qubits)
        for j, bit in enumerate(bit_string):
            bitGateMap(qc_bitstring, bit, j)
        qc_m.append(qc_bitstring.to_instruction(), shadow_register)
        qc_m.add_register(shadow_meas)
        qc_m.measure(shadow_register, shadow_meas)
        qc_m = qc_m.decompose(reps=4)
        #TODO: parallel transpilation of the circuits
        qc_m.name = name
        shadow_circs.append(qc_m)
        shadow_circ_name_pauli_dict[name] = bit_string
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
    import matplotlib.pyplot as plt
    from qiskit.visualization.state_visualization import plot_state_city
    from qiskit.quantum_info import DensityMatrix
    def cut_8x8_to_2x2(img: np.ndarray):
        # img: 8x8 image
        # return: 4x4x4 array, each element in the first 4x4 is a flattened patch
        patches = np.zeros((4, 4, 4))
        for i in range(4):
            for j in range(4):
                patches[i, j] = img[2 * i:2 * i + 2, 2 * j:2 * j + 2].flatten()
        return patches


    img = np.arange(64).reshape(8, 8)
    patches = cut_8x8_to_2x2(img)
    """
    print(patches)
    print(img)
    first_four_patches = patches[:2, :2]
    print(first_four_patches)
    for i in range(2):
        for j in range(2):
            print("Patch ", i, j)
            print(patches[i * 2:i * 2 + 2, j * 2:j * 2 + 2])
    """
    theta = np.random.randn(12)
    phi = np.random.randn(2)
    gamma = np.random.randn(12)
    omega = np.random.randn(1)
    eta = np.random.randn(12)

    nShadows = 1000

    backbone = backboneCircFourQubitFeature(patches,theta, phi, gamma, omega, eta)
    rho_shadow = pauliShadow(nShadows, 4, backbone, [0,1,2,3], transpile=True, parallel=True, simulation=True)
    #rho_shadow = cliffordShadow(nShadows, 4, backbone, [0,1,2,3], transpile=True, parallel=True, simulation=True)
    rho_actual = qiskit.quantum_info.partial_trace(DensityMatrix(backbone), [4, 5, 6, 7, 8, 9]).data
    print(complexMatrixDiff(rho_actual, rho_shadow))
    print(qiskit.quantum_info.state_fidelity(DensityMatrix(rho_shadow), DensityMatrix(rho_actual), validate= False))

    plt.subplot(121)
    plt.suptitle("Correct")
    plt.imshow(rho_actual.real, vmax=0.7, vmin=-0.7)
    plt.subplot(122)
    plt.imshow(rho_actual.imag, vmax=0.7, vmin=-0.7)
    plt.savefig("correct-pauli.png")

    plt.subplot(121)
    plt.suptitle(f"Shadow(pauli)-{nShadows}-shadows")
    plt.imshow(rho_shadow.real, vmax=0.7, vmin=-0.7)
    plt.subplot(122)
    plt.imshow(rho_shadow.imag, vmax=0.7, vmin=-0.7)
    plt.savefig(f"shadow-pauli-{nShadows}-shadows.png")

    plot_state_city(rho_actual, title="Correct").savefig("correct-city.png")
    plot_state_city(rho_shadow, title="Shadow (pauli)").savefig(
        f"shadow-pauli-{nShadows}-shadows-city.png")
    plt.close('all')




