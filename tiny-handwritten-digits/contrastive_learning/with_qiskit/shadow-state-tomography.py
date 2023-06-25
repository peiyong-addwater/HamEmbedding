# Based on https://github.com/ryanlevy/shadow-tutorial/blob/main/Tutorial_Shadow_State_Tomography.ipynb
import math
import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from concurrent.futures import ThreadPoolExecutor
from qiskit.quantum_info import random_clifford
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

def bitGateMap(qc, g, qi):
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

def cliffordShadow(n_shadows:int,
                   n_qubits:int,
                   base_circuit:QuantumCircuit,
                   shadow_register:Union[QuantumRegister, List[QuantumRegister], List[int]],
                   device_backend=Aer.get_backend('aer_simulator'),
                   reps:int=1,
                   seed = 1701,
                   parallel:bool=True,
                   simulation:bool=True
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
        print(i)
        shadow_meas = ClassicalRegister(n_qubits, name="shadow")
        clifford = cliffords[i]
        qc = base_circuit.copy()
        qc.add_register(shadow_meas)
        qc.append(clifford.to_instruction(), shadow_register)
        qc.measure(shadow_register, shadow_meas)
        #qc = transpile(qc.decompose(reps=4), device_backend)
        qc = qc.decompose(reps=4)
        qc.name = f"Shadow_{i}"
        cliffords_dict[f"Shadow_{i}"] = clifford
        shadow_circs.append(qc)
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

    nShadows = 512

    backbone = backboneCircFourQubitFeature(patches,theta, phi, gamma, omega, eta)
    rho_actual = qiskit.quantum_info.partial_trace(DensityMatrix(backbone), [4,5,6,7,8,9]).data
    rho_shadow = cliffordShadow(nShadows, 4, backbone, [0,1,2,3])
    print(complexMatrixDiff(rho_actual, rho_shadow))
    print(qiskit.quantum_info.state_fidelity(DensityMatrix(rho_shadow), DensityMatrix(rho_actual), validate= False))

    plt.subplot(121)
    plt.suptitle("Correct")
    plt.imshow(rho_actual.real, vmax=0.7, vmin=-0.7)
    plt.subplot(122)
    plt.imshow(rho_actual.imag, vmax=0.7, vmin=-0.7)
    plt.savefig("correct-clifford.png")

    plt.subplot(121)
    plt.suptitle(f"Shadow(Clifford)-{nShadows}-shadows")
    plt.imshow(rho_shadow.real, vmax=0.7, vmin=-0.7)
    plt.subplot(122)
    plt.imshow(rho_shadow.imag, vmax=0.7, vmin=-0.7)
    plt.savefig(f"shadow-clifford-{nShadows}-shadows.png")

    plot_state_city(rho_actual, title="Correct").savefig("correct-city.png")
    plot_state_city(rho_shadow, title="Shadow (clifford)").savefig(
        f"shadow-clifford-{nShadows}-shadows-city.png")
    plt.close('all')





