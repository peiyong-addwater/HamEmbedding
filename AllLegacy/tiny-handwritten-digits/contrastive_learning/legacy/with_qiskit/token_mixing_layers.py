import math
import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA, COBYLA
import json
import time
import shutup
import pickle
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from SPSAGradOptimiser.qiskit_opts.SPSA_Adam import ADAMSPSA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QFT
import os

def FourQubitParameterisedLayer(parameters:Union[ParameterVector, np.ndarray], to_gate = True):
    """
    A four-qubit parameterised layer, which ends with circular entanglement with CZ gates
    :param parameters: 12N-element array, one-dim, where N is the number of layers
    :param to_gate:
    :return:
    """
    circ = QuantumCircuit(4, name = "FourQPQC")
    layers = len(parameters) // 12
    assert len(parameters) % 12 == 0
    for i in range(layers):
        for j in range(4):
            circ.u(parameters[12 * i + 3 * j], parameters[12 * i + 3 * j + 1], parameters[12 * i + 3 * j + 2], j)
        circ.cz(0, 1)
        circ.cz(1, 2)
        circ.cz(2, 3)
        circ.cz(3, 0)

    return circ.to_instruction(label="FourQPQC") if to_gate else circ

def TwoQubitParameterisedLayer(parameters:Union[ParameterVector, np.ndarray], to_gate = True):
    """
    A parameterised layer for two qubits.
    :param parameters:
    :param to_gate:
    :return:
    """
    circ = QuantumCircuit(2, name="TwoQPQC")
    layers = len(parameters) // 6
    assert len(parameters) % 6 == 0
    for i in range(layers):
        for j in range(2):
            circ.u(parameters[6 * i + 3 * j], parameters[6 * i + 3 * j + 1], parameters[6 * i + 3 * j + 2], j)
        circ.cz(0, 1)
    return circ.to_instruction(label="TwoQPQC") if to_gate else circ




def PermutationInvariantFourQLayer(
    parameters:Union[ParameterVector, np.ndarray],
    to_gate = True
):
    """
    A permutation invariant four-qubit parameterised layer composed of RX, RY and IsingZZ gates.
    N layers have 3N parameters.
    :param parameters:
    :param to_gate:
    :return:
    """
    layers = len(parameters) // 3
    assert int(layers) == layers
    circ = QuantumCircuit(4, name="FourQLayerPermInvariant")
    for i in range(layers):
        for j in range(4):
            circ.rx(parameters[3 * i], j)
            circ.ry(parameters[3 * i + 1], j)
        circ.rzz(parameters[3 * i + 2], 0, 1)
        circ.rzz(parameters[3 * i + 2], 1, 2)
        circ.rzz(parameters[3 * i + 2], 2, 3)
        circ.rzz(parameters[3 * i + 2], 3, 0)
        circ.rzz(parameters[3 * i + 2], 0, 2)
        circ.rzz(parameters[3 * i + 2], 1, 3)
    return circ.to_instruction(label="FourQLayerPermInvariant") if to_gate else circ

def QFTTokenMixingLayer(n_wires=4):
    """
    Token mixing with quantum Fourier transform
    :param n_wires: number of qubits
    :return:
    """
    circ = QuantumCircuit(n_wires, name="QFTTokenMixingLayer")
    circ.compose(QFT(num_qubits=n_wires), range(n_wires), inplace=True)
    return circ.to_instruction(label="QFTTokenMixingLayer")