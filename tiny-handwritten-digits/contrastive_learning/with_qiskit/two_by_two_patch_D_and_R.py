# Deposit and reverse for the 2 x 2 patch encoding;
# D&R follows Fig 1 of https://arxiv.org/abs/2305.18961

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
import os

from two_by_two_patch_encode import FourPixelEncodeTwoQubits

def FourPixelDepositAndReverse(
        pixels:Union[list, np.ndarray, ParameterVector],
        encode_parameters:Union[List, np.ndarray, ParameterVector],
        phase_parameters:Union[List, np.ndarray, ParameterVector],
        to_gate = True
):
    """

    :param pixels: four-element array
    :param encode_parameters: 6N-element array, one-dim, where N is the number of "layers" of data-reuploading
    :param phase_parameters: L-element array, one-dim, where L is the number of D&R repetitions
    :param to_gate:
    :return: a gate or circuit, in which the first qubit will have the image patch encoded as phase information
    """
    circ = QuantumCircuit(3, name = "FourPixelDepositAndReverse")
    n_layers = len(phase_parameters)
    img_patch_encode_gate = FourPixelEncodeTwoQubits(pixels, encode_parameters, to_gate=to_gate)
    img_patch_encode_gate_inv = img_patch_encode_gate.inverse()
    circ.h(0)
    for i in range(n_layers):
        circ.append(img_patch_encode_gate, [1, 2])
        circ.cp(phase_parameters[i], 0, 1)
        circ.append(img_patch_encode_gate_inv, [1, 2])

    return circ.to_gate(label="FourPixelDepositAndReverse") if to_gate else circ

if __name__ == '__main__':

    x = ParameterVector('x', 4)
    theta = ParameterVector('θ', 12)
    phi = ParameterVector('φ', 2)
    circ = FourPixelDepositAndReverse(x, theta, phi, to_gate=False)
    circ.draw(output='mpl', style='bw', filename='FourPixelDepositAndReverse.png')

