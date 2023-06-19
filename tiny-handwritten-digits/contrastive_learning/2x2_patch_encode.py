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

def FourPixelEncodeTwoQubits(pixels:Union[list, np.ndarray], parameters:Union[List, np.ndarray]):
    """

    :param pixels: 4-element array
    :param parameters: 6N-element array, one-dim, where N is the number of "layers" of data-reuploading
    :return:
    """
    circ = QuantumCircuit(2)
    n_layers = len(parameters) // 6
    for i in range(n_layers):
        circ.rx(pixels[0], 0)
        circ.ry(pixels[1], 0)
        circ.rx(pixels[2], 1)
        circ.ry(pixels[3], 1)
        circ.cz(0, 1)
        circ.u(parameters[6*i], parameters[6*i+1], parameters[6*i+2], 0)
        circ.u(parameters[6*i+3], parameters[6*i+4], parameters[6*i+5], 1)
        circ.cz(0, 1)

    return circ.to_gate(label="PatchEncode")

