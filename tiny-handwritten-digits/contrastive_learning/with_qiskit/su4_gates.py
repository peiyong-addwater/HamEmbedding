import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
from typing import List, Tuple, Union
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle
import os.path

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from qiskit_aer import AerSimulator
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.circuit import ParameterVector
import json
import time
import shutup
import pickle
import sys

def SU4Gate(params:Union[ParameterVector, np.ndarray], to_gate = True):
    su4 = QuantumCircuit(2, name='SU4')
    su4.u(params[0], params[1], params[2], qubit=0)
    su4.u(params[3], params[4], params[5], qubit=1)
    su4.cx(0, 1)
    su4.ry(params[6], 0)
    su4.rz(params[7], 1)
    su4.cx(1, 0)
    su4.ry(params[8], 0)
    su4.cx(0, 1)
    su4.u(params[9], params[10], params[11], 0)
    su4.u(params[12], params[13], params[14], 1)

    return su4.to_gate(label="SU4") if to_gate else su4

