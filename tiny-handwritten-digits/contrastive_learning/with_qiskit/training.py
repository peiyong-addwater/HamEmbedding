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

DATA_FILE = "/home/peiyongw/Desktop/Research/QML-ImageClassification/tiny-handwritten-digits/contrastive_learning/with_qiskit/tiny-handwritten-as-rotation-angles-patches.pkl"


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

