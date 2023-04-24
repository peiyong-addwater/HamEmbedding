import os.path

import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA
import json
import time
import shutup
import pickle


shutup.please()

from qiskit_ibm_provider import IBMProvider
PROVIDER = IBMProvider()

