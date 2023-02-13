import os.path

import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from qiskit_aer import AerSimulator
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA
import json
import time
import shutup
import pickle

shutup.please()

WKDIR = "/scratch/pawsey0419/peiyongw/QML-ImageClassification/pawsey/tiny-images"
DATA_PATH = "/scratch/pawsey0419/peiyongw/QML-ImageClassification/data/mini-digits/tiny-handwritten.pkl"

