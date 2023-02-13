import os.path

import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from qiskit_aer import AerSimulator
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA
from qiskit_aer.noise import NoiseModel
import json
import time
import shutup
import pickle

shutup.please()

WKDIR = "/scratch/pawsey0419/peiyongw/QML-ImageClassification/pawsey/tiny-images"
DATA_PATH = "/scratch/pawsey0419/peiyongw/QML-ImageClassification/data/mini-digits/tiny-handwritten.pkl"
NOISE_MODEL_PATH = "/scratch/pawsey0419/peiyongw/QML-ImageClassification/pawsey/tiny-images/ibm_perth-20230213-161111.noise_model"

with open(NOISE_MODEL_PATH, 'rb') as f:
    ibm_perth_noise_model = pickle.load(f)

print(ibm_perth_noise_model)

