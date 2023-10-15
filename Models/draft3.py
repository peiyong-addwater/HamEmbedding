# Necessary imports

import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
#from qiskit_machine_learning.connectors import TorchConnector
from torch_connector import TorchConnector
import copy
from qiskit.algorithms.gradients import SPSASamplerGradient
from qiskit.primitives import BaseSampler, SamplerResult, Sampler

# Set seed for random generators
algorithm_globals.random_seed = 42

# Generate random dataset

# Select dataset dimension (num_inputs) and size (num_samples)
num_inputs = 2
num_samples = 20

# Generate random input coordinates (X) and binary labels (y)
X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}, y01 will be used for SamplerQNN example
y = 2 * y01 - 1  # in {-1, +1}, y will be used for EstimatorQNN example

# Convert to torch Tensors
X_ = Tensor(X)
y01_ = Tensor(y01).reshape(len(y)).long()
y_ = Tensor(y).reshape(len(y), 1)

# Set up a circuit
feature_map = ZZFeatureMap(num_inputs)
ansatz = RealAmplitudes(num_inputs)

qreg = QuantumRegister(num_inputs)
creg = ClassicalRegister(num_inputs)
creg2 = ClassicalRegister(2)

qc = QuantumCircuit(qreg, creg, creg2)
qc.compose(feature_map,qubits=qreg, inplace=True)
qc.compose(ansatz, qubits=qreg, inplace=True)
qc.measure(qreg[0], creg2[0])
qc.measure(qreg[1], creg2[1])
with qc.if_test((creg2[0],1)):
    qc.reset(qreg[0])
    qc.x(qreg[0])

