# Necessary imports

import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
#from qiskit_machine_learning.connectors import TorchConnector
from torch_connector import TorchConnector
import copy
from qiskit.algorithms.gradients import SPSASamplerGradient
from qiskit_aer.primitives import Sampler, Estimator
from qiskit.quantum_info import SparsePauliOp

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
input = ParameterVector('x', length=num_inputs)
params = ParameterVector('θ', length=6+3)

qreg = QuantumRegister(num_inputs)
creg = ClassicalRegister(4)
creg2 = ClassicalRegister(1)
qc = QuantumCircuit(qreg, creg, creg2)
qc.rx(input[0], qreg[0])
qc.rx(input[1], qreg[1])
qc.cz(qreg[0], qreg[1])
qc.u(params[0], params[1], params[2], qreg[0])
qc.u(params[3], params[4], params[5], qreg[1])
qc.cz(qreg[0], qreg[1])
qc.measure(qreg[1], creg2[0])
with qc.if_test((creg2[0],1)):
    qc.u(params[6], params[7], params[8], qreg[0])

qc.measure(qreg[0], creg[0])
qc.measure(qreg[1], creg[1])

observable1 = (SparsePauliOp("XI"), SparsePauliOp("IX"), SparsePauliOp("ZX"))
def parity(x):
    print("input to parity function")
    print(x)
    #print("{:b}".format(x).count("1"))
    return "{:b}".format(x).count("1")


qnn=SamplerQNN(circuit=qc, input_params=input, weight_params=params,interpret=parity, output_shape=4, sampler=Sampler())

res = qnn.forward(input_data=[0.1, 0.2], weights=[0.1, -0.5, 0.3, -0.7, 0.4, -0.2, 0.6, -0.8, 0.9])

print(res)

