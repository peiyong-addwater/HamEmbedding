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

num_inputs = 2
num_samples = 200

input = [0.1, 0.2]
params = [0.1, -0.5, 0.3, -0.7, 0.4, -0.2, 0.6, -0.8, 0.9]

sampler = Sampler()

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

#qc.measure(qreg[0], creg[0])
#qc.measure(qreg[1], creg[1])
"""
If the measurement to the 4-bit classical register is disabled,
then the mesurement results will only be 16 and 0.
It seems that creg2 will be the leading qubit, 
since creg is always zero
and creg2 could be one or zero
making the result either 10000 or 00000,
which 16 or 0 in decimal.
This indicates that the order of registers are reversed.
"""

job = sampler.run([qc]*num_samples)
result = job.result()
print(result)