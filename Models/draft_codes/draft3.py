import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS

from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
#from qiskit_machine_learning.connectors import TorchConnector
from torch_connector import TorchConnector
import copy
from qiskit.algorithms.gradients import SPSASamplerGradient
from Optimization.zero_order_gradient_estimation import RSGFSamplerGradient
from qiskit.primitives import BaseSampler, SamplerResult, Sampler

# Set seed for random generators
algorithm_globals.random_seed = 42

# Generate random dataset

# Select dataset dimension (num_inputs) and size (num_samples)
num_inputs = 2
num_samples = 20




num_qubits = 3
feature_map = ZZFeatureMap(feature_dimension=num_qubits)
ansatz = RealAmplitudes(num_qubits=num_qubits, reps=1)

qc = QuantumCircuit(num_qubits, num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)
for i in range(num_qubits):
    qc.measure(i, i)

qc1 = QuantumCircuit(num_qubits, num_qubits-1)
qc1.compose(feature_map, inplace=True)
qc1.compose(ansatz, inplace=True)


def parity(x):
    #print("input to parity function")
    #print(x)
    #print("{:b}".format(x).count("1"))
    return "{:b}".format(x).count("1")


qnn = SamplerQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    interpret=parity,
    output_shape=4,
    gradient = RSGFSamplerGradient(Sampler(),0.01, batch_size=1) # epsilon is the "c" in SPSA
)

res = qnn.forward(input_data=[1, 2, 3], weights=[1, 2, 3, 4, 5, 6])


sampler_qnn_input_grad, sampler_qnn_weight_grad = qnn.backward(
    [1, 2, 3], [1, 2, 3, 4, 5, 6]
)



print("sampler_qnn_weight_grad")
print(sampler_qnn_weight_grad)
print(sampler_qnn_weight_grad.shape)

