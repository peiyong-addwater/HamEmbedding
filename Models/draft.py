# Necessary imports

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
qc = QuantumCircuit(num_inputs)
qc.compose(feature_map, inplace=True)
qc.reset(0) # at least EstimatorQNN supports reset
qc.compose(ansatz, inplace=True)

#print(qc)

# Setup QNN
qnn1 = SamplerQNN(
    circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters
)

# Set up PyTorch module
# Note: If we don't explicitly declare the initial weights
# they are chosen uniformly at random from [-1, 1].
initial_weights = Tensor([0.1, -0.5, 0.3, -0.7, 0.4, -0.2, 0.6, -0.8])
model1 = TorchConnector(qnn1, initial_weights=initial_weights)
#print("Initial weights: ", initial_weights)

print(model1(X_[0, :]))

model2 = copy.deepcopy(model1)

model2.to("cuda")

print("="*20)

# looks like the Qiskit TorchConnector model can be deep copied
#print(model2(X_[0, :].to("cuda")))

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

from qiskit_machine_learning.neural_networks import SamplerQNN

from qiskit.quantum_info import SparsePauliOp



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
observable1 = (SparsePauliOp("XII"), SparsePauliOp("IXI"), SparsePauliOp("IIX"))
for c in observable1:
    print(c.num_qubits)

from qiskit_machine_learning.neural_networks import EstimatorQNN

estimator_qnn = EstimatorQNN(
    circuit=qc1, observables=observable1, input_params=feature_map.parameters, weight_params=ansatz.parameters
)
print(estimator_qnn)
estimator_qnn_input = algorithm_globals.random.random(estimator_qnn.num_inputs)
estimator_qnn_weights = algorithm_globals.random.random(estimator_qnn.num_weights)
print(
    f"Number of input features for EstimatorQNN: {estimator_qnn.num_inputs} \nInput: {estimator_qnn_input}"
)
print(
    f"Number of trainable weights for EstimatorQNN: {estimator_qnn.num_weights} \nWeights: {estimator_qnn_weights}"
)
estimator_qnn_forward = estimator_qnn.forward(estimator_qnn_input, estimator_qnn_weights)

print(
    f"Forward pass result for EstimatorQNN: {estimator_qnn_forward}. \nShape: {estimator_qnn_forward.shape}"
)



print("="*20)

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
    gradient = SPSASamplerGradient(Sampler(),0.01) # epsilon is the "c" in SPSA
)

res = qnn.forward(input_data=[1, 2, 3], weights=[1, 2, 3, 4, 5, 6])


sampler_qnn_input_grad, sampler_qnn_weight_grad = qnn.backward(
    [1, 2, 3], [1, 2, 3, 4, 5, 6]
)


print("Sampler QNN forward pass result:")
print(res)
print("sampler_qnn_input_grad")
print(sampler_qnn_input_grad)
print("sampler_qnn_weight_grad")
print(sampler_qnn_weight_grad)
print(sampler_qnn_weight_grad.shape)