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

