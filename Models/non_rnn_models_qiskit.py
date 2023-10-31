# models based on Qiskit and PyTorch
import qiskit
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit import Qubit
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union
from math import pi
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_algorithms.gradients import SPSASamplerGradient, SPSAEstimatorGradient
from qiskit.primitives import BackendSampler
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_ibm_runtime import Estimator as IBMRuntimeEstimator
from qiskit.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp
import torch
from torch import nn
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np


from .Layers.qk.qiskit_layers import createMemStateInitCirc, createMemCompCirc, createMemPatchInteract, simplePQC, allInOneAnsatz
from .PatchEncoding.qk.LowEntEmbedding import fourByFourPatchReUploadingResetPooling1Q
from .torch_connector import TorchConnector
from .Optimization.zero_order_gradient_estimation import RSGFSamplerGradient

QiskitParameter = Union[ParameterVector, List[Parameter], List[ParameterVectorElement]]
QiskitQubits = Union[List[int], List[Qubit], QuantumRegister]

def create8x8ImageBackbone(
        pixels: QiskitParameter,
        params: QiskitParameter,
        num_single_patch_reuploading: int=2,
)->QuantumCircuit:
    """
    Create a quantum circuit that encodes an 8x8 image into a four-qubit quantum state.
    The encoding is based on the re-uploading method, imnplemented in the fourByFourPatchReUploadingResetPooling1Q function
    from the LowEntEmbedding module.

    Args:
        pixels:
        params:
        num_single_patch_reuploading:

    Returns:

    """