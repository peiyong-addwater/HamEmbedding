# models based on Qiskit and PyTorch
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit import Qubit
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union
from math import pi
from Layers.qk.qiskit_layers import createMemStateInitCirc, createMemCompCirc, createMemPatchInteract
from PatchEncoding.qk.PatchEmbedding import fourByFourPatchReupload, create8x8ReUploading
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.algorithms.gradients import SPSASamplerGradient
from qiskit.primitives import BaseSampler, SamplerResult, Sampler
from qiskit.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp

QiskitParameter = Union[ParameterVector, List[Parameter], List[ParameterVectorElement]]
QiskitQubits = Union[List[int], List[Qubit], QuantumRegister]

def createBackbone8x8Image(
        pixels: QiskitParameter,
        params: QiskitParameter,
        num_single_patch_reuploading: int=2,
        num_mem_qubits:int = 2,
        num_mem_interact_qubits:int = 2,
        num_patch_interact_qubits:int = 2,
        num_mem_comp_layers:int=1
)->QuantumCircuit:
    """
    Creates a (num_mem_qubits+3)-qubit circuit that encodes an 8x8 image into num_mem_qubits qubits,
    with trainable parameters for data re-uploading, and trainable parameters for the memory-related computations.
    The trainable parameters include:

    Args:
        pixels: flattened 64 pixels of an eight by eight image. first 16 pixels are for the first patch, and so on
        params: trainable parameters for the backbone qnn.
        num_single_patch_reuploading:
        num_mem_qubits:
        num_mem_interact_qubits:
        num_patch_interact_qubits:
        num_mem_comp_layers:

    Returns:

    """