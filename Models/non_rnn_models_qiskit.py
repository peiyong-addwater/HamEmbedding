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
    Create a (4+3)-qubit quantum circuit that encodes an 8x8 image into a four-qubit quantum state.
    The encoding is based on the re-uploading method, imnplemented in the fourByFourPatchReUploadingResetPooling1Q function
    from the LowEntEmbedding module.
    After the encoding, there will be an allInOneAnsatz layer, which has 6*num_qubits parameters

    Args:
        pixels: flattened 64 pixels of an eight by eight image. first 16 pixels are for the first patch, and so on
        params:
        num_single_patch_reuploading:
        num_classification_layers:

    Returns:

    """
    num_single_patch_reuploading_params = 24 * num_single_patch_reuploading
    num_fc_qubits = 4
    num_qubits = 3 + num_fc_qubits
    num_classification_params = 6 * num_fc_qubits
    num_params = num_single_patch_reuploading_params + num_classification_params

    #params = ParameterVector('θ', length=num_params)
    #inputs = ParameterVector('x', length=64)

    assert len(pixels) == 64, f"pixels must be a 64-element list of parameters"
    assert len(params) == num_params, f"params must be a {num_params}-element list of parameters"

    encoding_params = params[:num_single_patch_reuploading_params]
    fc_params = params[num_single_patch_reuploading_params:]

    circ = QuantumCircuit(num_qubits)

    # encode the image
    circ.append(
        fourByFourPatchReUploadingResetPooling1Q(pixels[:16], encoding_params).to_instruction(),
        [0,1,2,3]
    )
    circ.barrier(label=f"Patch 1 Encoded")
    circ.append(
        fourByFourPatchReUploadingResetPooling1Q(pixels[16:32], encoding_params).to_instruction(),
        [1,2,3,4]
    )
    circ.barrier(label=f"Patch 2 Encoded")
    circ.append(
        fourByFourPatchReUploadingResetPooling1Q(pixels[32:48], encoding_params).to_instruction(),
        [2,3,4,5]
    )
    circ.barrier(label=f"Patch 3 Encoded")
    circ.append(
        fourByFourPatchReUploadingResetPooling1Q(pixels[48:], encoding_params).to_instruction(),
        [3,4,5,6]
    )
    circ.barrier(label=f"Patch 4 Encoded")

    # "fully-connected" layers
    circ.append(
        allInOneAnsatz(num_fc_qubits, fc_params).to_instruction(),
        [0,1,2,3]
    )

    return circ


