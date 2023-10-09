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
from qiskit.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp
import torch
from torch import nn

from .Layers.qk.qiskit_layers import createMemStateInitCirc, createMemCompCirc, createMemPatchInteract, simplePQC
from .PatchEncoding.qk.PatchEmbedding import fourByFourPatchReupload, create8x8ReUploading
from .torch_connector import TorchConnector

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
        num_single_patch_reuploading: number of re-uploading repetitions for each patch.
        num_mem_qubits: number of memory qubits
        num_mem_interact_qubits: number of interaction qubits in memory
        num_patch_interact_qubits: number of interaction qubits in patch
        num_mem_comp_layers: number of memory computation layers

    Returns:
        QuantumCircuit: a (num_mem_qubits+3)-qubit circuit (backbone QNN) that encodes an 8x8 image into num_mem_qubits qubits,
        with trainable parameters for data re-uploading, and trainable parameters for the memory-related computations.
    """
    num_single_patch_reuploading_params = 18*num_single_patch_reuploading # using the fourByFourPatchReupload function
    num_mem_state_init_params = 12*num_mem_qubits-9 # using the createMemStateInitCirc function
    num_mem_patch_interact_params = 9*(num_mem_interact_qubits*num_patch_interact_qubits) # using the createMemPatchInteract function
    num_mem_comp_params = num_mem_comp_layers*(15*num_mem_qubits - 12) # using the createMemCompCirc function
    num_total_params = num_single_patch_reuploading_params + num_mem_state_init_params + num_mem_patch_interact_params + num_mem_comp_params

    assert len(params) == num_total_params, f"The number of parameters must be {num_total_params}, got {len(params)}"
    assert num_mem_qubits > 0, "Too few memory qubits. The number of memory qubits must be positive"
    assert num_mem_interact_qubits > 0, "Too few memory interaction qubits. The number of memory interaction qubits must be positive"
    assert num_patch_interact_qubits > 0, "Too few patch interaction qubits. The number of patch interaction qubits must be positive"
    assert num_mem_interact_qubits<=num_mem_qubits, "Too many memory interaction qubits. The number of memory interaction qubits must be less than or equal to the number of memory qubits"
    assert num_patch_interact_qubits<=3, "Too many patch interaction qubits. The number of patch interaction qubits must be less than or equal to 3"
    assert num_mem_comp_layers > 0, "Too few memory computation layers. The number of memory computation layers must be positive"
    assert num_single_patch_reuploading > 0, "Too few re-uploading repetitions. The number of re-uploading repetitions must be positive"
    assert len(pixels) == 64, f"The number of pixels must be 64, got {len(pixels)}"

    all_qubit_indices = list(range(num_mem_qubits+3))
    mem_qubit_indices = list(range(num_mem_qubits))
    patch_qubit_indices = list(range(num_mem_qubits, num_mem_qubits+3))
    mem_patch_interact_qubit_indices = list(range(num_mem_qubits-num_mem_interact_qubits, num_mem_qubits+num_patch_interact_qubits))

    mem_init_params = params[:num_mem_state_init_params]
    patch_reuploading_params = params[num_mem_state_init_params:num_mem_state_init_params+num_single_patch_reuploading_params]
    mem_patch_interact_params = params[num_mem_state_init_params+num_single_patch_reuploading_params:num_mem_state_init_params+num_single_patch_reuploading_params+num_mem_patch_interact_params]
    mem_comp_params = params[num_mem_state_init_params+num_single_patch_reuploading_params+num_mem_patch_interact_params:]

    circ = QuantumCircuit(num_mem_qubits+3, name='Backbone8x8Image')

    # initialise memory state
    circ.append(createMemStateInitCirc(params[:num_mem_state_init_params]).to_instruction(), mem_qubit_indices)
    circ.barrier()
    # iterate over the four patches
    # they all share the same parameters
    for i in range(4):
        # first encode the patch into 3 qubits
        circ.append(fourByFourPatchReupload(pixels[16*i:16*(i+1)], patch_reuploading_params).to_instruction(), patch_qubit_indices)
        # then interact the patch with the memory
        circ.append(createMemPatchInteract(num_mem_interact_qubits, num_patch_interact_qubits, mem_patch_interact_params).to_instruction(), mem_patch_interact_qubit_indices)
        # then perform computation on the memory
        circ.append(createMemCompCirc(num_mem_qubits, mem_comp_params).to_instruction(), mem_qubit_indices)
        # reset the patch qubits
        circ.reset(patch_qubit_indices)
        circ.barrier(label=f"Patch {i+1} Encoded")
    return circ

def classification8x8Image10ClassesSamplerQNN(
        num_single_patch_reuploading: int=2,
        num_mem_qubits:int = 2,
        num_mem_interact_qubits:int = 1,
        num_patch_interact_qubits:int = 1,
        num_mem_comp_layers:int=1,
        num_classification_layers:int=1,
        spsa_batchsize:int=1,
        spsa_epsilon:float=0.2
)->(SamplerQNN, int, int):
    """
    Creates an EstimatorQNN that classifies an 8x8 image into 10 classes,
    with trainable parameters for data re-uploading,
    and trainable parameters for the memory-related computations,
    and trainable parameters for the 4-qubit classification layer at the end of the circuit.
    The classification is performed via measuring the bitstring of the 4-qubit classification layer,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, corresponding to the 10 classes.
    Any other bitstring is considered as class 9.
    Classification will be the index of the Pauli string with the largest expectation value.
    Gradient is calculated via SPSA.
    1673.3945541381836 seconds for SPSA gradient with batchseize = 100
    20.506850719451904 seconds for SPSA gradient with batchsize=1,
    87.28186655044556 for batchsize=5,
    170 seconds for batchsize=10,
    839.2028388977051 for batchsize=50
    parameter-shift gradient takes forever
    Args:
        num_single_patch_reuploading: number of re-uploading repetitions for each patch.
        num_mem_qubits: number of memory qubits
        num_mem_interact_qubits: number of interaction qubits in memory
        num_patch_interact_qubits: number of interaction qubits in patch
        num_mem_comp_layers: number of memory computation layers
        num_classification_layers: number of classification layers
        batchsize: batchsize for the SPSASamplerGradient
        spsa_epsilon: epsilon for the SPSASamplerGradient, the "c" in SPSA

    Returns:
        SamplerQNN, number of trainable parameters, input size

    """

    def parity(x):
        if x>=9:
            res = 9
        else:
            res = x
        return res

    sampler = AerSampler(
        backend_options={'method': 'statevector',
                         }
    )

    num_classification_qubits = 4
    num_total_qubits = num_mem_qubits + 3
    assert num_mem_qubits+3>=num_classification_qubits, "The number of memory qubits plus 3 must be greater than or equal to the number of classification qubits (4)"
    num_single_patch_reuploading_params = 18 * num_single_patch_reuploading  # using the fourByFourPatchReupload function
    num_mem_state_init_params = 12 * num_mem_qubits - 9  # using the createMemStateInitCirc function
    num_mem_patch_interact_params = 9 * (
            num_mem_interact_qubits * num_patch_interact_qubits)  # using the createMemPatchInteract function
    num_mem_comp_params = num_mem_comp_layers * (15 * num_mem_qubits - 12)  # using the createMemCompCirc function
    num_classification_params = num_classification_layers*3*num_classification_qubits  # using the simplePQC function
    num_total_params = num_single_patch_reuploading_params + num_mem_state_init_params + num_mem_patch_interact_params + num_mem_comp_params + num_classification_params

    params = ParameterVector('θ', length=num_total_params)
    inputs = ParameterVector('x', length=64)

    backbone_params = params[:num_single_patch_reuploading_params+num_mem_state_init_params+num_mem_patch_interact_params+num_mem_comp_params]
    classification_params = params[num_single_patch_reuploading_params+num_mem_state_init_params+num_mem_patch_interact_params+num_mem_comp_params:]
    circ = QuantumCircuit(num_total_qubits,num_classification_qubits, name='Sampler10ClassQNN')
    backbone = createBackbone8x8Image(inputs, backbone_params, num_single_patch_reuploading, num_mem_qubits, num_mem_interact_qubits, num_patch_interact_qubits, num_mem_comp_layers)
    circ.append(backbone.to_instruction(), list(range(num_total_qubits)))
    circ.append(simplePQC(num_classification_qubits, classification_params).to_instruction(), list(range(num_classification_qubits)))
    circ.measure(list(range(num_classification_qubits)), list(range(num_classification_qubits)))

    qnn = SamplerQNN(
        circuit=circ,
        input_params=inputs,
        weight_params=params,
        interpret=parity,
        output_shape=10,
        gradient = SPSASamplerGradient(sampler,spsa_epsilon, batch_size=spsa_batchsize), # epsilon is the "c" in SPSA
        sampler=sampler
    )

    return qnn, num_total_params, 64

class ClassificationSamplerQNN8x8Image(nn.Module):
    def __init__(self,
                 num_single_patch_reuploading: int = 2,
                 num_mem_qubits: int = 2,
                 num_mem_interact_qubits: int = 1,
                 num_patch_interact_qubits: int = 1,
                 num_mem_comp_layers: int = 1,
                 num_classification_layers: int = 1,
                 spsa_batchsize: int = 1,
                 spsa_epsilon: float = 0.2
                 ):
        super().__init__()
        self.qnn,_,_ = classification8x8Image10ClassesSamplerQNN(
            num_single_patch_reuploading,
            num_mem_qubits,
            num_mem_interact_qubits,
            num_patch_interact_qubits,
            num_mem_comp_layers,
            num_classification_layers,
            spsa_batchsize,
            spsa_epsilon
        )
        self.qnn_torch = TorchConnector(self.qnn)

    def forward(self, x):
        # x must be of shape (batchsize, 64)
        # each 16 elements of x is a 4 by 4 patch of the 8x8 image
        return self.qnn_torch.forward(x)

def classification8x8Image10ClassesEstimatorQNN(
        num_single_patch_reuploading: int=2,
        num_mem_qubits:int = 2,
        num_mem_interact_qubits:int = 1,
        num_patch_interact_qubits:int = 1,
        num_mem_comp_layers:int=1,
        num_classification_layers:int=1,
        spsa_batchsize:int=1,
        spsa_epsilon:float=0.2,
        observalbes:Sequence[SparsePauliOp]=None
)->(EstimatorQNN, int, int):
    """
    Creates an EstimatorQNN that classifies an 8x8 image into 10 classes.
    The classification is performed via measuring the bitstring at the end of the 4-qubit classification layer,
    the number of bitstrings is 10, corresponding to the 10 classes.
    the bitstring with the highest expectation value is the classification result.
    The number of operators in each bitstring must match the total number of qubits.
    Args:
        num_single_patch_reuploading: number of re-uploading repetitions for each patch.
        num_mem_qubits: number of memory qubits
        num_mem_interact_qubits: number of interaction qubits in memory
        num_patch_interact_qubits: number of interaction qubits in patch
        num_mem_comp_layers: number of memory computation layers
        num_classification_layers: number of classification layers
        batchsize: batchsize for the SPSAEstimatorGradient
        spsa_epsilon: epsilon for the SPSAEstimatorGradient, the "c" in SPSA
        observalbes: list of ten SparsePauliOp observables for the ten classes. The number of qubits in each observable must match the total number of qubits.

    Returns:
        EstimatorQNN, number of trainable parameters, input size
    """
    estimator = AerEstimator(
        backend_options={'method': 'statevector'}
    )
    num_classification_qubits = 4
    num_total_qubits = num_mem_qubits + 3
    for ob in observalbes:
        assert ob.num_qubits == num_total_qubits, f"The number of qubits in observable {ob} must match the total number of qubits, got {ob.num_qubits} and {num_total_qubits}"
    assert num_mem_qubits + 3 >= num_classification_qubits, "The number of memory qubits plus 3 must be greater than or equal to the number of classification qubits (4)"
    num_single_patch_reuploading_params = 18 * num_single_patch_reuploading  # using the fourByFourPatchReupload function
    num_mem_state_init_params = 12 * num_mem_qubits - 9  # using the createMemStateInitCirc function
    num_mem_patch_interact_params = 9 * (
            num_mem_interact_qubits * num_patch_interact_qubits)  # using the createMemPatchInteract function
    num_mem_comp_params = num_mem_comp_layers * (15 * num_mem_qubits - 12)  # using the createMemCompCirc function
    num_classification_params = num_classification_layers * 3 * num_classification_qubits  # using the simplePQC function
    num_total_params = num_single_patch_reuploading_params + num_mem_state_init_params + num_mem_patch_interact_params + num_mem_comp_params + num_classification_params

    params = ParameterVector('θ', length=num_total_params)
    inputs = ParameterVector('x', length=64)

    backbone_params = params[
                      :num_single_patch_reuploading_params + num_mem_state_init_params + num_mem_patch_interact_params + num_mem_comp_params]
    classification_params = params[
                            num_single_patch_reuploading_params + num_mem_state_init_params + num_mem_patch_interact_params + num_mem_comp_params:]
    circ = QuantumCircuit(num_total_qubits, name='Estimator10ClassQNN')
    backbone = createBackbone8x8Image(inputs, backbone_params, num_single_patch_reuploading, num_mem_qubits,
                                      num_mem_interact_qubits, num_patch_interact_qubits, num_mem_comp_layers)
    circ.append(backbone.to_instruction(), list(range(num_total_qubits)))
    circ.append(simplePQC(num_classification_qubits, classification_params).to_instruction(),
                list(range(num_classification_qubits)))

    qnn = EstimatorQNN(
        circuit=circ,
        observables=observalbes,
        input_params=inputs,
        weight_params=params,
        gradient=SPSAEstimatorGradient(estimator, spsa_epsilon, batch_size=spsa_batchsize),  # epsilon is the "c" in SPSA
        estimator = estimator
    )

    return qnn, num_total_params, 64

class ClassificationEstimatorQNN8x8Image(nn.Module):
    def __init__(self,
                 num_single_patch_reuploading: int = 2,
                 num_mem_qubits: int = 2,
                 num_mem_interact_qubits: int = 1,
                 num_patch_interact_qubits: int = 1,
                 num_mem_comp_layers: int = 1,
                 num_classification_layers: int = 1,
                 spsa_batchsize: int = 1,
                 spsa_epsilon: float = 0.2,
                 observables: Sequence[SparsePauliOp] = None
                 ):
        super().__init__()
        self.qnn, _, _ = classification8x8Image10ClassesEstimatorQNN(
            num_single_patch_reuploading=num_single_patch_reuploading,
            num_mem_qubits=num_mem_qubits,
            num_mem_interact_qubits=num_mem_interact_qubits,
            num_patch_interact_qubits=num_patch_interact_qubits,
            num_mem_comp_layers=num_mem_comp_layers,
            num_classification_layers=num_classification_layers,
            spsa_batchsize=spsa_batchsize,
            spsa_epsilon=spsa_epsilon,
            observalbes=observables
        )
        self.qnn_torch = TorchConnector(self.qnn)

    def forward(self, x):
        # x must be of shape (batchsize, 64)
        # each 16 elements of x is a 4 by 4 patch of the 8x8 image
        return self.qnn_torch.forward(x)



