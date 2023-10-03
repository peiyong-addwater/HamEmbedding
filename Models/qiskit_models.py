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



if __name__ == '__main__':

    flattened_8x8_patch = ParameterVector('x', length=64)
    num_single_patch_reuploading = 2
    num_mem_qubits = 2
    num_mem_interact_qubits = 1
    num_patch_interact_qubits = 1
    num_mem_comp_layers = 1
    num_single_patch_reuploading_params = 18 * num_single_patch_reuploading  # using the fourByFourPatchReupload function
    num_mem_state_init_params = 12 * num_mem_qubits - 9  # using the createMemStateInitCirc function
    num_mem_patch_interact_params = 9 * (
                num_mem_interact_qubits * num_patch_interact_qubits)  # using the createMemPatchInteract function
    num_mem_comp_params = num_mem_comp_layers * (15 * num_mem_qubits - 12)  # using the createMemCompCirc function
    num_total_params = num_single_patch_reuploading_params + num_mem_state_init_params + num_mem_patch_interact_params + num_mem_comp_params
    params = ParameterVector('θ', length=num_total_params)
    backbone_circ = createBackbone8x8Image(flattened_8x8_patch, params, num_single_patch_reuploading, num_mem_qubits, num_mem_interact_qubits, num_patch_interact_qubits, num_mem_comp_layers)
    backbone_circ.draw('mpl', filename=f'backbone_circ_8x8_image_{num_mem_qubits}q_mem_{num_total_params}_params.png', style='bw')