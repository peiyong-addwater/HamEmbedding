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
from qiskit.algorithms.gradients import SPSASamplerGradient
from qiskit.primitives import BaseSampler, SamplerResult, Sampler, BackendSampler
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
        spsa_batchsize:int=1
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

    Returns:
        SamplerQNN, number of trainable parameters, input size

    """

    def parity(x):
        if x>=9:
            res = 9
        else:
            res = x
        return res

    backend = AerSimulator(method='statevector')
    sampler = BackendSampler(backend)

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
        gradient = SPSASamplerGradient(sampler,0.01, batch_size=spsa_batchsize) # epsilon is the "c" in SPSA
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
                 spsa_batchsize: int = 1
                 ):
        super().__init__()
        self.qnn,_,_ = classification8x8Image10ClassesSamplerQNN(
            num_single_patch_reuploading,
            num_mem_qubits,
            num_mem_interact_qubits,
            num_patch_interact_qubits,
            num_mem_comp_layers,
            num_classification_layers,
            spsa_batchsize
        )
        self.qnn_torch = TorchConnector(self.qnn)

    def forward(self, x):
        # x must be of shape (batchsize, 64)
        # each 16 elements of x is a 4 by 4 patch of the 8x8 image
        return self.qnn_torch.forward(x)



if __name__ == '__main__':
    import time
    from data import PatchedDigitsDataset
    from torch.utils.data import DataLoader

    dataset = PatchedDigitsDataset()
    dataloader = DataLoader(dataset, batch_size=500,
                            shuffle=True, num_workers=0)

    flattened_8x8_patch = ParameterVector('x', length=64)
    num_single_patch_reuploading = 2
    num_mem_qubits = 2
    num_mem_interact_qubits = 1
    num_patch_interact_qubits = 1
    num_mem_comp_layers = 1
    num_classification_layers = 1
    num_single_patch_reuploading_params = 18 * num_single_patch_reuploading  # using the fourByFourPatchReupload function
    num_mem_state_init_params = 12 * num_mem_qubits - 9  # using the createMemStateInitCirc function
    num_mem_patch_interact_params = 9 * (
                num_mem_interact_qubits * num_patch_interact_qubits)  # using the createMemPatchInteract function
    num_mem_comp_params = num_mem_comp_layers * (15 * num_mem_qubits - 12)  # using the createMemCompCirc function
    num_total_params = num_single_patch_reuploading_params + num_mem_state_init_params + num_mem_patch_interact_params + num_mem_comp_params
    params = ParameterVector('θ', length=num_total_params)
    backbone_circ = createBackbone8x8Image(flattened_8x8_patch, params, num_single_patch_reuploading, num_mem_qubits, num_mem_interact_qubits, num_patch_interact_qubits, num_mem_comp_layers)
    backbone_circ.draw('mpl', filename=f'backbone_circ_8x8_image_{num_mem_qubits}q_mem_{num_total_params}_params.png', style='bw')

    """
    qnn, num_total_params, input_size = classification8x8Image10ClassesSamplerQNN(
        num_single_patch_reuploading,
        num_mem_qubits,
        num_mem_interact_qubits,
        num_patch_interact_qubits,
        num_mem_comp_layers,
        num_classification_layers,
        1
    )
    print(qnn)
    start = time.time()
    params = algorithm_globals.random.random(num_total_params)
    input = algorithm_globals.random.random((100,input_size))
    res = qnn.forward(input, params)
    print("Sampler QNN forward pass result:")
    #print(res)
    print(res.shape)
    sampler_qnn_input_grad, sampler_qnn_weight_grad = qnn.backward(
        input, params
    )
    end = time.time()
    print("sampler_qnn_input_grad")
    print(sampler_qnn_input_grad)
    print("sampler_qnn_weight_grad")
    #print(sampler_qnn_weight_grad)
    print(sampler_qnn_weight_grad.shape)
    print(f"Time taken: {end-start}")
    # 1673.3945541381836 seconds for SPSA gradient with 100 batch
    # 20.506850719451904 seconds for SPSA gradient with 1 batch,87.28186655044556 for 5 batchsize, 170 seconds for 10 batch, 839.2028388977051 for 50 batchsize
    # parameter-shift gradient takes forever
    """
    print("Testing the PyTorch model")
    model = ClassificationSamplerQNN8x8Image()
    print(model)
    for batch, (X, y) in enumerate(dataloader):
        start = time.time()
        model.train()
        print(batch, X.shape, y.shape)
        out = model(X)
        end = time.time()
        print(out.shape)
        print("Single forward pass takes: ", end-start)
        break

