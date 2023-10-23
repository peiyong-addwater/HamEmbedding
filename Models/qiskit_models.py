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


from .Layers.qk.qiskit_layers import createMemStateInitCirc, createMemCompCirc, createMemPatchInteract, simplePQC, allInOneAnsatz
from .PatchEncoding.qk.PatchEmbedding import fourByFourPatchReuploadResetPooling1Q, fourByFourPatchReupload, create8x8ReUploading, fourByFourPatchReuploadPoolingClassicalCtrl1Q
from .torch_connector import TorchConnector
from .Optimization.zero_order_gradient_estimation import RSGFSamplerGradient

QiskitParameter = Union[ParameterVector, List[Parameter], List[ParameterVectorElement]]
QiskitQubits = Union[List[int], List[Qubit], QuantumRegister]

def createRecurrentBackbone8x8Image(
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

def classification8x8Image10ClassesSamplerRecurrentQNN(
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
    Creates a SamplerQNN that classifies an 8x8 image into 10 classes,
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
    backbone = createRecurrentBackbone8x8Image(inputs, backbone_params, num_single_patch_reuploading, num_mem_qubits, num_mem_interact_qubits, num_patch_interact_qubits, num_mem_comp_layers)
    circ.append(backbone.to_instruction(), list(range(num_total_qubits)))
    circ.append(simplePQC(num_classification_qubits, classification_params).to_instruction(), list(range(num_classification_qubits)))
    circ.measure(list(range(num_classification_qubits)), list(range(num_classification_qubits)))

    qnn = SamplerQNN(
        circuit=circ,
        input_params=inputs,
        weight_params=params,
        interpret=parity,
        output_shape=10,
        gradient = RSGFSamplerGradient(sampler,spsa_epsilon, batch_size=spsa_batchsize), # epsilon is the "c" in SPSA
        sampler=sampler
    )

    return qnn, num_total_params, 64

class ClassificationSamplerRecurrentQNN8x8Image(nn.Module):
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
        self.qnn,_,_ = classification8x8Image10ClassesSamplerRecurrentQNN(
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


def createSimpleQRNNBackbone8x8Image(
        pixels: QiskitParameter,
        params: QiskitParameter,
        num_single_patch_reuploading: int=2,
        num_mem_qubits:int = 3,
        reset_between_reuploading: bool = False,
        spread_info_between_reuploading: bool = True
)->QuantumCircuit:
    """
    Creates a (num_mem_qubits+3)-qubit circuit that encodes an 8x8 image into num_mem_qubits qubits.
    The interaction between the memory and the encoded qubit state for the patch is provided
    by the allInOneAnsatz function.
    The patch encoding is provided by the fourByFourPatchReuploadPoolingClassicalCtrl1Q function.
    Args:
        pixels: flattened 64 pixels of an eight by eight image. first 16 pixels are for the first patch, and so on
        params: trainable parameters for the backbone qnn.
        num_single_patch_reuploading: number of re-uploading repetitions for each patch.
        num_mem_qubits: number of memory qubits
        reset_between_reuploading: whether to reset the bottom two patch qubits between re-uploading repetitions
        spread_info_between_reuploading: whether to spread the information from the first patch qubits to the reset two
            between re-uploading repetitions

    Returns:
        QuantumCircuit: a (num_mem_qubits+3)-qubit circuit (backbone QNN) that encodes an 8x8 image into num_mem_qubits qubits,
        with trainable parameters for data re-uploading, and trainable parameters for the memory-related computations.
    """
    num_single_patch_reuploading_params = 30*num_single_patch_reuploading # using the fourByFourPatchReuploadPoolingClassicalCtrl1Q function
    num_mem_params = 6*(num_mem_qubits+1) # using the allInOneAnsatz function
    num_total_params = num_single_patch_reuploading_params + num_mem_params
    assert len(params) == num_total_params, f"The number of parameters must be {num_total_params}, got {len(params)}"
    assert num_mem_qubits > 0, "Too few memory qubits. The number of memory qubits must be positive"
    assert num_single_patch_reuploading > 0, "Too few re-uploading repetitions. The number of re-uploading repetitions must be positive"
    assert len(pixels) == 64, f"The number of pixels must be 64, got {len(pixels)}"

    patch_encoding_params = params[:num_single_patch_reuploading_params]
    mem_params = params[num_single_patch_reuploading_params:]

    mem_qreg = QuantumRegister(num_mem_qubits, name='mem')
    patch_qreg = QuantumRegister(3, name='patch')
    patch_creg = ClassicalRegister(2, name='patch_classical')
    circ = QuantumCircuit(mem_qreg, patch_qreg, patch_creg, name='SimpQRNNBackbone8x8Image')
    # iterate over the four patches
    # they all share the same parameters
    for i in range(4):
        # first encode the patch into 3 qubits
        circ.append(fourByFourPatchReuploadPoolingClassicalCtrl1Q(pixels[16 * i:16 * (i + 1)], patch_encoding_params, reset_between_reuploading, spread_info_between_reuploading).to_instruction(), patch_qreg, patch_creg)
        # then interact the patch with the memory
        circ.append(allInOneAnsatz(num_mem_qubits+1, mem_params).to_instruction(), mem_qreg[:]+patch_qreg[:1])
        circ.barrier()
        # reset the patch qubits
        circ.reset(patch_qreg)
        circ.barrier(label=f"Patch {i+1} Encoded")
    return circ

def classification8x8Image10ClassesSamplerSimpleQRNN(
        num_single_patch_reuploading: int=2,
        num_mem_qubits:int = 3,
        reset_between_reuploading: bool = False,
        spread_info_between_reuploading: bool = True,
        num_classification_layers:int=1,
        spsa_batchsize:int=1,
        spsa_epsilon:float=0.2
)->(SamplerQNN, int, int):
    """
    Creates a SamplerQNN that classifies an 8x8 image into 10 classes, using the simple QRNN backbone,
    which has pooling in the patch encoding part,
    with trainable parameters for data re-uploading,
    and trainable parameters for the memory-related computations,
    and trainable parameters for the 4-qubit classification layer at the end of the circuit.
    The classification is performed via measuring the bitstring of the 4-qubit classification layer,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, corresponding to the 10 classes.
    Any other bitstring is considered as class 9.
    However, since there are additional two classical bits for patch encoding,
    Total number of classification qubits is 6.
    Since the sampler only output an integer for all the cregs,
    if we order the cregs as [patch_classical, classification] when creating the quantum circuit,
    and convert the output integer to fixed-length (length 6) binary,
    then the last two classical bits are for patch encoding, due to the little-endian convention.
    We can remove the last two bits to get the classification result in the parity function.
    Args:
        num_single_patch_reuploading: number of re-uploading repetitions for each patch.
        num_mem_qubits: number of memory qubits
        reset_between_reuploading: whether to reset the bottom two patch qubits between re-uploading repetitions
        spread_info_between_reuploading: whether to spread the information from the first patch qubits to the reset two
            between re-uploading repetitions
        num_classification_layers: number of classification layers
        spsa_batchsize: batchsize for the stochastic gradient estimator to be averaged over
        spsa_epsilon: smoothing factor for the stochastic gradient estimator, like the "c" in SPSA

    Returns:
        SamplerQNN, number of trainable parameters, input size

    """

    def parity(x):
        test_bit_string = "{0:06b}".format(x)
        cls_bit_string = test_bit_string[:-2]
        cls_int = int(cls_bit_string, 2)
        if cls_int>=9:
            res = 9
        else:
            res = cls_int
        return res

    sampler = AerSampler(
        backend_options={'method': 'statevector',
                         }
    )

    num_classification_qubits = 4
    num_total_qubits = num_mem_qubits + 3

    assert num_mem_qubits + 3 >= num_classification_qubits, "The number of memory qubits plus 3 must be greater than or equal to the number of classification qubits (4)"
    num_single_patch_reuploading_params = 30 * num_single_patch_reuploading  # using the fourByFourPatchReuploadPoolingClassicalCtrl1Q function
    num_mem_params = 6 * (num_mem_qubits + 1)  # using the allInOneAnsatz function
    num_classification_params = num_classification_layers * 3 * num_classification_qubits  # using the simplePQC function
    num_total_params = num_single_patch_reuploading_params + num_mem_params + num_classification_params
    num_backbone_params = num_single_patch_reuploading_params + num_mem_params

    params = ParameterVector('θ', length=num_total_params)
    inputs = ParameterVector('x', length=64)

    backbone_params = params[:num_backbone_params]
    classification_params = params[num_backbone_params:]

    qreg = QuantumRegister(num_total_qubits, name='q')
    patch_cre = ClassicalRegister(2, name='patch_classical')
    cls_creg = ClassicalRegister(num_classification_qubits, name='classification')

    circ = QuantumCircuit(qreg, patch_cre, cls_creg, name='Sampler10ClassQRNN')
    backbone = createSimpleQRNNBackbone8x8Image(inputs,
                                                backbone_params,
                                                num_single_patch_reuploading,
                                                num_mem_qubits,
                                                reset_between_reuploading,
                                                spread_info_between_reuploading)
    circ.append(backbone.to_instruction(), qargs=qreg, cargs=patch_cre)
    circ.append(simplePQC(num_classification_qubits, classification_params).to_instruction(), qargs=qreg[:num_classification_qubits])
    circ.measure(qreg[:num_classification_qubits], cls_creg[:num_classification_qubits])

    qnn = SamplerQNN(
        circuit=circ,
        input_params=inputs,
        weight_params=params,
        interpret=parity,
        output_shape=10,
        gradient=RSGFSamplerGradient(sampler, spsa_epsilon, batch_size=spsa_batchsize),  # epsilon is the "c" in SPSA
        sampler=sampler
    )

    return qnn, num_total_params, 64

class ClassificationSamplerSimpleQRNN8x8Image(nn.Module):
    def __init__(self,
                 num_single_patch_reuploading: int = 2,
                 num_mem_qubits: int = 3,
                 reset_between_reuploading: bool = False,
                 spread_info_between_reuploading: bool = True,
                 num_classification_layers: int = 1,
                 spsa_batchsize: int = 1,
                 spsa_epsilon: float = 0.2
                 ):
        super().__init__()
        self.qnn,_,_ = classification8x8Image10ClassesSamplerSimpleQRNN(
            num_single_patch_reuploading,
            num_mem_qubits,
            reset_between_reuploading,
            spread_info_between_reuploading,
            num_classification_layers,
            spsa_batchsize,
            spsa_epsilon
        )
        self.qnn_torch = TorchConnector(self.qnn)

    def forward(self, x):
        # x must be of shape (batchsize, 64)
        # each 16 elements of x is a 4 by 4 patch of the 8x8 image
        return self.qnn_torch.forward(x)

def createSimpleQRNNBackboneResetPooling8x8Image(
        pixels: QiskitParameter,
        params: QiskitParameter,
        num_single_patch_reuploading: int=2,
        num_mem_qubits:int = 3
)->QuantumCircuit:
    """
    Create a (num_mem_qubits+3)-qubit backbone QNN.
    3 qubits for patch encoding, each patch is 4 by 4, so 3 qubits are enough.
    Patch encoding is done by fourByFourPatchReuploadResetPooling1Q function.
    The interaction between memory and the encoded qubit state for the patch is provided
    by the allInOneAnsatz function
    Args:
        pixels:  flattened 64 pixels of an eight by eight image. first 16 pixels are for the first patch, and so on
        params: trainable parameters for the backbone qnn.
        num_single_patch_reuploading: number of re-uploading repetitions for each patch.
        num_mem_qubits: number of memory qubits

    Returns:
        QuantumCircuit: a (num_mem_qubits+3)-qubit circuit (backbone QNN) that encodes an 8x8 image into the states of
         the first num_mem_qubits qubits, which can be used for downstream tasks.
    """
    num_encode_params =  num_single_patch_reuploading*(3*3 + 4*(3-1))
    num_qubits = num_mem_qubits+3
    num_mem_params = 6*(num_mem_qubits+1) # using the allInOneAnsatz function
    num_total_params = num_encode_params + num_mem_params
    assert len(params) == num_total_params, f"The number of parameters must be {num_total_params}, got {len(params)}"
    assert num_mem_qubits > 0, "Too few memory qubits. The number of memory qubits must be positive"
    assert num_single_patch_reuploading > 0, "Too few re-uploading repetitions. The number of re-uploading repetitions must be positive"
    assert len(pixels) == 64, f"The number of pixels must be 64, got {len(pixels)}"

    encode_params = params[:num_encode_params]
    mem_params = params[num_encode_params:]

    mem_qreg = QuantumRegister(num_mem_qubits, name='mem')
    patch_qreg = QuantumRegister(3, name='patch')
    circ = QuantumCircuit(mem_qreg, patch_qreg, name='SimpQRNNBackboneResetPooling8x8Image')
    # iterate over the four patches
    # they all share the same parameters
    for i in range(4):
        # first append the patch encoding circuit to the patch qubits
        circ.append(fourByFourPatchReuploadResetPooling1Q(pixels[16 * i:16 * (i + 1)], encode_params).to_instruction(), patch_qreg)
        # then interact the first patch qubit with the memory
        circ.append(allInOneAnsatz(num_mem_qubits + 1, mem_params).to_instruction(), mem_qreg[:] + patch_qreg[:1])
        circ.barrier()
        # reset the patch qubits
        circ.reset(patch_qreg)
        circ.barrier(label=f"Patch {i + 1} Encoded")
    return circ

if __name__ == '__main__':

    from qiskit import transpile

    sim = AerSimulator(method="statevector")

    n_mem = 3
    n_reuploading = 2
    params_simpleQRNN = ParameterVector('θ', length=30*n_reuploading+6*(n_mem+1))
    pixels = ParameterVector('x', length=64)
    circ_simpleQRNN = createSimpleQRNNBackbone8x8Image(pixels, params_simpleQRNN, n_reuploading, n_mem)
    circ_simpleQRNN.draw(output='mpl', filename='simpleQRNN.png', style='iqx')

    circ_simpleQRNN = circ_simpleQRNN.bind_parameters(
        {params_simpleQRNN: list(range(30*n_reuploading+6*(n_mem+1))),
         pixels: list(range(64))
         }
    )

    circ = transpile(circ_simpleQRNN, sim)
    print(circ) # matplotlib draw has problems with this circuit
    job = sim.run(circ)
    result = job.result()
    print(result.get_counts())

    num_mem_qubits = n_mem
    num_single_patch_reuploading = n_reuploading
    num_classification_layers = 1

    num_classification_qubits = 4
    num_total_qubits = num_mem_qubits + 3

    assert num_mem_qubits + 3 >= num_classification_qubits, "The number of memory qubits plus 3 must be greater than or equal to the number of classification qubits (4)"
    num_single_patch_reuploading_params = 30 * num_single_patch_reuploading  # using the fourByFourPatchReuploadPoolingClassicalCtrl1Q function
    num_mem_params = 6 * (num_mem_qubits + 1)  # using the allInOneAnsatz function
    num_classification_params = num_classification_layers * 3 * num_classification_qubits  # using the simplePQC function
    num_total_params = num_single_patch_reuploading_params + num_mem_params + num_classification_params
    num_backbone_params = num_single_patch_reuploading_params + num_mem_params

    params = ParameterVector('θ', length=num_total_params)
    inputs = ParameterVector('x', length=64)

    backbone_params = params[:num_backbone_params]
    classification_params = params[num_backbone_params:]

    qreg = QuantumRegister(num_total_qubits, name='q')
    patch_cre = ClassicalRegister(2, name='patch_classical')
    cls_creg = ClassicalRegister(num_classification_qubits, name='classification')

    circ = QuantumCircuit(qreg, patch_cre, cls_creg, name='Sampler10ClassQRNN')
    backbone = createSimpleQRNNBackbone8x8Image(inputs,
                                                backbone_params,
                                                num_single_patch_reuploading,
                                                num_mem_qubits,
                                                False,
                                                True)
    circ.append(backbone.to_instruction(), qargs=qreg, cargs=patch_cre)
    circ.append(simplePQC(num_classification_qubits, classification_params).to_instruction(),
                qargs=qreg[:num_classification_qubits])
    circ.measure(qreg[:num_classification_qubits], cls_creg[:num_classification_qubits])

    circ.draw('mpl', style='iqx', filename='simpleQRNNCls.png')






