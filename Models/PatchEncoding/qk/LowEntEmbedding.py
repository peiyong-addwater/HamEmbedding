import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union
from math import pi

#from .su4 import createTaillessSU4, createHeadlessSU4, createSU4Circ

QiskitParameter = Union[ParameterVector, List[Parameter], List[ParameterVectorElement]]

def pqcUCU(
        params:QiskitParameter,
        num_qubits: int
)->QuantumCircuit:
    """
    Creates a PQC with U3 gates and CU3 gates. First there are U3 gates on each of the qubits,
    Then there are CU3 gates from bottom to top: 3->2, 2->1, 1->0.
    Args:
        params: (3*num_qubits + 4*(num_qubits-1))L-element list of parameters, where the number of
                layers L is inferred from the number of parameters.
        num_qubits: number of qubits in the circuit

    Returns:
        A PQC with U3 gates and CU3 gates.
    """
    num_params = len(params)
    num_layers = num_params//(3*num_qubits + 4*(num_qubits-1))
    assert num_params == num_layers*(3*num_qubits + 4*(num_qubits-1)), f"Number of parameters must be a multiple of 3*{num_qubits} + 3*({num_qubits}-1)"
    assert num_qubits >= 2, f"Number of qubits must be at least 2"
    assert num_layers >= 1, f"Number of layers must be at least 1"
    circ = QuantumCircuit(num_qubits, name='PQCUCU')
    for i in range(num_layers):
        layer_params = params[(3*num_qubits + 4*(num_qubits-1))*i:(3*num_qubits + 4*(num_qubits-1))*(i+1)]
        for j in range(num_qubits):
            circ.u(layer_params[3*j], layer_params[3*j+1], layer_params[3*j+2], j)
        for j in range(num_qubits-1):
            # inverse ordering
            circ.cu(layer_params[3*num_qubits+4*j], layer_params[3*num_qubits+4*j+1], layer_params[3*num_qubits+4*j+2],layer_params[3*num_qubits+4*j+3], num_qubits-1-j, num_qubits-2-j)
        circ.barrier()
    return circ

def fourByFourPatchReUploading4QCircResetPooling1Q(
        pixels: QiskitParameter,
        encoding_params: QiskitParameter,
)->QuantumCircuit:
    """
    Creates a four-qubit circuit that encodes 16 pixels into 1 qubit,
    with trainable parameters for data re-uploading.
    The pooling is achieved through resetting the bottom three qubits between re-uploading layers.
    After resetting, the information on the first qubit is spread to the other two qubits through bit-flip QEC style encoding.
    The trainable layer is a single pqcUCU layer for each data re-uploading repetition.
    Args:
        pixels: the 16 pixels to encode. 16-element list of parameters.
        encoding_params: (3*4+4*(4-1))*L = 24L-element list of parameters, where L is the number of data-reuploading repetitions

    Returns:
        A 4-qubit circuit that encodes 16 pixels into a 1-qubit state.
    """
    num_single_layer_encoding_params = 3*4 + 4*(4-1)

    circ = QuantumCircuit(4, name='FourByFourPatch4QReuploadResetPooling1Q')
    layers = len(encoding_params) // num_single_layer_encoding_params
    assert len(
        encoding_params) == layers * num_single_layer_encoding_params, f"Number of encoding parameters must be a multiple of {num_single_layer_encoding_params}"
    assert len(pixels) == 16, f"Number of pixels must be 16"

    for i in range(4):
        circ.h(i)

    for i in range(layers):
        layer_i_params = encoding_params[
                         num_single_layer_encoding_params * i:num_single_layer_encoding_params * (i + 1)]
        # first 8 pixels
        circ.u(pixels[0], pixels[1], pixels[2], 0)
        circ.u(pixels[3], pixels[4], pixels[5], 1)
        circ.rxx(pixels[6], 0, 1)
        circ.rzz(pixels[7], 0, 1)
        # second 8 pixels
        circ.u(pixels[8], pixels[9], pixels[10], 2)
        circ.u(pixels[11], pixels[12], pixels[13], 3)
        circ.rxx(pixels[14], 2, 3)
        circ.rzz(pixels[15], 2, 3)
        circ.barrier()
        # parameterized layer
        circ.append(pqcUCU(layer_i_params, 4).to_instruction(), [0, 1, 2, 3])
        circ.barrier()
        circ.reset(3)
        circ.reset(2)
        circ.reset(1)
        circ.barrier()

        if i < layers - 1:
            circ.h(1)
            circ.h(2)
            circ.h(3)

        circ.barrier()

    return circ

if __name__ == '__main__':
    pixels = ParameterVector('x', 16)
    encoding_params = ParameterVector('θ', 24*2)
    circ = fourByFourPatchReUploading4QCircResetPooling1Q(pixels, encoding_params)
    circ.draw('mpl', filename='FourByFourPatch4QReuploadResetPooling1Q.png', style='iqx')