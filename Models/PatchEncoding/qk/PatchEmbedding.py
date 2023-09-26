import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union
from su4 import createTaillessSU4

QiskitParameter = Union[ParameterVector, List[Parameter], List[ParameterVectorElement]]

def createFourPixelReupload(
        pixels: QiskitParameter,
        encoding_params: QiskitParameter
)->QuantumCircuit:
    """
    Creates a 2-qubit circuit that encodes 4 pixels into 2 qubits, with trainable parameters for data re-uploading.
    Trainable layers are composed of U3 gates and IsingXX, IsingYY, and IsingZZ gates.
    Args:
        pixels: The pixel values to encode. 4-element list of parameters.
        encoding_params: The encoding parameters. 9L-element list of parameters, where L is the number of data-reuploading repetitions

    Returns:
        A 2-qubit circuit that encodes 4 pixels into 2 qubits.
    """
    circ = QuantumCircuit(2)
    layers = len(encoding_params)//9
    for i in range(layers):
        circ.rx(pixels[0], 0)
        circ.rx(pixels[1], 1)
        circ.rz(pixels[2], 0)
        circ.rz(pixels[3], 1)
        circ.barrier()
        circ.u(encoding_params[0+9*i], encoding_params[1+9*i], encoding_params[2+9*i], 0)
        circ.u(encoding_params[3+9*i], encoding_params[4+9*i], encoding_params[5+9*i], 1)
        circ.rxx(encoding_params[6+9*i], 0, 1)
        circ.ryy(encoding_params[7+9*i], 0, 1)
        circ.rzz(encoding_params[8+9*i], 0, 1)
        circ.barrier()
        circ.barrier()
    return circ

def fourByFourPatchReupload(
        pixels: QiskitParameter,
        encoding_params: QiskitParameter
)->QuantumCircuit:
    """
    Creates a 3-qubit circuit that encodes 16 pixels into 3 qubits, with trainable parameters for data re-uploading.
    The trainable layer are two TaillessSU4 layers.
    Args:
        pixels: the 16 pixels to encode. 16-element list of parameters.
        encoding_params: 18*L-element list of parameters, where L is the number of data-reuploading repetitions

    Returns:
        A 3-qubit circuit that encodes 16 pixels into 3 qubits.
    """
    circ = QuantumCircuit(3)
    layers = len(encoding_params)//18
    for i in range(layers):
        circ.u(pixels[0], pixels[1], pixels[2], 0)
        circ.u(pixels[3], pixels[4], pixels[5], 1)
        circ.u(pixels[6], pixels[7], pixels[8], 2)
        circ.rxx(pixels[9], 0, 1)
        circ.ryy(pixels[10], 0, 1)
        circ.rzz(pixels[11], 0, 1)
        circ.rxx(pixels[12], 1, 2)
        circ.ryy(pixels[13], 1, 2)
        circ.rzz(pixels[14], 1, 2)
        circ.rxx(pixels[15], 0, 2)
        circ.barrier()
        circ.append(createTaillessSU4(encoding_params[0+18*i:9+18*i]).to_instruction(), [0,1])
        circ.append(createTaillessSU4(encoding_params[9+18*i:18+18*i]).to_instruction(), [1,2])
        circ.barrier()
        circ.barrier()
    return circ


if __name__ == '__main__':
    pixel = ParameterVector('p', 4)
    encoding_param = ParameterVector('e', 9*2)
    circ = createFourPixelReupload(pixel, encoding_param)
    circ.draw('mpl', filename='FourPixelReupload.png', style='bw')

    pixel2 = ParameterVector('p', 16)
    encoding_param2 = ParameterVector('e', 18*2)
    circ2 = fourByFourPatchReupload(pixel2, encoding_param2)
    circ2.draw('mpl', filename='FourByFourPatchReupload.png', style='bw')