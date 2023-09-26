import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union
from su4 import createTaillessSU4, createHeadlessSU4, createSU4Circ
from math import pi

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
    circ = QuantumCircuit(2, name='FourPixelReupload')
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
    circ = QuantumCircuit(3, name='FourByFourPatchReupload')
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

def createPQC64(
        params: QiskitParameter
)->QuantumCircuit:
    """
    A four-qubit PQC that has 64 parameters
    Args:
        params: 64-element list of parameters

    Returns:
        A four-qubit PQC that has 64 parameters "PQC64"
    """
    circ = QuantumCircuit(4, name='PQC64')
    circ.append(createSU4Circ(params[0:15]).to_instruction(), [0,1])
    circ.append(createSU4Circ(params[15:30]).to_instruction(), [2,3])
    circ.append(createHeadlessSU4(params[30:39]).to_instruction(), [1,2])
    circ.append(createHeadlessSU4(params[39:48]).to_instruction(), [0,2])
    circ.append(createHeadlessSU4(params[48:57]).to_instruction(), [1,3])
    circ.rxx(params[57], 0, 3)
    circ.ryy(params[58], 0, 3)
    circ.rzz(params[59], 0, 3)
    circ.u(pi/2, params[60], params[61], 0)
    circ.u(pi/2, params[62], params[63], 3)
    return circ

def create8x8ReUploading(
        pixels: QiskitParameter,
        encoding_params: QiskitParameter
)->QuantumCircuit:
    """
    Creates a 4-qubit circuit that encodes 64 pixels into 4 qubits, with trainable parameters for data re-uploading.
    Then followed by a chain of TailLessSU4 gates, from bottom up: 3->2, 2->1, 1->0.
    Each TailLessSU4 gate has 9 parameters.
    Total number of trainable parameters: 27*L
    Where L is the number of re-uploading layers.
    Args:
        pixels: flattened 8 by 8 patch, 64-element list of parameters
        encoding_params: 27*L-element list of parameters, where L is the number of data-reuploading repetitions

    Returns:
        A 4-qubit circuit that encodes 64 pixels into 4 qubits.
    """
    circ = QuantumCircuit(4, name='8x8ReUploading')
    layers = len(encoding_params)//27
    for i in range(layers):
        circ.append(createPQC64(pixels[0:64]).to_instruction(), [0,1,2,3])
        circ.barrier()
        circ.append(createTaillessSU4(encoding_params[0+27*i:9+27*i]).to_instruction(), [2,3])
        circ.append(createTaillessSU4(encoding_params[9+27*i:18+27*i]).to_instruction(), [1,2])
        circ.append(createTaillessSU4(encoding_params[18+27*i:27+27*i]).to_instruction(), [0,1])
        circ.barrier()
        circ.barrier()
    return circ

def createPQC100(
        params: QiskitParameter
)->QuantumCircuit:
    """
    Creates a 5-qubit circuit that has 100 parameters.
    Args:
        params: 100-element list of parameters

    Returns:
        A 5-qubit circuit that has 100 parameters, "PQC100"
    """
    circ = QuantumCircuit(5, name='PQC100')
    circ.u(params[0], params[1], params[2], 0)
    circ.u(params[3], params[4], params[5], 1)
    circ.u(params[6], params[7], params[8], 2)
    circ.u(params[9], params[10], params[11], 3)
    circ.u(params[12], params[13], params[14], 4)
    circ.append(createHeadlessSU4(params[15:24]).to_instruction(), [0,1])
    circ.append(createHeadlessSU4(params[24:33]).to_instruction(), [2,3])
    circ.append(createHeadlessSU4(params[33:42]).to_instruction(), [1,2])
    circ.append(createHeadlessSU4(params[42:51]).to_instruction(), [3,4])
    circ.append(createHeadlessSU4(params[51:60]).to_instruction(), [0,2])
    circ.append(createHeadlessSU4(params[60:69]).to_instruction(), [1,3])
    circ.append(createHeadlessSU4(params[69:78]).to_instruction(), [2,4])
    circ.append(createHeadlessSU4(params[78:87]).to_instruction(), [0,3])
    circ.append(createHeadlessSU4(params[87:96]).to_instruction(), [1,4])
    circ.rxx(params[96], 0, 4)
    circ.ry(params[97], 0)
    circ.ry(params[98], 4)
    circ.rzz(params[99], 0, 4)
    return circ

def create10x10Reuploading(
        pixels: QiskitParameter,
        encoding_params: QiskitParameter
)->QuantumCircuit:
    """
    Creates a 5-qubit circuit that encodes 100 pixels into 5 qubits, with trainable parameters for data re-uploading.
    Then followed by a chain of TailLessSU4 gates, from bottom up: 4->3, 3->2, 2->1, 1->0.
    Each TailLessSU4 gate has 9 parameters.
    Total number of trainable parameters: 36*L
    Where L is the number of re-uploading layers.
    Args:
        pixels: 100-element list of parameters
        encoding_params: 36*L-element list of parameters, where L is the number of data-reuploading repetitions

    Returns:
        A 5-qubit circuit that encodes 100 pixels into 5 qubits.
    """
    circ = QuantumCircuit(5, name='10x10ReUploading')
    layers = len(encoding_params)//36
    for i in range(layers):
        circ.append(createPQC100(pixels[0:100]).to_instruction(), [0,1,2,3,4])
        circ.barrier()
        circ.append(createTaillessSU4(encoding_params[0+36*i:9+36*i]).to_instruction(), [3,4])
        circ.append(createTaillessSU4(encoding_params[9+36*i:18+36*i]).to_instruction(), [2,3])
        circ.append(createTaillessSU4(encoding_params[18+36*i:27+36*i]).to_instruction(), [1,2])
        circ.append(createTaillessSU4(encoding_params[27+36*i:36+36*i]).to_instruction(), [0,1])
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

    params = ParameterVector('$\\theta$', 64)
    circ3 = createPQC64(params)
    circ3.draw('mpl', filename='PQC64.png', style='bw')

    pixel4 = ParameterVector('p', 64)
    encoding_param4 = ParameterVector('e', 27*2)
    circ4 = create8x8ReUploading(pixel4, encoding_param4)
    circ4.draw('mpl', filename='8x8ReUploading.png', style='bw')

    params5 = ParameterVector('$\\theta$', 100)
    circ5 = createPQC100(params5)
    circ5.draw('mpl', filename='PQC100.png', style='bw')

    pixel6 = ParameterVector('p', 100)
    encoding_param6 = ParameterVector('e', 36*2)
    circ6 = create10x10Reuploading(pixel6, encoding_param6)
    circ6.draw('mpl', filename='10x10Reuploading.png', style='bw')

