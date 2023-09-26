import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union

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


if __name__ == '__main__':
    pixels = ParameterVector('p', 4)
    encoding_params = ParameterVector('e', 9*2)
    circ = createFourPixelReupload(pixels, encoding_params)
    circ.draw('mpl', filename='FourPixelReupload.png', style='bw')