import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from two_by_two_patch_encode import FourPixelEncodeTwoQubits
from reset_gate import Reset0

def FourPixelOneQubit(
        pixels: torch.Tensor,
        encode_parameters: torch.Tensor,
        phase_parameters: torch.Tensor,
        wires: Union[List[int], Wires]
):
    """

    :param pixels: 4-element array
    :param encode_parameters: 6N*2 element array, where N is the number of data re-uploading repetitions
    :param phase_parameters: 2 element array
    :param wires: 3 qubits
    :return:
    """
    n_encode_params = len(encode_parameters)
    assert n_encode_params / 2 == n_encode_params // 2
    assert n_encode_params // 12 == n_encode_params / 12
    first_half_params = encode_parameters[:n_encode_params // 2]
    second_half_params = encode_parameters[n_encode_params // 2:]
    qml.Hadamard(wires[0])
    FourPixelEncodeTwoQubits(pixels, first_half_params, wires=[wires[1], wires[2]])
    qml.CPhase(phase_parameters[0], wires=[wires[0], wires[1]])
    Reset0(wires=[wires[1], wires[2]])
    qml.Barrier()
    # an X gate on the deposit qubit swap 0 and 1, making room for the new phase encoding with different parameters
    qml.PauliX(wires=wires[0])
    qml.Barrier()
    FourPixelEncodeTwoQubits(pixels, second_half_params, wires=[wires[1], wires[2]])
    qml.CPhase(phase_parameters[1], wires=[wires[0], wires[1]])
    Reset0(wires=[wires[1], wires[2]])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    qml.drawer.use_style("black_white")

    x = torch.randn(4)
    theta = torch.randn(12)
    phi = torch.randn(2)
    wires = Wires([0, 1, 2])
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev, interface="torch")
    def circuit(x, theta, phi):
        FourPixelOneQubit(x, theta, phi, wires=wires)
        return qml.state()

    fig, ax = qml.draw_mpl(circuit)(x, theta, phi)
    plt.savefig("four_pixel_one_qubit.png")
    plt.close(fig)