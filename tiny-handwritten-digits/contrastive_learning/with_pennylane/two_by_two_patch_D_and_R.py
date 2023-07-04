import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from two_by_two_patch_encode import FourPixelEncodeTwoQubits

def FourPixelDepositAndReverse(
        pixels: Union[list, np.ndarray, pnp.array],
        encode_parameters: Union[jnp.ndarray, np.ndarray, pnp.array],
        phase_parameters: Union[jnp.ndarray, np.ndarray, pnp.array],
        wires: Union[List[int], Wires]
):
    """
    :param pixels: 4-element array
    :param encode_parameters: 6NL-element array, one-dim, where N is the number of "layers" of data-reuploading and L is the number of D&R repetitions
    :param phase_parameters: L-element array
    :param wires: the wires that the circuit is applied to. Must be of length 3
    :return:
    """
    n_layers = len(encode_parameters) // 6
    for i in range(n_layers):
        FourPixelEncodeTwoQubits(pixels, encode_parameters[6 * i: 6 * (i + 1)], wires=[wires[1], wires[2]])
        qml.CPhase(phase_parameters[i], wires=[wires[0], wires[1]])
        qml.adjoint(FourPixelEncodeTwoQubits)(pixels, encode_parameters[6 * i: 6 * (i + 1)], wires=[wires[1], wires[2]])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    qml.drawer.use_style("black_white")

    x = np.random.randn(4)
    theta = np.random.randn(12)
    phi = np.random.randn(3)
    wires = Wires([0, 1, 2])
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit(x, theta, phi):
        FourPixelDepositAndReverse(x, theta, phi, wires=wires)
        return qml.state()

    fig, ax = qml.draw_mpl(circuit)(x, theta, phi)
    plt.savefig("four_pixel_deposit_and_reverse.png")
    plt.close(fig)

