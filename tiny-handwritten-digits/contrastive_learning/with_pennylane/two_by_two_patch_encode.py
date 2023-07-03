import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys

sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')


def FourPixelEncodeTwoQubits(
        pixels: Union[list, np.ndarray, pnp.array],
        parameters: Union[List, np.ndarray, pnp.array],
        wires: Union[List[int], Wires]
):
    """
    :param pixels: 4-element array
    :param parameters: 6N-element array, one-dim, where N is the number of "layers" of data-reuploading
    :return:
    """
    n_layers = len(parameters) // 6
    for i in range(n_layers):
        qml.RX(pixels[0], wires=wires[0])
        qml.RY(pixels[1], wires=wires[0])
        qml.RX(pixels[2], wires=wires[1])
        qml.RY(pixels[3], wires=wires[1])
        qml.CZ(wires=[wires[0], wires[1]])
        qml.U3(parameters[6 * i], parameters[6 * i + 1], parameters[6 * i + 2], wires=wires[0])
        qml.U3(parameters[6 * i + 3], parameters[6 * i + 4], parameters[6 * i + 5], wires=wires[1])
        qml.CZ(wires=[wires[0], wires[1]])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    qml.drawer.use_style("black_white")

    x = np.random.randn(4)
    theta = np.random.randn(12)
    dev = qml.device("default.qubit", wires=2)
    wires = Wires([0, 1])


    @qml.qnode(dev)
    def circuit(x, theta):
        FourPixelEncodeTwoQubits(x, theta, wires=wires)
        return qml.state()


    fig, ax = qml.draw_mpl(circuit)(x, theta)
    plt.savefig("four_pixel_two_qubits_encode.png")
    plt.close(fig)
