import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

def FourQubitParameterisedLayer(
        parameters: Union[np.ndarray, pnp.array, jnp.ndarray],
        wires: Union[List[int], Wires]
):
    """
    A generic four-qubit parameterised layer composed of RX, RZ and CZ gates.
    :param parameters: 12N-element array, one-dim, where N is the number of layers
    :param wires: Four wires that the layer is applied to
    :return:
    """
    layers = len(parameters) // 12
    for i in range(layers):
        for j in range(4):
            qml.U3(parameters[12 * i + 3 * j], parameters[12 * i + 3 * j + 1], parameters[12 * i + 3 * j + 2], wires=wires[j])
        qml.CZ(wires=[wires[0], wires[1]])
        qml.CZ(wires=[wires[1], wires[2]])
        qml.CZ(wires=[wires[2], wires[3]])
        qml.CZ(wires=[wires[3], wires[0]])


def PermutationInvariantFourQLayer(
        parameters: Union[np.ndarray, pnp.array, jnp.ndarray],
        wires: Union[List[int], Wires]
):
    """
    A permutation invariant four-qubit parameterised layer composed of RX, RY and IsingZZ gates.
    :param parameters: 3N-element array, one-dim, where N is the number of layers
    :param wires:
    :return:
    """
    layers = len(parameters) // 3
    for i in range(layers):
        for j in range(4):
            qml.RX(parameters[3 * i], wires=wires[j])
            qml.RY(parameters[3 * i + 1], wires=wires[j])
        qml.IsingZZ(parameters[3 * i + 2], wires=[wires[0], wires[1]])
        qml.IsingZZ(parameters[3 * i + 2], wires=[wires[1], wires[2]])
        qml.IsingZZ(parameters[3 * i + 2], wires=[wires[2], wires[3]])
        qml.IsingZZ(parameters[3 * i + 2], wires=[wires[3], wires[0]])
        qml.IsingZZ(parameters[3 * i + 2], wires=[wires[0], wires[2]])
        qml.IsingZZ(parameters[3 * i + 2], wires=[wires[1], wires[3]])

def QFTTokenMixing(
        wires: Union[List[int], Wires]
):
    """
    Token mixing with QFT, mimicking the token mixing in the paper "FNet: Mixing Tokens with Fourier Transforms"
    :param wires:
    :return:
    """
    qml.QFT(wires=wires)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    qml.drawer.use_style("black_white")

    wires = Wires([0, 1, 2, 3])
    dev = qml.device("default.qubit", wires=wires)

    generic_layer_param = np.random.randn(12)
    pv_layer_param = np.random.randn(3)

    @qml.qnode(dev)
    def circuit():
        FourQubitParameterisedLayer(generic_layer_param, wires=wires)
        qml.Barrier()
        PermutationInvariantFourQLayer(pv_layer_param, wires=wires)
        qml.Barrier()
        QFTTokenMixing(wires=wires)
        qml.Barrier()
        return qml.state()

    fig, ax = qml.draw_mpl(circuit)()
    plt.savefig('different_four_qubit_token_mixing_layers.png')
    plt.close(fig)
