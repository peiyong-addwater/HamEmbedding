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
        encode_parameters: Union[jnp.ndarray, np.ndarray, pnp.ndarray],
        phase_parameters: Union[jnp.ndarray, np.ndarray, pnp.ndarray],
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
    qml.Hadamard(wires[0])
    for i in range(n_layers):
        FourPixelEncodeTwoQubits(pixels, encode_parameters[6 * i: 6 * (i + 1)], wires=[wires[1], wires[2]])
        qml.CPhase(phase_parameters[i], wires=[wires[0], wires[1]])
        qml.adjoint(FourPixelEncodeTwoQubits)(pixels, encode_parameters[6 * i: 6 * (i + 1)], wires=[wires[1], wires[2]])

def FourPixelDepositAndReverseFixed(
        pixels: Union[list, np.ndarray, pnp.array],
        encode_parameters: Union[jnp.ndarray, np.ndarray, pnp.ndarray],
        phase_parameters: Union[jnp.ndarray, np.ndarray, pnp.ndarray],
        wires: Union[List[int], Wires]
):
    """
    A fixed size version of the single patch D&R. There are two D&R in total. Each D&R contains N repetitions of encoding
    :param pixels: 4-element array
    :param encode_parameters: 6N*2 element array
    :param phase_parameters: 2 element array
    :param wires:
    :return:
    """
    n_encode_params = len(encode_parameters)
    assert n_encode_params/2 == n_encode_params//2
    assert n_encode_params//12 == n_encode_params/12
    first_half_params = encode_parameters[:n_encode_params//2]
    second_half_params = encode_parameters[n_encode_params//2:]
    qml.Hadamard(wires[0])
    FourPixelEncodeTwoQubits(pixels, first_half_params, wires=[wires[1], wires[2]])
    qml.CPhase(phase_parameters[0], wires=[wires[0], wires[1]])
    qml.adjoint(FourPixelEncodeTwoQubits)(pixels, first_half_params, wires=[wires[1], wires[2]])
    qml.Barrier()
    # an X gate on the deposit qubit swap 0 and 1, making room for the new phase encoding with different parameters
    qml.PauliX(wires=wires[0])
    qml.Barrier()
    FourPixelEncodeTwoQubits(pixels, second_half_params, wires=[wires[1], wires[2]])
    qml.CPhase(phase_parameters[1], wires=[wires[0], wires[1]])
    qml.adjoint(FourPixelEncodeTwoQubits)(pixels, second_half_params, wires=[wires[1], wires[2]])




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    qml.drawer.use_style("black_white")

    x = pnp.random.randn(4)
    theta = pnp.random.randn(12)
    phi = pnp.random.randn(2)
    wires = Wires([0, 1, 2])
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit(x, theta, phi):
        FourPixelDepositAndReverse(x, theta, phi, wires=wires)
        return qml.state()

    fig, ax = qml.draw_mpl(circuit)(x, theta, phi)
    plt.savefig("four_pixel_deposit_and_reverse.png")
    plt.close(fig)


    @qml.qnode(dev)
    def circuit(x, theta, phi):
        FourPixelDepositAndReverseFixed(x, theta, phi, wires=wires)
        return qml.state()


    fig, ax = qml.draw_mpl(circuit)(x, theta, phi)
    plt.savefig("four_pixel_deposit_and_reverse_fixed_d_and_p_layers.png")
    plt.close(fig)
