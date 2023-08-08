import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys

sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

def FourPixelReUploading(
        pixels: torch.Tensor,
        encode_params: torch.Tensor,
        L1: int,
        wires: Union[Wires, List[int]]
):
    """
    Data re-uploading layer for four pixels. Pixel data is encoded as rotation parameters of RX and RY gates;
    Entanglement is achieved by CZ gates.
    Trainable parameters are the rotation parameters of Rot gates
    :param pixels: input pixel data. Four element array
    :param encode_params: trainable parameters for encoding. 6*L1 element array
    :param L1: number of data re-uploading layers
    :param wires: Qubits to be used. Number of qubits: 2
    :return:
    """
    assert len(wires) == 2 # Number of qubits must be 2
    for i in range(L1):
        qml.RX(pixels[0], wires=wires[0])
        qml.RY(pixels[1], wires=wires[0])
        qml.RX(pixels[2], wires=wires[1])
        qml.RY(pixels[3], wires=wires[1])
        qml.CZ(wires=[wires[0], wires[1]])
        qml.Rot(encode_params[6*i], encode_params[6*i+1], encode_params[6*i+2], wires=wires[0])
        qml.Rot(encode_params[6*i+3], encode_params[6*i+4], encode_params[6*i+5], wires=wires[1])
        qml.CZ(wires=[wires[0], wires[1]])
        qml.Barrier()


if __name__ == '__main__':
    dev2q = qml.device('default.qubit', wires=2)
    dev4q = qml.device('default.qubit', wires=4)
    @qml.qnode(dev2q, interface='torch')
    def four_pixel_encode(pixels, encode_params, L1):
        FourPixelReUploading(pixels, encode_params, L1, wires=[0, 1])
        return qml.probs()

    pixels = torch.randn(4,5)
    L1 = 2
    encode_params = torch.randn(L1*6)

    print(four_pixel_encode(pixels, encode_params, L1))
    fig, ax = qml.draw_mpl(four_pixel_encode, style='sketch')(pixels[:,0], encode_params, L1)
    fig.savefig('four_pixel_reupload.png')