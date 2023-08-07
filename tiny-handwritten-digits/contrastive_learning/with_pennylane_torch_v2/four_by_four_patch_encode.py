import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys

sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

def FourByFourPatchDataCirc(
        data: torch.Tensor,
        wires: Union[Wires, List[int]],
):
    """
    This function uses a modified SU(4) gate to encode 16 pixels into 3 qubits
    Since we are dealing with 8 by 8 images and we want to encode 4 by 4 patches,
    plus the positional encoding, which is a two-element vector with elements in {0,1},
    for example, the patch on the upper left corner will have positional encoding (0,0),
    which will be transformed into (0*pi/2, 0*pi/2) = (0,0) when acting as rotation angles of the gates.
    then the input data dimension is (batchsize, 4x4+2) = (batchsize, 18)
    :param data: (batchsize, 18) tensor, containing the pixel values in the first 16 elements, the last two elements contain the positional encoding.
    :param wires:
    :return:
    """
    assert len(wires) == 3
    assert data.shape[1] == 18