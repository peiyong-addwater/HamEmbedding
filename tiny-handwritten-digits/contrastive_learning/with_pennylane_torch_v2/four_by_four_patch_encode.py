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
        loc: torch.Tensor,
        wires: Union[Wires, List[int]],
):
    #TODO: change to 18-dim data input while the last two elements are the position encoding.
    """
    This function uses a modified SU(4) gate to encode 16 pixels into 3 qubits
    Since we are dealing with 8 by 8 images and we want to encode 4 by 4 patches,
    the location tensor is a 2D tensor of shape (batchsize, 2),
    the location
    :param data: (batchsize, 16) tensor, containing the pixel values
    :param loc:  (batchsize, 2) tensor, containing the location of the patch.
    :param wires:
    :return:
    """
    assert len(wires) == 3