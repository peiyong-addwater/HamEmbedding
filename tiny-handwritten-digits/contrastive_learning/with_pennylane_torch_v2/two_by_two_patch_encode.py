import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys

sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

def FourPixelsOnTwoQubits(
        pixels: torch.Tensor,
        parameters: torch.Tensor,
        wires:Union[List[int], Wires]
):
    """

    :param pixels:
    :param parameters:
    :param wires:
    :return:
    """