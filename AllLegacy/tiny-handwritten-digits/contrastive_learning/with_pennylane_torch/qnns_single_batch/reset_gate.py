import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
from pennylane.operation import Operation, AnyWires, Channel

class ResetZeroState(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, wires):
        super().__init__(wires=wires)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(wires):
        return [qml.ResetError(p0=1, p1=0, wires = wire) for wire in wires]