import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
from pennylane.operation import Operation, AnyWires
from reset_gate import ResetZeroState

sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

class SU4(Operation):
    """
    SU4 gate.
    If the gate is leading gate (15 params), it will start with U3 gates, otherwise (9 params) it will start with the entangling gate.
    """
    num_wires = 2
    grad_method = None

    def __init__(self, weights, wires, leading_gate = True, do_queue = None, id=None):
        interface = qml.math.get_interface(weights)
        shape = qml.math.shape(weights)
        if not (len(shape)==1 or len(shape)==2): # 2 is when batching, 1 is not batching
            raise ValueError("Weights tensor must be 1D or 2D.")
        if shape[-1] != 15 and shape[-1] != 9:
            raise ValueError("Weights tensor must have 15 or 9 elements.")
        if not ((shape[-1] == 15 and leading_gate==True) or (shape[-1] == 9 and leading_gate==False)):
            raise ValueError("Weights tensor must have 15 elements if leading_gate is True, or 9 elements if leading_gate is False.")

        self._hyperparameters = {"leading_gate": leading_gate}
