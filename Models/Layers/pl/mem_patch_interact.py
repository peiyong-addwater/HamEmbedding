import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
from pennylane.operation import Operation, AnyWires
from .SU4 import SU4, TailLessSU4

# sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

class MemPatchInteract2to2(Operation):
    """
    Interaction between two of the memory qubits and two patch feature qubits,
    Composed of non-leading SU4 gates.
    Let the first two qubits be the memory qubits, and the last two be the patch qubits.
    Interaction map: (0,2), (1,3), (0,3), (1,2)
    Meaning there will be 4 SU4 gates.-> 4*9 = 36 parameters
    After the SU4 gates, there are also four CZ gates following the interaction map.
    """
    num_wires = 4
    grad_method = None

    def __init__(self, params, wires,id=None):
        """

        :param params:
        :param wires:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(params)
        if not (len(params_shape)==1 or len(params_shape)==2): # 2 when is batching, 1 when not
            raise ValueError(f"params must be a 1D or 2D array, got shape {params_shape}")
        if n_wires != 4:
            raise ValueError(f"num_wires must be 4, got {n_wires}")
        if params_shape[-1] != 36:
            raise ValueError(f"params must be an array of shape (..., 36), got {params_shape}")
        super().__init__(params, wires=wires,  id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(params, wires):
        op_list = []
        op_list.append(SU4(params[..., :9], wires=[wires[0], wires[2]], leading_gate=False))
        op_list.append(SU4(params[..., 9:18], wires=[wires[1], wires[3]], leading_gate=False))
        op_list.append(SU4(params[..., 18:27], wires=[wires[0], wires[3]], leading_gate=False))
        op_list.append(SU4(params[..., 27:36], wires=[wires[1], wires[2]], leading_gate=False))
        op_list.append(qml.CZ(wires=[wires[0], wires[2]]))
        op_list.append(qml.CZ(wires=[wires[1], wires[3]]))
        op_list.append(qml.CZ(wires=[wires[0], wires[3]]))
        op_list.append(qml.CZ(wires=[wires[1], wires[2]]))
        return op_list
