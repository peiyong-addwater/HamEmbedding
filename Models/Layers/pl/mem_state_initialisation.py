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

class InitialiseMemState(Operation):
    """
    Initial state of the memory qubits. Trainable.
    Start with a layer of H gates,
    then a layer of U3 gates,
    then a chain of non-leading SU4 gates.
    Total number of parameters: 3 * num_wires + 9 * (num_wires-1)
    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, parmas, wires, id=None):
        """

        :param parmas: shape of (..., 3 * num_wires + 9 * (num_wires-1)= 12 * num_wires - 9)
        :param wires:
        :param id:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(parmas)
        if not (len(params_shape)==1 or len(params_shape)==2): # 2 when is batching, 1 when not
            raise ValueError(f"params must be a 1D or 2D array, got shape {params_shape}")
        if params_shape[-1] != 12 * n_wires - 9:
            raise ValueError(f"params must be an array of shape (..., 12 * {n_wires} - 9), got {params_shape}")
        super().__init__(parmas, wires=wires,  id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(params, wires):
        n_wires = len(wires)
        u3_params = params[..., :3 * n_wires]
        su4_params = params[..., 3 * n_wires:]
        op_list = []
        for wire in wires:
            op_list.append(qml.Hadamard(wire))

        for i in range(n_wires):
            op_list.append(qml.U3(*u3_params[..., 3 * i: 3 * (i + 1)], wires=wires[i]))

        for i in range(n_wires - 1):
            op_list.append(SU4(su4_params[..., 9 * i: 9 * (i + 1)], wires=[wires[i], wires[i + 1]], leading_gate = False))

        return op_list