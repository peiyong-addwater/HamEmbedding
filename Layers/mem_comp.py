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

class MemComputation(Operation):
    """
    Computation on the memory qubits.
    0- Starting with n_wire - 1 CZ gates from bottom up,
    1- then a layer of U3 gates,
    2- then a chain of non-leading SU4 gates, from bottom up,
    3- then ends with a chain of CZ gates from bottom up.
    Steps 1,2,3 are repeated L_MC times.
    total number of parameters per layer: 3 * num_wires + 9 * (num_wires-1)
    parameter shape: (..., L_MC, 3 * num_wires + 9 * (num_wires-1))
    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires, L_MC,  id=None):
        """

        :param params: shape of (..., L_MC, 3 * num_wires + 9 * (num_wires-1))
        :param wires:
        :param L_MC:
        :param id:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(params)
        if not (len(params_shape)==2 or len(params_shape)==3):
            raise ValueError(f"params must be a 2D or 3D array, got shape {params_shape}")
        if params_shape[-1] != 12 * n_wires - 9 or params_shape[-2] != L_MC:
            raise ValueError(f"params must be an array of shape (...,{L_MC}, 12 * {n_wires} - 9), got {params_shape}")

        self._hyperparameters = {"L_MC": L_MC}
        super().__init__(params, wires=wires,  id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(params, wires, L_MC):
        n_wires = len(wires)
        op_list = []
        for i in range(1, n_wires):
            op_list.append(qml.CZ(wires=[wires[-i], wires[-i-1]]))
        for i in range(L_MC):
            layer_params = params[...,i,:]
            u3_params = layer_params[..., :3 * n_wires]
            su4_params = layer_params[..., 3 * n_wires:]
            for i in range(n_wires):
                op_list.append(qml.U3(*u3_params[..., 3 * i: 3 * (i + 1)], wires=wires[i]))
            for i in range(1, n_wires):
                op_list.append(SU4(su4_params[..., 9 * (i-1): 9 * i], wires=[wires[-i], wires[-i-1]], leading_gate=False))
            for i in range(1, n_wires):
                op_list.append(qml.CZ(wires=[wires[-i], wires[-i-1]]))
        return op_list