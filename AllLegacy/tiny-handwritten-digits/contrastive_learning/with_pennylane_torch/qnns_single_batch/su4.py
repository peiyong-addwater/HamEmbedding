import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
from pennylane.operation import Operation, AnyWires
from .reset_gate import ResetZeroState

# sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

class SU4(Operation):
    """
    SU4 gate.
    If the gate is leading gate (15 params), it will start with U3 gates, otherwise (9 params) it will start with the entangling gate.
    """
    num_wires = 2
    grad_method = None

    def __init__(self, weights, wires, leading_gate = True, do_queue = None, id=None):
        # interface = qml.math.get_interface(weights)


        self._hyperparameters = {"leading_gate": leading_gate}

        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weights, wires, leading_gate):
        op_list = []
        if leading_gate:
            op_list.append(qml.U3(weights[...,0], weights[...,1], weights[...,2], wires=wires[0]))
            op_list.append(qml.U3(weights[...,3], weights[...,4], weights[...,5], wires=wires[1]))
            op_list.append(qml.IsingXX(weights[...,6], wires=[wires[0], wires[1]]))
            op_list.append(qml.IsingYY(weights[...,7], wires=[wires[0], wires[1]]))
            op_list.append(qml.IsingZZ(weights[...,8], wires=[wires[0], wires[1]]))
            op_list.append(qml.U3(weights[...,9], weights[...,10], weights[...,11], wires=wires[0]))
            op_list.append(qml.U3(weights[...,12], weights[...,13], weights[...,14], wires=wires[1]))
        else:
            op_list.append(qml.IsingXX(weights[...,0], wires=[wires[0], wires[1]]))
            op_list.append(qml.IsingYY(weights[...,1], wires=[wires[0], wires[1]]))
            op_list.append(qml.IsingZZ(weights[...,2], wires=[wires[0], wires[1]]))
            op_list.append(qml.U3(weights[...,3], weights[...,4], weights[...,5], wires=wires[0]))
            op_list.append(qml.U3(weights[...,6], weights[...,7], weights[...,8], wires=wires[1]))
        return op_list

