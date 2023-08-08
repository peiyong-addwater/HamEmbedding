import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
from pennylane.operation import Operation, AnyWires
from reset_gate import ResetZeroState
import math
from four_by_four_patch_encode import FourByFourPatchReUpload
from su4 import SU4

sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

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

    def __init__(self, parmas, wires, do_queue=None, id=None):
        """

        :param parmas: shape of (..., 3 * num_wires + 9 * (num_wires-1)= 12 * num_wires - 9)
        :param wires:
        :param do_queue:
        :param id:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(parmas)
        if not (len(params_shape)==1 or len(params_shape)==2): # 2 when is batching, 1 when not
            raise ValueError(f"params must be a 1D or 2D array, got shape {params_shape}")
        if params_shape[-1] != 12 * n_wires - 9:
            raise ValueError(f"params must be an array of shape (..., 12 * {n_wires} - 9), got {params_shape}")
        super().__init__(parmas, wires=wires, do_queue=do_queue, id=id)

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

    def __init__(self, params, wires,do_queue=None, id=None):
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
        super().__init__(params, wires=wires, do_queue=do_queue, id=id)

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

    def __init__(self, params, wires, L_MC, do_queue=None, id=None):
        """

        :param params: shape of (..., L_MC, 3 * num_wires + 9 * (num_wires-1))
        :param wires:
        :param L_MC:
        :param do_queue:
        :param id:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(params)
        if not (len(params_shape)==2 or len(params_shape)==3):
            raise ValueError(f"params must be a 2D or 3D array, got shape {params_shape}")
        if params_shape[-1] != 12 * n_wires - 9 or params_shape[-2] != L_MC:
            raise ValueError(f"params must be an array of shape (...,{L_MC}, 12 * {n_wires} - 9), got {params_shape}")

        self._hyperparameters = {"L_MC": L_MC}
        super().__init__(params, wires=wires, do_queue=do_queue, id=id)

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





if __name__ == '__main__':
    import matplotlib.pyplot as plt

    qml.disable_return()  # Turn of the experimental return feature,
    # see https://docs.pennylane.ai/en/stable/code/api/pennylane.enable_return.html#pennylane.enable_return
    dev4q = qml.device('default.mixed', wires=4)
    dev5q = qml.device('default.mixed', wires=5)
    dev6q = qml.device('default.mixed', wires=6)

    mem_init_params = torch.randn( 12 * 6 - 9)
    mem_patch_interact_params = torch.randn(36)
    L_MC = 2
    mem_computation_params = torch.randn(L_MC, 12 * 6 - 9)

    @qml.qnode(dev6q)
    def mem_init(params):
        InitialiseMemState.compute_decomposition(params, wires=range(6))
        return qml.probs()

    print(mem_init(mem_init_params).shape)
    fig, ax = qml.draw_mpl(mem_init, style = 'sketch')(mem_init_params)
    fig.savefig('mem_init.png')
    plt.close(fig)

    @qml.qnode(dev4q)
    def mem_patch_interact(params):
        MemPatchInteract2to2.compute_decomposition(params, wires=range(4))
        return qml.probs()

    print(mem_patch_interact(mem_patch_interact_params).shape)
    fig, ax = qml.draw_mpl(mem_patch_interact, style = 'sketch')(mem_patch_interact_params)
    fig.savefig('mem_patch_interact.png')
    plt.close(fig)

    @qml.qnode(dev6q)
    def mem_computation(params):
        MemComputation.compute_decomposition(params, wires=range(6), L_MC=L_MC)
        return qml.probs()

    print(mem_computation(mem_computation_params).shape)
    fig, ax = qml.draw_mpl(mem_computation, style = 'sketch', expand='device')(mem_computation_params)
    fig.savefig('mem_computation.png')



